"""FastAPI application exposing /predict and /health endpoints."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Attributes:
        features: A list of feature dicts, one per sample.
    """

    features: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of feature dicts, one per sample.",
        examples=[[{"mean_radius": 14.0, "mean_texture": 19.0}]],
    )

    @model_validator(mode="after")
    def check_consistent_keys(self) -> "PredictRequest":
        """Ensure all feature dicts share the same set of keys.

        Returns:
            self if valid.

        Raises:
            ValueError: If feature dicts have inconsistent keys.
        """
        if len(self.features) > 1:
            first_keys = set(self.features[0].keys())
            for i, sample in enumerate(self.features[1:], start=1):
                if set(sample.keys()) != first_keys:
                    raise ValueError(
                        f"Feature keys of sample {i} differ from sample 0. "
                        f"Expected: {sorted(first_keys)}, got: {sorted(sample.keys())}"
                    )
        return self


class PredictionResult(BaseModel):
    """Single prediction result.

    Attributes:
        prediction: Hard class prediction.
        probabilities: Optional dict mapping class name to probability.
    """

    prediction: int | str
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    """Response body for POST /predict.

    Attributes:
        predictions: List of prediction results.
        model_name: Name of the model that produced the predictions.
        model_version: Version string from the MLflow registry.
    """

    predictions: list[PredictionResult]
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Response body for GET /health.

    Attributes:
        status: ``"ok"`` if the service is healthy.
        model_loaded: Whether a model is currently loaded.
        model_name: Name of the loaded model.
        model_version: Version of the loaded model.
    """

    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class ModelRegistry:
    """Holds the loaded MLflow pyfunc model and its metadata.

    Args:
        tracking_uri: MLflow tracking server URI.
        model_name: Registered model name.
        model_stage: Registry stage to load.
    """

    def __init__(self, tracking_uri: str, model_name: str, model_stage: str = "Production") -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_stage = model_stage
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.model_version: str = "unknown"

    def load(self) -> None:
        """Load the model from the MLflow registry.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        logger.info("Loading model from %s", model_uri)
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
            self.model_version = versions[0].version if versions else "unknown"
            logger.info("Loaded model '%s' version %s", self.model_name, self.model_version)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Model loading failed: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""
        return self.model is not None


def _build_registry() -> ModelRegistry:
    """Construct a ModelRegistry from environment variables."""
    return ModelRegistry(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        model_name=os.environ.get("MODEL_NAME", "ml-project-model"),
        model_stage=os.environ.get("MODEL_STAGE", "Production"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup; release on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI once startup is complete.
    """
    registry: ModelRegistry = app.state.registry  # type: ignore[attr-defined]
    try:
        registry.load()
    except RuntimeError as exc:
        logger.warning("Startup model load failed (continuing): %s", exc)
    yield
    logger.info("Shutting down serving API.")


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    """Construct and configure the FastAPI application.

    Args:
        registry: Optional pre-built ModelRegistry.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="ML Project Template — Prediction API",
        version="0.1.0",
        description="Serve predictions from a model registered in MLflow.",
        lifespan=lifespan,
    )
    app.state.registry = registry or _build_registry()
    return app


app = create_app()


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["ops"])
async def health() -> HealthResponse:
    """Return service health and model load status."""
    registry: ModelRegistry = app.state.registry
    return HealthResponse(
        status="ok",
        model_loaded=registry.is_loaded,
        model_name=registry.model_name,
        model_version=registry.model_version,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch prediction",
    tags=["inference"],
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Run inference on a batch of feature dicts.

    Args:
        request: PredictRequest containing a list of feature dicts.

    Returns:
        PredictResponse with one PredictionResult per input sample.

    Raises:
        HTTPException 503: If no model is currently loaded.
        HTTPException 422: If input features are malformed.
        HTTPException 500: If inference raises an unexpected error.
    """
    registry: ModelRegistry = app.state.registry

    if not registry.is_loaded or registry.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check server logs.",
        )

    try:
        input_df = pd.DataFrame(request.features)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not construct DataFrame from features: {exc}",
        ) from exc

    try:
        raw_preds = registry.model.predict(input_df)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    probabilities: list[dict[str, float] | None] = [None] * len(request.features)
    try:
        underlying = registry.model._model_impl  # noqa: SLF001
        if hasattr(underlying, "predict_proba"):
            proba_arr: np.ndarray = underlying.predict_proba(input_df)
            classes = [str(c) for c in underlying.classes_]
            probabilities = [dict(zip(classes, row.tolist())) for row in proba_arr]
    except Exception:  # noqa: BLE001
        pass

    results = [
        PredictionResult(prediction=pred, probabilities=prob)
        for pred, prob in zip(raw_preds.tolist(), probabilities)
    ]

    return PredictResponse(
        predictions=results,
        model_name=registry.model_name,
        model_version=registry.model_version,
    )
