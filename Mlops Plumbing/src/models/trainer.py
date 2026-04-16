"""Training orchestration with MLflow experiment tracking and early stopping."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.features.engineering import build_feature_pipeline, infer_column_types
from src.models.base import BaseModel

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class SklearnModel(BaseModel):
    """Wraps an sklearn estimator to implement the BaseModel interface.

    Args:
        name: Model identifier used in logging and MLflow.
        params: Hyperparameter dict.
        estimator: A pre-constructed sklearn estimator.
    """

    def __init__(self, name: str, params: dict[str, Any], estimator: Any = None) -> None:
        super().__init__(name=name, params=params)
        self.estimator = estimator

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SklearnModel":
        """Train the estimator.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            self.
        """
        if self.estimator is None:
            raise RuntimeError("No estimator set.")
        logger.info("Fitting %s on %d samples…", self.name, len(X_train))
        self.estimator.fit(X_train, y_train)
        self._is_fitted = True
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            logger.info("Validation metrics: %s", val_metrics)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class labels.
        """
        return self.estimator.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, n_classes).
        """
        return self.estimator.predict_proba(X)  # type: ignore[no-any-return]

    def save(self, path: Path) -> None:
        """Persist the estimator via joblib.

        Args:
            path: Destination path.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SklearnModel":
        """Load a joblib-serialised estimator.

        Args:
            path: Path to the .joblib file.

        Returns:
            SklearnModel instance with loaded estimator.
        """
        import joblib

        estimator = joblib.load(path)
        instance = cls(name=type(estimator).__name__, params={}, estimator=estimator)
        instance._is_fitted = True
        return instance


def build_model(cfg: DictConfig) -> SklearnModel:
    """Instantiate the correct sklearn estimator from a Hydra config.

    Args:
        cfg: Full project config.

    Returns:
        Unfitted SklearnModel.

    Raises:
        ValueError: If ``cfg.model.type`` is not recognised.
    """
    model_type = cfg.model.type
    seed = cfg.project.seed

    if model_type == "random_forest":
        p = cfg.model.random_forest
        estimator = RandomForestClassifier(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            min_samples_split=p.min_samples_split,
            min_samples_leaf=p.min_samples_leaf,
            n_jobs=p.n_jobs,
            random_state=seed,
        )
    elif model_type == "gradient_boosting":
        p = cfg.model.gradient_boosting
        estimator = GradientBoostingClassifier(
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            subsample=p.subsample,
            random_state=seed,
        )
    elif model_type == "logistic_regression":
        p = cfg.model.logistic_regression
        estimator = LogisticRegression(
            C=p.C, max_iter=p.max_iter, solver=p.solver, random_state=seed
        )
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            "Choose from: random_forest, gradient_boosting, logistic_regression"
        )

    return SklearnModel(name=model_type, params=dict(cfg.model[model_type]), estimator=estimator)


class Trainer:
    """Orchestrates the full train → evaluate → log → register workflow.

    Args:
        cfg: Hydra/OmegaConf config object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking and registry URIs from config."""
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", self.cfg.mlflow.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg.project.experiment_name)
        logger.info("MLflow tracking URI: %s", tracking_uri)

    def _log_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, split: str = "val"
    ) -> None:
        """Generate and log a confusion matrix plot as an MLflow artifact.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions.
            split: Name of the data split.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title(f"Confusion Matrix — {split}")
        artifact_path = f"confusion_matrix_{split}.png"
        fig.savefig(artifact_path, bbox_inches="tight")
        mlflow.log_artifact(artifact_path)
        plt.close(fig)
        try:
            Path(artifact_path).unlink()
        except OSError:
            pass

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[SklearnModel, dict[str, float]]:
        """Execute the full training and evaluation pipeline.

        Args:
            train_df: Training DataFrame (features + target column).
            val_df: Validation DataFrame.
            test_df: Test DataFrame.

        Returns:
            Tuple of (fitted_model, test_metrics_dict).
        """
        target_col = self.cfg.data.target_column
        feature_cols = list(self.cfg.data.feature_columns) or None

        if feature_cols:
            X_train_raw = train_df[feature_cols]
            X_val_raw = val_df[feature_cols]
            X_test_raw = test_df[feature_cols]
        else:
            X_train_raw = train_df.drop(columns=[target_col])
            X_val_raw = val_df.drop(columns=[target_col])
            X_test_raw = test_df.drop(columns=[target_col])

        y_train = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        numeric_cols, categorical_cols = infer_column_types(X_train_raw)
        feature_pipeline = build_feature_pipeline(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            scaler_name=self.cfg.preprocessing.scaler,
            imputer_strategy=self.cfg.preprocessing.imputer_strategy,
            max_categories=self.cfg.preprocessing.max_categories,
        )

        logger.info("Fitting feature pipeline…")
        X_train = feature_pipeline.fit_transform(X_train_raw)
        X_val = feature_pipeline.transform(X_val_raw)
        X_test = feature_pipeline.transform(X_test_raw)

        model = build_model(self.cfg)

        with mlflow.start_run(run_name=f"{model.name}_{int(time.time())}"):
            mlflow.log_params(model.params)
            mlflow.log_params(
                {
                    "scaler": self.cfg.preprocessing.scaler,
                    "imputer_strategy": self.cfg.preprocessing.imputer_strategy,
                    "model_type": self.cfg.model.type,
                    "seed": self.cfg.project.seed,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                }
            )
            mlflow.sklearn.autolog(log_input_examples=False)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            val_metrics = model.evaluate(X_val, y_val)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

            test_metrics = model.evaluate(X_test, y_test)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            logger.info("Test metrics: %s", test_metrics)

            if self.cfg.evaluation.plot_confusion_matrix:
                self._log_confusion_matrix(y_val, model.predict(X_val), split="val")
                self._log_confusion_matrix(y_test, model.predict(X_test), split="test")

            ckpt_dir = Path(self.cfg.training.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{model.name}_best.joblib"
            model.save(ckpt_path)
            mlflow.log_artifact(str(ckpt_path))

            mlflow.sklearn.log_model(
                sk_model=model.estimator,
                artifact_path="model",
                registered_model_name=self.cfg.mlflow.model_name,
            )

        return model, test_metrics
