"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.serving.api import ModelRegistry, create_app


@pytest.fixture(scope="session")
def breast_cancer_df() -> pd.DataFrame:
    """Return the sklearn breast-cancer dataset as a DataFrame with a 'target' column."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df["target"] = raw.target
    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Return a tiny synthetic DataFrame for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.uniform(0, 10, n),
            "cat_x": rng.choice(["foo", "bar", "baz"], n),
            "target": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def xy_arrays(
    small_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val splits as numpy arrays from small_df."""
    from sklearn.model_selection import train_test_split

    X = small_df.drop(columns=["target"]).select_dtypes(include=[np.number]).to_numpy()
    y = small_df["target"].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


@pytest.fixture
def mock_registry() -> ModelRegistry:
    """Return a ModelRegistry with a mocked MLflow pyfunc model."""
    registry = ModelRegistry(
        tracking_uri="http://mock",
        model_name="test-model",
        model_stage="Production",
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])

    mock_underlying = MagicMock()
    mock_underlying.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
    )
    mock_underlying.classes_ = np.array([0, 1])
    mock_model._model_impl = mock_underlying

    registry.model = mock_model
    registry.model_version = "1"
    return registry


@pytest.fixture
def api_client(mock_registry: ModelRegistry) -> Generator[TestClient, None, None]:
    """Return a TestClient backed by a mocked model registry."""
    app = create_app(registry=mock_registry)
    with TestClient(app) as client:
        yield client
