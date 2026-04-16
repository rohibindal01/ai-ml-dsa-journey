"""Abstract base model defining the fit / predict / evaluate contract."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all ML models in this project."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        """Initialise the model with a name and hyperparameter dict.

        Args:
            name: Human-readable model identifier.
            params: Hyperparameter dict passed to the underlying estimator.
        """
        self.name = name
        self.params = params
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """Train the model on *(X_train, y_train)*.

        Args:
            X_train: Feature matrix for training.
            y_train: Target array for training.
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            self.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions for *X*.

        Args:
            X: Feature matrix.

        Returns:
            1-D array of predicted class labels.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for *X*.

        Args:
            X: Feature matrix.

        Returns:
            2-D array of shape (n_samples, n_classes).
        """
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute a standard set of classification metrics on *(X, y)*.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dictionary mapping metric names to scalar values.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before evaluate().")

        y_pred = self.predict(X)
        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        }

        try:
            y_proba = self.predict_proba(X)
            n_classes = y_proba.shape[1]
            auc = roc_auc_score(
                y,
                y_proba if n_classes > 2 else y_proba[:, 1],
                multi_class="ovr" if n_classes > 2 else "raise",
            )
            metrics["roc_auc"] = float(auc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not compute ROC-AUC: %s", exc)

        logger.info("Evaluation metrics for '%s': %s", self.name, metrics)
        return metrics

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to *path*.

        Args:
            path: File path to save the model artefact.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a model from *path* and return a fitted instance.

        Args:
            path: File path of the saved model artefact.

        Returns:
            Loaded, ready-to-predict model instance.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, params={self.params!r})"
