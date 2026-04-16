"""Evaluation metrics and plots for classification and regression tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "macro",
) -> dict[str, float]:
    """Compute a standard suite of classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_proba: Optional predicted probabilities, shape (n_samples, n_classes).
        average: Averaging strategy passed to sklearn metrics.

    Returns:
        Dict mapping metric name to scalar value.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        f"f1_{average}": f1_score(y_true, y_pred, average=average, zero_division=0),
        f"precision_{average}": precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        f"recall_{average}": recall_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1]
            auc = roc_auc_score(
                y_true,
                y_proba if n_classes > 2 else y_proba[:, 1],
                multi_class="ovr" if n_classes > 2 else "raise",
            )
            metrics["roc_auc"] = float(auc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ROC-AUC computation failed: %s", exc)

    logger.info("Classification metrics: %s", metrics)
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth continuous targets.
        y_pred: Predicted values.

    Returns:
        Dict with ``rmse``, ``mae``, and ``r2`` keys.
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics: dict[str, float] = {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    logger.info("Regression metrics: %s", metrics)
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Generate an interactive Plotly confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class label strings.
        title: Figure title.
        output_path: If provided, saves the figure as an HTML file.

    Returns:
        Plotly Figure.
    """
    labels = class_names or [str(c) for c in sorted(np.unique(y_true))]
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    text_annotations = [
        [f"{cm[i, j]}<br>({cm_pct[i, j]:.1f}%)" for j in range(len(labels))]
        for i in range(len(labels))
    ]

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title=title,
        text_auto=False,
        aspect="auto",
    )
    fig.update_traces(text=text_annotations, texttemplate="%{text}")

    if output_path is not None:
        fig.write_html(str(output_path))
        logger.info("Saved confusion matrix to %s", output_path)

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "ROC Curves",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true: Ground-truth integer labels.
        y_proba: Predicted probabilities, shape (n_samples, n_classes).
        class_names: Optional class name list.
        title: Figure title.
        output_path: If provided, save as HTML.

    Returns:
        Plotly Figure containing all ROC curves.
    """
    n_classes = y_proba.shape[1]
    labels = class_names or [str(i) for i in range(n_classes)]
    fig = go.Figure()

    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        try:
            auc = roc_auc_score(y_bin, y_proba[:, i])
        except Exception:  # noqa: BLE001
            auc = float("nan")
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Class {labels[i]} (AUC={auc:.3f})")
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Random",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
    )

    if output_path is not None:
        fig.write_html(str(output_path))
        logger.info("Saved ROC curve to %s", output_path)

    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> str:
    """Return and log a formatted sklearn classification report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional class name list.

    Returns:
        The classification report as a string.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    logger.info("\n%s", report)
    return report


def metrics_to_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    """Convert a flat metrics dict to a tidy two-column DataFrame.

    Args:
        metrics: Mapping of metric name to value.

    Returns:
        DataFrame with columns ``metric`` and ``value``.
    """
    return pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
