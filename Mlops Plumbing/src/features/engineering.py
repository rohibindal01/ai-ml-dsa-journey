"""Sklearn Pipeline-compatible feature transformers for the ML project."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerType = Union[StandardScaler, MinMaxScaler, RobustScaler]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a subset of columns from a DataFrame.

    Args:
        columns: List of column names to retain.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":
        """Validate that all requested columns exist.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            Fitted self.

        Raises:
            ValueError: If any column is missing from X.
        """
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Return a copy of X containing only the selected columns.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            DataFrame restricted to *self.columns*.
        """
        return X[self.columns].copy()


class TypeCaster(BaseEstimator, TransformerMixin):
    """Cast DataFrame columns to specified dtypes.

    Args:
        cast_map: Mapping from column name to target dtype string.
    """

    def __init__(self, cast_map: dict[str, str]) -> None:
        self.cast_map = cast_map

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TypeCaster":
        """No-op; returns self."""
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply dtype casts.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            DataFrame with columns cast according to *cast_map*.
        """
        X = X.copy()
        for col, dtype in self.cast_map.items():
            if col in X.columns:
                X[col] = X[col].astype(dtype)
        return X


class HighCardinalityDropper(BaseEstimator, TransformerMixin):
    """Drop categorical columns with more unique values than *max_categories*.

    Args:
        max_categories: Columns with more unique values are dropped.
    """

    def __init__(self, max_categories: int = 20) -> None:
        self.max_categories = max_categories
        self.cols_to_drop_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "HighCardinalityDropper":
        """Identify columns exceeding the cardinality threshold.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            Fitted self.
        """
        self.cols_to_drop_ = [
            col
            for col in X.select_dtypes(include=["object", "category"]).columns
            if X[col].nunique() > self.max_categories
        ]
        if self.cols_to_drop_:
            logger.info("HighCardinalityDropper will drop: %s", self.cols_to_drop_)
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Drop high-cardinality columns identified during fit.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            DataFrame without the identified high-cardinality columns.
        """
        return X.drop(columns=self.cols_to_drop_, errors="ignore").copy()


def build_scaler(scaler_name: str) -> ScalerType:
    """Instantiate a sklearn scaler by name.

    Args:
        scaler_name: One of ``"standard"``, ``"minmax"``, or ``"robust"``.

    Returns:
        Unfitted scaler instance.

    Raises:
        ValueError: If *scaler_name* is not recognised.
    """
    registry: dict[str, ScalerType] = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if scaler_name not in registry:
        raise ValueError(f"Unknown scaler '{scaler_name}'. Choose from: {list(registry)}")
    return registry[scaler_name]


def build_feature_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scaler_name: str = "standard",
    imputer_strategy: str = "mean",
    max_categories: int = 20,
) -> Pipeline:
    """Build a full sklearn feature-engineering Pipeline.

    Args:
        numeric_columns: List of numeric feature column names.
        categorical_columns: List of categorical feature column names.
        scaler_name: Scaler type: ``"standard"``, ``"minmax"``, or ``"robust"``.
        imputer_strategy: sklearn SimpleImputer strategy for numeric columns.
        max_categories: Max unique categories before a column is OHE'd.

    Returns:
        An unfitted sklearn Pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", build_scaler(scaler_name)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=max_categories,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("high_card_drop", HighCardinalityDropper(max_categories=max_categories)),
            ("preprocessor", preprocessor),
        ]
    )


def infer_column_types(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    exclude_columns: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical column lists from *df*.

    Args:
        df: Input DataFrame.
        target_column: If given, this column is excluded from feature lists.
        exclude_columns: Additional columns to exclude.

    Returns:
        Tuple of (numeric_columns, categorical_columns).
    """
    exclude: set[str] = set()
    if target_column:
        exclude.add(target_column)
    if exclude_columns:
        exclude.update(exclude_columns)

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    categorical_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in exclude
    ]
    return numeric_cols, categorical_cols
