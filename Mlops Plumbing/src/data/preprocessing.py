"""Data preprocessing: cleaning, splitting, and persisting processed datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing encounters an unrecoverable problem."""


def drop_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows from *df*.

    Args:
        df: Input DataFrame.
        subset: Column names to consider for identifying duplicates.

    Returns:
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)
    return df


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where the fraction of null values exceeds *threshold*.

    Args:
        df: Input DataFrame.
        threshold: Columns with null fraction > threshold are dropped (0–1).

    Returns:
        DataFrame with offending columns removed.
    """
    null_fractions = df.isnull().mean()
    cols_to_drop = null_fractions[null_fractions > threshold].index.tolist()
    if cols_to_drop:
        logger.info("Dropping %d high-null columns: %s", len(cols_to_drop), cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


def cast_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce *columns* to numeric, turning parse failures into NaN.

    Args:
        df: Input DataFrame.
        columns: Column names to coerce.

    Returns:
        DataFrame with specified columns cast to float64.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train, validation, and test sets.

    Args:
        df: Full dataset including the target column.
        target_column: Name of the target/label column.
        test_size: Fraction of total data for the test set.
        val_size: Fraction of *training* data reserved for validation.
        random_state: Random seed for reproducibility.
        stratify: If True, preserve target distribution across splits.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        PreprocessingError: If the target column is not present.
    """
    if target_column not in df.columns:
        raise PreprocessingError(
            f"Target column '{target_column}' not found. Available: {list(df.columns)}"
        )

    stratify_col = df[target_column] if stratify else None
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    effective_val_size = val_size / (1.0 - test_size)
    stratify_col_tv = train_val_df[target_column] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=effective_val_size,
        random_state=random_state,
        stratify=stratify_col_tv,
    )

    logger.info(
        "Dataset split — train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df)
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist train/val/test splits as Parquet files.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        output_dir: Directory to write the three Parquet files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = output_dir / f"{name}.parquet"
        split_df.to_parquet(out_path, index=False)
        logger.info("Saved %s split → %s (%d rows)", name, out_path, len(split_df))


def load_splits(
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved train/val/test Parquet splits from *processed_dir*.

    Args:
        processed_dir: Directory containing train.parquet, val.parquet, test.parquet.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        PreprocessingError: If any expected Parquet file is missing.
    """
    processed_dir = Path(processed_dir)
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise PreprocessingError(f"Expected split file not found: {path}")
        splits[name] = pd.read_parquet(path)
        logger.info("Loaded %s split from %s (%d rows)", name, path, len(splits[name]))
    return splits["train"], splits["val"], splits["test"]


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for imbalanced classification.

    Args:
        y: 1-D integer array of class labels.

    Returns:
        Mapping from class index to weight.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weights = {int(cls): total / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    logger.debug("Class weights: %s", weights)
    return weights
