"""Tests for src/data/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    PreprocessingError,
    cast_numeric_columns,
    compute_class_weights,
    drop_duplicates,
    drop_high_null_columns,
    split_dataset,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.integers(0, 5, n).astype(float),
            "target": rng.integers(0, 2, n),
        }
    )


class TestDropDuplicates:
    def test_removes_exact_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = drop_duplicates(df)
        assert len(result) == 3

    def test_subset_deduplication(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        result = drop_duplicates(df, subset=["a"])
        assert len(result) == 2


class TestDropHighNullColumns:
    def test_drops_columns_above_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "ok": [1, 2, 3, 4, 5],
                "mostly_null": [np.nan, np.nan, np.nan, np.nan, 1.0],
            }
        )
        result = drop_high_null_columns(df, threshold=0.5)
        assert "ok" in result.columns
        assert "mostly_null" not in result.columns

    def test_retains_columns_below_threshold(self) -> None:
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = drop_high_null_columns(df, threshold=0.5)
        assert set(result.columns) == {"a", "b"}


class TestCastNumericColumns:
    def test_casts_string_to_float(self) -> None:
        df = pd.DataFrame({"x": ["1.1", "2.2", "3.3"]})
        result = cast_numeric_columns(df, columns=["x"])
        assert result["x"].dtype == np.float64

    def test_invalid_values_become_nan(self) -> None:
        df = pd.DataFrame({"x": ["1.0", "not_a_number", "3.0"]})
        result = cast_numeric_columns(df, columns=["x"])
        assert result["x"].isna().sum() == 1

    def test_skips_missing_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        result = cast_numeric_columns(df, columns=["a", "z_nonexistent"])
        assert "z_nonexistent" not in result.columns


class TestSplitDataset:
    def test_split_sizes(self, small_df: pd.DataFrame) -> None:
        train, val, test = split_dataset(small_df, target_column="target")
        assert len(train) + len(val) + len(test) == len(small_df)
        assert len(test) == pytest.approx(len(small_df) * 0.2, abs=2)

    def test_raises_on_missing_target(self, small_df: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="Target column"):
            split_dataset(small_df, target_column="nonexistent")

    def test_no_index_overlap(self, small_df: pd.DataFrame) -> None:
        train, val, test = split_dataset(small_df, target_column="target", stratify=False)
        all_indices = set(train.index) | set(val.index) | set(test.index)
        assert len(all_indices) == len(train) + len(val) + len(test)


class TestComputeClassWeights:
    def test_balanced_classes_equal_weights(self) -> None:
        y = np.array([0, 0, 1, 1])
        weights = compute_class_weights(y)
        assert abs(weights[0] - weights[1]) < 1e-9

    def test_imbalanced_classes(self) -> None:
        y = np.array([0, 0, 0, 1])
        weights = compute_class_weights(y)
        assert weights[1] > weights[0]

    def test_all_classes_present(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(y)
        assert set(weights.keys()) == {0, 1, 2}
