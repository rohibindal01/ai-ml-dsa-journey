"""Tests for src/features/engineering.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.engineering import (
    ColumnSelector,
    HighCardinalityDropper,
    TypeCaster,
    build_feature_pipeline,
    build_scaler,
    infer_column_types,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    return pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.uniform(0, 5, n),
            "cat_x": rng.choice(["foo", "bar", "baz"], n),
            "target": rng.integers(0, 2, n),
        }
    )


class TestColumnSelector:
    def test_selects_specified_columns(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        selector = ColumnSelector(columns=["a", "c"])
        result = selector.fit_transform(df)
        assert list(result.columns) == ["a", "c"]

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        selector = ColumnSelector(columns=["a", "z"])
        with pytest.raises(ValueError, match="Missing columns"):
            selector.fit(df)


class TestTypeCaster:
    def test_casts_correctly(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        caster = TypeCaster(cast_map={"x": "float32"})
        result = caster.fit_transform(df)
        assert result["x"].dtype == np.float32

    def test_ignores_absent_columns(self) -> None:
        df = pd.DataFrame({"x": [1.0]})
        caster = TypeCaster(cast_map={"y": "float32"})
        result = caster.fit_transform(df)
        assert list(result.columns) == ["x"]


class TestHighCardinalityDropper:
    def test_drops_high_cardinality_column(self) -> None:
        df = pd.DataFrame(
            {
                "low": ["a", "b", "a", "b"] * 5,
                "high": [str(i) for i in range(20)],
            }
        )
        dropper = HighCardinalityDropper(max_categories=5)
        result = dropper.fit_transform(df)
        assert "low" in result.columns
        assert "high" not in result.columns

    def test_retains_numeric_columns(self) -> None:
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0] * 7, "cat": ["x", "y", "z"] * 7})
        dropper = HighCardinalityDropper(max_categories=10)
        result = dropper.fit_transform(df)
        assert "num" in result.columns


class TestBuildScaler:
    @pytest.mark.parametrize("name", ["standard", "minmax", "robust"])
    def test_valid_names(self, name: str) -> None:
        scaler = build_scaler(name)
        assert scaler is not None

    def test_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown scaler"):
            build_scaler("zscore_v2")


class TestBuildFeaturePipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = build_feature_pipeline(
            numeric_columns=["a", "b"],
            categorical_columns=["c"],
        )
        assert isinstance(pipeline, Pipeline)

    def test_fit_transform_shape(self, small_df: pd.DataFrame) -> None:
        numeric_cols = ["num_a", "num_b"]
        cat_cols = ["cat_x"]
        pipeline = build_feature_pipeline(
            numeric_columns=numeric_cols,
            categorical_columns=cat_cols,
        )
        X = small_df.drop(columns=["target"])
        result = pipeline.fit_transform(X)
        assert result.shape[0] == len(small_df)
        assert result.shape[1] >= len(numeric_cols)

    def test_no_nans_in_output(self, small_df: pd.DataFrame) -> None:
        small_df = small_df.copy()
        small_df.loc[0, "num_a"] = np.nan
        pipeline = build_feature_pipeline(
            numeric_columns=["num_a", "num_b"],
            categorical_columns=["cat_x"],
        )
        result = pipeline.fit_transform(small_df.drop(columns=["target"]))
        assert not np.isnan(result).any()


class TestInferColumnTypes:
    def test_separates_types(self, small_df: pd.DataFrame) -> None:
        numeric, categorical = infer_column_types(small_df, target_column="target")
        assert "num_a" in numeric
        assert "num_b" in numeric
        assert "cat_x" in categorical
        assert "target" not in numeric
        assert "target" not in categorical
