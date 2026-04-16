"""Tests for src/serving/api.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import ModelRegistry, create_app


class TestHealthEndpoint:
    def test_healthy_with_model(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "test-model"

    def test_healthy_without_model(self) -> None:
        registry = ModelRegistry(
            tracking_uri="http://mock",
            model_name="test-model",
            model_stage="Production",
        )
        app = create_app(registry=registry)
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is False


class TestPredictEndpoint:
    def test_single_sample_prediction(self, api_client: TestClient) -> None:
        payload = {"features": [{"feature_a": 1.0, "feature_b": 2.0}]}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 1
        assert "prediction" in body["predictions"][0]

    def test_batch_prediction(self, api_client: TestClient) -> None:
        payload = {
            "features": [
                {"feature_a": 1.0, "feature_b": 2.0},
                {"feature_a": 3.0, "feature_b": 4.0},
                {"feature_a": 5.0, "feature_b": 6.0},
            ]
        }
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 3

    def test_probabilities_returned(self, api_client: TestClient) -> None:
        payload = {
            "features": [
                {"feature_a": 1.0, "feature_b": 2.0},
                {"feature_a": 3.0, "feature_b": 4.0},
                {"feature_a": 5.0, "feature_b": 6.0},
            ]
        }
        response = api_client.post("/predict", json=payload)
        body = response.json()
        first = body["predictions"][0]
        assert first["probabilities"] is not None
        assert "0" in first["probabilities"] or "1" in first["probabilities"]

    def test_503_when_model_not_loaded(self) -> None:
        registry = ModelRegistry(
            tracking_uri="http://mock",
            model_name="test-model",
            model_stage="Production",
        )
        app = create_app(registry=registry)
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": [{"a": 1.0}]})
        assert response.status_code == 503

    def test_empty_features_rejected(self, api_client: TestClient) -> None:
        payload = {"features": []}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_inconsistent_keys_rejected(self, api_client: TestClient) -> None:
        payload = {
            "features": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "c": 4.0},
            ]
        }
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_model_name_and_version_in_response(self, api_client: TestClient) -> None:
        payload = {"features": [{"x": 1.0}]}
        response = api_client.post("/predict", json=payload)
        body = response.json()
        assert body["model_name"] == "test-model"
        assert body["model_version"] == "1"

    def test_inference_error_returns_500(self) -> None:
        registry = ModelRegistry(
            tracking_uri="http://mock",
            model_name="test-model",
            model_stage="Production",
        )
        broken_model = MagicMock()
        broken_model.predict.side_effect = RuntimeError("GPU OOM")
        registry.model = broken_model
        registry.model_version = "2"

        app = create_app(registry=registry)
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": [{"x": 1.0}]})
        assert response.status_code == 500
