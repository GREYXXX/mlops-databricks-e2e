"""Tests for FastAPI backend endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    # Reset module-level singletons before each test
    import app.backend.services.jobs_service as js
    import app.backend.services.mlflow_service as ms

    ms._client = None
    js._ws = None

    from app.backend.main import app

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for /health."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPipelineEndpoints:
    """Tests for /api/pipeline/*."""

    @patch("app.backend.services.jobs_service._get_client")
    def test_pipeline_status_no_jobs(self, mock_get_client, client):
        mock_ws = MagicMock()
        mock_ws.jobs.list.return_value = []
        mock_get_client.return_value = mock_ws

        response = client.get("/api/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NOT_FOUND"

    @patch("app.backend.services.jobs_service._get_client")
    def test_pipeline_history_no_jobs(self, mock_get_client, client):
        mock_ws = MagicMock()
        mock_ws.jobs.list.return_value = []
        mock_get_client.return_value = mock_ws

        response = client.get("/api/pipeline/history")
        assert response.status_code == 200
        assert response.json() == []

    @patch("app.backend.services.jobs_service._get_client")
    def test_pipeline_history_with_limit(self, mock_get_client, client):
        mock_ws = MagicMock()
        mock_ws.jobs.list.return_value = []
        mock_get_client.return_value = mock_ws

        response = client.get("/api/pipeline/history?limit=5")
        assert response.status_code == 200


class TestExperimentEndpoints:
    """Tests for /api/experiments/*."""

    @patch("app.backend.services.mlflow_service._get_client")
    def test_list_runs_no_experiment(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_get_client.return_value = mock_client

        response = client.get("/api/experiments/runs")
        assert response.status_code == 200
        assert response.json() == []

    @patch("app.backend.services.mlflow_service._get_client")
    def test_list_runs_with_results(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_client.get_experiment_by_name.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "run_1"
        mock_run.info.run_name = "test_run"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1000
        mock_run.info.end_time = 2000
        mock_run.data.params = {"lr": "0.1"}
        mock_run.data.metrics = {"rmse": 0.5}
        mock_run.data.tags = {"user": "test"}

        mock_client.search_runs.return_value = [mock_run]
        mock_get_client.return_value = mock_client

        response = client.get("/api/experiments/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["run_id"] == "run_1"
        assert data[0]["metrics"]["rmse"] == 0.5

    @patch("app.backend.services.mlflow_service._get_client")
    def test_get_run_not_found(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.get_run.side_effect = Exception("not found")
        mock_get_client.return_value = mock_client

        response = client.get("/api/experiments/runs/nonexistent")
        # The route calls mlflow_service.get_run_details which propagates the
        # exception.  FastAPI catches unhandled exceptions as 500.
        assert response.status_code in (404, 500)


class TestModelEndpoints:
    """Tests for /api/models/*."""

    @patch("app.backend.services.mlflow_service._get_client")
    def test_list_versions_empty(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.search_model_versions.side_effect = Exception("not found")
        mock_get_client.return_value = mock_client

        response = client.get("/api/models/versions")
        assert response.status_code == 200
        assert response.json() == []

    @patch("app.backend.services.mlflow_service._get_alias_map")
    @patch("app.backend.services.mlflow_service._list_uc_model_versions")
    def test_list_versions_with_results(self, mock_list_uc, mock_alias_map, client):
        # get_model_versions uses Databricks SDK (_list_uc_model_versions), not
        # MlflowClient.search_model_versions — patch the UC list + alias map.
        mock_mv = MagicMock()
        mock_mv.version = 1
        mock_mv.model_name = "main.mlops_e2e.california_housing_model"
        mock_mv.created_at = 1000
        mock_mv.updated_at = 2000
        mock_status = MagicMock()
        mock_status.value = "READY"
        mock_mv.status = mock_status
        mock_mv.source = "runs:/abc/model"
        mock_mv.run_id = "abc"

        mock_list_uc.return_value = [mock_mv]
        mock_alias_map.return_value = {"1": ["Champion"]}

        response = client.get("/api/models/versions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["version"] == "1"
        assert "Champion" in data[0]["aliases"]

    @patch("app.backend.services.mlflow_service._get_client")
    def test_champion_not_found(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.side_effect = Exception("not found")
        mock_get_client.return_value = mock_client

        response = client.get("/api/models/champion")
        assert response.status_code == 404


class TestComparisonEndpoint:
    """Tests for /api/models/comparison."""

    @patch("app.backend.services.mlflow_service._get_client")
    def test_comparison_no_models(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.side_effect = Exception("not found")
        mock_get_client.return_value = mock_client

        response = client.get("/api/models/comparison")
        assert response.status_code == 200
        data = response.json()
        assert data["promotion_status"] == "no_models"
        assert data["champion"]["version"] is None
        assert data["challenger"]["version"] is None

    @patch(
        "app.backend.routes.comparison.resolve_metrics_profile",
        return_value="regression",
    )
    @patch("app.backend.services.mlflow_service._get_client")
    def test_comparison_champion_only(
        self, mock_get_client, _mock_profile, client
    ):
        mock_client = MagicMock()

        mock_champion_mv = MagicMock()
        mock_champion_mv.version = "1"
        mock_champion_mv.name = "model"
        mock_champion_mv.creation_timestamp = 1000
        mock_champion_mv.last_updated_timestamp = 2000
        mock_champion_mv.status = "READY"
        mock_champion_mv.source = "runs:/abc/model"
        mock_champion_mv.run_id = "abc"
        mock_champion_mv.aliases = ["Champion"]

        mock_run = MagicMock()
        mock_run.info.run_id = "abc"
        mock_run.info.run_name = "run"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1000
        mock_run.info.end_time = 2000
        mock_run.info.artifact_uri = "dbfs:/artifacts"
        mock_run.data.params = {}
        mock_run.data.metrics = {"test_rmse": 0.4}
        mock_run.data.tags = {}

        def get_by_alias(name, alias):
            if alias == "Champion":
                return mock_champion_mv
            raise Exception("not found")

        mock_client.get_model_version_by_alias.side_effect = get_by_alias
        mock_client.get_run.return_value = mock_run
        mock_client.list_artifacts.return_value = []
        mock_get_client.return_value = mock_client

        response = client.get("/api/models/comparison")
        assert response.status_code == 200
        data = response.json()
        assert data["promotion_status"] == "promoted"
        assert data["champion"]["version"] == "1"
