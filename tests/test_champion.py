"""Tests for champion module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import mlflow.exceptions
import numpy as np
import pytest


class TestGetChampionVersion:
    """Tests for get_champion_version()."""

    def test_returns_version_when_champion_exists(self):
        from mlops_e2e.champion import get_champion_version

        mock_mv = MagicMock()
        mock_mv.version = "3"

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_client.get_model_version_by_alias.return_value = mock_mv
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            result = get_champion_version("cat.sch.model")

        assert result == "3"
        mock_client.get_model_version_by_alias.assert_called_once_with("cat.sch.model", "Champion")

    def test_returns_none_when_no_champion(self):
        from mlops_e2e.champion import get_champion_version

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
                "not found"
            )
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            mock_mlflow.exceptions = mlflow.exceptions

            result = get_champion_version("cat.sch.model")

        assert result is None


class TestCompareModels:
    """Tests for compare_models()."""

    def test_returns_both_rmse_values(self, sample_test_data):
        from mlops_e2e.champion import compare_models

        y_test = sample_test_data["y_test"]

        mock_champion = MagicMock()
        mock_champion.predict.return_value = y_test + 0.5  # Worse

        mock_challenger = MagicMock()
        mock_challenger.predict.return_value = y_test + 0.1  # Better

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_mlflow.lightgbm.load_model.side_effect = [mock_champion, mock_challenger]

            result = compare_models("1", "2", "cat.sch.model", sample_test_data)

        assert "champion_rmse" in result
        assert "challenger_rmse" in result
        assert result["champion_rmse"] > 0
        assert result["challenger_rmse"] > 0

    def test_challenger_wins_when_lower_rmse(self, sample_test_data):
        from mlops_e2e.champion import compare_models

        y_test = sample_test_data["y_test"]

        mock_champion = MagicMock()
        mock_champion.predict.return_value = y_test + 0.5

        mock_challenger = MagicMock()
        mock_challenger.predict.return_value = y_test + 0.1

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_mlflow.lightgbm.load_model.side_effect = [mock_champion, mock_challenger]

            result = compare_models("1", "2", "cat.sch.model", sample_test_data)

        assert result["challenger_rmse"] < result["champion_rmse"]

    def test_champion_retains_when_lower_rmse(self, sample_test_data):
        from mlops_e2e.champion import compare_models

        y_test = sample_test_data["y_test"]

        mock_champion = MagicMock()
        mock_champion.predict.return_value = y_test + 0.1  # Better

        mock_challenger = MagicMock()
        mock_challenger.predict.return_value = y_test + 0.5  # Worse

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_mlflow.lightgbm.load_model.side_effect = [mock_champion, mock_challenger]

            result = compare_models("1", "2", "cat.sch.model", sample_test_data)

        assert result["champion_rmse"] < result["challenger_rmse"]


class TestPromoteChallenger:
    """Tests for promote_challenger()."""

    def test_sets_champion_alias_and_removes_challenger(self):
        from mlops_e2e.champion import promote_challenger

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            result = promote_challenger("cat.sch.model", "5")

        mock_client.set_registered_model_alias.assert_called_once_with(
            name="cat.sch.model",
            alias="Champion",
            version="5",
        )
        mock_client.delete_registered_model_alias.assert_called_once_with(
            name="cat.sch.model",
            alias="Challenger",
        )
        assert result is None  # No former champion

    def test_archives_former_champion_with_timestamp_alias(self):
        from mlops_e2e.champion import promote_challenger

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            result = promote_challenger("cat.sch.model", "5", former_champion_version="3")

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        expected_alias = f"Champion-{now_str}"

        # Should set archive alias on former champion, then Champion on new version
        alias_calls = mock_client.set_registered_model_alias.call_args_list
        assert len(alias_calls) == 2
        assert alias_calls[0] == call(name="cat.sch.model", alias=expected_alias, version="3")
        assert alias_calls[1] == call(name="cat.sch.model", alias="Champion", version="5")
        assert result == expected_alias

    def test_handles_missing_challenger_alias_gracefully(self):
        from mlops_e2e.champion import promote_challenger

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_client.delete_registered_model_alias.side_effect = (
                mlflow.exceptions.MlflowException("not found")
            )
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            mock_mlflow.exceptions = mlflow.exceptions

            # Should not raise
            promote_challenger("cat.sch.model", "5")

        mock_client.set_registered_model_alias.assert_called_once()


class TestRunChampionManagement:
    """Tests for run_champion_management()."""

    def test_auto_promotes_when_no_champion(self, sample_test_data, tmp_path):
        from mlops_e2e.champion import run_champion_management

        # Save test data
        test_data_path = str(tmp_path / "test.npz")
        np.savez(test_data_path, **sample_test_data)

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            mock_mlflow.exceptions = mlflow.exceptions

            # Challenger exists
            mock_challenger_mv = MagicMock()
            mock_challenger_mv.version = "1"
            mock_challenger_mv.run_id = "run_abc"

            def get_by_alias(name, alias):
                if alias == "Challenger":
                    return mock_challenger_mv
                raise mlflow.exceptions.MlflowException("not found")

            mock_client.get_model_version_by_alias.side_effect = get_by_alias
            mock_client.download_artifacts.return_value = str(tmp_path)

            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result = run_champion_management("cat.sch.model", "cat", "sch")

        assert result["action"] == "auto_promoted"
        assert result["challenger_version"] == "1"

    def test_promotes_when_challenger_better(self, sample_test_data, tmp_path):
        from mlops_e2e.champion import run_champion_management

        test_data_path = str(tmp_path / "test.npz")
        np.savez(test_data_path, **sample_test_data)

        y_test = sample_test_data["y_test"]

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            mock_mlflow.exceptions = mlflow.exceptions

            mock_challenger_mv = MagicMock()
            mock_challenger_mv.version = "2"
            mock_challenger_mv.run_id = "run_def"

            mock_champion_mv = MagicMock()
            mock_champion_mv.version = "1"

            def get_by_alias(name, alias):
                if alias == "Challenger":
                    return mock_challenger_mv
                if alias == "Champion":
                    return mock_champion_mv
                raise mlflow.exceptions.MlflowException("not found")

            mock_client.get_model_version_by_alias.side_effect = get_by_alias
            mock_client.download_artifacts.return_value = str(tmp_path)

            # Champion worse, challenger better
            mock_champion_model = MagicMock()
            mock_champion_model.predict.return_value = y_test + 0.5
            mock_challenger_model = MagicMock()
            mock_challenger_model.predict.return_value = y_test + 0.1

            mock_mlflow.lightgbm.load_model.side_effect = [
                mock_champion_model,
                mock_challenger_model,
            ]
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result = run_champion_management("cat.sch.model", "cat", "sch")

        assert result["action"] == "promoted"
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        assert result["archived_champion_alias"] == f"Champion-{now_str}"

    def test_rejects_when_champion_better(self, sample_test_data, tmp_path):
        from mlops_e2e.champion import run_champion_management

        test_data_path = str(tmp_path / "test.npz")
        np.savez(test_data_path, **sample_test_data)

        y_test = sample_test_data["y_test"]

        with patch("mlops_e2e.champion.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            mock_mlflow.exceptions = mlflow.exceptions

            mock_challenger_mv = MagicMock()
            mock_challenger_mv.version = "2"
            mock_challenger_mv.run_id = "run_ghi"

            mock_champion_mv = MagicMock()
            mock_champion_mv.version = "1"

            def get_by_alias(name, alias):
                if alias == "Challenger":
                    return mock_challenger_mv
                if alias == "Champion":
                    return mock_champion_mv
                raise mlflow.exceptions.MlflowException("not found")

            mock_client.get_model_version_by_alias.side_effect = get_by_alias
            mock_client.download_artifacts.return_value = str(tmp_path)

            # Champion better, challenger worse
            mock_champion_model = MagicMock()
            mock_champion_model.predict.return_value = y_test + 0.1
            mock_challenger_model = MagicMock()
            mock_challenger_model.predict.return_value = y_test + 0.5

            mock_mlflow.lightgbm.load_model.side_effect = [
                mock_champion_model,
                mock_challenger_model,
            ]
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result = run_champion_management("cat.sch.model", "cat", "sch")

        assert result["action"] == "rejected"
