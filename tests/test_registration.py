"""Tests for registration module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRegisterModelToUC:
    """Tests for register_model_to_uc()."""

    def test_calls_register_model_with_correct_args(self):
        from mlops_e2e.housing.registration import register_model_to_uc

        mock_mv = MagicMock()
        mock_mv.version = "1"

        with patch("mlops_e2e.housing.registration.mlflow") as mock_mlflow:
            mock_mlflow.register_model.return_value = mock_mv

            version = register_model_to_uc(
                run_id="abc123",
                model_name="catalog.schema.my_model",
            )

        mock_mlflow.set_registry_uri.assert_called_once_with("databricks-uc")
        mock_mlflow.register_model.assert_called_once_with(
            "runs:/abc123/model",
            "catalog.schema.my_model",
        )
        assert version == "1"

    def test_returns_version_string(self):
        from mlops_e2e.housing.registration import register_model_to_uc

        mock_mv = MagicMock()
        mock_mv.version = "5"

        with patch("mlops_e2e.housing.registration.mlflow") as mock_mlflow:
            mock_mlflow.register_model.return_value = mock_mv

            version = register_model_to_uc("run123", "cat.sch.model")

        assert version == "5"


class TestSetModelAlias:
    """Tests for set_model_alias()."""

    def test_calls_client_with_correct_args(self):
        from mlops_e2e.housing.registration import set_model_alias

        with patch("mlops_e2e.housing.registration.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            set_model_alias("cat.sch.model", "3", "Challenger")

        mock_client.set_registered_model_alias.assert_called_once_with(
            name="cat.sch.model",
            alias="Challenger",
            version="3",
        )

    def test_champion_alias(self):
        from mlops_e2e.housing.registration import set_model_alias

        with patch("mlops_e2e.housing.registration.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            set_model_alias("cat.sch.model", "7", "Champion")

        mock_client.set_registered_model_alias.assert_called_once_with(
            name="cat.sch.model",
            alias="Champion",
            version="7",
        )
