"""Model registration to Unity Catalog."""

from __future__ import annotations

import logging

import mlflow

logger = logging.getLogger(__name__)


def register_model_to_uc(run_id: str, model_name: str) -> str:
    """Register a model from an MLflow run to Unity Catalog.

    Args:
        run_id: MLflow run ID containing the model artifact.
        model_name: Fully-qualified UC model name (catalog.schema.model).

    Returns:
        The registered model version string.
    """
    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering model from %s to %s", model_uri, model_name)

    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version

    logger.info("Registered model version %s for %s", version, model_name)
    return version


def set_model_alias(model_name: str, version: str, alias: str) -> None:
    """Set an alias on a model version in Unity Catalog.

    Args:
        model_name: Fully-qualified UC model name.
        version: Model version to alias.
        alias: Alias to set (e.g. "Challenger", "Champion").
    """
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )
    logger.info("Set alias '%s' on %s version %s", alias, model_name, version)
