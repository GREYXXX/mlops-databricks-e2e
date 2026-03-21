"""Champion/challenger model comparison and promotion logic."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def get_champion_version(model_name: str) -> str | None:
    """Get the current Champion model version.

    Args:
        model_name: Fully-qualified UC model name.

    Returns:
        Version string if a Champion exists, None otherwise.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, "Champion")
        logger.info("Found Champion version %s for %s", mv.version, model_name)
        return mv.version
    except mlflow.exceptions.MlflowException:
        logger.info("No Champion version found for %s", model_name)
        return None


def compare_models(
    champion_version: str,
    challenger_version: str,
    model_name: str,
    test_data: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compare Champion and Challenger models on the test set.

    Args:
        champion_version: Champion model version.
        challenger_version: Challenger model version.
        model_name: Fully-qualified UC model name.
        test_data: Dictionary with 'X_test' and 'y_test' arrays.

    Returns:
        Dictionary with champion_rmse and challenger_rmse.
    """
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    # Score Champion
    champion_uri = f"models:/{model_name}@Champion"
    champion_model = mlflow.lightgbm.load_model(champion_uri)
    champion_pred = champion_model.predict(X_test)
    champion_rmse = float(np.sqrt(mean_squared_error(y_test, champion_pred)))

    # Score Challenger
    challenger_uri = f"models:/{model_name}@Challenger"
    challenger_model = mlflow.lightgbm.load_model(challenger_uri)
    challenger_pred = challenger_model.predict(X_test)
    challenger_rmse = float(np.sqrt(mean_squared_error(y_test, challenger_pred)))

    logger.info(
        "Comparison - Champion RMSE: %.4f, Challenger RMSE: %.4f",
        champion_rmse,
        challenger_rmse,
    )

    return {
        "champion_rmse": champion_rmse,
        "challenger_rmse": challenger_rmse,
    }


def _make_timestamp_alias(prefix: str) -> str:
    """Create a timestamped alias like ``Champion-20260225-1430``."""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def promote_challenger(
    model_name: str,
    challenger_version: str,
    former_champion_version: str | None = None,
) -> str | None:
    """Promote the Challenger to Champion by moving aliases.

    If there is a former Champion, it receives a timestamped alias
    (e.g. ``Champion-20260225-143052``) so it can be used for rollback.

    Args:
        model_name: Fully-qualified UC model name.
        challenger_version: Version to promote to Champion.
        former_champion_version: Version of the outgoing Champion (if any).

    Returns:
        The archive alias assigned to the former Champion, or None.
    """
    client = mlflow.tracking.MlflowClient()
    archive_alias: str | None = None

    # Archive former Champion with a timestamped alias for easy rollback
    if former_champion_version is not None:
        archive_alias = _make_timestamp_alias("Champion")
        client.set_registered_model_alias(
            name=model_name,
            alias=archive_alias,
            version=former_champion_version,
        )
        logger.info(
            "Archived former Champion version %s with alias '%s'",
            former_champion_version,
            archive_alias,
        )

    # Set Champion alias on the challenger version
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=challenger_version,
    )

    # Remove Challenger alias
    try:
        client.delete_registered_model_alias(
            name=model_name,
            alias="Challenger",
        )
    except mlflow.exceptions.MlflowException:
        logger.warning("Could not remove Challenger alias - it may have already been removed")

    logger.info("Promoted version %s to Champion for %s", challenger_version, model_name)
    return archive_alias


def archive_challenger(model_name: str, challenger_version: str) -> str:
    """Archive a losing Challenger with a timestamped alias.

    Args:
        model_name: Fully-qualified UC model name.
        challenger_version: Version of the rejected Challenger.

    Returns:
        The archive alias assigned (e.g. ``Challenger-20260225-1430``).
    """
    client = mlflow.tracking.MlflowClient()
    archive_alias = _make_timestamp_alias("Challenger")
    client.set_registered_model_alias(
        name=model_name,
        alias=archive_alias,
        version=challenger_version,
    )
    # Remove the active Challenger alias
    try:
        client.delete_registered_model_alias(name=model_name, alias="Challenger")
    except mlflow.exceptions.MlflowException:
        pass
    logger.info(
        "Archived rejected Challenger version %s with alias '%s'",
        challenger_version,
        archive_alias,
    )
    return archive_alias


def run_champion_management(
    model_name: str,
    catalog: str,
    schema: str,
) -> dict[str, str]:
    """Full champion management pipeline.

    Compares the Challenger to the current Champion (if one exists) and
    promotes the Challenger if it performs better on the test set.

    Args:
        model_name: Fully-qualified UC model name.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.

    Returns:
        Dictionary with promotion decision details.
    """
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()

    # Get Challenger version
    challenger_mv = client.get_model_version_by_alias(model_name, "Challenger")
    challenger_version = challenger_mv.version
    challenger_run_id = challenger_mv.run_id
    logger.info("Challenger: version %s (run %s)", challenger_version, challenger_run_id)

    # Load test data from the challenger's run
    local_dir = client.download_artifacts(challenger_run_id, "test_data")
    npz_files = [f for f in os.listdir(local_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No .npz test data found in {local_dir}")
    test_data_path = os.path.join(local_dir, npz_files[0])
    test_data = dict(np.load(test_data_path))

    # Check for existing Champion
    champion_version = get_champion_version(model_name)
    result: dict[str, str] = {"challenger_version": challenger_version}

    if champion_version is None:
        # First model - auto-promote
        logger.info(
            "No existing Champion - auto-promoting Challenger version %s", challenger_version
        )
        promote_challenger(model_name, challenger_version)
        result["action"] = "auto_promoted"
        result["reason"] = "No existing Champion model"
    else:
        # Compare models
        result["champion_version"] = champion_version
        comparison = compare_models(champion_version, challenger_version, model_name, test_data)
        result["champion_rmse"] = str(comparison["champion_rmse"])
        result["challenger_rmse"] = str(comparison["challenger_rmse"])

        if comparison["challenger_rmse"] < comparison["champion_rmse"]:
            improvement = comparison["champion_rmse"] - comparison["challenger_rmse"]
            logger.info(
                "Challenger wins! RMSE improved by %.4f. Promoting version %s.",
                improvement,
                challenger_version,
            )
            archive_alias = promote_challenger(
                model_name, challenger_version, former_champion_version=champion_version
            )
            result["action"] = "promoted"
            ch_rmse = comparison["challenger_rmse"]
            champ_rmse = comparison["champion_rmse"]
            result["reason"] = f"Challenger RMSE ({ch_rmse:.4f}) < Champion RMSE ({champ_rmse:.4f})"
            if archive_alias:
                result["archived_champion_alias"] = archive_alias
        else:
            logger.info(
                "Champion retains title. Champion RMSE: %.4f <= Challenger RMSE: %.4f",
                comparison["champion_rmse"],
                comparison["challenger_rmse"],
            )
            challenger_alias = archive_challenger(model_name, challenger_version)
            result["action"] = "rejected"
            champ_r = comparison["champion_rmse"]
            chal_r = comparison["challenger_rmse"]
            result["reason"] = f"Champion RMSE ({champ_r:.4f}) <= Challenger RMSE ({chal_r:.4f})"
            result["archived_challenger_alias"] = challenger_alias

    # Log comparison summary as artifact on the challenger's run
    with mlflow.start_run(run_id=challenger_run_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "champion_comparison.json")
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            mlflow.log_artifact(summary_path)

    logger.info("Champion management complete: %s", result["action"])
    return result
