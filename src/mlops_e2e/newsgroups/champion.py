"""Champion/challenger model comparison and promotion logic for the newsgroups pipeline.

Uses weighted F1 as the promotion metric (higher is better).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import mlflow
import numpy as np

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def get_champion_version(model_name: str) -> str | None:
    """Get the current Champion model version, or None if none exists."""
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, "Champion")
        logger.info("Found Champion version %s for %s", mv.version, model_name)
        return mv.version
    except mlflow.exceptions.MlflowException:
        logger.info("No Champion version found for %s", model_name)
        return None


def _get_weighted_f1(model_uri: str, test_texts: list[str], test_labels: list[int]) -> float:
    """Load a pyfunc model and compute weighted F1 on the test set."""
    from sklearn.metrics import f1_score

    model = mlflow.pyfunc.load_model(model_uri)
    labels = model.predict(test_texts, params={"return_mode": "labels"})
    return float(f1_score(test_labels, labels, average="weighted", zero_division=0))


def compare_models(
    champion_version: str,
    challenger_version: str,
    model_name: str,
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, float]:
    """Compare Champion and Challenger on the test set using weighted F1.

    Args:
        champion_version: Current Champion model version.
        challenger_version: New Challenger model version.
        model_name: Fully-qualified UC model name.
        test_texts: Raw text for evaluation.
        test_labels: True class labels.

    Returns:
        Dict with champion_f1 and challenger_f1.
    """
    champion_f1 = _get_weighted_f1(
        f"models:/{model_name}@Champion", test_texts, test_labels
    )
    challenger_f1 = _get_weighted_f1(
        f"models:/{model_name}@Challenger", test_texts, test_labels
    )
    logger.info(
        "Comparison — Champion F1: %.4f, Challenger F1: %.4f",
        champion_f1,
        challenger_f1,
    )
    return {"champion_f1": champion_f1, "challenger_f1": challenger_f1}


def _make_timestamp_alias(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def promote_challenger(
    model_name: str,
    challenger_version: str,
    former_champion_version: str | None = None,
) -> str | None:
    """Promote Challenger to Champion, archive the former Champion."""
    client = mlflow.tracking.MlflowClient()
    archive_alias: str | None = None

    if former_champion_version is not None:
        archive_alias = _make_timestamp_alias("Champion")
        client.set_registered_model_alias(
            name=model_name, alias=archive_alias, version=former_champion_version
        )
        logger.info("Archived former Champion version %s → '%s'", former_champion_version, archive_alias)

    client.set_registered_model_alias(name=model_name, alias="Champion", version=challenger_version)
    try:
        client.delete_registered_model_alias(name=model_name, alias="Challenger")
    except mlflow.exceptions.MlflowException:
        pass

    logger.info("Promoted version %s to Champion for %s", challenger_version, model_name)
    return archive_alias


def archive_challenger(model_name: str, challenger_version: str) -> str:
    """Archive a losing Challenger with a timestamped alias."""
    client = mlflow.tracking.MlflowClient()
    archive_alias = _make_timestamp_alias("Challenger")
    client.set_registered_model_alias(name=model_name, alias=archive_alias, version=challenger_version)
    try:
        client.delete_registered_model_alias(name=model_name, alias="Challenger")
    except mlflow.exceptions.MlflowException:
        pass
    logger.info("Archived rejected Challenger version %s → '%s'", challenger_version, archive_alias)
    return archive_alias


def run_champion_management(
    spark: SparkSession,
    catalog: str,
    schema: str,
    model_name: str,
) -> dict[str, str]:
    """Full champion management pipeline for the newsgroups ensemble.

    Compares Challenger against the current Champion using weighted F1.
    Promotes the Challenger if it scores higher.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        model_name: Fully-qualified UC model name.

    Returns:
        Dict with promotion decision details (action, reason, versions, metrics).
    """
    from mlops_e2e.newsgroups.data import load_raw_table

    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()

    challenger_mv = client.get_model_version_by_alias(model_name, "Challenger")
    challenger_version = challenger_mv.version
    challenger_run_id = challenger_mv.run_id
    logger.info("Challenger: version %s (run %s)", challenger_version, challenger_run_id)

    _, test_texts, _, test_labels = load_raw_table(spark, catalog, schema)

    champion_version = get_champion_version(model_name)
    result: dict[str, str] = {"challenger_version": challenger_version}

    if champion_version is None:
        logger.info("No existing Champion — auto-promoting Challenger version %s", challenger_version)
        promote_challenger(model_name, challenger_version)
        result["action"] = "auto_promoted"
        result["reason"] = "No existing Champion model"
    else:
        result["champion_version"] = champion_version
        comparison = compare_models(
            champion_version, challenger_version, model_name, test_texts, test_labels
        )
        result["champion_f1"] = str(comparison["champion_f1"])
        result["challenger_f1"] = str(comparison["challenger_f1"])

        if comparison["challenger_f1"] > comparison["champion_f1"]:
            improvement = comparison["challenger_f1"] - comparison["champion_f1"]
            logger.info(
                "Challenger wins! Weighted F1 improved by %.4f. Promoting version %s.",
                improvement,
                challenger_version,
            )
            archive_alias = promote_challenger(
                model_name, challenger_version, former_champion_version=champion_version
            )
            result["action"] = "promoted"
            result["reason"] = (
                f"Challenger F1 ({comparison['challenger_f1']:.4f}) > "
                f"Champion F1 ({comparison['champion_f1']:.4f})"
            )
            if archive_alias:
                result["archived_champion_alias"] = archive_alias
        else:
            logger.info(
                "Champion retains title. Champion F1: %.4f >= Challenger F1: %.4f",
                comparison["champion_f1"],
                comparison["challenger_f1"],
            )
            challenger_alias = archive_challenger(model_name, challenger_version)
            result["action"] = "rejected"
            result["reason"] = (
                f"Champion F1 ({comparison['champion_f1']:.4f}) >= "
                f"Challenger F1 ({comparison['challenger_f1']:.4f})"
            )
            result["archived_challenger_alias"] = challenger_alias

    with mlflow.start_run(run_id=challenger_run_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "champion_comparison.json")
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            mlflow.log_artifact(summary_path)

    logger.info("Champion management complete: %s", result["action"])
    return result
