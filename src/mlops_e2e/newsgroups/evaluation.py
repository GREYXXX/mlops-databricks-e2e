"""Model evaluation: reload ensemble from MLflow, score test set, log metrics.

This stage re-evaluates the ensemble on the test set and logs a full
classification report for every sub-model and the ensemble itself.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING

import mlflow
import numpy as np

from mlops_e2e.newsgroups.config import EXPERIMENT_NAME

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def evaluate_ensemble(
    spark: SparkSession,
    catalog: str,
    schema: str,
    training_run_id: str,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict[str, float]:
    """Load the ensemble pyfunc, score the test set, and log full metrics.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        training_run_id: MLflow run ID from the training stage.
        experiment_name: MLflow experiment path.

    Returns:
        Dict with scalar metrics (accuracy, precision, recall, f1_weighted)
        for each sub-model and the ensemble.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    from mlops_e2e.newsgroups.data import load_raw_table

    _, test_texts, _, test_labels = load_raw_table(spark, catalog, schema)
    class_names = fetch_20newsgroups(subset="train").target_names

    logger.info("Loading ensemble pyfunc from run %s", training_run_id)
    model_uri = f"runs:/{training_run_id}/model"
    ensemble = mlflow.pyfunc.load_model(model_uri)

    logger.info("Running inference on %d test samples…", len(test_texts))
    result = ensemble.predict(test_texts, params={"return_mode": "all"})

    scalar_metrics: dict[str, float] = {}

    with mlflow.start_run(run_id=training_run_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            for prefix, proba in [
                ("lr", result["lr_proba"]),
                ("rf", result["rf_proba"]),
                ("cnn", result["cnn_proba"]),
                ("ensemble", result["ensemble_proba"]),
            ]:
                y_pred = np.array(proba).argmax(axis=1)
                metrics = {
                    f"eval_{prefix}_accuracy": float(accuracy_score(test_labels, y_pred)),
                    f"eval_{prefix}_precision": float(
                        precision_score(test_labels, y_pred, average="weighted", zero_division=0)
                    ),
                    f"eval_{prefix}_recall": float(
                        recall_score(test_labels, y_pred, average="weighted", zero_division=0)
                    ),
                    f"eval_{prefix}_f1_weighted": float(
                        f1_score(test_labels, y_pred, average="weighted", zero_division=0)
                    ),
                }
                mlflow.log_metrics(metrics)
                scalar_metrics.update(metrics)

                report = classification_report(
                    test_labels, y_pred, target_names=class_names, output_dict=True
                )
                report_path = os.path.join(tmpdir, f"eval_{prefix}_cls_report.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(report_path, "evaluation_reports")
                logger.info(
                    "%s weighted F1: %.4f", prefix, metrics[f"eval_{prefix}_f1_weighted"]
                )

    return scalar_metrics
