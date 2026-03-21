"""Model evaluation: metrics computation and plot generation."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary with RMSE, MAE, R2, MAPE, and MedianAE.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    median_ae = float(median_absolute_error(y_true, y_pred))

    # MAPE - handle zero values
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("inf")

    return {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "test_mape": mape,
        "test_median_ae": median_ae,
    }


def generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importances: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, "matplotlib.figure.Figure"]:
    """Generate evaluation plots.

    Args:
        y_true: Ground truth values.
        y_pred: Predictions.
        feature_importances: Feature importance values (optional).
        feature_names: Feature names (optional).

    Returns:
        Dictionary mapping plot names to matplotlib Figure objects.
    """
    plots: Dict[str, matplotlib.figure.Figure] = {}

    # 1. Residual plot
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    plots["residual_plot"] = fig

    # 2. Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    plots["predicted_vs_actual"] = fig

    # 3. Feature importance
    if feature_importances is not None and feature_names is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importances)
        ax.barh(
            [feature_names[i] for i in sorted_idx],
            feature_importances[sorted_idx],
        )
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        fig.tight_layout()
        plots["feature_importance"] = fig

    # 4. Error distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    fig.tight_layout()
    plots["error_distribution"] = fig

    return plots


def evaluate_model(run_id: str, test_data_path: Optional[str] = None) -> Dict[str, float]:
    """Load model from an MLflow run, score test set, and log metrics and plots.

    Args:
        run_id: MLflow run ID containing the model.
        test_data_path: Path to .npz test data file. If None, downloads from
            the run's artifacts.

    Returns:
        Dictionary of computed metrics.
    """
    logger.info("Evaluating model from run %s", run_id)

    # Load test data
    if test_data_path is None:
        client = mlflow.tracking.MlflowClient()
        local_dir = client.download_artifacts(run_id, "test_data")
        # Find the .npz file in the downloaded directory
        npz_files = [f for f in os.listdir(local_dir) if f.endswith(".npz")]
        if not npz_files:
            raise FileNotFoundError(f"No .npz test data found in {local_dir}")
        test_data_path = os.path.join(local_dir, npz_files[0])

    test_data = np.load(test_data_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    feature_names = list(test_data["feature_names"])

    # Load model and predict
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    logger.info("Evaluation metrics: %s", metrics)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Generate plots
    plots = generate_plots(y_test, y_pred, feature_importances, feature_names)

    # Log to the run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

        # Log plots as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, fig in plots.items():
                path = os.path.join(tmpdir, f"{name}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(path, "evaluation_plots")

            # Log summary JSON
            summary_path = os.path.join(tmpdir, "evaluation_summary.json")
            with open(summary_path, "w") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(summary_path)

    logger.info("Evaluation complete for run %s", run_id)
    return metrics
