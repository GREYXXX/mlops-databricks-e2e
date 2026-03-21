export interface ImportedFunction {
  name: string;
  module: string;
}

export interface StageCodeEntry {
  key: string;
  label: string;
  description: string;
  sourceFile: string;
  notebookFile: string;
  sourceCode: string;
  notebookCode: string;
  importedFunctions?: ImportedFunction[];
}

/**
 * Extract a top-level Python function from source code by name.
 * Captures from `def funcName(` until the next top-level `def ` or end of string.
 */
export function extractFunctionCode(
  sourceCode: string,
  funcName: string
): string | null {
  const lines = sourceCode.split("\n");
  let startIdx = -1;

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].match(new RegExp(`^def ${funcName}\\b`))) {
      startIdx = i;
      break;
    }
  }
  if (startIdx === -1) return null;

  let endIdx = lines.length;
  for (let i = startIdx + 1; i < lines.length; i++) {
    if (lines[i].match(/^def \w/) || lines[i].match(/^class \w/)) {
      // Walk back to skip blank lines before the next def/class
      let j = i - 1;
      while (j > startIdx && lines[j].trim() === "") j--;
      endIdx = j + 1;
      break;
    }
  }

  return lines.slice(startIdx, endIdx).join("\n");
}

export const STAGE_CODE: Record<string, StageCodeEntry> = {
  data_preparation: {
    key: "data_preparation",
    label: "Data Preparation",
    description: "Load the California Housing dataset and persist as a Delta table in Unity Catalog.",
    sourceFile: "src/mlops_e2e/data_prep.py",
    notebookFile: "notebooks/01_data_preparation.py",
    sourceCode: `"""Data preparation: load California Housing dataset and write to Delta table."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.datasets import fetch_california_housing

from mlops_e2e.config import RAW_TABLE_NAME, get_full_table_name

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def load_california_housing(spark: SparkSession) -> DataFrame:
    """Load the California Housing dataset and return as a Spark DataFrame.

    Args:
        spark: Active SparkSession.

    Returns:
        Spark DataFrame with all features and the MedHouseVal target column.
    """
    data = fetch_california_housing(as_frame=True)
    pdf: pd.DataFrame = data.frame  # type: ignore[union-attr]
    logger.info("Loaded California Housing dataset: %d rows, %d columns", len(pdf), len(pdf.columns))
    spark_df = spark.createDataFrame(pdf)
    return spark_df


def save_raw_table(
    spark: SparkSession,
    df: DataFrame,
    catalog: str,
    schema: str,
) -> str:
    """Write the DataFrame as a Delta table in Unity Catalog.

    Args:
        spark: Active SparkSession.
        df: Spark DataFrame to persist.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.

    Returns:
        Fully-qualified table name that was written.
    """
    table_name = get_full_table_name(catalog, schema, RAW_TABLE_NAME)
    logger.info("Writing raw data to %s", table_name)
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    row_count = spark.table(table_name).count()
    logger.info("Wrote %d rows to %s", row_count, table_name)
    return table_name`,
    notebookCode: `# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 1: Data Preparation
# MAGIC Load the California Housing dataset and persist as a Delta table in Unity Catalog.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

# Ensure the schema exists (catalog is pre-provisioned by the workspace)
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

from mlops_e2e.data_prep import load_california_housing, save_raw_table

# COMMAND ----------

# Load the California Housing dataset
df = load_california_housing(spark)
display(df)

# COMMAND ----------

# Save to Delta table
table_name = save_raw_table(spark, df, catalog, schema)
print(f"Saved raw data to: {table_name}")

# COMMAND ----------

# Verify the table
row_count = spark.table(table_name).count()
print(f"Table {table_name} has {row_count} rows")`,
    importedFunctions: [
      { name: "load_california_housing", module: "mlops_e2e.data_prep" },
      { name: "save_raw_table", module: "mlops_e2e.data_prep" },
    ],
  },

  feature_engineering: {
    key: "feature_engineering",
    label: "Feature Engineering",
    description: "Transform raw features into a training-ready feature table with derived ratios and log transforms.",
    sourceFile: "src/mlops_e2e/feature_eng.py",
    notebookFile: "notebooks/02_feature_engineering.py",
    sourceCode: `"""Feature engineering: derived features, log transforms, and feature table creation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import pyspark.sql.functions as F

from mlops_e2e.config import (
    FEATURES_TABLE_NAME,
    RAW_TABLE_NAME,
    get_full_table_name,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def add_derived_features(df: DataFrame) -> DataFrame:
    """Add derived features to the DataFrame.

    New columns:
        - rooms_per_household: AveRooms * AveOccup (feature interaction term)
        - bedrooms_ratio: AveBedrms / AveRooms
        - population_per_household: Population / AveOccup (approx. number of households)

    Args:
        df: Spark DataFrame with raw California Housing columns.

    Returns:
        DataFrame with three additional derived columns.
    """
    df = df.withColumn("rooms_per_household", F.col("AveRooms") * F.col("AveOccup"))
    df = df.withColumn(
        "bedrooms_ratio",
        F.when(F.col("AveRooms") != 0, F.col("AveBedrms") / F.col("AveRooms")).otherwise(0.0),
    )
    df = df.withColumn("population_per_household", F.col("Population") / F.col("AveOccup"))
    logger.info("Added derived features: rooms_per_household, bedrooms_ratio, population_per_household")
    return df


def log_transform_skewed(df: DataFrame, columns: List[str]) -> DataFrame:
    """Apply log1p transformation to specified skewed columns.

    Args:
        df: Spark DataFrame.
        columns: List of column names to transform.

    Returns:
        DataFrame with specified columns replaced by their log1p values.
    """
    for col_name in columns:
        df = df.withColumn(col_name, F.log1p(F.col(col_name)))
    logger.info("Applied log1p transform to columns: %s", columns)
    return df


def create_feature_table(spark: SparkSession, catalog: str, schema: str) -> str:
    """Full feature engineering pipeline: read raw table, transform, write features table.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.

    Returns:
        Fully-qualified feature table name.
    """
    raw_table = get_full_table_name(catalog, schema, RAW_TABLE_NAME)
    feature_table = get_full_table_name(catalog, schema, FEATURES_TABLE_NAME)

    logger.info("Reading raw data from %s", raw_table)
    df = spark.table(raw_table)

    # Add derived features
    df = add_derived_features(df)

    # Log-transform skewed columns
    skewed_columns = ["Population", "AveRooms", "AveBedrms"]
    df = log_transform_skewed(df, skewed_columns)

    # Add lineage timestamp
    df = df.withColumn("_feature_timestamp", F.current_timestamp())

    # Write feature table
    logger.info("Writing feature table to %s", feature_table)
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(feature_table)

    row_count = spark.table(feature_table).count()
    logger.info("Wrote %d rows to %s", row_count, feature_table)
    return feature_table`,
    notebookCode: `# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 2: Feature Engineering
# MAGIC Transform raw features into a training-ready feature table with derived ratios and log transforms.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

from mlops_e2e.feature_eng import create_feature_table

# COMMAND ----------

# Run the full feature engineering pipeline
feature_table = create_feature_table(spark, catalog, schema)
print(f"Feature table created: {feature_table}")

# COMMAND ----------

# Verify the feature table
features_df = spark.table(feature_table)
print(f"Feature table has {features_df.count()} rows and {len(features_df.columns)} columns")
print(f"Columns: {features_df.columns}")
display(features_df.limit(10))`,
    importedFunctions: [
      { name: "create_feature_table", module: "mlops_e2e.feature_eng" },
    ],
  },

  model_training: {
    key: "model_training",
    label: "Model Training",
    description: "Train a LightGBM model with Optuna hyperparameter tuning, logging everything to MLflow.",
    sourceFile: "src/mlops_e2e/training.py",
    notebookFile: "notebooks/03_model_training.py",
    sourceCode: `"""Model training with LightGBM and Optuna hyperparameter tuning."""

from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING, Callable, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlops_e2e.config import FEATURES_TABLE_NAME, get_full_table_name

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def create_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for LightGBM hyperparameter tuning.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        X_val: Validation feature matrix.
        y_val: Validation target array.

    Returns:
        Callable objective function that returns validation RMSE.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        return rmse

    return objective


def _split_data(
    pdf: "np.ndarray", target_col: str, feature_cols: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split pandas DataFrame into train/val/test (70/15/15)."""
    import pandas as pd

    df = pd.DataFrame(pdf, columns=feature_cols + [target_col]) if not isinstance(pdf, pd.DataFrame) else pdf
    X = df[feature_cols].values
    y = df[target_col].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_with_tuning(
    spark: "SparkSession",
    catalog: str,
    schema: str,
    experiment_name: str,
    n_trials: int = 50,
) -> str:
    """Full training pipeline: read features, tune, log best model.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        experiment_name: MLflow experiment name/path.
        n_trials: Number of Optuna trials.

    Returns:
        The best MLflow run ID.
    """
    feature_table = get_full_table_name(catalog, schema, FEATURES_TABLE_NAME)
    logger.info("Reading feature table from %s", feature_table)

    df = spark.table(feature_table).drop("_feature_timestamp").toPandas()

    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        df, target_col, feature_cols
    )

    logger.info(
        "Data split: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Enable LightGBM autologging
    mlflow.lightgbm.autolog(log_models=False)

    # Run Optuna study inside a parent MLflow run
    with mlflow.start_run(run_name="optuna_tuning") as parent_run:
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("feature_table", feature_table)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="val_rmse",
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )

        study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
        objective = create_optuna_objective(X_train, y_train, X_val, y_val)
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

        # Log best trial info to parent run
        best_trial = study.best_trial
        mlflow.log_metric("best_val_rmse", best_trial.value)
        for key, value in best_trial.params.items():
            mlflow.log_param(f"best_{key}", value)

        # Train final model with best params and log it
        logger.info("Training final model with best params: %s", best_trial.params)
        best_model = lgb.LGBMRegressor(**best_trial.params, verbose=-1, random_state=42)
        best_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Log the final model
        mlflow.lightgbm.log_model(
            best_model,
            name="model",
            input_example=X_train[:5],
        )

        # Save test data for evaluation stage
        test_data_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        test_data_path = test_data_file.name
        test_data_file.close()
        np.savez(
            test_data_path,
            X_test=X_test,
            y_test=y_test,
            feature_names=np.array(feature_cols),
        )
        mlflow.log_artifact(test_data_path, "test_data")

        best_run_id = parent_run.info.run_id
        logger.info("Best run ID: %s with val RMSE: %.4f", best_run_id, best_trial.value)

    mlflow.lightgbm.autolog(disable=True)
    return best_run_id`,
    notebookCode: `# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 3: Model Training
# MAGIC Train a LightGBM model with Optuna hyperparameter tuning, logging everything to MLflow.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

from mlops_e2e.training import train_with_tuning

# COMMAND ----------

# Build experiment path under current user's namespace to avoid conflicts
username = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{username}/mlops_e2e_california_housing"

# Run the full training pipeline with Optuna tuning
best_run_id = train_with_tuning(
    spark=spark,
    catalog=catalog,
    schema=schema,
    experiment_name=experiment_name,
    n_trials=50,
)

print(f"Best run ID: {best_run_id}")

# COMMAND ----------

# Pass the best_run_id to downstream tasks via task values
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
print(f"Set task value best_run_id={best_run_id}")`,
    importedFunctions: [
      { name: "train_with_tuning", module: "mlops_e2e.training" },
    ],
  },

  model_evaluation: {
    key: "model_evaluation",
    label: "Model Evaluation",
    description: "Compute evaluation metrics and generate comparison artifacts for the best model.",
    sourceFile: "src/mlops_e2e/evaluation.py",
    notebookFile: "notebooks/04_model_evaluation.py",
    sourceCode: `"""Model evaluation: metrics computation and plot generation."""

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
    return metrics`,
    notebookCode: `# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 4: Model Evaluation
# MAGIC Compute evaluation metrics and generate comparison artifacts for the best model.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

# Get best_run_id from the training stage
best_run_id = dbutils.jobs.taskValues.get(
    taskKey="model_training",
    key="best_run_id",
)
print(f"Evaluating model from run: {best_run_id}")

# COMMAND ----------

from mlops_e2e.evaluation import evaluate_model

# COMMAND ----------

# Evaluate the model
metrics = evaluate_model(run_id=best_run_id)
print("Evaluation metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

# COMMAND ----------

# Pass best_run_id to downstream stages
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)`,
    importedFunctions: [
      { name: "evaluate_model", module: "mlops_e2e.evaluation" },
    ],
  },

  model_registration: {
    key: "model_registration",
    label: "Model Registration",
    description: "Register the best model to Unity Catalog and assign the \"Challenger\" alias.",
    sourceFile: "src/mlops_e2e/registration.py",
    notebookFile: "notebooks/05_model_registration.py",
    sourceCode: `"""Model registration to Unity Catalog."""

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
    logger.info("Set alias '%s' on %s version %s", alias, model_name, version)`,
    notebookCode: `# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 5: Model Registration
# MAGIC Register the best model to Unity Catalog and assign the "Challenger" alias.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")
dbutils.widgets.text("model_name", "california_housing_model", "Model Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

full_model_name = f"{catalog}.{schema}.{model_name}"
print(f"Registering to: {full_model_name}")

# COMMAND ----------

# Get best_run_id from upstream task
best_run_id = dbutils.jobs.taskValues.get(
    taskKey="model_evaluation",
    key="best_run_id",
)
print(f"Registering model from run: {best_run_id}")

# COMMAND ----------

from mlops_e2e.registration import register_model_to_uc, set_model_alias

# COMMAND ----------

# Register model to Unity Catalog
version = register_model_to_uc(run_id=best_run_id, model_name=full_model_name)
print(f"Registered model version: {version}")

# COMMAND ----------

# Set Challenger alias
set_model_alias(model_name=full_model_name, version=version, alias="Challenger")
print(f"Set alias 'Challenger' on version {version}")

# COMMAND ----------

# Pass model version to downstream stages
dbutils.jobs.taskValues.set(key="model_version", value=version)
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)`,
    importedFunctions: [
      { name: "register_model_to_uc", module: "mlops_e2e.registration" },
      { name: "set_model_alias", module: "mlops_e2e.registration" },
    ],
  },

  champion_management: {
    key: "champion_management",
    label: "Champion Management",
    description: "Compare the Challenger model against the current Champion and promote if better.",
    sourceFile: "src/mlops_e2e/champion.py",
    notebookFile: "notebooks/06_champion_management.py",
    sourceCode: `"""Champion/challenger model comparison and promotion logic."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Dict, Optional

import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def get_champion_version(model_name: str) -> Optional[str]:
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
    test_data: Dict[str, np.ndarray],
) -> Dict[str, float]:
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


def promote_challenger(model_name: str, challenger_version: str) -> None:
    """Promote the Challenger to Champion by moving aliases.

    Args:
        model_name: Fully-qualified UC model name.
        challenger_version: Version to promote to Champion.
    """
    client = mlflow.tracking.MlflowClient()

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


def run_champion_management(
    model_name: str,
    catalog: str,
    schema: str,
) -> Dict[str, str]:
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
    result: Dict[str, str] = {"challenger_version": challenger_version}

    if champion_version is None:
        # First model - auto-promote
        logger.info("No existing Champion - auto-promoting Challenger version %s", challenger_version)
        promote_challenger(model_name, challenger_version)
        result["action"] = "auto_promoted"
        result["reason"] = "No existing Champion model"
    else:
        # Compare models
        result["champion_version"] = champion_version
        comparison = compare_models(
            champion_version, challenger_version, model_name, test_data
        )
        result["champion_rmse"] = str(comparison["champion_rmse"])
        result["challenger_rmse"] = str(comparison["challenger_rmse"])

        if comparison["challenger_rmse"] < comparison["champion_rmse"]:
            improvement = comparison["champion_rmse"] - comparison["challenger_rmse"]
            logger.info(
                "Challenger wins! RMSE improved by %.4f. Promoting version %s.",
                improvement,
                challenger_version,
            )
            promote_challenger(model_name, challenger_version)
            result["action"] = "promoted"
            result["reason"] = f"Challenger RMSE ({comparison['challenger_rmse']:.4f}) < Champion RMSE ({comparison['champion_rmse']:.4f})"
        else:
            logger.info(
                "Champion retains title. Champion RMSE: %.4f <= Challenger RMSE: %.4f",
                comparison["champion_rmse"],
                comparison["challenger_rmse"],
            )
            result["action"] = "rejected"
            result["reason"] = f"Champion RMSE ({comparison['champion_rmse']:.4f}) <= Challenger RMSE ({comparison['challenger_rmse']:.4f})"

    # Log comparison summary as artifact on the challenger's run
    with mlflow.start_run(run_id=challenger_run_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "champion_comparison.json")
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            mlflow.log_artifact(summary_path)

    logger.info("Champion management complete: %s", result["action"])
    return result`,
    notebookCode: `# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 6: Champion Management
# MAGIC Compare the Challenger model against the current Champion and promote if better.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")
dbutils.widgets.text("model_name", "california_housing_model", "Model Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

full_model_name = f"{catalog}.{schema}.{model_name}"
print(f"Managing champion for: {full_model_name}")

# COMMAND ----------

from mlops_e2e.champion import run_champion_management

# COMMAND ----------

# Run the champion management pipeline
result = run_champion_management(
    model_name=full_model_name,
    catalog=catalog,
    schema=schema,
)

# COMMAND ----------

print(f"Action: {result['action']}")
print(f"Reason: {result['reason']}")
print(f"Challenger version: {result['challenger_version']}")
if "champion_version" in result:
    print(f"Champion version: {result['champion_version']}")`,
    importedFunctions: [
      { name: "run_champion_management", module: "mlops_e2e.champion" },
    ],
  },
};
