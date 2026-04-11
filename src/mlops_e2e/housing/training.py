"""Model training with LightGBM and Optuna hyperparameter tuning."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlops_e2e.config import get_full_table_name
from mlops_e2e.housing.config import FEATURES_TABLE_NAME

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
    pdf: np.ndarray, target_col: str, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split pandas DataFrame into train/val/test (70/15/15)."""
    import pandas as pd

    df = (
        pd.DataFrame(pdf, columns=feature_cols + [target_col])
        if not isinstance(pdf, pd.DataFrame)
        else pdf
    )
    X = df[feature_cols].values
    y = df[target_col].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.176,
        random_state=42,  # 0.176 * 0.85 ≈ 0.15
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_with_tuning(
    spark: SparkSession,
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

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(df, target_col, feature_cols)

    logger.info("Data split: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    mlflow.set_experiment(experiment_name)
    mlflow.lightgbm.autolog(log_models=False)

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

        best_trial = study.best_trial
        mlflow.log_metric("best_val_rmse", best_trial.value)
        for key, value in best_trial.params.items():
            mlflow.log_param(f"best_{key}", value)

        logger.info("Training final model with best params: %s", best_trial.params)
        best_model = lgb.LGBMRegressor(**best_trial.params, verbose=-1, random_state=42)
        best_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        mlflow.lightgbm.log_model(
            best_model,
            name="model",
            input_example=X_train[:5],
        )

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
    return best_run_id
