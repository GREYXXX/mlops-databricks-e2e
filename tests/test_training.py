"""Tests for training module."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TestCreateOptunaObjective:
    """Tests for create_optuna_objective()."""

    def test_returns_callable(self):
        from mlops_e2e.housing.training import create_optuna_objective

        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)

        obj = create_optuna_objective(X_train, y_train, X_val, y_val)
        assert callable(obj)

    def test_objective_returns_float(self):
        import optuna

        from mlops_e2e.housing.training import create_optuna_objective

        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)

        obj = create_optuna_objective(X_train, y_train, X_val, y_val)

        study = optuna.create_study(direction="minimize")
        study.optimize(obj, n_trials=2, show_progress_bar=False)

        assert len(study.trials) == 2
        assert isinstance(study.best_value, float)
        assert study.best_value > 0  # RMSE is always positive

    def test_objective_minimizes_over_trials(self):
        """Verify Optuna can find params that produce lower RMSE (over a few trials)."""
        import optuna

        from mlops_e2e.housing.training import create_optuna_objective

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 3)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n) * 0.1

        obj = create_optuna_objective(X[:400], y[:400], X[400:], y[400:])

        study = optuna.create_study(direction="minimize")
        study.optimize(obj, n_trials=5, show_progress_bar=False)

        # Best RMSE should be reasonable (well below random baseline)
        assert study.best_value < 1.0


class TestTrainWithTuning:
    """Tests for train_with_tuning()."""

    def test_returns_run_id(self, spark, sample_housing_spark):
        from mlops_e2e.housing.training import train_with_tuning

        db_name = "test_training"

        # Clean up any stale state from prior runs (including filesystem)
        spark.sql(f"DROP TABLE IF EXISTS {db_name}.california_housing_features")
        spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
        db_path = Path("/tmp/spark-warehouse-test") / f"{db_name}.db"
        if db_path.exists():
            shutil.rmtree(db_path)
        spark.sql(f"CREATE DATABASE {db_name}")

        # Create the feature table (with derived + log-transformed columns)
        import pyspark.sql.functions as F

        from mlops_e2e.housing.feature_eng import add_derived_features, log_transform_skewed

        df = add_derived_features(sample_housing_spark)
        df = log_transform_skewed(df, ["Population", "AveRooms", "AveBedrms"])
        df = df.withColumn("_feature_timestamp", F.current_timestamp())
        df.write.mode("overwrite").saveAsTable(f"{db_name}.california_housing_features")

        with patch(
            "mlops_e2e.housing.training.get_full_table_name",
            return_value=f"{db_name}.california_housing_features",
        ):
            run_id = train_with_tuning(
                spark=spark,
                catalog="c",
                schema="s",
                experiment_name="/test_mlops_e2e_training",
                n_trials=3,  # Small for speed
            )

        assert isinstance(run_id, str)
        assert len(run_id) == 32  # MLflow run IDs are 32 hex chars

        # Verify artifacts exist in the run
        import mlflow

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert "best_val_rmse" in run.data.metrics
        assert float(run.data.metrics["best_val_rmse"]) > 0

        # Check test data artifact
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "test_data" in artifacts

        # Verify model was logged (MLflow 3.x stores named models separately)
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        assert model is not None

        # Cleanup
        spark.sql(f"DROP TABLE IF EXISTS {db_name}.california_housing_features")
        spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
        if db_path.exists():
            shutil.rmtree(db_path)


class TestSplitData:
    """Tests for _split_data() helper."""

    def test_split_ratios(self):
        import pandas as pd

        from mlops_e2e.housing.training import _split_data

        np.random.seed(42)
        n = 1000
        pdf = pd.DataFrame(
            {
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )

        X_train, X_val, X_test, y_train, y_val, y_test = _split_data(pdf, "target", ["f1", "f2"])

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == n

        # Approximate 70/15/15 split (allow some tolerance)
        assert 0.65 < len(X_train) / n < 0.75
        assert 0.10 < len(X_val) / n < 0.20
        assert 0.10 < len(X_test) / n < 0.20
