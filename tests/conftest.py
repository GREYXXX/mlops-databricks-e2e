"""Shared fixtures and mocks for MLOps E2E tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    from pyspark.sql import SparkSession

    java_opts = (
        "-Dderby.system.home=/tmp/derby-test "
        "-Djava.security.manager=allow"
    )
    session = (
        SparkSession.builder.master("local[2]")
        .appName("mlops_e2e_tests")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse-test")
        .config("spark.driver.extraJavaOptions", java_opts)
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_housing_pdf() -> pd.DataFrame:
    """Return a small sample California Housing-like DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "MedInc": np.random.uniform(0.5, 15.0, n),
        "HouseAge": np.random.uniform(1, 52, n),
        "AveRooms": np.random.uniform(1, 15, n),
        "AveBedrms": np.random.uniform(0.3, 5, n),
        "Population": np.random.uniform(3, 35000, n),
        "AveOccup": np.random.uniform(1, 6, n),
        "Latitude": np.random.uniform(32, 42, n),
        "Longitude": np.random.uniform(-124, -114, n),
        "MedHouseVal": np.random.uniform(0.15, 5.0, n),
    })


@pytest.fixture
def sample_housing_spark(spark, sample_housing_pdf):
    """Return a Spark DataFrame with sample housing data."""
    return spark.createDataFrame(sample_housing_pdf)


@pytest.fixture
def sample_test_data() -> dict:
    """Return sample test data arrays for evaluation/champion tests."""
    np.random.seed(42)
    n = 100
    n_features = 11
    X_test = np.random.randn(n, n_features)
    y_test = np.random.uniform(0.15, 5.0, n)
    feature_names = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
        "AveOccup", "Latitude", "Longitude",
        "rooms_per_household", "bedrooms_ratio", "population_per_household",
    ]
    return {
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": np.array(feature_names),
    }


@pytest.fixture
def mock_dbutils():
    """Return a mock dbutils object with task values support."""
    dbutils = MagicMock()
    task_values = {}

    def set_task_value(key, value):
        task_values[key] = value

    def get_task_value(taskKey, key, debugValue=None):
        return task_values.get(key, debugValue)

    dbutils.jobs.taskValues.set = MagicMock(side_effect=set_task_value)
    dbutils.jobs.taskValues.get = MagicMock(side_effect=get_task_value)
    dbutils.widgets.get = MagicMock(side_effect=lambda k: {"catalog": "test_catalog", "schema": "test_schema", "model_name": "test_model"}.get(k, ""))
    dbutils._task_values = task_values
    return dbutils


@pytest.fixture
def mock_mlflow_client():
    """Return a mock MlflowClient."""
    client = MagicMock()
    return client
