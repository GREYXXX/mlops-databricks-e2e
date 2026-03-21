"""Configuration constants for the MLOps E2E pipeline."""

from __future__ import annotations

import os


def _get_widget_or_env(key: str, default: str) -> str:
    """Read a value from Databricks notebook widgets, env vars, or use default.

    Attempts dbutils.widgets.get first (when running in a notebook context),
    then falls back to environment variables, then to the provided default.
    """
    # Try dbutils widgets (notebook context)
    try:
        from pyspark.dbutils import DBUtils  # type: ignore[import-untyped]
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is not None:
            dbutils = DBUtils(spark)
            return dbutils.widgets.get(key)
    except Exception:
        pass

    # Fall back to environment variable
    return os.environ.get(key.upper(), default)


# Unity Catalog coordinates
DEFAULT_CATALOG = "workspace"
DEFAULT_SCHEMA = "mlops_e2e"
DEFAULT_MODEL_NAME = "california_housing_model"

# Table names
RAW_TABLE_NAME = "california_housing_raw"
FEATURES_TABLE_NAME = "california_housing_features"

# MLflow experiment
EXPERIMENT_NAME = "/mlops_e2e_california_housing"


def get_catalog(override: str | None = None) -> str:
    """Get the Unity Catalog catalog name."""
    if override:
        return override
    return _get_widget_or_env("catalog", DEFAULT_CATALOG)


def get_schema(override: str | None = None) -> str:
    """Get the Unity Catalog schema name."""
    if override:
        return override
    return _get_widget_or_env("schema", DEFAULT_SCHEMA)


def get_model_name(override: str | None = None) -> str:
    """Get the registered model name."""
    if override:
        return override
    return _get_widget_or_env("model_name", DEFAULT_MODEL_NAME)


def get_full_table_name(catalog: str, schema: str, table: str) -> str:
    """Return fully-qualified 3-level table name."""
    return f"{catalog}.{schema}.{table}"


def get_full_model_name(catalog: str, schema: str, model_name: str) -> str:
    """Return fully-qualified 3-level model name."""
    return f"{catalog}.{schema}.{model_name}"
