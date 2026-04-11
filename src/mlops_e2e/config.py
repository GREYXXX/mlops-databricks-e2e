"""Shared configuration utilities for all MLOps E2E pipelines.

Pipeline-specific constants (table names, model names, experiment paths) live in
their respective subpackage configs:
  - mlops_e2e.housing.config   — California Housing regression pipeline
  - mlops_e2e.newsgroups.config — 20 Newsgroups ensemble classifier pipeline
"""

from __future__ import annotations

import os


def _get_widget_or_env(key: str, default: str) -> str:
    """Read a value from Databricks notebook widgets, env vars, or use default.

    Attempts dbutils.widgets.get first (when running in a notebook context),
    then falls back to environment variables, then to the provided default.
    """
    try:
        from pyspark.dbutils import DBUtils  # type: ignore[import-untyped]
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is not None:
            dbutils = DBUtils(spark)
            return dbutils.widgets.get(key)
    except Exception:
        pass

    return os.environ.get(key.upper(), default)


def get_catalog(override: str | None = None) -> str:
    """Get the Unity Catalog catalog name."""
    if override:
        return override
    return _get_widget_or_env("catalog", "workspace")


def get_schema(override: str | None = None) -> str:
    """Get the Unity Catalog schema name."""
    if override:
        return override
    return _get_widget_or_env("schema", "mlops_e2e")


def get_model_name(override: str | None = None) -> str:
    """Get the registered model name."""
    if override:
        return override
    return _get_widget_or_env("model_name", "")


def get_full_table_name(catalog: str, schema: str, table: str) -> str:
    """Return fully-qualified 3-level table name."""
    return f"{catalog}.{schema}.{table}"


def get_full_model_name(catalog: str, schema: str, model_name: str) -> str:
    """Return fully-qualified 3-level model name."""
    return f"{catalog}.{schema}.{model_name}"
