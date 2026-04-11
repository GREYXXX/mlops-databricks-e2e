"""Configuration constants for the California Housing regression pipeline."""

from __future__ import annotations

# Unity Catalog defaults
DEFAULT_CATALOG = "workspace"
DEFAULT_SCHEMA = "mlops_e2e"
DEFAULT_MODEL_NAME = "california_housing_model"

# Table names
RAW_TABLE_NAME = "california_housing_raw"
FEATURES_TABLE_NAME = "california_housing_features"

# MLflow experiment
EXPERIMENT_NAME = "/mlops_e2e_california_housing"
