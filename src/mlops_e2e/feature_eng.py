"""Feature engineering: derived features, log transforms, and feature table creation."""

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
    return feature_table
