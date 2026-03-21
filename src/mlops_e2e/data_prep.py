"""Data preparation: load California Housing dataset and write to Delta table."""

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
    logger.info(
        "Loaded California Housing dataset: %d rows, %d columns", len(pdf), len(pdf.columns)
    )
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
    return table_name
