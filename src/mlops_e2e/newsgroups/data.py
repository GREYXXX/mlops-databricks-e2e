"""Data preparation: fetch 20 Newsgroups dataset and write to Delta table."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from mlops_e2e.config import get_full_table_name
from mlops_e2e.newsgroups.config import RAW_TABLE_NAME, SEED

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def load_newsgroups(spark: SparkSession) -> DataFrame:
    """Fetch the 20 Newsgroups dataset and return as a Spark DataFrame.

    Strips headers/footers/quotes so the classification task is non-trivial.
    Returns both train and test splits in a single table with a ``split`` column.

    Args:
        spark: Active SparkSession.

    Returns:
        Spark DataFrame with columns: text (str), label (int), split (str).
    """
    remove = ("headers", "footers", "quotes")
    train = fetch_20newsgroups(subset="train", remove=remove, random_state=SEED)
    test = fetch_20newsgroups(subset="test", remove=remove, random_state=SEED)

    train_pdf = pd.DataFrame({"text": train.data, "label": train.target, "split": "train"})
    test_pdf = pd.DataFrame({"text": test.data, "label": test.target, "split": "test"})
    pdf = pd.concat([train_pdf, test_pdf], ignore_index=True)

    logger.info(
        "Loaded 20 Newsgroups: train=%d, test=%d, classes=%d",
        len(train_pdf),
        len(test_pdf),
        len(train.target_names),
    )
    return spark.createDataFrame(pdf)


def save_raw_table(
    spark: SparkSession,
    df: DataFrame,
    catalog: str,
    schema: str,
) -> str:
    """Write the newsgroups DataFrame as a Delta table in Unity Catalog.

    Args:
        spark: Active SparkSession.
        df: Spark DataFrame to persist.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.

    Returns:
        Fully-qualified table name that was written.
    """
    table_name = get_full_table_name(catalog, schema, RAW_TABLE_NAME)
    logger.info("Writing raw newsgroups data to %s", table_name)
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    row_count = spark.table(table_name).count()
    logger.info("Wrote %d rows to %s", row_count, table_name)
    return table_name


def load_raw_table(
    spark: SparkSession,
    catalog: str,
    schema: str,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Read the raw newsgroups Delta table and return train/test splits.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.

    Returns:
        Tuple of (train_texts, test_texts, train_labels, test_labels).
    """
    table_name = get_full_table_name(catalog, schema, RAW_TABLE_NAME)
    df = spark.table(table_name).toPandas()

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    logger.info("Loaded %d train, %d test rows from %s", len(train), len(test), table_name)
    return (
        train["text"].tolist(),
        test["text"].tolist(),
        train["label"].tolist(),
        test["label"].tolist(),
    )
