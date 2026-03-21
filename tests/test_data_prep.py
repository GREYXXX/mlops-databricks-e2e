"""Tests for data_prep module."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestLoadCaliforniaHousing:
    """Tests for load_california_housing()."""

    def test_returns_spark_dataframe(self, spark):
        from mlops_e2e.data_prep import load_california_housing

        df = load_california_housing(spark)
        assert df is not None
        assert df.count() > 0

    def test_has_expected_columns(self, spark):
        from mlops_e2e.data_prep import load_california_housing

        df = load_california_housing(spark)
        expected_cols = {
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal",
        }
        assert set(df.columns) == expected_cols

    def test_no_nulls_in_target(self, spark):
        from mlops_e2e.data_prep import load_california_housing

        df = load_california_housing(spark)
        null_count = df.filter(df["MedHouseVal"].isNull()).count()
        assert null_count == 0

    def test_row_count_matches_sklearn(self, spark):
        from sklearn.datasets import fetch_california_housing

        from mlops_e2e.data_prep import load_california_housing

        expected_count = len(fetch_california_housing(as_frame=True).frame)
        df = load_california_housing(spark)
        assert df.count() == expected_count


class TestSaveRawTable:
    """Tests for save_raw_table()."""

    def test_writes_table_and_returns_name(self, spark, sample_housing_spark, tmp_path):
        from mlops_e2e.data_prep import save_raw_table

        # Use a temp database for isolation
        db_name = "test_save_raw"
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

        with patch("mlops_e2e.data_prep.get_full_table_name", return_value=f"{db_name}.california_housing_raw"):
            result = save_raw_table(spark, sample_housing_spark, "test_cat", "test_schema")

        assert result == f"{db_name}.california_housing_raw"
        count = spark.table(f"{db_name}.california_housing_raw").count()
        assert count == 200

        # Cleanup
        spark.sql(f"DROP TABLE IF EXISTS {db_name}.california_housing_raw")
        spark.sql(f"DROP DATABASE IF EXISTS {db_name}")

    def test_overwrite_mode(self, spark, sample_housing_spark):
        """Verify that writing twice doesn't double the rows."""
        from mlops_e2e.data_prep import save_raw_table

        db_name = "test_overwrite"
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

        with patch("mlops_e2e.data_prep.get_full_table_name", return_value=f"{db_name}.california_housing_raw"):
            save_raw_table(spark, sample_housing_spark, "c", "s")
            save_raw_table(spark, sample_housing_spark, "c", "s")

        count = spark.table(f"{db_name}.california_housing_raw").count()
        assert count == 200

        spark.sql(f"DROP TABLE IF EXISTS {db_name}.california_housing_raw")
        spark.sql(f"DROP DATABASE IF EXISTS {db_name}")
