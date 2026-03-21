"""Tests for feature_eng module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pyspark.sql.functions as F


class TestAddDerivedFeatures:
    """Tests for add_derived_features()."""

    def test_adds_three_columns(self, sample_housing_spark):
        from mlops_e2e.feature_eng import add_derived_features

        result = add_derived_features(sample_housing_spark)
        new_cols = {"rooms_per_household", "bedrooms_ratio", "population_per_household"}
        assert new_cols.issubset(set(result.columns))

    def test_rooms_per_household_formula(self, spark):
        from mlops_e2e.feature_eng import add_derived_features

        df = spark.createDataFrame(
            [
                (5.0, 3.0, 2.0, 1.0, 100.0, 3.0, 34.0, -118.0, 2.0),
            ],
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
                "MedHouseVal",
            ],
        )

        result = add_derived_features(df).collect()[0]
        # rooms_per_household = AveRooms * AveOccup = 2.0 * 3.0 = 6.0
        assert abs(result["rooms_per_household"] - 6.0) < 1e-6

    def test_bedrooms_ratio_formula(self, spark):
        from mlops_e2e.feature_eng import add_derived_features

        df = spark.createDataFrame(
            [
                (5.0, 3.0, 4.0, 1.0, 100.0, 3.0, 34.0, -118.0, 2.0),
            ],
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
                "MedHouseVal",
            ],
        )

        result = add_derived_features(df).collect()[0]
        # bedrooms_ratio = AveBedrms / AveRooms = 1.0 / 4.0 = 0.25
        assert abs(result["bedrooms_ratio"] - 0.25) < 1e-6

    def test_bedrooms_ratio_zero_rooms(self, spark):
        from mlops_e2e.feature_eng import add_derived_features

        df = spark.createDataFrame(
            [
                (5.0, 3.0, 0.0, 1.0, 100.0, 3.0, 34.0, -118.0, 2.0),
            ],
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
                "MedHouseVal",
            ],
        )

        result = add_derived_features(df).collect()[0]
        assert result["bedrooms_ratio"] == 0.0

    def test_population_per_household_formula(self, spark):
        from mlops_e2e.feature_eng import add_derived_features

        df = spark.createDataFrame(
            [
                (5.0, 3.0, 4.0, 1.0, 300.0, 3.0, 34.0, -118.0, 2.0),
            ],
            [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
                "MedHouseVal",
            ],
        )

        result = add_derived_features(df).collect()[0]
        # population_per_household = Population / AveOccup = 300.0 / 3.0 = 100.0
        assert abs(result["population_per_household"] - 100.0) < 1e-6

    def test_preserves_existing_columns(self, sample_housing_spark):
        from mlops_e2e.feature_eng import add_derived_features

        original_cols = set(sample_housing_spark.columns)
        result = add_derived_features(sample_housing_spark)
        assert original_cols.issubset(set(result.columns))

    def test_no_nan_in_derived(self, sample_housing_spark):
        from mlops_e2e.feature_eng import add_derived_features

        result = add_derived_features(sample_housing_spark)
        for col_name in ["rooms_per_household", "bedrooms_ratio", "population_per_household"]:
            nan_count = result.filter(F.isnan(F.col(col_name)) | F.col(col_name).isNull()).count()
            assert nan_count == 0, f"Found NaN/null in {col_name}"


class TestLogTransformSkewed:
    """Tests for log_transform_skewed()."""

    def test_applies_log1p(self, spark):
        from mlops_e2e.feature_eng import log_transform_skewed

        df = spark.createDataFrame([(10.0, 20.0)], ["a", "b"])
        result = log_transform_skewed(df, ["a"]).collect()[0]

        expected = np.log1p(10.0)
        assert abs(result["a"] - expected) < 1e-6
        # b should be unchanged
        assert result["b"] == 20.0

    def test_multiple_columns(self, spark):
        from mlops_e2e.feature_eng import log_transform_skewed

        df = spark.createDataFrame([(5.0, 10.0, 99.0)], ["a", "b", "c"])
        result = log_transform_skewed(df, ["a", "b"]).collect()[0]

        assert abs(result["a"] - np.log1p(5.0)) < 1e-6
        assert abs(result["b"] - np.log1p(10.0)) < 1e-6
        assert result["c"] == 99.0

    def test_zero_value(self, spark):
        from mlops_e2e.feature_eng import log_transform_skewed

        df = spark.createDataFrame([(0.0,)], ["a"])
        result = log_transform_skewed(df, ["a"]).collect()[0]
        assert abs(result["a"] - np.log1p(0.0)) < 1e-6


class TestCreateFeatureTable:
    """Tests for create_feature_table()."""

    def test_end_to_end_pipeline(self, spark, sample_housing_spark):
        from mlops_e2e.feature_eng import create_feature_table

        db_name = "test_feat"
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")

        # Write source table
        sample_housing_spark.write.mode("overwrite").saveAsTable(
            f"{db_name}.california_housing_raw"
        )

        with patch("mlops_e2e.feature_eng.get_full_table_name") as mock_name:
            mock_name.side_effect = lambda cat, sch, tbl: f"{db_name}.{tbl}"
            result = create_feature_table(spark, "c", "s")

        assert result == f"{db_name}.california_housing_features"
        feat_df = spark.table(f"{db_name}.california_housing_features")
        assert feat_df.count() == 200
        assert "rooms_per_household" in feat_df.columns
        assert "bedrooms_ratio" in feat_df.columns
        assert "population_per_household" in feat_df.columns
        assert "_feature_timestamp" in feat_df.columns

        spark.sql(f"DROP TABLE IF EXISTS {db_name}.california_housing_raw")
        spark.sql(f"DROP TABLE IF EXISTS {db_name}.california_housing_features")
        spark.sql(f"DROP DATABASE IF EXISTS {db_name}")
