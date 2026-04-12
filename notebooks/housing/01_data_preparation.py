# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 1: Data Preparation
# MAGIC Load the California Housing dataset and persist as a Delta table in Unity Catalog.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------

from mlops_e2e.housing.data_prep import load_california_housing, save_raw_table

# COMMAND ----------

df = load_california_housing(spark)
display(df)

# COMMAND ----------

table_name = save_raw_table(spark, df, catalog, schema)
print(f"Saved raw data to: {table_name}")

# COMMAND ----------

row_count = spark.table(table_name).count()
print(f"Table {table_name} has {row_count} rows")
