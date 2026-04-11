# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 1: Data Preparation — 20 Newsgroups
# MAGIC Fetch the 20 Newsgroups dataset and persist it as a Delta table in Unity Catalog.
# MAGIC Both train and test splits are stored together with a `split` column.

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

from mlops_e2e.newsgroups.data import load_newsgroups, save_raw_table

# COMMAND ----------

df = load_newsgroups(spark)
display(df)

# COMMAND ----------

table_name = save_raw_table(spark, df, catalog, schema)
print(f"Saved raw data to: {table_name}")

# COMMAND ----------

train_count = spark.table(table_name).filter("split = 'train'").count()
test_count = spark.table(table_name).filter("split = 'test'").count()
print(f"Train: {train_count} rows | Test: {test_count} rows")
