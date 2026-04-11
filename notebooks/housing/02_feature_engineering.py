# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 2: Feature Engineering
# MAGIC Transform raw features into a training-ready feature table with derived ratios and log transforms.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

from mlops_e2e.housing.feature_eng import create_feature_table

# COMMAND ----------

feature_table = create_feature_table(spark, catalog, schema)
print(f"Feature table created: {feature_table}")

# COMMAND ----------

features_df = spark.table(feature_table)
print(f"Feature table has {features_df.count()} rows and {len(features_df.columns)} columns")
print(f"Columns: {features_df.columns}")
display(features_df.limit(10))
