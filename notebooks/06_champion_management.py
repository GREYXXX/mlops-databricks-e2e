# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 6: Champion Management
# MAGIC Compare the Challenger model against the current Champion and promote if better.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")
dbutils.widgets.text("model_name", "california_housing_model", "Model Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

full_model_name = f"{catalog}.{schema}.{model_name}"
print(f"Managing champion for: {full_model_name}")

# COMMAND ----------

from mlops_e2e.champion import run_champion_management

# COMMAND ----------

# Run the champion management pipeline
result = run_champion_management(
    model_name=full_model_name,
    catalog=catalog,
    schema=schema,
)

# COMMAND ----------

print(f"Action: {result['action']}")
print(f"Reason: {result['reason']}")
print(f"Challenger version: {result['challenger_version']}")
if "champion_version" in result:
    print(f"Champion version: {result['champion_version']}")
