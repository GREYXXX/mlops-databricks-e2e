# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 5: Model Registration
# MAGIC Register the best model to Unity Catalog and assign the "Challenger" alias.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")
dbutils.widgets.text("model_name", "california_housing_model", "Model Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

full_model_name = f"{catalog}.{schema}.{model_name}"
print(f"Registering to: {full_model_name}")

# COMMAND ----------

# Get best_run_id from upstream task
best_run_id = dbutils.jobs.taskValues.get(
    taskKey="model_evaluation",
    key="best_run_id",
)
print(f"Registering model from run: {best_run_id}")

# COMMAND ----------

from mlops_e2e.registration import register_model_to_uc, set_model_alias

# COMMAND ----------

# Register model to Unity Catalog
version = register_model_to_uc(run_id=best_run_id, model_name=full_model_name)
print(f"Registered model version: {version}")

# COMMAND ----------

# Set Challenger alias
set_model_alias(model_name=full_model_name, version=version, alias="Challenger")
print(f"Set alias 'Challenger' on version {version}")

# COMMAND ----------

# Pass model version to downstream stages
dbutils.jobs.taskValues.set(key="model_version", value=version)
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
