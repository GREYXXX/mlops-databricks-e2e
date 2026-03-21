# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 3: Model Training
# MAGIC Train a LightGBM model with Optuna hyperparameter tuning, logging everything to MLflow.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

from mlops_e2e.training import train_with_tuning

# COMMAND ----------

# Build experiment path under current user's namespace to avoid conflicts
username = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{username}/mlops_e2e_california_housing"

# Run the full training pipeline with Optuna tuning
best_run_id = train_with_tuning(
    spark=spark,
    catalog=catalog,
    schema=schema,
    experiment_name=experiment_name,
    n_trials=50,
)

print(f"Best run ID: {best_run_id}")

# COMMAND ----------

# Pass the best_run_id to downstream tasks via task values
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
print(f"Set task value best_run_id={best_run_id}")
