# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 4: Model Evaluation
# MAGIC Reload the ensemble pyfunc from MLflow, score the held-out test set,
# MAGIC and log detailed per-model and ensemble classification reports.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

training_run_id = dbutils.jobs.taskValues.get(
    taskKey="ensemble_training",
    key="training_run_id",
)
experiment_name = dbutils.jobs.taskValues.get(
    taskKey="ensemble_training",
    key="experiment_name",
)
print(f"Evaluating ensemble from run: {training_run_id}")

# COMMAND ----------

from mlops_e2e.newsgroups.evaluation import evaluate_ensemble

# COMMAND ----------

metrics = evaluate_ensemble(
    spark=spark,
    catalog=catalog,
    schema=schema,
    training_run_id=training_run_id,
    experiment_name=experiment_name,
)

print("Evaluation metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="training_run_id", value=training_run_id)
dbutils.jobs.taskValues.set(key="experiment_name", value=experiment_name)
