# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 4: Model Evaluation
# MAGIC Compute evaluation metrics and generate comparison artifacts for the best model.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

# Get best_run_id from the training stage
best_run_id = dbutils.jobs.taskValues.get(
    taskKey="model_training",
    key="best_run_id",
)
print(f"Evaluating model from run: {best_run_id}")

# COMMAND ----------

from mlops_e2e.evaluation import evaluate_model

# COMMAND ----------

# Evaluate the model
metrics = evaluate_model(run_id=best_run_id)
print("Evaluation metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

# COMMAND ----------

# Pass best_run_id to downstream stages
dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
