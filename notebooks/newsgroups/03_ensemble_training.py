# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 3: Ensemble Training
# MAGIC Train LR+TF-IDF, RF+TF-IDF, and TextCNN using FastText embeddings from Stage 2.
# MAGIC All three models and the ensemble are logged to MLflow as a single `pyfunc` model.
# MAGIC
# MAGIC Per-model metrics logged:
# MAGIC   - accuracy, precision (weighted), recall (weighted), F1 (weighted)
# MAGIC   - full classification report JSON (20-class breakdown)
# MAGIC
# MAGIC Inference modes (pass via `params={"return_mode": ...}`):
# MAGIC   - `"labels"` (default) — predicted class index per sample
# MAGIC   - `"proba"`            — ensemble probability matrix
# MAGIC   - `"all"`              — dict with all sub-model probabilities + labels

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

feature_run_id = dbutils.jobs.taskValues.get(
    taskKey="feature_engineering",
    key="feature_run_id",
)
experiment_name = dbutils.jobs.taskValues.get(
    taskKey="feature_engineering",
    key="experiment_name",
)
print(f"Feature run ID: {feature_run_id}")

# COMMAND ----------

from mlops_e2e.newsgroups.training import train_ensemble

# COMMAND ----------

training_run_id = train_ensemble(
    spark=spark,
    catalog=catalog,
    schema=schema,
    feature_run_id=feature_run_id,
    experiment_name=experiment_name,
)

print(f"Training run ID: {training_run_id}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="training_run_id", value=training_run_id)
dbutils.jobs.taskValues.set(key="experiment_name", value=experiment_name)
print(f"Set task values: training_run_id={training_run_id}")
