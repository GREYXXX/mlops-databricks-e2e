# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 2: Feature Engineering — FastText + Vocab
# MAGIC Tokenize the corpus, train FastText embeddings (unsupervised on all text),
# MAGIC and build the token vocabulary from training data only.
# MAGIC
# MAGIC Artifacts are saved to MLflow. The `feature_run_id` is passed to Stage 3
# MAGIC via task values so the training stage can download them.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Using catalog={catalog}, schema={schema}")

# COMMAND ----------

from mlops_e2e.newsgroups.feature_eng import run_feature_engineering

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{username}/mlops_e2e_newsgroups_ensemble"

feature_run_id = run_feature_engineering(
    spark=spark,
    catalog=catalog,
    schema=schema,
    experiment_name=experiment_name,
)

print(f"Feature run ID: {feature_run_id}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="feature_run_id", value=feature_run_id)
dbutils.jobs.taskValues.set(key="experiment_name", value=experiment_name)
print(f"Set task values: feature_run_id={feature_run_id}")
