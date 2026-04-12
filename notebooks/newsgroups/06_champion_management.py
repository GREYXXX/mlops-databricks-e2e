# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 6: Champion Management
# MAGIC Compare the Challenger ensemble against the current Champion using weighted F1.
# MAGIC Promotes the Challenger if it scores higher; archives it otherwise.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "mlops_e2e", "Schema")
dbutils.widgets.text("model_name", "newsgroups_ensemble_model", "Model Name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

full_model_name = f"{catalog}.{schema}.{model_name}"
print(f"Managing champion for: {full_model_name}")

# COMMAND ----------

from mlops_e2e.newsgroups.champion import run_champion_management

# COMMAND ----------

result = run_champion_management(
    spark=spark,
    catalog=catalog,
    schema=schema,
    model_name=full_model_name,
)

# COMMAND ----------

print(f"Action:             {result['action']}")
print(f"Reason:             {result['reason']}")
print(f"Challenger version: {result['challenger_version']}")
if "champion_version" in result:
    print(f"Champion version:   {result['champion_version']}")
if "challenger_f1" in result:
    print(f"Challenger F1:      {result['challenger_f1']}")
    print(f"Champion F1:        {result['champion_f1']}")
