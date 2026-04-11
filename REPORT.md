# MLOps on Databricks — End-to-End Repository Report

> Generated: 2026-04-11  
> Purpose: Deep-dive reference for understanding how enterprise MLOps works on Databricks, aligned with the **Databricks Certified Machine Learning Professional** exam objectives.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Databricks Asset Bundles (DAB)](#3-databricks-asset-bundles-dab)
4. [Pipeline Architecture — California Housing](#4-pipeline-architecture--california-housing)
5. [Pipeline Architecture — 20 Newsgroups Ensemble](#5-pipeline-architecture--20-newsgroups-ensemble)
6. [Core Python Package (`src/mlops_e2e/`)](#6-core-python-package-srcmlops_e2e)
7. [Notebooks — Thin Wrappers Pattern](#7-notebooks--thin-wrappers-pattern)
8. [Unity Catalog — Tables, Models & Governance](#8-unity-catalog--tables-models--governance)
9. [MLflow — Experiment Tracking & Model Registry](#9-mlflow--experiment-tracking--model-registry)
10. [Champion / Challenger Model Promotion](#10-champion--challenger-model-promotion)
11. [Databricks App — FastAPI + React Dashboard](#11-databricks-app--fastapi--react-dashboard)
12. [Testing Strategy](#12-testing-strategy)
13. [Key Databricks Concepts & Exam Alignment](#13-key-databricks-concepts--exam-alignment)

---

## 1. Project Overview

This repository is a **production-style MLOps reference implementation** on Databricks. It runs two independent end-to-end machine learning pipelines:

| Pipeline | Problem Type | Dataset | Model |
|---|---|---|---|
| **California Housing** | Regression | 20,640 rows, 8 features | LightGBM + Optuna tuning |
| **20 Newsgroups Ensemble** | Multi-class Text Classification | 18,846 documents, 20 classes | Soft-vote ensemble (LR + RF + TextCNN) |

Both pipelines share the same **6-stage orchestration pattern**: Data Preparation → Feature Engineering → Model Training → Evaluation → Model Registration → Champion Management.

### Technologies Involved

| Layer | Technology |
|---|---|
| Infrastructure-as-Code | Databricks Asset Bundles (DAB) |
| Orchestration | Databricks Jobs (multi-task, scheduled) |
| Experiment Tracking | MLflow (nested runs, autolog) |
| Model Registry | MLflow + Unity Catalog (3-part namespace, aliases) |
| Data Storage | Delta Lake tables in Unity Catalog |
| Hyperparameter Tuning | Optuna (housing) |
| Embeddings | FastText via Gensim (newsgroups) |
| Deep Learning | PyTorch + PyTorch Lightning (TextCNN) |
| Dashboard Backend | FastAPI (Python) |
| Dashboard Frontend | React + TypeScript + Tailwind |
| Deployment | Databricks Apps |
| Python packaging | Setuptools wheel uploaded as DAB artifact |

---

## 2. Repository Structure

```
mlops-databricks-e2e/
│
├── databricks.yml                   # DAB root: variables, targets, artifact build
│
├── resources/                       # DAB resource declarations
│   ├── pipeline_job.yml             # Housing 6-stage Job definition
│   ├── ensemble_pipeline_job.yml    # Newsgroups 6-stage Job definition
│   ├── experiment.yml               # MLflow experiment resource
│   └── app.yml                      # Databricks App resource
│
├── notebooks/                       # Databricks Notebooks (thin wrappers only)
│   ├── housing/
│   │   ├── 01_data_preparation.py
│   │   ├── 02_feature_engineering.py
│   │   ├── 03_model_training.py
│   │   ├── 04_model_evaluation.py
│   │   ├── 05_model_registration.py
│   │   └── 06_champion_management.py
│   └── newsgroups/
│       ├── 01_data_preparation.py
│       ├── 02_feature_engineering.py
│       ├── 03_ensemble_training.py
│       ├── 04_model_evaluation.py
│       ├── 05_model_registration.py
│       └── 06_champion_management.py
│
├── src/mlops_e2e/                   # Pure Python package — all testable logic lives here
│   ├── config.py                    # Shared config helpers (widget → env → default)
│   ├── housing/                     # Housing pipeline logic
│   │   ├── config.py               # Table names, model name, experiment name
│   │   ├── data_prep.py            # Load from sklearn, write Delta table
│   │   ├── feature_eng.py          # Derived features, log transforms
│   │   ├── training.py             # Optuna + LightGBM + MLflow
│   │   ├── evaluation.py           # Metrics, plots, artifact logging
│   │   ├── registration.py         # Register model to Unity Catalog
│   │   └── champion.py             # Champion/Challenger comparison & promotion
│   └── newsgroups/
│       ├── config.py               # Embed dim, vocab size, sequence length constants
│       ├── data.py                 # Load 20 Newsgroups, Delta read/write
│       ├── feature_eng.py          # Tokenization, FastText training, vocab building
│       ├── models.py               # TextCNN (PyTorch Lightning), LR/RF helpers
│       ├── training.py             # _EnsembleModel pyfunc, train_ensemble()
│       ├── evaluation.py           # Per-model + ensemble evaluation
│       └── champion.py             # Champion/Challenger for F1 metric
│
├── app/                            # Databricks App (dashboard)
│   ├── backend/
│   │   ├── main.py                # FastAPI app, CORS, static file mount
│   │   ├── routes/                # API route handlers
│   │   └── services/              # MLflow, Jobs, Catalog service clients
│   └── frontend/
│       ├── src/                   # React + TypeScript source
│       └── dist/                  # Built static files (synced to Databricks)
│
├── tests/                         # pytest unit tests (no Databricks required)
│   ├── conftest.py                # Fixtures: local Spark, mock MLflow, mock dbutils
│   └── test_*.py                  # One file per module
│
├── ensemble/
│   └── ens_cls.py                 # Standalone local ensemble demo (no Databricks)
│
├── setup.py                       # Python package definition
├── requirements.txt               # Runtime + dev dependencies
└── .databricksignore              # Files excluded from DAB sync
```

---

## 3. Databricks Asset Bundles (DAB)

### What is a DAB?

A **Databricks Asset Bundle** is the official infrastructure-as-code framework for Databricks. It lets you define jobs, experiments, apps, and variables in YAML files, then deploy them to multiple environments (dev/staging/prod) with a single CLI command.

```bash
databricks bundle validate -t dev --profile <PROFILE>
databricks bundle deploy  -t dev --profile <PROFILE>
databricks bundle run     -t dev --profile <PROFILE> mlops_e2e_pipeline
```

### `databricks.yml` — Root Configuration

```
bundle:
  name: mlops_e2e_example

variables:
  catalog:              default = main
  schema:               default = mlops_e2e
  model_name:           default = california_housing_model
  ensemble_model_name:  default = newsgroups_ensemble_model
  warehouse_id:         (required — SQL warehouse for app queries)

workspace:
  root_path: /Users/${workspace.current_user.userName}/.bundle/...

artifacts:
  mlops_e2e_wheel:
    type: whl
    build: python setup.py bdist_wheel    ← builds the package wheel
    files: [{source: dist/*.whl}]         ← uploaded to Databricks

sync:
  include:
    - app/frontend/dist/**                ← built React app is synced too

include:
  - resources/*.yml                       ← loads all job/experiment/app definitions

targets:
  dev:     (default) user-specific paths, dev catalog
  staging: staging catalog, shared paths
  prod:    prod catalog, runs as service principal `mlops-e2e-sp`
```

**Key insight**: Variables allow the same bundle to deploy to dev/staging/prod by just changing `catalog` and `schema`. The `run_as` stanza in prod ensures the pipeline runs with a service principal, not a user's credentials.

### DAB Artifact Build

Before deploying, DAB automatically runs `python setup.py bdist_wheel` to produce `dist/mlops_e2e-0.2.0-py3-none-any.whl`. This wheel is uploaded to the Databricks workspace and installed in each job's Python environment. This is why all the business logic lives in `src/mlops_e2e/` as a proper Python package — notebooks just `import` from it.

---

## 4. Pipeline Architecture — California Housing

### Job Resource (`resources/pipeline_job.yml`)

The housing pipeline is a **Databricks Job** with 6 tasks arranged in a linear dependency chain, scheduled daily at 08:00 Pacific.

```
data_preparation
      ↓
feature_engineering
      ↓
model_training
      ↓
model_evaluation
      ↓
model_registration
      ↓
champion_management
```

Each task specifies:
- `notebook_task.notebook_path`: the notebook to run
- `depends_on`: the upstream task name
- `job_cluster_key` or `environment_key`: compute to use
- `base_parameters`: widget values passed to the notebook

### Inter-Task Communication via `dbutils.jobs.taskValues`

Tasks pass data to downstream tasks using **task values** — a Databricks-native key-value store scoped to a job run.

| Set by | Key | Read by |
|---|---|---|
| `model_training` | `best_run_id` | `model_evaluation`, `model_registration` |
| `model_registration` | `model_version` | `champion_management` |

```python
# In model_training notebook:
dbutils.jobs.taskValues.set(key="best_run_id", value=run_id)

# In model_evaluation notebook:
run_id = dbutils.jobs.taskValues.get(
    taskKey="model_training",
    key="best_run_id",
    debugValue="some-fallback-run-id"   # used when running interactively
)
```

**Exam note**: `taskValues` are the correct way to share state between tasks in a Databricks multi-task job. They are scoped to the job run and not persisted after the run ends.

### Job Environment

The job uses a **job environment** (`ml_env`) rather than a cluster definition. This is the modern DAB approach:

```yaml
environments:
  - environment_key: ml_env
    spec:
      client: "4"
      dependencies:
        - lightgbm
        - optuna
        - mlflow
        - /Workspace/.../mlops_e2e-*.whl   ← the built package
```

### Stage-by-Stage Breakdown

#### Stage 1: Data Preparation

- **What**: Loads the California Housing dataset from `sklearn.datasets.fetch_california_housing()` (20,640 rows, 8 numeric features + `MedHouseVal` target).
- **Output**: Delta table `{catalog}.{schema}.california_housing_raw`
- **Databricks tool**: `spark.createDataFrame()` + Delta write with `overwriteSchema=true`

#### Stage 2: Feature Engineering

- **What**: Reads the raw Delta table, computes 3 derived features, applies log transforms to skewed columns, adds a `_feature_timestamp` column.
- **Derived features**:
  - `rooms_per_household = AveRooms × AveOccup`
  - `bedrooms_ratio = AveBedrms / AveRooms` (zero-safe)
  - `population_per_household = Population / AveOccup`
- **Log transforms** (`log1p`): `Population`, `AveRooms`, `AveBedrms`
- **Output**: Delta table `{catalog}.{schema}.california_housing_features`

#### Stage 3: Model Training

- **What**: Reads the feature table, splits train (70%) / val (15%) / test (15%), runs Optuna hyperparameter search for 50 trials, trains the final LightGBM model with the best hyperparameters.
- **MLflow**:
  - `mlflow.set_experiment(experiment_name)` — sets the experiment
  - `mlflow.lightgbm.autolog()` — automatically logs params, metrics, model
  - Each Optuna trial is a **nested child run** under the parent run
  - Final model logged as `model` artifact
  - Test data logged as `test_data.npz` artifact (for use in evaluation)
- **Output**: `best_run_id` set as task value

**Optuna hyperparameters tuned**:
- `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`
- `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

**Objective**: Minimize validation RMSE (uses early stopping on val set)

#### Stage 4: Model Evaluation

- **What**: Downloads `test_data.npz` from the MLflow run's artifacts, loads the logged model, runs inference, computes metrics, generates 4 plots.
- **Metrics logged to MLflow**:
  - `test_rmse`, `test_mae`, `test_r2`, `test_mape`, `test_median_ae`
- **Plots logged as PNG artifacts**:
  - Residuals plot (predicted − actual vs. predicted)
  - Predicted vs. Actual scatter
  - Feature importance bar chart
  - Error distribution histogram

#### Stage 5: Model Registration

- **What**: Registers the trained model to the **Unity Catalog** model registry.
- **MLflow API**:
  ```python
  mlflow.set_registry_uri("databricks-uc")
  mlflow.register_model(f"runs:/{run_id}/model", f"{catalog}.{schema}.{model_name}")
  ```
- **Alias**: Newly registered version gets the `"Challenger"` alias
- **Output**: `model_version` set as task value

#### Stage 6: Champion Management

- **What**: Compares the new Challenger against the current Champion (if any) on held-out test RMSE. Promotes or rejects the Challenger.
- **Details**: See [Section 10](#10-champion--challenger-model-promotion)

---

## 5. Pipeline Architecture — 20 Newsgroups Ensemble

### Job Resource (`resources/ensemble_pipeline_job.yml`)

The newsgroups pipeline follows the same 6-stage pattern, scheduled daily at 09:00 Pacific.

```
data_preparation
      ↓
feature_engineering  (trains FastText embeddings, builds vocab)
      ↓
ensemble_training    (LR + RF + TextCNN trained and packaged as pyfunc)
      ↓
model_evaluation
      ↓
model_registration
      ↓
champion_management
```

Inter-task values:

| Set by | Key | Read by |
|---|---|---|
| `feature_engineering` | `feature_run_id`, `experiment_name` | `ensemble_training` |
| `ensemble_training` | `training_run_id`, `experiment_name` | `model_evaluation`, `model_registration` |

### Stage-by-Stage Breakdown

#### Stage 1: Data Preparation

- Fetches 20 Newsgroups (train + test) with `remove=("headers", "footers", "quotes")` — strips metadata that would make the task trivially easy.
- Combines into one Spark DataFrame with a `split` column (`"train"` or `"test"`).
- Saves to Delta: `{catalog}.{schema}.newsgroups_raw`

#### Stage 2: Feature Engineering

This stage is unique: it trains **unsupervised embeddings** before any labeled model training.

- **Tokenization**: `re.findall(r"[a-z]+", text.lower())` — simple regex tokenizer
- **FastText training** (via Gensim):
  - Trained on ALL text (train + test) — unsupervised, so using test text is valid
  - `sg=1` (skip-gram), `vector_size=100`, `min_count=2`, `window=5`, 10 epochs
- **Vocabulary**: Built from training text only (top 30,000 tokens by frequency)
  - Index `0` = PAD token
  - Indices `1..30000` = real tokens
- **MLflow logging**: `fasttext_model.bin` + `vocab.json` as artifacts in the feature run
- **Output task value**: `feature_run_id` (so training stage can download these artifacts)

**Why this matters for the exam**: This demonstrates the pattern of logging non-model artifacts (embeddings, vocabularies, preprocessing artifacts) to MLflow for reproducibility and downstream consumption.

#### Stage 3: Ensemble Training

Three models trained sequentially, all logged under one MLflow run:

| Model | Vectorizer | Classifier | Key Params |
|---|---|---|---|
| **LR** | TF-IDF (50K features, unigrams+bigrams) | LogisticRegression | C=5.0, solver=saga |
| **RF** | TF-IDF (20K features, unigrams) | RandomForestClassifier | 300 trees |
| **TextCNN** | FastText embeddings | Conv1D + max-pool + FC | 128 filters, kernels (2,3,4,5), 10 epochs |

**Ensemble**: Soft voting — average the probability matrices from all 3 models, take argmax.

**MLflow pyfunc model** (`_EnsembleModel`): All 3 sub-models + vocab + CNN config are serialized and packaged as a single `mlflow.pyfunc` model. This is the correct way to package custom multi-component models for the Databricks registry.

```python
class _EnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Deserialize LR pipeline, RF pipeline, CNN state dict, vocab
    
    def predict(self, context, model_input, params=None):
        # params["return_mode"]: "labels" | "proba" | "all"
        # "all" returns per-model + ensemble probabilities
```

**Per-model metrics logged**:
- `{model}_accuracy`, `{model}_precision`, `{model}_recall`, `{model}_f1`
- Classification report as JSON artifact

#### Stages 4–6: Evaluation, Registration, Champion Management

Mirror the housing pipeline, with two differences:
- Ensemble pyfunc loaded with `mlflow.pyfunc.load_model()`
- Champion comparison metric: **weighted F1** (higher is better, vs. RMSE lower is better)

---

## 6. Core Python Package (`src/mlops_e2e/`)

### The Two-Layer Design

This is the most important architectural decision in the repo:

```
Databricks Notebook  →  thin wrapper (widgets, task values, spark)
         ↓
  src/mlops_e2e/     →  all business logic (pure Python, testable locally)
```

The notebooks contain almost no logic — they just read parameters, call a function, and set task values. All the real work is in the Python package. This enables:
- **Local unit testing** without a Databricks cluster
- **Separation of concerns** between orchestration and logic
- **Reusability** (functions can be called from notebooks, tests, or scripts)

### `config.py` — Widget → Env Var → Default Fallback

```python
def _get_widget_or_env(key: str, default: str) -> str:
    try:
        import dbutils
        return dbutils.widgets.get(key)   # Running in Databricks notebook
    except:
        return os.environ.get(key, default)  # Running locally or in tests
```

This pattern appears throughout the package. It allows the same code to work in a Databricks notebook (where `dbutils` is available), in a local test (where it falls back to env vars), and in CI (where env vars can be set).

**Exam note**: This is the recommended pattern for making notebook code testable.

### `housing/training.py` — Optuna + MLflow Integration

```python
def create_optuna_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[early_stopping(50)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    return objective
```

The `MLflowCallback` from `optuna_integration` logs each trial as a **nested MLflow run**, creating a parent run with 50 child runs. This gives you a full hyperparameter search history in the MLflow UI.

---

## 7. Notebooks — Thin Wrappers Pattern

Every notebook follows the same structure:

```python
# Databricks notebook source
# COMMAND ----------

# 1. Import from the package (installed via wheel)
from mlops_e2e.housing import data_prep
from mlops_e2e import config

# COMMAND ----------

# 2. Read parameters from widgets (set by the Job or interactively)
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "mlops_e2e")
catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")

# COMMAND ----------

# 3. (If needed) Read task values from upstream stages
run_id = dbutils.jobs.taskValues.get(
    taskKey="model_training",
    key="best_run_id",
    debugValue="local-debug-run"
)

# COMMAND ----------

# 4. Call the function (all logic is in the package)
result = data_prep.save_raw_table(spark, df, catalog, schema)

# COMMAND ----------

# 5. (If needed) Set task values for downstream stages
dbutils.jobs.taskValues.set(key="best_run_id", value=run_id)
```

**Cell separator**: `# COMMAND ----------` — this is the Databricks notebook source format. Each `# COMMAND ----------` line starts a new cell when the file is imported into a Databricks workspace.

---

## 8. Unity Catalog — Tables, Models & Governance

### 3-Level Namespace

Unity Catalog uses a three-level naming convention: `catalog.schema.table_or_model`.

| Resource | Full Name (dev) |
|---|---|
| Raw housing table | `main.mlops_e2e.california_housing_raw` |
| Feature table | `main.mlops_e2e.california_housing_features` |
| Registered model | `main.mlops_e2e.california_housing_model` |
| Raw newsgroups table | `main.mlops_e2e.newsgroups_raw` |
| Ensemble model | `main.mlops_e2e.newsgroups_ensemble_model` |

In staging/prod, `main` is replaced with `staging_catalog` or `prod_catalog` via DAB variables.

### Delta Tables

All data is stored as **Delta Lake** tables:
- ACID transactions — safe concurrent reads/writes
- `overwriteSchema=true` — allows schema evolution on re-runs
- Tables are queryable from Databricks SQL, notebooks, and the Databricks SDK

### Unity Catalog Model Registry

Models are registered to UC using:
```python
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/{run_id}/model", "catalog.schema.model_name")
```

UC model versions support **aliases** — named pointers to a specific version:
- `"Champion"` — currently serving/best model
- `"Challenger"` — newly trained, under evaluation
- `"Champion-20260411-120000"` — archived former champion (timestamped)
- `"Challenger-20260411-120000"` — archived failed challenger

**Exam note**: UC model registry aliases replace the deprecated Stage-based workflow (`Staging`, `Production`) from the workspace model registry. Aliases are the current best practice.

---

## 9. MLflow — Experiment Tracking & Model Registry

### Experiment Setup

The MLflow experiment is declared as a DAB resource (`resources/experiment.yml`):

```yaml
resources:
  experiments:
    mlops_e2e_experiment:
      name: /Users/${workspace.current_user.userName}/mlops_e2e_california_housing
      permissions:
        - level: CAN_MANAGE
          group_name: users
```

This creates the experiment at deploy time and sets permissions.

### Run Hierarchy

For the housing pipeline, the MLflow run structure is:

```
Experiment: /Users/{user}/mlops_e2e_california_housing
│
└── Parent Run: "lgbm_optuna_tuning" (the main training run)
    ├── Child Run: trial_0  (Optuna trial 0)
    ├── Child Run: trial_1  (Optuna trial 1)
    ├── ...
    └── Child Run: trial_49 (Optuna trial 49)
```

The parent run is where the final model, test data, evaluation metrics, and plots are logged. Child runs each contain the hyperparameters and validation RMSE for one Optuna trial.

### What Gets Logged

| Stage | Logged to MLflow |
|---|---|
| Training | Model (`model/`), test data (`test_data.npz`), hyperparams, train/val metrics |
| Evaluation | `test_rmse`, `test_mae`, `test_r2`, `test_mape`, `test_median_ae`, 4 PNG plots, `metrics_summary.json` |
| Feature Eng (newsgroups) | `fasttext_model.bin`, `vocab.json` |
| Ensemble Training | Per-model metrics, classification reports (JSON), pyfunc model |
| Champion Management | Comparison summary JSON |

### MLflow Autolog

```python
mlflow.lightgbm.autolog()
```

This single line automatically captures:
- All LightGBM parameters
- Training and validation metrics (per iteration if `eval_set` is provided)
- Feature importance
- The model itself

### Downloading Artifacts Between Stages

Because test data is logged in training and downloaded in evaluation, stages communicate through MLflow artifacts:

```python
# In evaluation stage:
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="test_data.npz"
)
data = np.load(local_path)
```

This is a clean pattern: each stage is self-contained and reads what it needs from MLflow, rather than passing large arrays through task values.

---

## 10. Champion / Challenger Model Promotion

### Concept

Every time the pipeline runs, it produces a new model version. Rather than blindly deploying it, the pipeline compares it against the current production model:

```
New Model Run
     ↓
Register to UC → assigned "Challenger" alias
     ↓
Champion Management Stage
     ├── If no Champion exists → auto-promote to Champion
     ├── If Challenger RMSE < Champion RMSE → promote Challenger
     │       Archive old Champion: "Champion-YYYYMMDD-HHMMSS"
     │       Set Challenger → "Champion"
     └── If Champion RMSE ≤ Challenger RMSE → reject Challenger
             Archive: "Challenger-YYYYMMDD-HHMMSS"
```

### Implementation

```python
def run_champion_management(model_name, catalog, schema) -> dict:
    # 1. Load Challenger by alias
    client = MlflowClient(registry_uri="databricks-uc")
    challenger = client.get_model_version_by_alias(model_name, "Challenger")
    challenger_version = challenger.version
    
    # 2. Download test data from Challenger's run
    run_id = challenger.run_id
    test_data = download_test_data(run_id)
    
    # 3. Check for existing Champion
    try:
        champion = client.get_model_version_by_alias(model_name, "Champion")
        champion_version = champion.version
    except:
        # No champion exists → auto-promote
        promote_challenger(model_name, challenger_version, None)
        return {"action": "auto_promoted", ...}
    
    # 4. Load both models and compute RMSE
    champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")
    challenger_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Challenger")
    
    champion_rmse = compute_rmse(champion_model, test_data)
    challenger_rmse = compute_rmse(challenger_model, test_data)
    
    # 5. Decision
    if challenger_rmse < champion_rmse:
        promote_challenger(model_name, challenger_version, champion_version)
        return {"action": "promoted", ...}
    else:
        archive_challenger(model_name, challenger_version)
        return {"action": "rejected", ...}
```

### Loading Models by Alias

```python
# Load by alias (recommended — alias always points to the right version)
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")

# Load by version number
model = mlflow.pyfunc.load_model(f"models:/{model_name}/5")
```

**Exam note**: Loading by alias is preferred in production because it decouples the code from specific version numbers. When you promote a new champion, all downstream consumers automatically get the new model on next load.

### Alias Management

```python
# Set an alias
client.set_registered_model_alias(model_name, "Champion", version)

# Delete an alias
client.delete_registered_model_alias(model_name, "Challenger")
```

Timestamped archive aliases serve as an audit trail — you can always load any past champion or challenger by its archive alias.

---

## 11. Databricks App — FastAPI + React Dashboard

### Architecture

```
Browser (React)
      ↓ HTTP
Databricks App
      ├── FastAPI Backend
      │   ├── /api/pipeline/*    → Databricks Jobs API
      │   ├── /api/experiments/* → MLflow Tracking API
      │   ├── /api/models/*      → MLflow Registry API + Databricks SDK
      │   └── /api/config        → Config endpoint
      └── Static Files (React build)
```

### Databricks App Resource (`resources/app.yml`)

```yaml
resources:
  apps:
    mlops_e2e_app:
      name: "mlops-e2e-dashboard"
      source_code_path: ../app
      resources:
        - name: mlops-e2e-job
          job:
            id: ${resources.jobs.mlops_e2e_pipeline.id}
            permission: CAN_MANAGE_RUN
        - name: mlops-e2e-warehouse
          sql_warehouse:
            id: ${var.warehouse_id}
            permission: CAN_USE
```

The App automatically gets a service principal and the listed permissions. It can trigger pipeline runs, query the model registry, and read experiments — without any manual IAM setup.

### FastAPI Backend (`app/backend/`)

**`services/mlflow_service.py`** — Core service, communicates with both MLflow and Databricks SDK:

```python
mlflow.set_registry_uri("databricks-uc")

def get_model_versions(model_name) -> list[dict]:
    # Uses Databricks SDK (WorkspaceClient) for reliable auth inside Apps
    client = WorkspaceClient()
    versions = client.model_versions.list(full_name=model_name)
    # Fetch aliases to annotate each version
    ...

def promote_to_champion(model_name, version) -> dict:
    # Performs the same champion/challenger alias dance as the pipeline
    # But triggered manually from the UI
    ...
```

**`services/jobs_service.py`** — Pipeline monitoring and triggering:

```python
def get_pipeline_status(job_name) -> dict:
    client = WorkspaceClient()
    jobs = client.jobs.list(name=job_name)
    latest_run = client.jobs.list_runs(job_id=job.job_id, limit=1)
    # Returns task-level status (6 stages with individual status/timing)
    ...

def trigger_pipeline_run(job_name) -> dict:
    client.jobs.run_now(job_id=job.job_id)
    ...
```

**API Routes**:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/pipeline/status` | Current run status (all 6 task statuses) |
| `POST` | `/api/pipeline/run` | Trigger a new pipeline run |
| `GET` | `/api/pipeline/history` | Last N run results |
| `GET` | `/api/experiments/runs` | All MLflow runs in experiment |
| `GET` | `/api/experiments/runs/{run_id}` | Single run details + artifacts |
| `GET` | `/api/models/versions` | All UC model versions with aliases |
| `GET` | `/api/models/champion` | Current Champion version |
| `GET` | `/api/models/comparison` | Champion vs Challenger metric comparison |
| `GET` | `/api/models/training-history` | All versions with their training metrics |
| `GET` | `/api/models/history` | Paginated version history |
| `POST` | `/api/models/{version}/promote` | Manually promote a version to Champion |
| `DELETE` | `/api/models/{version}` | Delete a model version |

### React Frontend (`app/frontend/`)

Built with React + TypeScript + Tailwind CSS. Three main views:

1. **Pipeline View** — 6-stage visual pipeline. Shows current stage status (running/success/failed), timing, and links to logs.
2. **Model Comparison** — Side-by-side Champion vs. Challenger cards with metric deltas and a promote button.
3. **Run History** — Paginated table of all model versions with aliases, training metrics, and hyperparameters.

### Deployment Flow

```bash
# 1. Build React frontend
cd app/frontend && npm install && npm run build
# → produces app/frontend/dist/

# 2. Deploy bundle (uploads wheel + dist/ to Databricks)
databricks bundle deploy -t dev --profile <PROFILE>
# DAB syncs app/frontend/dist/ via sync.include

# 3. FastAPI serves the built files
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```

The `.databricksignore` file excludes `app/frontend/dist/` from gitignore while keeping it in the DAB sync. This is intentional — dist is git-ignored but needs to be deployed.

---

## 12. Testing Strategy

### Design Goal: Test Without Databricks

All tests run **locally** with no Databricks connection needed. This is achieved through:

1. **Local Spark** (`pyspark` installed as dev dependency, local master)
2. **Mocked MLflow** (`unittest.mock.patch` for `mlflow.*` calls)
3. **Mocked dbutils** (custom `MockDbutils` class simulating task values and widgets)
4. **Mocked Databricks SDK** (`unittest.mock.patch` for `WorkspaceClient`)

### `conftest.py` — Shared Fixtures

```python
@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[*]").appName("test").getOrCreate()

@pytest.fixture
def mock_dbutils():
    class MockDbutils:
        class jobs:
            _values = {}
            @staticmethod
            def taskValues.set(key, value): MockDbutils.jobs._values[key] = value
            @staticmethod
            def taskValues.get(taskKey, key, debugValue=None): ...
        class widgets:
            @staticmethod
            def get(key): return os.environ.get(key, "test_default")
    return MockDbutils()
```

### Test Coverage

| File | What's Tested |
|---|---|
| `test_data_prep.py` | Load from sklearn, DataFrame schema, Delta write |
| `test_feature_eng.py` | Derived feature values, log transforms, null handling |
| `test_training.py` | Optuna trial execution, LightGBM fit, run ID returned |
| `test_evaluation.py` | RMSE/MAE/R2 computed correctly, plots generated |
| `test_registration.py` | MLflow register_model called with correct args |
| `test_champion.py` | Promote/reject logic, alias setting, archive naming |
| `test_api.py` | All 32 FastAPI endpoints return correct status codes |

### Running Tests

```bash
pip install -e ".[dev]"          # Install package + dev deps
pytest tests/ -v --tb=short     # Run all tests
pytest tests/test_evaluation.py::TestComputeMetrics::test_known_rmse -v  # Single test
```

---

## 13. Key Databricks Concepts & Exam Alignment

This section maps the repo components to exam topic areas.

### MLflow Experiment Tracking

| Concept | Where Used in Repo |
|---|---|
| `mlflow.set_experiment()` | `housing/training.py` |
| `mlflow.start_run()` | All training stages |
| `mlflow.log_metric()`, `log_param()`, `log_artifact()` | Evaluation, training stages |
| `mlflow.lightgbm.autolog()` | `housing/training.py` |
| Nested runs | Optuna trials as child runs |
| `mlflow.pyfunc.PythonModel` | `newsgroups/training.py` (`_EnsembleModel`) |
| `mlflow.artifacts.download_artifacts()` | Evaluation stage fetches test data |

### Unity Catalog & Model Registry

| Concept | Where Used |
|---|---|
| 3-level namespace | All table/model names |
| `mlflow.set_registry_uri("databricks-uc")` | Registration + evaluation stages |
| `mlflow.register_model()` | `housing/registration.py` |
| Model aliases (Champion/Challenger) | `housing/champion.py` |
| `client.set_registered_model_alias()` | Champion management |
| `client.delete_registered_model_alias()` | Challenger archiving |
| Loading model by alias `@Champion` | Champion comparison |

### Databricks Jobs & Orchestration

| Concept | Where Used |
|---|---|
| Multi-task jobs | `resources/pipeline_job.yml` |
| Task dependencies (`depends_on`) | All tasks in the job |
| `dbutils.jobs.taskValues.set/get` | Training → Evaluation → Registration → Champion |
| Job environments (vs. cluster definitions) | `ml_env`, `ensemble_env` in resource YAMLs |
| Scheduled jobs (cron) | Both pipelines run daily |
| `databricks bundle run` | Local triggering of job runs |

### Databricks Asset Bundles

| Concept | Where Used |
|---|---|
| `databricks.yml` targets | dev, staging, prod targets |
| Bundle variables | `catalog`, `schema`, `model_name`, `warehouse_id` |
| Artifact build (wheel) | `artifacts.mlops_e2e_wheel` |
| `sync.include` | Frontend dist directory |
| Resource includes | `resources/*.yml` |
| `run_as` service principal | prod target |
| `databricks bundle validate/deploy/run` | CI/CD workflow |

### Delta Lake & Data Patterns

| Concept | Where Used |
|---|---|
| Delta table write with `overwriteSchema` | Data prep stages |
| Reading Delta tables as Pandas | Training stage (`.toPandas()`) |
| Delta table as feature store | Feature engineering output |
| Time travel (implicit, via `_feature_timestamp`) | Feature table |

### Model Serving Patterns

| Concept | Where Used |
|---|---|
| Champion/Challenger pattern | `housing/champion.py`, `newsgroups/champion.py` |
| Model alias for serving | `@Champion` alias for stable reference |
| Audit trail via timestamped aliases | `Champion-YYYYMMDD-HHMMSS` |
| Manual promotion via API | Dashboard `POST /api/models/{version}/promote` |

### Custom pyfunc Models

The newsgroups ensemble is an excellent example of the **pyfunc** pattern for packaging custom inference logic:

```python
class _EnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Deserialize all sub-models from artifacts/
        self.lr_pipeline = pickle.load(open(context.artifacts["lr_pipeline"], "rb"))
        self.rf_pipeline = pickle.load(open(context.artifacts["rf_pipeline"], "rb"))
        # Load CNN from state dict + config
        # Load vocab
    
    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist()
        lr_proba  = self.lr_pipeline.predict_proba(texts)
        rf_proba  = self.rf_pipeline.predict_proba(texts)
        cnn_proba = self._cnn_predict_proba(texts)
        ensemble  = (lr_proba + rf_proba + cnn_proba) / 3.0
        labels    = np.argmax(ensemble, axis=1)
        # Return based on params["return_mode"]
```

**Key pyfunc concepts demonstrated**:
- `load_context` for lazy loading (called once at model load time)
- `context.artifacts` to access logged artifact paths
- `params` for inference-time configuration (no retraining needed to change behavior)
- Packaging heterogeneous sub-models as a single registered model

### Databricks Apps

| Concept | Where Used |
|---|---|
| App resource in DAB | `resources/app.yml` |
| Resource permissions (job, warehouse) | App granted `CAN_MANAGE_RUN` on job |
| Service principal auto-creation | Managed by Databricks when app is deployed |
| FastAPI + static files | `app/backend/main.py` |
| Databricks SDK inside App | `WorkspaceClient()` for auth (preferred over MLflow client) |

---

### Quick Reference: Data Flow Across the Pipeline

```
sklearn.fetch_california_housing()
          ↓
  Delta: catalog.schema.california_housing_raw
          ↓
  Delta: catalog.schema.california_housing_features
          ↓
  MLflow Run (parent):
    ├─ model artifact (LightGBM)
    ├─ test_data.npz
    └─ child runs (Optuna trials)
          ↓  [task value: best_run_id]
  MLflow Run (same):
    ├─ test_rmse, test_mae, test_r2, ...
    └─ residuals.png, predictions.png, ...
          ↓  [task value: best_run_id]
  UC Model Registry: catalog.schema.model_name
    version N → alias: "Challenger"
          ↓  [task value: model_version]
  Champion Management:
    ├─ Load @Champion, @Challenger
    ├─ Compare RMSE on test set
    └─ Set @Champion alias on winner
          ↓
  Dashboard reads @Champion via MLflow API
  Users see metrics, can trigger reruns
```

---

*This report covers the full implementation of a production MLOps system on Databricks, including all the patterns and tools you need to know for the Databricks Certified Machine Learning Professional exam.*
