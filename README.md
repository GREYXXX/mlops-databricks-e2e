# MLOps End-to-End Example (Databricks)

This repository demonstrates a full model lifecycle on **Databricks**: data preparation, feature engineering, training, evaluation, **Unity Catalog** registration, and **champion / challenger** promotion. Logic is packaged as a Python library under `src/mlops_e2e/`; notebooks are thin wrappers. Deployment is driven by a **Databricks Asset Bundle (DAB)**. A **Databricks App** (FastAPI + React) provides operational visibility.

The project contains **two independent six-stage jobs**:

| Pipeline | Domain | Model |
|----------|--------|--------|
| **Housing** | California Housing regression | LightGBM + Optuna |
| **Newsgroups** | 20 Newsgroups text classification | LR + RF + TextCNN ensemble (MLflow pyfunc) |

---

## Table of contents

1. [Databricks Asset Bundle layout](#databricks-asset-bundle-layout)
2. [Pipeline stages](#pipeline-stages)
3. [Dashboard application](#dashboard-application)
4. [Prerequisites](#prerequisites)
5. [Workspace authentication](#workspace-authentication)
6. [Deploy and run](#deploy-and-run)
7. [Databricks App service principal permissions](#databricks-app-service-principal-permissions)
8. [Local development](#local-development)
9. [Bundle variables](#bundle-variables)
10. [Repository layout](#repository-layout)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

---

## Databricks Asset Bundle layout

The root file **`databricks.yml`** defines the bundle name, **variables**, **artifacts**, **targets**, and pulls in everything under `resources/*.yml`. All `${var.*}` placeholders are resolved against the variable definitions and the active **target** (e.g. `dev`, `staging`, `prod`).

```
databricks.yml
│
├── variables
│       catalog, schema, model_name, ensemble_model_name, warehouse_id
│       (referenced as ${var.*} in included resource files)
│
├── artifacts
│       mlops_e2e_wheel — build: python setup.py bdist_wheel
│       The wheel is built during deploy and uploaded to the workspace for job environments.
│
├── include: resources/*.yml
│
├── pipeline_job.yml
│       Resource: jobs.mlops_e2e_pipeline
│       Name pattern: "[${bundle.target}] MLOps E2E Pipeline"
│       Schedule: 08:00 (America/Los_Angeles)
│       Six tasks → environment ml_env (LightGBM, Optuna, project wheel)
│       Notebook base_parameters: catalog, schema; registration/champion use model_name
│
├── ensemble_pipeline_job.yml
│       Resource: jobs.mlops_e2e_newsgroups_pipeline
│       Name pattern: "[${bundle.target}] MLOps E2E Newsgroups Ensemble Pipeline"
│       Schedule: 09:00 (America/Los_Angeles)
│       Six tasks → environment ensemble_env (scikit-learn, gensim, PyTorch, project wheel)
│       Notebook base_parameters: catalog, schema; registration/champion use ensemble_model_name
│
├── experiment.yml
│       Resource: experiments.mlops_e2e_experiment
│       MLflow path: /Users/${workspace.current_user.userName}/mlops_e2e_california_housing
│       Permissions: users group — CAN_MANAGE
│
└── app.yml
        Resource: apps.mlops_e2e_dashboard
        App name: mlops-e2e-dashboard
        source_code_path: ../app (FastAPI backend + React frontend; build frontend before deploy)
        Entitlements (applied automatically for the app’s service principal):
          - SQL warehouse — CAN_USE (${var.warehouse_id})
          - Job — CAN_MANAGE_RUN (${resources.jobs.mlops_e2e_newsgroups_pipeline.id})
```

The **housing** MLflow experiment is declared in `experiment.yml`. The **Newsgroups** pipeline uses an experiment path constructed in the notebooks (see `src/mlops_e2e/newsgroups/config.py` and `notebooks/newsgroups/`).

The App is currently wired to the **Newsgroups** job for run permissions. To point the dashboard at the housing job instead, change the job reference in `resources/app.yml` and set the App’s environment variables (e.g. `PIPELINE_JOB_NAME`, `UC_MODEL_NAME`) accordingly. See `app/app.yaml` for a template.

---

## Pipeline stages

Each job follows the same stage pattern; task keys and notebooks differ by pipeline.

### Housing (regression)

| Stage | Description | Key technologies |
|-------|-------------|-------------------|
| 1. Data preparation | Load California Housing; persist Delta table | scikit-learn, Delta Lake |
| 2. Feature engineering | Log transforms, ratios, scaling | PySpark, Delta Lake |
| 3. Model training | LightGBM + Optuna with MLflow tracking | LightGBM, Optuna, MLflow |
| 4. Model evaluation | RMSE, MAE, R², MAPE; plots | MLflow, matplotlib |
| 5. Model registration | Register best run to Unity Catalog | MLflow, Unity Catalog |
| 6. Champion management | Compare Challenger vs Champion (RMSE); promote or archive | MLflow, UC aliases |

### Newsgroups (classification)

| Stage | Description | Key technologies |
|-------|-------------|-------------------|
| 1. Data preparation | Fetch 20 Newsgroups; Delta table with `split` | scikit-learn, Delta Lake |
| 2. Feature engineering | FastText embeddings, vocabulary | Gensim, MLflow artifacts |
| 3. Ensemble training | LR + RF + TextCNN; single pyfunc model | scikit-learn, PyTorch, MLflow |
| 4. Model evaluation | Per-model and ensemble classification metrics | MLflow |
| 5. Model registration | Register ensemble to Unity Catalog | MLflow, Unity Catalog |
| 6. Champion management | Weighted F1 comparison; promote or archive | MLflow, UC aliases |

---

## Dashboard application

The **Databricks App** serves a React (TypeScript) UI and a FastAPI API.

- **Pipeline view** — Six-stage status, timings, and links to workspace notebooks; optional inline source for each stage.
- **Model comparison** — Champion vs Challenger metrics and promotion context. The UI selects **regression** (RMSE, MAE, R², …) or **classification** (weighted F1, accuracy, precision, recall) based on `/api/config` (`metrics_profile`), derived from `UC_MODEL_NAME` or `DASHBOARD_METRICS_PROFILE`.
- **Training history** — Registered model versions linked to MLflow runs, with key metrics and parameters.

For local development, run the API from the repository root and the Vite dev server from `app/frontend` (see [Local development](#local-development)).

---

## Prerequisites

- Databricks CLI with access to a workspace
- Python 3.10+
- Node.js 18+ (frontend build and local dev)

---

## Workspace authentication

Create a **named profile** for each workspace. Use that profile on every `databricks bundle` and `databricks apps` command.

```bash
# 1. OAuth login; stores credentials under the profile name
databricks auth login --host https://<workspace-url> --profile <PROFILE>

# 2. Confirm host
databricks auth env --profile <PROFILE>

# 3. Confirm connectivity
databricks current-user me --profile <PROFILE>
```

Profile file (typical location: `~/.databrickscfg`):

```ini
[mlops-e2e]
host = https://my-workspace.cloud.databricks.com
auth_type = databricks-cli
```

For service principals or token-based auth, see `databricks configure --help`.

---

## Deploy and run

```bash
# Validate
databricks bundle validate -t dev --profile <PROFILE>

# Deploy (builds wheel, syncs bundle including app/frontend/dist if present)
databricks bundle deploy -t dev --profile <PROFILE>

# Run housing pipeline
databricks bundle run -t dev --profile <PROFILE> mlops_e2e_pipeline

# Run Newsgroups ensemble pipeline
databricks bundle run -t dev --profile <PROFILE> mlops_e2e_newsgroups_pipeline
```

Build the frontend before deploy so static assets are included:

```bash
cd app/frontend && npm install && npm run build
```

---

## Databricks App service principal permissions

Apps run as a workspace **service principal**. After first deploy, grant that principal access to Unity Catalog, the MLflow experiment, and registered models the API reads.

1. **Identify the App principal**

   ```bash
   databricks apps get mlops-e2e-dashboard --profile <PROFILE>
   ```

   Use `service_principal_name` from the response (e.g. `app-mlops-e2e-dashboard-...`).

2. **Unity Catalog** (SQL warehouse or notebook)

   ```sql
   GRANT USE CATALOG ON CATALOG <catalog> TO `<service_principal_name>`;
   GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<service_principal_name>`;
   GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<service_principal_name>`;

   -- Repeat per registered model the dashboard manages (housing and/or newsgroups)
   GRANT MANAGE ON FUNCTION <catalog>.<schema>.california_housing_model TO `<service_principal_name>`;
   GRANT MANAGE ON FUNCTION <catalog>.<schema>.newsgroups_ensemble_model TO `<service_principal_name>`;
   ```

3. **MLflow experiment** — Grant **CAN_READ** (or equivalent) on the experiment the backend resolves via `MLFLOW_EXPERIMENT_NAME` / discovery in `mlflow_service`.

4. **Model registry / UC** — If your workspace requires extra privileges for alias updates, apply per your admin guidance (e.g. tag permissions on the schema).

| Resource | Permission | Purpose |
|----------|------------|---------|
| Catalog / schema | `USE CATALOG`, `USE SCHEMA`, `SELECT` | Read pipeline Delta tables |
| Registered models | `MANAGE` | Read versions, set aliases, promote/delete |
| MLflow experiment | `CAN_READ` | Read runs and metrics |
| Job | `CAN_MANAGE_RUN` | Trigger pipeline (declared in `app.yml` for the bound job) |
| SQL warehouse | `CAN_USE` | Queries from the app (declared in `app.yml`) |

Replace placeholders with your catalog, schema, and principal name.

---

## Local development

```bash
pip install -e ".[dev]"
pytest tests/ -v

# Production-style frontend bundle
cd app/frontend && npm install && npm run build
```

**API** (repository root):

```bash
uvicorn app.backend.main:app --reload
```

**Frontend** (separate terminal; proxies API to `127.0.0.1:8000`):

```bash
cd app/frontend && npm install && npm run dev
```

Configure a `.env` file at the repository root for `UC_MODEL_NAME`, `PIPELINE_JOB_NAME`, `MLFLOW_EXPERIMENT_NAME`, `NOTEBOOK_ROOT_PATH`, and optional `DATABRICKS_CONFIG_PROFILE`. Restart the API after changes; `mlflow_service` reads several values at import time.

---

## Bundle variables

| Variable | Default (definition) | Description |
|----------|----------------------|-------------|
| `catalog` | `main` | Unity Catalog catalog |
| `schema` | `mlops_e2e` | Unity Catalog schema |
| `model_name` | `california_housing_model` | Housing registered model (short name) |
| `ensemble_model_name` | `newsgroups_ensemble_model` | Newsgroups registered model (short name) |
| `warehouse_id` | — | SQL warehouse ID for the App |

Targets in `databricks.yml` override variables (e.g. `dev` may set `catalog: workspace` and a concrete `warehouse_id`). Override at deploy time:

```bash
databricks bundle deploy -t dev --profile <PROFILE> --var="catalog=my_catalog"
```

---

## Repository layout

```
mlops-databricks-e2e/
├── databricks.yml          # Bundle root: variables, artifacts, targets, includes
├── resources/              # Jobs, app, experiment
├── src/mlops_e2e/          # Shared Python package (housing/, newsgroups/, config)
├── notebooks/              # housing/ and newsgroups/ pipeline notebooks
├── app/
│   ├── backend/            # FastAPI
│   ├── frontend/           # React + TypeScript + Vite
│   └── app.yaml            # Example env for Databricks App deployment
├── tests/                  # pytest
└── CLAUDE.md               # Contributor / agent notes
```

---

## Testing

Pipeline logic is tested locally with pytest; Spark-backed tests use a local session, and Databricks / MLflow clients are mocked where needed.

```bash
pytest tests/ -v --tb=short
```

---

## Troubleshooting

### MLflow experiment name conflict

If deploy fails because an experiment already exists, this project namespaces the **housing** experiment under `/Users/{username}/mlops_e2e_california_housing` via `resources/experiment.yml` and `notebooks/housing/03_model_training.py`.

To bind an existing experiment to the bundle:

```bash
databricks experiments list --profile <PROFILE> | grep mlops_e2e_california_housing
databricks bundle deployment bind mlops_e2e_experiment <EXPERIMENT_ID> -t dev --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
```

### Databricks App name conflict

If deploy reports that an app with the same name already exists:

```bash
databricks bundle deployment bind mlops_e2e_dashboard mlops-e2e-dashboard --auto-approve -t dev --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
```
