# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end MLOps example on Databricks using Asset Bundles (DAB), MLflow, Unity Catalog, and a Databricks App dashboard. Contains two independent 6-stage pipelines:

- **Housing** (`src/mlops_e2e/housing/`): LightGBM regression on California Housing with Optuna tuning
- **Newsgroups** (`src/mlops_e2e/newsgroups/`): Soft-vote ensemble classifier (LR + RF + TextCNN) on 20 Newsgroups

## Commands

```bash
# Install package in dev mode (required before running tests)
pip install -e ".[dev]"

# Lint
ruff check src/ tests/ app/backend/

# Run all tests
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_champion.py -v

# Run a specific test
pytest tests/test_evaluation.py::TestComputeMetrics::test_known_rmse -v

# Run the FastAPI backend locally (from repo root)
uvicorn app.backend.main:app --reload

# Frontend dev server (in a separate terminal)
cd app/frontend && npm install && npm run dev

# Frontend build — MUST run before bundle deploy
cd app/frontend && npm run build

# DAB workflow
databricks bundle validate -t dev --profile <PROFILE>
databricks bundle deploy  -t dev --profile <PROFILE>   # builds wheel + syncs dist/
databricks bundle run     -t dev --profile <PROFILE> mlops_e2e_pipeline
databricks bundle run     -t dev --profile <PROFILE> mlops_e2e_newsgroups_pipeline
```

> `app/frontend/dist/` is git-ignored but included in DAB sync via `.databricksignore` + `sync.include` in `databricks.yml`. Always build the frontend before deploying.

## Architecture

### Two-layer design (critical)

All business logic lives in `src/mlops_e2e/` as a proper Python package — notebooks are thin wrappers that only read widgets, call one function, and set task values. This enables local unit testing without a Databricks cluster.

```
notebooks/housing/03_model_training.py   ← reads widgets + task values, calls ↓
src/mlops_e2e/housing/training.py        ← all real logic, testable locally
```

### Subpackages

- `src/mlops_e2e/housing/` — `data_prep`, `feature_eng`, `training`, `evaluation`, `registration`, `champion`
- `src/mlops_e2e/newsgroups/` — `data`, `feature_eng`, `models`, `training`, `evaluation`, `champion`
- `src/mlops_e2e/config.py` — shared `_get_widget_or_env()` helper used by both subpackages

### Pipeline task flow

Both pipelines are 6-task Databricks Jobs with linear dependencies. State is passed between tasks via `dbutils.jobs.taskValues`:

| Task | Sets task value | Read by |
|---|---|---|
| `model_training` | `best_run_id` | `model_evaluation`, `model_registration` |
| `feature_engineering` (newsgroups) | `feature_run_id`, `experiment_name` | `ensemble_training` |
| `model_registration` | `model_version` | `champion_management` |

### Champion/Challenger promotion

Every pipeline run registers the new model as `"Challenger"` in Unity Catalog, then compares it against `"Champion"` on held-out test data. The winner gets the `"Champion"` alias; the loser is archived with a timestamped alias (`Champion-YYYYMMDD-HHMMSS` / `Challenger-YYYYMMDD-HHMMSS`). If no Champion exists, the Challenger is auto-promoted. Housing uses RMSE (lower wins); newsgroups uses weighted F1 (higher wins).

### Newsgroups pyfunc model

The ensemble is packaged as a single `mlflow.pyfunc.PythonModel` (`_EnsembleModel` in `newsgroups/training.py`). All three sub-models (LR pipeline, RF pipeline, TextCNN state dict) plus the vocab are serialized as MLflow artifacts and deserialized in `load_context`. The `predict` method accepts a `params={"return_mode": "labels"|"proba"|"all"}` argument for inference-time control.

### Config resolution order

`config.py:_get_widget_or_env()` tries: `dbutils.widgets.get(key)` → `os.environ[KEY.upper()]` → default. This makes the same code work in notebooks, local tests (set env vars), and CI.

### DAB structure

- `databricks.yml` — variables (`catalog`, `schema`, `model_name`, `ensemble_model_name`, `warehouse_id`), targets (`dev`/`staging`/`prod`), artifact build (`python setup.py bdist_wheel`)
- `resources/pipeline_job.yml` — housing job + `ml_env` (LightGBM, Optuna, wheel)
- `resources/ensemble_pipeline_job.yml` — newsgroups job + `ensemble_env` (scikit-learn, gensim, torch, wheel)
- `resources/experiment.yml` — MLflow experiment resource
- `resources/app.yml` — Databricks App with job + warehouse permissions

Prod target sets `run_as: service_principal_name: mlops-e2e-sp`.

### App backend

FastAPI (`app/backend/`) with three services:
- `mlflow_service.py` — experiment runs, model versions, alias management (uses `WorkspaceClient` from Databricks SDK for auth inside Apps)
- `jobs_service.py` — pipeline status, trigger run, run history
- `catalog_service.py` — Unity Catalog queries

`.env` at repo root is loaded by `main.py` before route imports because `mlflow_service` reads `UC_MODEL_NAME` at import time.

## Key conventions

- Unity Catalog 3-level namespace: `{catalog}.{schema}.{table_or_model}`
- Notebook format: `# Databricks notebook source` header, `# COMMAND ----------` cell separators
- All functions have type hints
- Test fixtures in `conftest.py`: `spark` (local Spark), `mock_dbutils` (simulates task values + widgets), `mock_mlflow_client`
- MLflow artifacts (test data `.npz`, FastText model, vocab) are the cross-stage communication mechanism for large data — task values only carry IDs/names

## Skills

- `/deploy` - Build frontend, upload to workspace, and deploy Databricks App (uses `mlops-e2e` profile). See `.claude/skills/deploy.md`.
