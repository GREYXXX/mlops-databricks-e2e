# MLOps E2E Example

End-to-end MLOps pipeline on Databricks demonstrating the full model lifecycle: data preparation, feature engineering, model training with hyperparameter tuning, evaluation, Unity Catalog registration, and champion/challenger model promotion.

Built as a **Databricks Asset Bundle (DAB)** for portable deployment across workspaces, with a **Databricks App** dashboard for operational visibility.

## Pipeline Stages

| Stage | Description | Key Technology |
|-------|-------------|----------------|
| 1. Data Preparation | Load California Housing dataset, save as Delta table | sklearn, Delta Lake |
| 2. Feature Engineering | Log transforms, derived ratios, scaling | PySpark, Delta Lake |
| 3. Model Training | LightGBM + Optuna (50 trials) with MLflow tracking | LightGBM, Optuna, MLflow |
| 4. Model Evaluation | RMSE, MAE, R2, MAPE metrics + artifact plots | MLflow, matplotlib |
| 5. Model Registration | Register best model to Unity Catalog | MLflow, Unity Catalog |
| 6. Champion Management | Compare Challenger vs Champion, promote if better | MLflow Model Registry |

## Dashboard App

React + TypeScript frontend with FastAPI backend, deployed as a **Databricks App**.

### Pipeline View

Real-time 6-stage pipeline visualization with status tracking, execution times, and direct links to workspace notebooks. Click any stage to view its source code and imported functions inline.

![Pipeline View](screenshots/dashboard-pipeline-code.png)

### Model Comparison

Side-by-side Champion vs Challenger metrics (RMSE, MAE, R², MAPE, Median AE) with delta indicators and promotion decision logic. Links directly to the Unity Catalog registered model.

![Model Comparison](screenshots/dashboard-models.png)

### Training History

Model version table showing the relationship between registered models and their training experiment runs. Each row displays version, aliases, training date, metrics (RMSE, MAE, R²), and key hyperparameters. Click any row for full details with artifacts, or use the link icon to open the run directly in MLflow.

![Training History](screenshots/dashboard-training-history.png)

## Quick Start

### Prerequisites

- Databricks CLI configured with a workspace
- Python 3.10+
- Node.js 18+ (for frontend)

### Workspace Login

Each user must create a **named profile** linked to a specific workspace. This profile name is required for all subsequent CLI commands.

```bash
# 1. Create a profile linked to your target workspace
#    This opens a browser for OAuth login and saves credentials under the profile name.
databricks auth login --host https://<workspace-url> --profile <PROFILE>

# 2. Verify the profile is configured correctly
databricks auth env --profile <PROFILE>
#   → Should show DATABRICKS_HOST pointing to your workspace URL

# 3. Verify you can connect to the workspace
databricks current-user me --profile <PROFILE>
```

The profile is stored in `~/.databrickscfg`. You can inspect or edit it directly:

```ini
# ~/.databrickscfg
[mlops-e2e]
host  = https://my-workspace.cloud.databricks.com
auth_type = databricks-cli
```

> Replace `<PROFILE>` with a name of your choice (e.g., `mlops-e2e`). If you work with multiple workspaces, create a separate profile for each one. For service principal or token-based auth, see `databricks configure --help`.

### Deploy

All bundle commands require `--profile <PROFILE>` to target the correct workspace.

```bash
# Validate bundle
databricks bundle validate -t dev --profile <PROFILE>

# Deploy resources and code
databricks bundle deploy -t dev --profile <PROFILE>

# Run the pipeline
databricks bundle run -t dev --profile <PROFILE> mlops_e2e_pipeline
```

### App Service Principal Permissions

Databricks Apps run under an auto-generated **Service Principal** (SP). After the first deploy, you must grant this SP access to the resources the dashboard reads. Without these permissions, the app will return empty data or errors.

1. **Find the App Service Principal**: Go to the app detail page in the workspace UI, or:
   ```bash
   databricks apps get mlops-e2e-dashboard --profile <PROFILE>
   ```
   The SP name is shown under `service_principal_name` (e.g., `app-mlops-e2e-dashboard-...`).

2. **Grant Unity Catalog permissions** (run in a SQL warehouse or notebook):
   ```sql
   -- Schema-level read access (covers all tables)
   GRANT USE CATALOG ON CATALOG <catalog> TO `<service_principal_name>`;
   GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<service_principal_name>`;
   GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<service_principal_name>`;

   -- Model read + manage (for re-promote / rollback / delete)
   GRANT MANAGE ON FUNCTION <catalog>.<schema>.california_housing_model TO `<service_principal_name>`;
   ```

3. **Grant MLflow Experiment access**: The experiment is created under the deploying user's directory. Grant CAN_READ via the workspace UI or CLI:
   ```bash
   databricks experiments set-permissions <EXPERIMENT_ID> \
     --access-control-list '[{"service_principal_name":"<service_principal_name>","permission_level":"CAN_READ"}]' \
     --profile <PROFILE>
   ```

4. **Grant Model Registry access** (UC model alias management for re-promote):
   ```sql
   GRANT APPLY TAG ON SCHEMA <catalog>.<schema> TO `<service_principal_name>`;
   ```

| Resource | Permission | Reason |
|----------|-----------|--------|
| `<catalog>` | `USE CATALOG` | Access the catalog |
| `<catalog>.<schema>` | `USE SCHEMA`, `SELECT` | Read pipeline tables (`california_housing_raw`, `california_housing_features`) |
| `<catalog>.<schema>.<model_name>` | `MANAGE` | Read/write model versions, aliases, promote/delete |
| MLflow Experiment | `CAN_READ` | Read experiment runs and metrics |
| Job (pipeline) | `CAN_MANAGE_RUN` | Trigger pipeline (already configured in `app.yml`) |
| SQL Warehouse | `CAN_USE` | Execute queries (already configured in `app.yml`) |

> Replace `<catalog>`, `<schema>`, `<service_principal_name>` with your actual values (e.g., `main`, `mlops_e2e`, `app-mlops-e2e-dashboard-1234`).

### Local Development

```bash
# Install Python package
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build frontend
cd app/frontend && npm install && npm run build
```

### DAB Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `catalog` | `main` | Unity Catalog catalog name |
| `schema` | `mlops_e2e` | Unity Catalog schema name |
| `model_name` | `california_housing_model` | Registered model name |
| `warehouse_id` | — | SQL warehouse ID for app |

Override per target in `databricks.yml` or via CLI: `databricks bundle deploy -t dev --profile <PROFILE> --var="catalog=my_catalog"`

## Project Structure

```
mlops_e2e_example/
├── databricks.yml              # DAB root config (targets, variables, artifacts)
├── src/mlops_e2e/              # Python package (testable pipeline logic)
├── notebooks/                  # Databricks notebooks (thin wrappers)
├── resources/                  # DAB resource definitions (job, app, experiment)
├── app/                        # Databricks App (FastAPI + React)
│   ├── backend/                # FastAPI API + services
│   └── frontend/               # React + TypeScript + Tailwind
└── tests/                      # pytest unit tests (58 tests)
```

## Testing

All pipeline logic lives in `src/mlops_e2e/` as pure Python, enabling local testing without Databricks:

```bash
pytest tests/ -v --tb=short    # 58 tests, ~23s
```

Tests use real local Spark for DataFrame operations and mock MLflow/Databricks SDK calls.

## Troubleshooting

### MLflow Experiment name conflict

If `databricks bundle deploy` fails with an experiment name conflict, this is because an MLflow experiment with the same name already exists in the workspace.

This project avoids the issue by namespacing the experiment under each user's directory (`/Users/{username}/mlops_e2e_california_housing`). The DAB resource in `resources/experiment.yml` and the training notebook (`notebooks/03_model_training.py`) both construct the path using the current user, so each user gets their own experiment automatically.

If you still encounter a conflict (e.g., from a previous deployment), bind the existing experiment to the bundle:

```bash
# Find the existing experiment ID
databricks experiments list --profile <PROFILE> | grep mlops_e2e_california_housing

# Bind the experiment to the bundle resource
databricks bundle deployment bind mlops_e2e_experiment <EXPERIMENT_ID> -t dev --profile <PROFILE>

# Deploy again
databricks bundle deploy -t dev --profile <PROFILE>
```

### Databricks App name conflict

If `databricks bundle deploy` fails with `An app with the same name already exists`, bind the existing app to the bundle:

```bash
# Bind the existing app to the bundle resource
databricks bundle deployment bind mlops_e2e_dashboard mlops-e2e-dashboard --auto-approve -t dev --profile <PROFILE>

# Deploy again
databricks bundle deploy -t dev --profile <PROFILE>
```
