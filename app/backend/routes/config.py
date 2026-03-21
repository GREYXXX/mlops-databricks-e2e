import os

from fastapi import APIRouter

from ..services import jobs_service, mlflow_service

router = APIRouter(prefix="/api", tags=["config"])

NOTEBOOK_MAP = {
    "data_preparation": "01_data_preparation",
    "feature_engineering": "02_feature_engineering",
    "model_training": "03_model_training",
    "model_evaluation": "04_model_evaluation",
    "model_registration": "05_model_registration",
    "champion_management": "06_champion_management",
}


@router.get("/config")
def get_config():
    ws = jobs_service._get_client()
    host = str(ws.config.host).rstrip("/")

    model_name = os.getenv(
        "UC_MODEL_NAME", "main.mlops_e2e.california_housing_model"
    )
    parts = model_name.split(".")
    model_url = (
        f"{host}/explore/data/models/{'/'.join(parts)}"
        if len(parts) == 3
        else None
    )

    experiment_id = None
    experiment_url = None
    try:
        exp_name = mlflow_service._resolve_experiment_name()
        client = mlflow_service._get_client()
        exp = client.get_experiment_by_name(exp_name)
        if exp:
            experiment_id = exp.experiment_id
            experiment_url = f"{host}/ml/experiments/{experiment_id}"
    except Exception:
        pass

    notebook_root = os.getenv("NOTEBOOK_ROOT_PATH", "")
    notebook_urls: dict[str, str] = {}
    if notebook_root:
        for stage_key, notebook_file in NOTEBOOK_MAP.items():
            path = f"{notebook_root}/{notebook_file}"
            notebook_urls[stage_key] = f"{host}/#workspace{path}"

    return {
        "workspace_url": host,
        "model_name": model_name,
        "model_url": model_url,
        "experiment_id": experiment_id,
        "experiment_url": experiment_url,
        "notebook_urls": notebook_urls,
    }
