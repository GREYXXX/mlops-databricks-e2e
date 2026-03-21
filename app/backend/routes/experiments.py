from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..services import mlflow_service

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/runs")
def list_runs(
    max_results: int = Query(default=50, ge=1, le=200),
    order_by: str = Query(default=None),
):
    return mlflow_service.get_experiment_runs(
        max_results=max_results,
        order_by=order_by,
    )


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    details = mlflow_service.get_run_details(run_id)
    if details is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return details


@router.get("/runs/{run_id}/artifacts/{path:path}")
def get_artifact(run_id: str, path: str):
    local_path = mlflow_service.get_run_artifact(run_id, path)
    if local_path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = "application/octet-stream"
    if path.endswith(".png"):
        media_type = "image/png"
    elif path.endswith(".jpg") or path.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif path.endswith(".json"):
        media_type = "application/json"
    elif path.endswith(".csv"):
        media_type = "text/csv"

    return FileResponse(local_path, media_type=media_type)
