import logging

from fastapi import APIRouter, HTTPException, Query

from ..services import mlflow_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/versions")
def list_versions():
    return mlflow_service.get_model_versions()


@router.get("/champion")
def get_champion():
    champion = mlflow_service.get_model_by_alias(alias="Champion")
    if champion is None:
        raise HTTPException(status_code=404, detail="No champion model found")
    return champion


@router.get("/history")
def list_version_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    return mlflow_service.get_version_history(page=page, page_size=page_size)


@router.get("/training-history")
def list_training_history():
    return mlflow_service.get_training_history()


@router.post("/promote/{version}")
def promote_version(version: str):
    try:
        return mlflow_service.promote_to_champion(version=version)
    except Exception as e:
        logger.exception("Failed to promote version %s", version)
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/versions/{version}")
def delete_version(version: str):
    try:
        return mlflow_service.delete_model_version(version=version)
    except Exception as e:
        logger.exception("Failed to delete version %s", version)
        raise HTTPException(status_code=400, detail=str(e))
