from fastapi import APIRouter, HTTPException, Query

from ..services import jobs_service

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

RUNNING_STATES = {"RUNNING", "PENDING", "QUEUED", "BLOCKED"}


@router.get("/status")
def get_pipeline_status(job_name: str = Query(default=None)):
    kwargs = {}
    if job_name:
        kwargs["job_name"] = job_name
    return jobs_service.get_pipeline_status(**kwargs)


@router.post("/run")
def trigger_pipeline_run(job_name: str = Query(default=None)):
    kwargs = {}
    if job_name:
        kwargs["job_name"] = job_name

    # Check if pipeline is already running
    status = jobs_service.get_pipeline_status(**kwargs)
    if status.get("status") in RUNNING_STATES:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running",
        )

    result = jobs_service.trigger_pipeline_run(**kwargs)
    if "error" in result:
        status_code = 404 if result["error"] == "NOT_FOUND" else 500
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


@router.get("/history")
def get_pipeline_history(
    job_name: str = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100),
):
    kwargs = {"limit": limit}
    if job_name:
        kwargs["job_name"] = job_name
    return jobs_service.get_pipeline_history(**kwargs)
