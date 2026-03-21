import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

JOB_NAME = os.getenv("PIPELINE_JOB_NAME", "MLOps E2E Pipeline")

_ws: WorkspaceClient | None = None


def _get_client() -> WorkspaceClient:
    global _ws
    if _ws is None:
        _ws = WorkspaceClient()
    return _ws


def get_pipeline_status(job_name: str = JOB_NAME):
    ws = _get_client()

    jobs = list(ws.jobs.list(name=job_name))
    if not jobs:
        return {"job_name": job_name, "status": "NOT_FOUND", "tasks": []}

    job = jobs[0]
    job_id = job.job_id

    runs = list(ws.jobs.list_runs(job_id=job_id, limit=1))
    if not runs:
        return {"job_name": job_name, "job_id": job_id, "status": "NO_RUNS", "tasks": []}

    # Get full run details (list_runs doesn't include task-level info)
    latest_run = ws.jobs.get_run(runs[0].run_id)

    tasks = []
    if latest_run.tasks:
        for task in latest_run.tasks:
            state = task.state
            task_info = {
                "task_key": task.task_key,
                "status": _lifecycle_to_str(state.life_cycle_state) if state else "UNKNOWN",
                "result": _result_to_str(state.result_state)
                if state and state.result_state
                else None,
                "start_time": task.start_time,
                "end_time": task.end_time,
                "duration_ms": (
                    (task.end_time - task.start_time) if task.end_time and task.start_time else None
                ),
                "attempt_number": task.attempt_number,
            }
            tasks.append(task_info)

    run_state = latest_run.state
    return {
        "job_name": job_name,
        "job_id": job_id,
        "run_id": latest_run.run_id,
        "run_name": latest_run.run_name,
        "status": _lifecycle_to_str(run_state.life_cycle_state) if run_state else "UNKNOWN",
        "result": _result_to_str(run_state.result_state)
        if run_state and run_state.result_state
        else None,
        "start_time": latest_run.start_time,
        "end_time": latest_run.end_time,
        "tasks": tasks,
    }


def trigger_pipeline_run(job_name: str = JOB_NAME) -> dict:
    ws = _get_client()

    jobs = list(ws.jobs.list(name=job_name))
    if not jobs:
        return {"error": "NOT_FOUND", "message": f"Job '{job_name}' not found"}

    job_id = jobs[0].job_id
    try:
        wait_response = ws.jobs.run_now(job_id)
        return {"run_id": wait_response.run_id, "job_id": job_id}
    except Exception as e:
        return {"error": "TRIGGER_FAILED", "message": str(e)}


def get_pipeline_history(job_name: str = JOB_NAME, limit: int = 10):
    ws = _get_client()

    jobs = list(ws.jobs.list(name=job_name))
    if not jobs:
        return []

    job_id = jobs[0].job_id
    runs = list(ws.jobs.list_runs(job_id=job_id, limit=limit))

    results = []
    for run in runs:
        state = run.state
        results.append(
            {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "status": _lifecycle_to_str(state.life_cycle_state) if state else "UNKNOWN",
                "result": _result_to_str(state.result_state)
                if state and state.result_state
                else None,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration_ms": (
                    (run.end_time - run.start_time) if run.end_time and run.start_time else None
                ),
            }
        )
    return results


def _lifecycle_to_str(state: RunLifeCycleState | None) -> str:
    if state is None:
        return "UNKNOWN"
    return state.value


def _result_to_str(state: RunResultState | None) -> str:
    if state is None:
        return "UNKNOWN"
    return state.value
