import logging
import os
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _log_client_issue(operation: str, exc: BaseException, **context: Any) -> None:
    """Log a non-fatal MLflow / Databricks SDK error (single warning line + optional context)."""
    if context:
        details = " ".join(f"{k}={v!r}" for k, v in context.items())
        logger.warning("%s [%s]: %s", operation, details, exc)
    else:
        logger.warning("%s: %s", operation, exc)


mlflow.set_registry_uri("databricks-uc")

_ws_client: WorkspaceClient | None = None


def _get_ws_client() -> WorkspaceClient:
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client


_EXPERIMENT_NAME_ENV = os.getenv("MLFLOW_EXPERIMENT_NAME", "")


def _resolve_experiment_name() -> str:
    """Resolve the experiment name, searching if default path doesn't exist."""
    client = _get_client()

    # 1. Use env var if set and the experiment actually exists
    if _EXPERIMENT_NAME_ENV:
        exp = client.get_experiment_by_name(_EXPERIMENT_NAME_ENV)
        if exp is not None:
            return exp.name

    # 2. Derive user-scoped path from NOTEBOOK_ROOT_PATH
    #    e.g. /Workspace/Users/user@db.com/.bundle/... -> /Users/user@db.com/mlops_e2e_...
    notebook_root = os.getenv("NOTEBOOK_ROOT_PATH", "")
    if "/Users/" in notebook_root:
        username = notebook_root.split("/Users/")[1].split("/")[0]
        user_exp = f"/Users/{username}/mlops_e2e_california_housing"
        exp = client.get_experiment_by_name(user_exp)
        if exp is not None:
            return exp.name

    # 3. Search by pattern, prefer most recently updated
    try:
        results = client.search_experiments(
            filter_string="name LIKE '%mlops_e2e_california_housing'"
        )
        if results:
            best = max(results, key=lambda e: e.last_update_time or 0)
            return best.name
    except Exception as e:
        _log_client_issue("search_experiments for mlops_e2e_california_housing", e)

    return _EXPERIMENT_NAME_ENV or "/mlops_e2e_california_housing"


EXPERIMENT_NAME: str = ""  # resolved lazily
MODEL_NAME = os.getenv("UC_MODEL_NAME", "main.mlops_e2e.california_housing_model")

_client: MlflowClient | None = None


def _get_client() -> MlflowClient:
    global _client
    if _client is None:
        _client = MlflowClient()
    return _client


def get_experiment_runs(
    experiment_name: str = "",
    max_results: int = 50,
    order_by: str | None = None,
):
    if not experiment_name:
        global EXPERIMENT_NAME
        if not EXPERIMENT_NAME:
            EXPERIMENT_NAME = _resolve_experiment_name()
        experiment_name = EXPERIMENT_NAME
    client = _get_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    order_by_list = None
    if order_by:
        order_by_list = [order_by]

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=order_by_list or ["start_time DESC"],
    )

    results = []
    for run in runs:
        results.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "duration_ms": (
                    (run.info.end_time - run.info.start_time)
                    if run.info.end_time and run.info.start_time
                    else None
                ),
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
            }
        )
    return results


def get_run_details(run_id: str):
    client = _get_client()
    try:
        run = client.get_run(run_id)
    except Exception as e:
        _log_client_issue("get_run", e, run_id=run_id)
        return None

    artifacts = []
    try:
        artifact_list = client.list_artifacts(run_id)
        for artifact in artifact_list:
            artifacts.append(
                {
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size,
                }
            )
    except Exception as e:
        _log_client_issue("list_artifacts", e, run_id=run_id)

    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "duration_ms": (
            (run.info.end_time - run.info.start_time)
            if run.info.end_time and run.info.start_time
            else None
        ),
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "artifacts": artifacts,
        "artifact_uri": run.info.artifact_uri,
    }


def get_run_artifact(run_id: str, artifact_path: str) -> str | None:
    client = _get_client()
    try:
        return client.download_artifacts(run_id, artifact_path)
    except Exception as e:
        _log_client_issue("download_artifacts", e, run_id=run_id, artifact_path=artifact_path)
        return None


def _get_alias_map(model_name: str = MODEL_NAME) -> dict:
    """Return {version_number: [alias_name, ...]} from registered model."""
    ws = _get_ws_client()
    alias_map: dict = {}
    try:
        model = ws.registered_models.get(full_name=model_name, include_aliases=True)
        if model.aliases:
            for a in model.aliases:
                ver = str(a.version_num)
                alias_map.setdefault(ver, []).append(a.alias_name)
    except Exception as e:
        _log_client_issue("registered_models.get (aliases)", e, model_name=model_name)
    return alias_map


def _list_uc_model_versions(model_name: str = MODEL_NAME):
    """List model versions using Databricks SDK (works with UC Service Principal)."""
    ws = _get_ws_client()
    try:
        return list(ws.model_versions.list(full_name=model_name))
    except Exception as e:
        _log_client_issue("model_versions.list", e, model_name=model_name)
        return []


def get_model_versions(model_name: str = MODEL_NAME):
    versions = _list_uc_model_versions(model_name)
    alias_map = _get_alias_map(model_name)
    results = []
    for v in versions:
        ver_str = str(v.version)
        results.append(
            {
                "version": ver_str,
                "name": v.model_name,
                "creation_timestamp": v.created_at,
                "last_updated_timestamp": v.updated_at,
                "status": v.status.value if v.status else "READY",
                "source": v.source or "",
                "run_id": v.run_id or "",
                "aliases": alias_map.get(ver_str, []),
            }
        )
    return sorted(results, key=lambda x: int(x["version"]), reverse=True)


def get_model_by_alias(model_name: str = MODEL_NAME, alias: str = "Champion"):
    client = _get_client()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        run_details = get_run_details(mv.run_id) if mv.run_id else None

        return {
            "version": mv.version,
            "name": mv.name,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
            "status": mv.status,
            "source": mv.source,
            "run_id": mv.run_id,
            "aliases": list(mv.aliases) if mv.aliases else [],
            "run_details": run_details,
        }
    except Exception as e:
        _log_client_issue("get_model_version_by_alias", e, model_name=model_name, alias=alias)
        return None


def get_model_metrics(model_name: str = MODEL_NAME, version: str = ""):
    client = _get_client()
    try:
        mv = client.get_model_version(model_name, version)
        if mv.run_id:
            run = client.get_run(mv.run_id)
            return dict(run.data.metrics)
    except Exception as e:
        _log_client_issue("get_model_metrics", e, model_name=model_name, version=version)
    return {}


def get_training_history(model_name: str = MODEL_NAME):
    """Return all model versions with their training run details.

    Shows the relationship between registered model versions and the
    MLflow experiment runs that produced them.
    """
    mlflow_client = _get_client()
    uc_versions = _list_uc_model_versions(model_name)
    alias_map = _get_alias_map(model_name)

    items = []
    for v in sorted(uc_versions, key=lambda x: x.version, reverse=True):
        ver_str = str(v.version)
        aliases = alias_map.get(ver_str, [])

        run_metrics: dict = {}
        run_params: dict = {}
        run_start_time = None
        run_end_time = None
        run_name = None
        if v.run_id:
            try:
                run = mlflow_client.get_run(v.run_id)
                run_metrics = dict(run.data.metrics)
                run_params = dict(run.data.params)
                run_start_time = run.info.start_time
                run_end_time = run.info.end_time
                run_name = run.info.run_name
            except Exception as e:
                _log_client_issue(
                    "get_run (training_history)",
                    e,
                    run_id=v.run_id,
                    model_name=model_name,
                    version=ver_str,
                )

        items.append(
            {
                "version": ver_str,
                "aliases": aliases,
                "run_id": v.run_id or "",
                "run_name": run_name,
                "creation_timestamp": v.created_at,
                "training_start_time": run_start_time,
                "training_end_time": run_end_time,
                "metrics": run_metrics,
                "params": run_params,
            }
        )

    return items


def _classify_version(aliases: list[str]) -> tuple[str, str]:
    """Return (role, role_timestamp) for a version based on its aliases.

    role is one of: champion, challenger, past_champion, past_challenger, none.
    role_timestamp is the archive timestamp string (or empty).
    """
    import re

    pat = re.compile(r"^(Champion|Challenger)-(\d{8}-\d{4,6})$", re.IGNORECASE)
    legacy_pat = re.compile(r"^Champion-(\d{4}-\d{2}-\d{2})$", re.IGNORECASE)

    for alias in aliases:
        if alias.lower() == "champion":
            return "champion", ""
        if alias.lower() == "challenger":
            return "challenger", ""
    for alias in aliases:
        m = pat.match(alias)
        if m:
            prefix = m.group(1).lower()
            return f"past_{prefix}", m.group(2)
        m = legacy_pat.match(alias)
        if m:
            return "past_champion", m.group(1)
    return "none", ""


def get_version_history(
    model_name: str = MODEL_NAME,
    page: int = 1,
    page_size: int = 10,
):
    """Return all model versions with aliases, metrics, and role info (paginated).

    Excludes the current Champion and Challenger (shown in comparison cards).
    """
    mlflow_client = _get_client()
    uc_versions = _list_uc_model_versions(model_name)
    alias_map = _get_alias_map(model_name)

    all_items = []
    for v in sorted(uc_versions, key=lambda x: x.version, reverse=True):
        ver_str = str(v.version)
        aliases = alias_map.get(ver_str, [])
        role, role_ts = _classify_version(aliases)
        # Skip current Champion / Challenger — they are in the comparison cards
        if role in ("champion", "challenger"):
            continue
        metrics: dict = {}
        if v.run_id:
            try:
                run = mlflow_client.get_run(v.run_id)
                metrics = dict(run.data.metrics)
            except Exception as e:
                _log_client_issue(
                    "get_run (version_history)",
                    e,
                    run_id=v.run_id,
                    model_name=model_name,
                    version=ver_str,
                )
        all_items.append(
            {
                "version": ver_str,
                "aliases": aliases,
                "role": role,
                "role_timestamp": role_ts,
                "run_id": v.run_id,
                "creation_timestamp": v.created_at,
                "metrics": metrics,
            }
        )

    total = len(all_items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    items = all_items[start : start + page_size]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def promote_to_champion(model_name: str = MODEL_NAME, version: str = ""):
    """Promote a specific version to Champion.

    Archives the current Champion and Challenger (if they exist and differ
    from the target) with timestamped aliases.  Uses the Databricks SDK
    (WorkspaceClient) for reliable auth in Databricks App environments.
    """
    from datetime import datetime, timezone

    ws = _get_ws_client()
    version_num = int(version)
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Read current aliases — must succeed before we modify anything
    model = ws.registered_models.get(full_name=model_name, include_aliases=True)

    raw_aliases = [(a.alias_name, a.version_num) for a in (model.aliases or [])]
    print(f"[PROMOTE] target=v{version_num}, raw aliases={raw_aliases}", flush=True)

    champion_ver: int | None = None
    challenger_ver: int | None = None
    challenger_alias_name: str = ""
    target_other_aliases: list[str] = []  # all non-Champion aliases on target
    for a in model.aliases or []:
        if a.alias_name.lower() == "champion":
            champion_ver = a.version_num
        elif a.alias_name.lower() == "challenger":
            challenger_ver = a.version_num
            challenger_alias_name = a.alias_name
        # Collect every alias on the target version (except Champion/Challenger)
        if a.version_num == version_num and a.alias_name.lower() not in ("champion", "challenger"):
            target_other_aliases.append(a.alias_name)

    print(
        f"[PROMOTE] champion_ver={champion_ver} (type={type(champion_ver).__name__}), "
        f"challenger_ver={challenger_ver}, target_other_aliases={target_other_aliases}",
        flush=True,
    )

    # Archive current Champion (the one being dethroned)
    if champion_ver is not None and champion_ver != version_num:
        archive_alias = f"Champion-{now_str}"
        print(
            f"[PROMOTE] Archiving current Champion v{champion_ver} -> '{archive_alias}'", flush=True
        )
        ws.registered_models.set_alias(
            full_name=model_name,
            alias=archive_alias,
            version_num=champion_ver,
        )
    else:
        print(
            f"[PROMOTE] SKIP archive: champion_ver={champion_ver}, version_num={version_num}, "
            f"equal={champion_ver == version_num}",
            flush=True,
        )

    # Archive current Challenger
    if challenger_ver is not None and challenger_ver != version_num:
        print(f"[PROMOTE] Archiving current Challenger v{challenger_ver}", flush=True)
        ws.registered_models.set_alias(
            full_name=model_name,
            alias=f"Challenger-{now_str}",
            version_num=challenger_ver,
        )
        ws.registered_models.delete_alias(full_name=model_name, alias=challenger_alias_name)

    # Remove ALL other aliases from the target version before crowning
    for alias_name in target_other_aliases:
        print(f"[PROMOTE] Removing alias '{alias_name}' from target v{version_num}", flush=True)
        ws.registered_models.delete_alias(full_name=model_name, alias=alias_name)

    # Set Champion alias on the target version
    print(f"[PROMOTE] Setting Champion alias on v{version_num}", flush=True)
    ws.registered_models.set_alias(full_name=model_name, alias="Champion", version_num=version_num)

    # Verify final state
    model_after = ws.registered_models.get(full_name=model_name, include_aliases=True)
    final_aliases = [(a.alias_name, a.version_num) for a in (model_after.aliases or [])]
    print(f"[PROMOTE] DONE: final aliases={final_aliases}", flush=True)

    return {"version": version, "alias": "Champion"}


def delete_model_version(model_name: str = MODEL_NAME, version: str = ""):
    """Delete a model version from the registry.

    Uses the Databricks SDK (WorkspaceClient) for reliable auth in
    Databricks App environments.
    """
    ws = _get_ws_client()
    version_num = int(version)

    # Remove any aliases first (required before deletion)
    try:
        model = ws.registered_models.get(full_name=model_name, include_aliases=True)
        for a in model.aliases or []:
            if a.version_num == version_num:
                try:
                    ws.registered_models.delete_alias(full_name=model_name, alias=a.alias_name)
                except Exception as e:
                    _log_client_issue(
                        "delete_alias before version delete",
                        e,
                        model_name=model_name,
                        alias=a.alias_name,
                        version=version_num,
                    )
    except Exception as e:
        _log_client_issue(
            "registered_models.get before version delete",
            e,
            model_name=model_name,
            version=version_num,
        )

    ws.model_versions.delete(full_name=model_name, version=version_num)
    return {"version": version, "deleted": True}
