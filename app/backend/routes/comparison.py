from fastapi import APIRouter

from ..metrics_profile import (
    comparison_metric_keys,
    metric_mlflow_keys_for_profile,
    primary_comparison_metric_key,
    resolve_metrics_profile,
)
from ..services import mlflow_service

router = APIRouter(prefix="/api/models", tags=["comparison"])


@router.get("/comparison")
def get_comparison():
    profile = resolve_metrics_profile()
    key_map = metric_mlflow_keys_for_profile(profile)
    metric_keys = comparison_metric_keys(profile)

    champion = mlflow_service.get_model_by_alias(alias="Champion")
    challenger = mlflow_service.get_model_by_alias(alias="Challenger")

    def _extract_metrics(model_info):
        if model_info is None:
            return {}
        run_details = model_info.get("run_details")
        if run_details is None:
            return {}
        all_metrics = run_details.get("metrics", {})
        result: dict[str, float] = {}
        for api_key, mlflow_keys in key_map.items():
            for mk in mlflow_keys:
                if mk in all_metrics:
                    result[api_key] = all_metrics[mk]
                    break
        return result

    champion_metrics = _extract_metrics(champion)
    challenger_metrics = _extract_metrics(challenger)

    lower_is_better_keys = {"rmse", "mae", "mape", "median_ae"}

    deltas = {}
    for key in metric_keys:
        c_val = champion_metrics.get(key)
        ch_val = challenger_metrics.get(key)
        if c_val is not None and ch_val is not None:
            diff = ch_val - c_val
            lower_is_better = key in lower_is_better_keys
            improved = diff < 0 if lower_is_better else diff > 0
            deltas[key] = {
                "champion": c_val,
                "challenger": ch_val,
                "delta": diff,
                "improved": improved,
            }

    promotion_status = "no_models"
    promotion_reason = ""
    if champion and challenger:
        if champion.get("version") == challenger.get("version"):
            promotion_status = "promoted"
            promotion_reason = (
                f"Version {champion.get('version')} was promoted from "
                "Challenger to Champion (same model)."
            )
        elif deltas:
            primary = primary_comparison_metric_key(profile)
            primary_delta = deltas.get(primary)
            if isinstance(primary_delta, dict):
                if primary_delta.get("improved"):
                    promotion_status = "challenger_wins"
                    promotion_reason = (
                        f"Challenger (v{challenger.get('version')}) has better "
                        f"metrics than Champion (v{champion.get('version')})."
                    )
                else:
                    promotion_status = "champion_wins"
                    promotion_reason = (
                        f"Champion (v{champion.get('version')}) retains title "
                        f"over Challenger (v{challenger.get('version')})."
                    )
            else:
                promotion_status = "pending"
                promotion_reason = "Evaluation pending (primary metric not available on both runs)."
        else:
            promotion_status = "pending"
            promotion_reason = "Evaluation pending."
    elif champion and not challenger:
        promotion_status = "promoted"
        promotion_reason = (
            f"Champion is version {champion.get('version')}. "
            "The last Challenger was promoted and its alias was removed."
        )
    elif not champion and challenger:
        promotion_status = "pending"
        promotion_reason = (
            f"Challenger version {challenger.get('version')} awaiting promotion evaluation."
        )

    return {
        "champion": {
            "version": champion.get("version") if champion else None,
            "aliases": champion.get("aliases", []) if champion else [],
            "metrics": champion_metrics,
            "run_id": champion.get("run_id") if champion else None,
        },
        "challenger": {
            "version": challenger.get("version") if challenger else None,
            "aliases": challenger.get("aliases", []) if challenger else [],
            "metrics": challenger_metrics,
            "run_id": challenger.get("run_id") if challenger else None,
        },
        "deltas": deltas,
        "promotion_status": promotion_status,
        "promotion_reason": promotion_reason,
    }
