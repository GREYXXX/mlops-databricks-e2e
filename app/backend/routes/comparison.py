from fastapi import APIRouter

from ..services import mlflow_service

router = APIRouter(prefix="/api/models", tags=["comparison"])

# Exact mapping: API response key -> MLflow metric key
METRIC_KEY_MAP = {
    "rmse": "test_rmse",
    "mae": "test_mae",
    "r2": "test_r2",
    "mape": "test_mape",
    "median_ae": "test_median_ae",
}

COMPARISON_METRICS = list(METRIC_KEY_MAP.keys())


@router.get("/comparison")
def get_comparison():
    champion = mlflow_service.get_model_by_alias(alias="Champion")
    challenger = mlflow_service.get_model_by_alias(alias="Challenger")

    def _extract_metrics(model_info):
        if model_info is None:
            return {}
        run_details = model_info.get("run_details")
        if run_details is None:
            return {}
        all_metrics = run_details.get("metrics", {})
        result = {}
        for api_key, mlflow_key in METRIC_KEY_MAP.items():
            if mlflow_key in all_metrics:
                result[api_key] = all_metrics[mlflow_key]
        return result

    champion_metrics = _extract_metrics(champion)
    challenger_metrics = _extract_metrics(challenger)

    deltas = {}
    for key in COMPARISON_METRICS:
        c_val = champion_metrics.get(key)
        ch_val = challenger_metrics.get(key)
        if c_val is not None and ch_val is not None:
            diff = ch_val - c_val
            lower_is_better = key in ("rmse", "mae", "mape", "median_ae")
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
        else:
            # Both exist with different versions - check if we have deltas
            if deltas:
                # Determine from deltas whether challenger was better
                rmse_delta = deltas.get("rmse", {})
                if isinstance(rmse_delta, dict) and rmse_delta.get("improved"):
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
