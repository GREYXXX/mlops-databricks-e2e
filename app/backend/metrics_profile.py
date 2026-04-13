"""Dashboard metrics profile: regression (housing) vs classification (newsgroups)."""

from __future__ import annotations

import os
from typing import Literal

MetricsProfile = Literal["regression", "classification"]


def resolve_metrics_profile() -> MetricsProfile:
    """Resolve profile from env or UC model name.

    Set ``DASHBOARD_METRICS_PROFILE`` to ``regression`` or ``classification`` to
    override. When unset, ``newsgroups`` in ``UC_MODEL_NAME`` selects classification.
    """
    explicit = os.getenv("DASHBOARD_METRICS_PROFILE", "").strip().lower()
    if explicit in ("regression", "classification"):
        return explicit  # type: ignore[return-value]
    model = os.getenv("UC_MODEL_NAME", "")
    if "newsgroups" in model.lower():
        return "classification"
    return "regression"


def metric_mlflow_keys_for_profile(
    profile: MetricsProfile,
) -> dict[str, tuple[str, ...]]:
    """Map API metric keys to MLflow metric names (first existing value wins)."""
    if profile == "classification":
        return {
            "f1_weighted": ("eval_ensemble_f1_weighted", "ensemble_f1_weighted"),
            "accuracy": ("eval_ensemble_accuracy", "ensemble_accuracy"),
            "precision": ("eval_ensemble_precision", "ensemble_precision"),
            "recall": ("eval_ensemble_recall", "ensemble_recall"),
        }
    return {
        "rmse": ("test_rmse",),
        "mae": ("test_mae",),
        "r2": ("test_r2",),
        "mape": ("test_mape",),
        "median_ae": ("test_median_ae",),
    }


def comparison_metric_keys(profile: MetricsProfile) -> tuple[str, ...]:
    return tuple(metric_mlflow_keys_for_profile(profile).keys())


def primary_comparison_metric_key(profile: MetricsProfile) -> str:
    return "f1_weighted" if profile == "classification" else "rmse"
