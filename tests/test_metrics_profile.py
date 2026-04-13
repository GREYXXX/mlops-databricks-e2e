"""Tests for dashboard metrics profile resolution."""

from __future__ import annotations

import os

import pytest

from app.backend.metrics_profile import (
    metric_mlflow_keys_for_profile,
    resolve_metrics_profile,
)


@pytest.fixture
def clear_profile_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHBOARD_METRICS_PROFILE", raising=False)


def test_explicit_regression(clear_profile_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHBOARD_METRICS_PROFILE", "regression")
    monkeypatch.setenv("UC_MODEL_NAME", "workspace.mlops_e2e.newsgroups_ensemble_model")
    assert resolve_metrics_profile() == "regression"


def test_explicit_classification(clear_profile_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHBOARD_METRICS_PROFILE", "classification")
    monkeypatch.setenv("UC_MODEL_NAME", "workspace.mlops_e2e.california_housing_model")
    assert resolve_metrics_profile() == "classification"


def test_auto_newsgroups_in_name(clear_profile_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UC_MODEL_NAME", "main.mlops_e2e.newsgroups_ensemble_model")
    assert resolve_metrics_profile() == "classification"


def test_auto_housing_default(clear_profile_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UC_MODEL_NAME", "main.mlops_e2e.california_housing_model")
    assert resolve_metrics_profile() == "regression"


def test_classification_key_map_includes_eval_and_fallback():
    m = metric_mlflow_keys_for_profile("classification")
    assert m["f1_weighted"][0] == "eval_ensemble_f1_weighted"
    assert "ensemble_f1_weighted" in m["f1_weighted"]
