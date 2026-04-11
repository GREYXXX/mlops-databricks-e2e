"""Local tests for newsgroups evaluation helpers (no Spark / Databricks)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestPredictAllDict:
    """Guards _predict_all_dict against MLflow returning ndarray instead of dict."""

    def test_returns_dict_unchanged_when_pyfunc_predict_returns_dict(self):
        from mlops_e2e.newsgroups.evaluation import _predict_all_dict

        lr = np.array([[0.9, 0.1], [0.2, 0.8]])
        rf = np.array([[0.8, 0.2], [0.3, 0.7]])
        cnn = np.array([[0.85, 0.15], [0.25, 0.75]])
        ens = (lr + rf + cnn) / 3.0
        expected = {
            "lr_proba": lr,
            "rf_proba": rf,
            "cnn_proba": cnn,
            "ensemble_proba": ens,
        }

        ensemble = MagicMock()
        ensemble.predict.return_value = expected

        texts = ["a", "b"]
        out = _predict_all_dict(ensemble, texts)

        ensemble.predict.assert_called_once_with(texts, params={"return_mode": "all"})
        ensemble.unwrap_python_model.assert_not_called()
        assert out is expected
        assert np.array_equal(out["ensemble_proba"], ens)

    def test_unwraps_when_pyfunc_returns_ndarray_labels(self):
        """Simulates Databricks: params ignored → predict returns 1d label array."""
        from mlops_e2e.newsgroups.evaluation import _predict_all_dict

        lr = np.ones((2, 3)) / 3.0
        rf = np.ones((2, 3)) / 3.0
        cnn = np.ones((2, 3)) / 3.0
        ens = (lr + rf + cnn) / 3.0
        direct = {
            "lr_proba": lr,
            "rf_proba": rf,
            "cnn_proba": cnn,
            "ensemble_proba": ens,
        }

        py_model = MagicMock()
        py_model.predict.return_value = direct

        ensemble = MagicMock()
        ensemble.predict.return_value = np.array([0, 1], dtype=np.int64)
        ensemble.unwrap_python_model.return_value = py_model

        texts = ["x", "y"]
        out = _predict_all_dict(ensemble, texts)

        ensemble.predict.assert_called_once_with(texts, params={"return_mode": "all"})
        ensemble.unwrap_python_model.assert_called_once()
        py_model.predict.assert_called_once_with(None, texts, {"return_mode": "all"})
        assert out == direct

    def test_dataframe_with_lr_proba_column(self):
        from mlops_e2e.newsgroups.evaluation import _predict_all_dict

        lr = np.array([[1.0, 0.0], [0.0, 1.0]])
        rf = np.array([[0.5, 0.5], [0.5, 0.5]])
        cnn = np.array([[0.6, 0.4], [0.4, 0.6]])
        ens = (lr + rf + cnn) / 3.0

        df = pd.DataFrame(
            [
                {
                    "lr_proba": lr,
                    "rf_proba": rf,
                    "cnn_proba": cnn,
                    "ensemble_proba": ens,
                }
            ]
        )
        ensemble = MagicMock()
        ensemble.predict.return_value = df

        out = _predict_all_dict(ensemble, ["t1", "t2"])
        assert np.array_equal(out["lr_proba"], lr)
        assert np.array_equal(out["ensemble_proba"], ens)
        ensemble.unwrap_python_model.assert_not_called()

    def test_raises_if_unwrap_still_not_dict(self):
        from mlops_e2e.newsgroups.evaluation import _predict_all_dict

        py_model = MagicMock()
        py_model.predict.return_value = "bad"

        ensemble = MagicMock()
        ensemble.predict.return_value = np.array([0])
        ensemble.unwrap_python_model.return_value = py_model

        with pytest.raises(TypeError, match="Expected dict"):
            _predict_all_dict(ensemble, ["a"])
