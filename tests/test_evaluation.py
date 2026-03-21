"""Tests for evaluation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_returns_all_five_metrics(self):
        from mlops_e2e.evaluation import compute_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        metrics = compute_metrics(y_true, y_pred)
        expected_keys = {"test_rmse", "test_mae", "test_r2", "test_mape", "test_median_ae"}
        assert set(metrics.keys()) == expected_keys

    def test_perfect_prediction(self):
        from mlops_e2e.evaluation import compute_metrics

        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y)

        assert metrics["test_rmse"] == 0.0
        assert metrics["test_mae"] == 0.0
        assert metrics["test_r2"] == 1.0
        assert metrics["test_mape"] == 0.0
        assert metrics["test_median_ae"] == 0.0

    def test_rmse_is_positive(self):
        from mlops_e2e.evaluation import compute_metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["test_rmse"] > 0

    def test_r2_in_valid_range(self):
        from mlops_e2e.evaluation import compute_metrics

        np.random.seed(42)
        y_true = np.random.uniform(1, 5, 100)
        y_pred = y_true + np.random.normal(0, 0.3, 100)

        metrics = compute_metrics(y_true, y_pred)
        assert 0 < metrics["test_r2"] <= 1.0

    def test_mape_handles_zeros(self):
        from mlops_e2e.evaluation import compute_metrics

        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["test_mape"] == float("inf")

    def test_known_rmse(self):
        from mlops_e2e.evaluation import compute_metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])

        metrics = compute_metrics(y_true, y_pred)
        # RMSE of errors [1,1,1] = 1.0
        assert abs(metrics["test_rmse"] - 1.0) < 1e-6


class TestGeneratePlots:
    """Tests for generate_plots()."""

    def test_generates_basic_plots(self):
        from mlops_e2e.evaluation import generate_plots

        y_true = np.random.uniform(1, 5, 50)
        y_pred = y_true + np.random.normal(0, 0.3, 50)

        plots = generate_plots(y_true, y_pred)

        assert "residual_plot" in plots
        assert "predicted_vs_actual" in plots
        assert "error_distribution" in plots
        # No feature importance without feature_importances arg
        assert "feature_importance" not in plots

    def test_generates_feature_importance_when_provided(self):
        from mlops_e2e.evaluation import generate_plots

        y_true = np.random.uniform(1, 5, 50)
        y_pred = y_true + np.random.normal(0, 0.3, 50)
        importances = np.array([0.5, 0.3, 0.2])
        names = ["f1", "f2", "f3"]

        plots = generate_plots(y_true, y_pred, importances, names)
        assert "feature_importance" in plots

    def test_plots_are_figures(self):
        import matplotlib.figure

        from mlops_e2e.evaluation import generate_plots

        y_true = np.random.uniform(1, 5, 50)
        y_pred = y_true + np.random.normal(0, 0.3, 50)

        plots = generate_plots(y_true, y_pred)
        for name, fig in plots.items():
            assert isinstance(fig, matplotlib.figure.Figure), f"{name} is not a Figure"


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_evaluate_logs_metrics_and_plots(self, sample_test_data, tmp_path):
        from mlops_e2e.evaluation import evaluate_model

        # Save test data to disk
        test_data_path = str(tmp_path / "test_data.npz")
        np.savez(test_data_path, **sample_test_data)

        # Create a mock model that returns predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = sample_test_data["y_test"] + np.random.normal(
            0, 0.1, len(sample_test_data["y_test"])
        )
        mock_model.feature_importances_ = np.random.rand(len(sample_test_data["feature_names"]))

        with patch("mlops_e2e.evaluation.mlflow") as mock_mlflow:
            mock_mlflow.lightgbm.load_model.return_value = mock_model
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            metrics = evaluate_model(run_id="fake_run_id", test_data_path=test_data_path)

        assert "test_rmse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        assert "test_mape" in metrics
        assert "test_median_ae" in metrics

        mock_mlflow.log_metrics.assert_called_once()
        assert mock_mlflow.log_artifact.call_count >= 2  # plots + summary JSON
