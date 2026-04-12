"""Ensemble training: LR+TF-IDF, RF+TF-IDF, TextCNN — logged as a single pyfunc model.

The pyfunc ``predict`` method supports a ``params`` dict with a ``return_mode`` key:
  - "labels"   (default) → 1-D array of predicted class indices
  - "proba"              → 2-D array of ensemble probabilities (n_samples, n_classes)
  - "all"                → dict with lr_proba, rf_proba, cnn_proba, ensemble_proba, labels

Per-model metrics (accuracy, precision, recall, weighted F1, classification report JSON)
are logged to the parent MLflow run so you can inspect any sub-model in the UI.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.pyfunc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from mlops_e2e.newsgroups.config import (
    EMBED_DIM,
    EXPERIMENT_NAME,
    MAX_SEQ_LEN,
    MAX_VOCAB,
    SEED,
    TEXTCNN_BATCH_SIZE,
    TEXTCNN_EPOCHS,
)
from mlops_e2e.newsgroups.data import load_raw_table
from mlops_e2e.newsgroups.feature_eng import load_feature_artifacts
from mlops_e2e.newsgroups.models import (
    TextCNN,
    TextDataset,
    build_embedding_matrix,
    preprocess_corpus,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# =============================================================================
# Base model trainers
# =============================================================================

def _train_lr(train_texts: list[str], train_labels: list[int]) -> tuple:
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="saga",
        random_state=SEED,
    )
    model.fit(X_train, train_labels)
    return vectorizer, model


def _train_rf(train_texts: list[str], train_labels: list[int]) -> tuple:
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=3,
    )
    X_train = vectorizer.fit_transform(train_texts)
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(X_train, train_labels)
    return vectorizer, model


def _train_textcnn(
    train_tokens: list[list[str]],
    train_labels: list[int],
    val_tokens: list[list[str]],
    val_labels: list[int],
    vocab: dict[str, int],
    embed_matrix: np.ndarray,
    num_classes: int,
    embed_dim: int = EMBED_DIM,
    epochs: int = TEXTCNN_EPOCHS,
    batch_size: int = TEXTCNN_BATCH_SIZE,
) -> TextCNN:
    import pytorch_lightning as pl

    pl.seed_everything(SEED, workers=True)

    train_ds = TextDataset(train_tokens, train_labels, vocab, max_len=MAX_SEQ_LEN)
    val_ds = TextDataset(val_tokens, val_labels, vocab, max_len=MAX_SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = TextCNN(
        vocab_size=len(vocab) + 1,
        embed_dim=embed_dim,
        num_classes=num_classes,
        embedding_matrix=embed_matrix,
    )

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
    )
    trainer.fit(model, train_dl, val_dl)
    return model


# =============================================================================
# Inference helpers
# =============================================================================

def _proba_lr(vec: TfidfVectorizer, model: LogisticRegression, texts: list[str]) -> np.ndarray:
    return model.predict_proba(vec.transform(texts))


def _proba_rf(vec: TfidfVectorizer, model: RandomForestClassifier, texts: list[str]) -> np.ndarray:
    return model.predict_proba(vec.transform(texts))


def _proba_cnn(
    cnn: TextCNN,
    tokens: list[list[str]],
    labels: list[int],
    vocab: dict[str, int],
    batch_size: int = TEXTCNN_BATCH_SIZE,
) -> np.ndarray:
    import pytorch_lightning as pl

    ds = TextDataset(tokens, labels, vocab, max_len=MAX_SEQ_LEN)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    trainer = pl.Trainer(
        accelerator="auto", devices="auto", logger=False, enable_progress_bar=False
    )
    batches = trainer.predict(cnn, dl)
    return np.vstack(batches)


# =============================================================================
# Metrics helper
# =============================================================================

def _compute_cls_metrics(
    y_true: list[int],
    y_proba: np.ndarray,
    class_names: list[str],
    prefix: str,
) -> dict[str, Any]:
    y_pred = y_proba.argmax(axis=1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}_cls_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
    }


# =============================================================================
# pyfunc wrapper
# =============================================================================

class _EnsembleModel(mlflow.pyfunc.PythonModel):
    """pyfunc model that wraps LR+TF-IDF, RF+TF-IDF, and TextCNN.

    Artifacts expected in context:
        lr_model     — pickled (vectorizer, LogisticRegression)
        rf_model     — pickled (vectorizer, RandomForestClassifier)
        cnn_state    — torch state_dict (.pt)
        vocab        — JSON token→index mapping
        cnn_config   — JSON with vocab_size, embed_dim, num_classes

    params (at inference time):
        return_mode: "labels" | "proba" | "all"  (default: "labels")
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import torch

        with open(context.artifacts["lr_model"], "rb") as f:
            self._lr_vec, self._lr_model = pickle.load(f)

        with open(context.artifacts["rf_model"], "rb") as f:
            self._rf_vec, self._rf_model = pickle.load(f)

        with open(context.artifacts["vocab"]) as f:
            self._vocab: dict[str, int] = json.load(f)

        with open(context.artifacts["cnn_config"]) as f:
            cfg = json.load(f)

        self._cnn = TextCNN(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_classes=cfg["num_classes"],
        )
        self._cnn.load_state_dict(
            torch.load(context.artifacts["cnn_state"], map_location="cpu", weights_only=True)
        )
        self._cnn.eval()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: list[Any],
        params: dict | None = None,
    ) -> Any:
        return_mode = (params or {}).get("return_mode", "labels")

        if isinstance(model_input, list):
            texts = model_input
        elif hasattr(model_input, "to_dict"):
            texts = model_input.iloc[:, 0].tolist()
        else:
            texts = list(model_input)

        dummy_labels = [0] * len(texts)
        tokens = preprocess_corpus(texts)

        lr_proba = _proba_lr(self._lr_vec, self._lr_model, texts)
        rf_proba = _proba_rf(self._rf_vec, self._rf_model, texts)
        cnn_proba = _proba_cnn(self._cnn, tokens, dummy_labels, self._vocab)

        ensemble_proba = (lr_proba + rf_proba + cnn_proba) / 3.0
        labels = ensemble_proba.argmax(axis=1)

        if return_mode == "proba":
            return ensemble_proba
        if return_mode == "all":
            return {
                "lr_proba": lr_proba,
                "rf_proba": rf_proba,
                "cnn_proba": cnn_proba,
                "ensemble_proba": ensemble_proba,
                "labels": labels,
            }
        return labels  # "labels" (default)


# =============================================================================
# Main training function
# =============================================================================

def train_ensemble(
    spark: SparkSession,
    catalog: str,
    schema: str,
    feature_run_id: str,
    experiment_name: str = EXPERIMENT_NAME,
    embed_dim: int = EMBED_DIM,
    textcnn_epochs: int = TEXTCNN_EPOCHS,
) -> str:
    """Train the full ensemble and log it as a pyfunc model to MLflow.

    Reads raw text from the newsgroups Delta table, downloads FastText + vocab
    from the feature engineering run, then trains:
      1. Logistic Regression + TF-IDF
      2. Random Forest + TF-IDF
      3. TextCNN with FastText embeddings

    All per-model metrics (accuracy, precision, recall, weighted F1,
    classification report JSON) are logged to the parent run.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        feature_run_id: MLflow run ID from the feature engineering stage.
        experiment_name: MLflow experiment path.
        embed_dim: Embedding dimensionality (must match feature eng stage).
        textcnn_epochs: TextCNN training epochs.

    Returns:
        MLflow run ID of the ensemble training run.
    """
    from sklearn.datasets import fetch_20newsgroups

    train_texts, test_texts, train_labels, test_labels = load_raw_table(spark, catalog, schema)
    class_names = fetch_20newsgroups(subset="train").target_names
    num_classes = len(class_names)

    logger.info("Loading feature artifacts from run %s", feature_run_id)
    ft_model, vocab = load_feature_artifacts(feature_run_id)

    embed_matrix = build_embedding_matrix(ft_model, vocab, embed_dim)

    train_tokens = preprocess_corpus(train_texts)
    test_tokens = preprocess_corpus(test_texts)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="ensemble_training") as run:
        mlflow.log_param("feature_run_id", feature_run_id)
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("textcnn_epochs", textcnn_epochs)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("train_size", len(train_texts))
        mlflow.log_param("test_size", len(test_texts))

        # ── LR ──────────────────────────────────────────────────────────────
        logger.info("Training Logistic Regression + TF-IDF…")
        lr_vec, lr_model = _train_lr(train_texts, train_labels)
        lr_proba_test = _proba_lr(lr_vec, lr_model, test_texts)
        lr_metrics = _compute_cls_metrics(test_labels, lr_proba_test, class_names, "lr")
        mlflow.log_metrics({k: v for k, v in lr_metrics.items() if isinstance(v, float)})
        logger.info("LR weighted F1: %.4f", lr_metrics["lr_f1_weighted"])

        # ── RF ──────────────────────────────────────────────────────────────
        logger.info("Training Random Forest + TF-IDF…")
        rf_vec, rf_model = _train_rf(train_texts, train_labels)
        rf_proba_test = _proba_rf(rf_vec, rf_model, test_texts)
        rf_metrics = _compute_cls_metrics(test_labels, rf_proba_test, class_names, "rf")
        mlflow.log_metrics({k: v for k, v in rf_metrics.items() if isinstance(v, float)})
        logger.info("RF weighted F1: %.4f", rf_metrics["rf_f1_weighted"])

        # ── TextCNN ─────────────────────────────────────────────────────────
        logger.info("Training TextCNN (epochs=%d)…", textcnn_epochs)
        cnn_model = _train_textcnn(
            train_tokens, train_labels,
            test_tokens, test_labels,
            vocab, embed_matrix, num_classes,
            embed_dim=embed_dim, epochs=textcnn_epochs,
        )
        cnn_proba_test = _proba_cnn(cnn_model, test_tokens, test_labels, vocab)
        cnn_metrics = _compute_cls_metrics(test_labels, cnn_proba_test, class_names, "cnn")
        mlflow.log_metrics({k: v for k, v in cnn_metrics.items() if isinstance(v, float)})
        logger.info("CNN weighted F1: %.4f", cnn_metrics["cnn_f1_weighted"])

        # ── Ensemble ────────────────────────────────────────────────────────
        ens_proba = (lr_proba_test + rf_proba_test + cnn_proba_test) / 3.0
        ens_metrics = _compute_cls_metrics(test_labels, ens_proba, class_names, "ensemble")
        mlflow.log_metrics({k: v for k, v in ens_metrics.items() if isinstance(v, float)})
        logger.info("Ensemble weighted F1: %.4f", ens_metrics["ensemble_f1_weighted"])

        # ── Save classification reports as JSON artifacts ────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            for prefix, metrics in [
                ("lr", lr_metrics),
                ("rf", rf_metrics),
                ("cnn", cnn_metrics),
                ("ensemble", ens_metrics),
            ]:
                report_path = os.path.join(tmpdir, f"{prefix}_classification_report.json")
                with open(report_path, "w") as f:
                    json.dump(metrics[f"{prefix}_cls_report"], f, indent=2)
                mlflow.log_artifact(report_path, "classification_reports")

        # ── Serialize sub-models ─────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            import torch

            lr_path = os.path.join(tmpdir, "lr_model.pkl")
            with open(lr_path, "wb") as f:
                pickle.dump((lr_vec, lr_model), f)

            rf_path = os.path.join(tmpdir, "rf_model.pkl")
            with open(rf_path, "wb") as f:
                pickle.dump((rf_vec, rf_model), f)

            cnn_state_path = os.path.join(tmpdir, "cnn_state.pt")
            torch.save(cnn_model.state_dict(), cnn_state_path)

            vocab_path = os.path.join(tmpdir, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(vocab, f)

            cnn_cfg_path = os.path.join(tmpdir, "cnn_config.json")
            with open(cnn_cfg_path, "w") as f:
                json.dump(
                    {"vocab_size": len(vocab) + 1, "embed_dim": embed_dim, "num_classes": num_classes},
                    f,
                )

            artifacts = {
                "lr_model": lr_path,
                "rf_model": rf_path,
                "cnn_state": cnn_state_path,
                "vocab": vocab_path,
                "cnn_config": cnn_cfg_path,
            }

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_EnsembleModel(),
                artifacts=artifacts,
                input_example=["example newsgroup post text"],
                pip_requirements=[
                    "scikit-learn>=1.3.0",
                    "numpy>=1.24.0",
                    "torch>=2.1.0",
                    "pytorch-lightning>=2.1.0",
                    "gensim>=4.3.0",
                ],
            )

        training_run_id = run.info.run_id
        logger.info("Ensemble training run ID: %s", training_run_id)

    return training_run_id
