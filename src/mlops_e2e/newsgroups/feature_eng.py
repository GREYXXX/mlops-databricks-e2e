"""Feature engineering: tokenize corpus, train FastText, build vocab.

This stage is intentionally decoupled from classifier training because
FastText is unsupervised and expensive. You can update embeddings on a
different schedule than retraining the classifiers.

Artifacts saved to MLflow:
  - fasttext_model/   — gensim FastText model files
  - vocab.json        — token→index mapping
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING

import mlflow

from mlops_e2e.newsgroups.config import EMBED_DIM, EXPERIMENT_NAME, FASTTEXT_EPOCHS, MAX_VOCAB
from mlops_e2e.newsgroups.data import load_raw_table
from mlops_e2e.newsgroups.models import build_vocab, preprocess_corpus, train_fasttext

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def run_feature_engineering(
    spark: SparkSession,
    catalog: str,
    schema: str,
    experiment_name: str = EXPERIMENT_NAME,
    embed_dim: int = EMBED_DIM,
    max_vocab: int = MAX_VOCAB,
    fasttext_epochs: int = FASTTEXT_EPOCHS,
) -> str:
    """Tokenize corpus, train FastText, build vocab, log artifacts to MLflow.

    Reads from the raw newsgroups Delta table. FastText is trained on ALL text
    (train + test) because it is unsupervised — this gives better subword
    representations without leaking label information.

    The vocab is built from training text only to respect the train/test split.

    Args:
        spark: Active SparkSession.
        catalog: Unity Catalog catalog name.
        schema: Unity Catalog schema name.
        experiment_name: MLflow experiment path.
        embed_dim: FastText embedding dimensionality.
        max_vocab: Maximum vocabulary size.
        fasttext_epochs: Number of FastText training epochs.

    Returns:
        MLflow run ID containing the FastText model and vocab artifacts.
    """
    train_texts, test_texts, _, _ = load_raw_table(spark, catalog, schema)

    logger.info("Tokenizing corpus (%d train + %d test docs)…", len(train_texts), len(test_texts))
    train_tokens = preprocess_corpus(train_texts)
    test_tokens = preprocess_corpus(test_texts)
    all_tokens = train_tokens + test_tokens

    logger.info("Training FastText (embed_dim=%d, epochs=%d)…", embed_dim, fasttext_epochs)
    ft_model = train_fasttext(all_tokens, embed_dim=embed_dim, epochs=fasttext_epochs)
    logger.info("FastText trained. Vocab size: %d", len(ft_model.wv))

    logger.info("Building vocab (max_vocab=%d) from training text only…", max_vocab)
    vocab = build_vocab(train_tokens, max_vocab=max_vocab)
    logger.info("Vocab built: %d tokens", len(vocab))

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="feature_engineering") as run:
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("max_vocab", max_vocab)
        mlflow.log_param("fasttext_epochs", fasttext_epochs)
        mlflow.log_param("fasttext_vocab_size", len(ft_model.wv))
        mlflow.log_param("built_vocab_size", len(vocab))
        mlflow.log_param("train_docs", len(train_texts))
        mlflow.log_param("test_docs", len(test_texts))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save FastText model
            ft_path = os.path.join(tmpdir, "fasttext.model")
            ft_model.save(ft_path)
            mlflow.log_artifacts(tmpdir, artifact_path="fasttext_model")

            # Save vocab as JSON (compact, human-readable)
            vocab_path = os.path.join(tmpdir, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(vocab, f)
            mlflow.log_artifact(vocab_path)

        feature_run_id = run.info.run_id
        logger.info("Feature engineering run ID: %s", feature_run_id)

    return feature_run_id


def load_feature_artifacts(feature_run_id: str) -> tuple:
    """Download FastText model and vocab from an MLflow feature run.

    Args:
        feature_run_id: MLflow run ID from ``run_feature_engineering``.

    Returns:
        Tuple of (ft_model, vocab) — gensim FastText and token→index dict.
    """
    from gensim.models import FastText

    client = mlflow.tracking.MlflowClient()

    ft_dir = client.download_artifacts(feature_run_id, "fasttext_model")
    ft_model_file = os.path.join(ft_dir, "fasttext.model")
    ft_model = FastText.load(ft_model_file)
    logger.info("Loaded FastText model from run %s", feature_run_id)

    vocab_path = client.download_artifacts(feature_run_id, "vocab.json")
    with open(vocab_path) as f:
        vocab: dict[str, int] = json.load(f)
    logger.info("Loaded vocab (%d tokens) from run %s", len(vocab), feature_run_id)

    return ft_model, vocab
