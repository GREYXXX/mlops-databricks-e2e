"""Configuration constants for the 20 Newsgroups ensemble classifier pipeline."""

from __future__ import annotations

# Unity Catalog defaults
DEFAULT_CATALOG = "workspace"
DEFAULT_SCHEMA = "mlops_e2e"
DEFAULT_MODEL_NAME = "newsgroups_ensemble_model"

# Table names
RAW_TABLE_NAME = "newsgroups_raw"

# MLflow experiment
EXPERIMENT_NAME = "/mlops_e2e_newsgroups_ensemble"

# Model hyperparameters
EMBED_DIM = 300
MAX_VOCAB = 30_000
MAX_SEQ_LEN = 200
TEXTCNN_EPOCHS = 30
TEXTCNN_BATCH_SIZE = 64
FASTTEXT_EPOCHS = 30
SEED = 42
