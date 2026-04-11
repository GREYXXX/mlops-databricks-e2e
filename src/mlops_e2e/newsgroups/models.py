"""Model components: tokenization, FastText, vocab, TextDataset, TextCNN.

All logic is pure Python / PyTorch — no Spark or MLflow dependencies.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from gensim.models import FastText
from torch.utils.data import Dataset

from mlops_e2e.newsgroups.config import EMBED_DIM, FASTTEXT_EPOCHS, MAX_VOCAB, SEED

if TYPE_CHECKING:
    pass


# =============================================================================
# Tokenization
# =============================================================================

def basic_tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alpha chars. Returns list of tokens."""
    return re.findall(r"[a-z]+", text.lower())


def preprocess_corpus(texts: list[str]) -> list[list[str]]:
    """Tokenize all texts. Used for FastText training and dataset building."""
    return [basic_tokenize(t) for t in texts]


# =============================================================================
# Vocabulary
# =============================================================================

def build_vocab(tokenized_corpus: list[list[str]], max_vocab: int = MAX_VOCAB) -> dict[str, int]:
    """Count token frequencies and keep the top ``max_vocab`` tokens.

    Index 0 is reserved for <PAD>. Tokens are assigned indices 1..max_vocab-1.

    Args:
        tokenized_corpus: List of tokenized documents.
        max_vocab: Maximum vocabulary size (including PAD).

    Returns:
        Dict mapping token → integer index (1-based).
    """
    counter = Counter(token for doc in tokenized_corpus for token in doc)
    most_common = counter.most_common(max_vocab - 1)
    return {token: idx + 1 for idx, (token, _) in enumerate(most_common)}


def build_embedding_matrix(
    ft_model: FastText,
    vocab: dict[str, int],
    embed_dim: int = EMBED_DIM,
) -> np.ndarray:
    """Build a (vocab_size, embed_dim) matrix aligned with the token→index vocab.

    Row 0 is the <PAD> zero vector. Token indices are 1..len(vocab).

    Args:
        ft_model: Trained gensim FastText model.
        vocab: Token→index mapping from ``build_vocab``.
        embed_dim: Embedding dimensionality.

    Returns:
        Float32 numpy array of shape (len(vocab)+1, embed_dim).
    """
    vocab_size = len(vocab) + 1  # index 0 = PAD, 1..len(vocab) = tokens
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for token, idx in vocab.items():
        if token in ft_model.wv:
            matrix[idx] = ft_model.wv[token]
    return matrix


# =============================================================================
# FastText training
# =============================================================================

def train_fasttext(
    tokenized_corpus: list[list[str]],
    embed_dim: int = EMBED_DIM,
    epochs: int = FASTTEXT_EPOCHS,
) -> FastText:
    """Train a FastText model on the tokenized corpus (unsupervised).

    Args:
        tokenized_corpus: All tokenized documents (train + test for unsupervised).
        embed_dim: Embedding vector size.
        epochs: Number of training epochs.

    Returns:
        Trained gensim FastText model.
    """
    ft_model = FastText(
        sentences=tokenized_corpus,
        vector_size=embed_dim,
        window=5,
        min_count=2,
        workers=4,
        epochs=epochs,
        sg=1,
        seed=SEED,
    )
    return ft_model


# =============================================================================
# PyTorch Dataset
# =============================================================================

class TextDataset(Dataset):
    """Tokenized text dataset for PyTorch DataLoader.

    Args:
        tokenized_texts: List of token lists.
        labels: Integer class labels.
        vocab: Token→index mapping.
        max_len: Maximum sequence length (truncate/pad to this length).
    """

    def __init__(
        self,
        tokenized_texts: list[list[str]],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int = 200,
    ) -> None:
        self.data = tokenized_texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[idx][: self.max_len]
        ids = [self.vocab.get(t, 0) for t in tokens]
        ids += [0] * (self.max_len - len(ids))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# =============================================================================
# TextCNN (Kim 2014)
# =============================================================================

class TextCNN(pl.LightningModule):
    """Classic TextCNN wrapped as a PyTorch Lightning module.

    Args:
        vocab_size: Vocabulary size (including PAD at index 0).
        embed_dim: Embedding vector size.
        num_classes: Number of output classes.
        embedding_matrix: Pre-trained embedding weights (optional).
        kernel_sizes: Convolution kernel sizes for different n-gram windows.
        num_filters: Number of convolutional filters per kernel size.
        dropout: Dropout rate applied before the final linear layer.
        lr: Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        embedding_matrix: np.ndarray | None = None,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        num_filters: int = 128,
        dropout: float = 0.5,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = lr

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        pooled = [torch.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        out = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(out))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        ids, labels = batch
        loss = self.loss_fn(self(ids), labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        ids, labels = batch
        logits = self(ids)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def predict_step(self, batch: tuple, batch_idx: int) -> np.ndarray:
        return torch.softmax(self(batch[0]), dim=1).cpu().numpy()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
