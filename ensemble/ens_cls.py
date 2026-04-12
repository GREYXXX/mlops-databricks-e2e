"""
Ensemble Text Classifier — 20 Newsgroups
=========================================
Base models:
  1. Logistic Regression  + TF-IDF
  2. Random Forest        + TF-IDF
  3. TextCNN              + FastText embeddings (trained on corpus)

Ensemble: soft voting (average of softmax probability vectors)
"""

import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import Tfi
dfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import FastText
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=False)


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_data():
    """
    Load 20 Newsgroups, stripping metadata so the task is genuinely hard.
    Returns raw text lists and integer labels.
    """
    remove = ("headers", "footers", "quotes")
    train = fetch_20newsgroups(subset="train", remove=remove, random_state=SEED)
    test  = fetch_20newsgroups(subset="test",  remove=remove, random_state=SEED)

    print(f"Train: {len(train.data)} samples | Test: {len(test.data)} samples")
    print(f"Classes: {len(train.target_names)}")
    return train.data, test.data, train.target, test.target, train.target_names


# =============================================================================
# 2.  TEXT PREPROCESSING
# =============================================================================

def basic_tokenize(text):
    """Lowercase and split on non-alpha chars. Returns list of tokens."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def preprocess_corpus(texts):
    """Tokenize all texts. Used for FastText training."""
    return [basic_tokenize(t) for t in texts]


# =============================================================================
# 3.  MODEL 1 — Logistic Regression + TF-IDF
# =============================================================================

def train_lr_tfidf(train_texts, train_labels):
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # log-scale TF — helps with long documents
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)

    model = LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="saga",        # fast for large sparse matrices
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(X_train, train_labels)
    print("LR+TF-IDF trained.")
    return vectorizer, model


def predict_proba_lr(vectorizer, model, texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)   # shape: (n_samples, n_classes)


# =============================================================================
# 4.  MODEL 2 — Random Forest + TF-IDF
# =============================================================================

def train_rf_tfidf(train_texts, train_labels):
    # Re-use a simpler TF-IDF for RF — fewer features keeps it tractable on CPU
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=3,
    )
    X_train = vectorizer.fit_transform(train_texts)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(X_train, train_labels)
    print("RF+TF-IDF trained.")
    return vectorizer, model


def predict_proba_rf(vectorizer, model, texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)


# =============================================================================
# 5.  MODEL 3 — TextCNN + FastText embeddings
# =============================================================================

# ── 5a. Train FastText on the corpus ─────────────────────────────────────────

def train_fasttext(tokenized_corpus, embed_dim=100):
    """
    Train FastText on the raw training corpus (unsupervised).
    FastText learns subword embeddings — good for handling OOV tokens.
    """
    ft_model = FastText(
        sentences=tokenized_corpus,
        vector_size=embed_dim,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
        sg=1,
        # min_n=3,      # minimum char n-gram size (default is 3, so actually already on)
        # max_n=6,      # maximum char n-gram size (default is 6)
        seed=SEED,
    )
    print(f"FastText trained. Vocab size: {len(ft_model.wv)}")
    return ft_model


def build_embedding_matrix(ft_model, vocab, embed_dim):
    """
    Build a (vocab_size, embed_dim) matrix aligned with our token→index vocab.
    Row 0 is <PAD>; token indices are 1..len(vocab) (see build_vocab).
    Unknown tokens get a zero vector.
    """
    vocab_size = len(vocab) + 1  # match TextCNN: index 0 = PAD, 1..len(vocab) = tokens
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for token, idx in vocab.items():
        if token in ft_model.wv:
            matrix[idx] = ft_model.wv[token]
    return matrix


# ── 5b. Vocabulary builder ───────────────────────────────────────────────────

def build_vocab(tokenized_corpus, max_vocab=30_000):
    """
    Count token frequencies and keep the top-max_vocab tokens.
    Index 0 = <PAD>.
    """
    from collections import Counter
    counter = Counter(token for doc in tokenized_corpus for token in doc)
    most_common = counter.most_common(max_vocab - 1)   # leave 0 for PAD
    vocab = {token: idx + 1 for idx, (token, _) in enumerate(most_common)}
    return vocab


# ── 5c. PyTorch Dataset ──────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab, max_len=200):
        self.data   = tokenized_texts
        self.labels = labels
        self.vocab  = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx][: self.max_len]
        ids = [self.vocab.get(t, 0) for t in tokens]   # 0 = <PAD> / unknown
        # Pad to max_len
        ids += [0] * (self.max_len - len(ids))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── 5d. TextCNN as a Lightning Module ────────────────────────────────────────

class TextCNN(pl.LightningModule):
    """
    Classic TextCNN (Kim 2014) wrapped in a Lightning module.

    Lightning handles: device placement, train loop, logging, val loop.
    We only define: model layers, forward, loss, optimizer.
    """
    def __init__(self, vocab_size, embed_dim, num_classes,
                 embedding_matrix=None,
                 kernel_sizes=(2, 3, 4, 5),
                 num_filters=128,
                 dropout=0.5,
                 lr=1e-3):
        super().__init__()
        self.lr = lr

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = True   # fine-tune allowed

        # One Conv1d per kernel size — captures different n-gram windows
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x).permute(0, 2, 1)   # → (batch, embed_dim, seq_len)

        # Conv → ReLU → max-over-time for each kernel size
        pooled = [torch.relu(conv(embedded)).max(dim=2).values for conv in self.convs]

        out = torch.cat(pooled, dim=1)   # (batch, num_filters * n_kernels)
        return self.fc(self.dropout(out))

    def training_step(self, batch, batch_idx):
        ids, labels = batch
        loss = self.loss_fn(self(ids), labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, labels = batch
        logits = self(ids)
        loss   = self.loss_fn(logits, labels)
        acc    = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ── 5e. Train + predict with Lightning Trainer ───────────────────────────────

def train_textcnn(train_tokens, train_labels, test_tokens, test_labels,
                  vocab, embedding_matrix, num_classes, embed_dim,
                  epochs=10, batch_size=64, lr=1e-3):

    train_ds = TextDataset(train_tokens, train_labels, vocab)
    test_ds  = TextDataset(test_tokens,  test_labels,  vocab)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    model = TextCNN(
        vocab_size=len(vocab) + 1,   # +1 for PAD index 0
        embed_dim=embed_dim,
        num_classes=num_classes,
        embedding_matrix=embedding_matrix,
        lr=lr,
    )

    # Trainer auto-selects GPU if available, falls back to CPU otherwise
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",    # "gpu" when available, "cpu" otherwise
        devices="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,          # no TensorBoard clutter
    )

    trainer.fit(model, train_dl, test_dl)
    print("TextCNN trained.")
    return model, test_dl


def predict_proba_cnn(model, test_dl):
    """Run inference and return softmax probabilities — Lightning handles device."""
    trainer = Trainer(accelerator="auto", devices="auto",
                      logger=False, enable_progress_bar=False)
    batches = trainer.predict(model, test_dl)
    return np.vstack(batches)   # (n_samples, n_classes)


# Needed by Lightning's predict_step
TextCNN.predict_step = lambda self, batch, _: (
    torch.softmax(self(batch[0]), dim=1).cpu().numpy()
)


# =============================================================================
# 6.  ENSEMBLE — Soft Voting
# =============================================================================

def ensemble_predict(proba_list):
    """
    Average the probability vectors from each model, then argmax.
    proba_list: list of arrays, each (n_samples, n_classes)
    """
    avg_proba = np.mean(proba_list, axis=0)   # (n_samples, n_classes)
    return avg_proba.argmax(axis=1)


# =============================================================================
# 7.  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Ensemble Text Classifier — 20 Newsgroups")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    train_texts, test_texts, train_labels, test_labels, class_names = load_data()
    num_classes = len(class_names)

    # ── Train LR + TF-IDF ───────────────────────────────────────────────────
    print("\n[2/6] Training Logistic Regression + TF-IDF...")
    lr_vec, lr_model = train_lr_tfidf(train_texts, train_labels)
    lr_proba_test    = predict_proba_lr(lr_vec, lr_model, test_texts)
    lr_acc = accuracy_score(test_labels, lr_proba_test.argmax(axis=1))
    print(f"  LR accuracy: {lr_acc:.4f}")

    # ── Train RF + TF-IDF ───────────────────────────────────────────────────
    print("\n[3/6] Training Random Forest + TF-IDF...")
    rf_vec, rf_model = train_rf_tfidf(train_texts, train_labels)
    rf_proba_test    = predict_proba_rf(rf_vec, rf_model, test_texts)
    rf_acc = accuracy_score(test_labels, rf_proba_test.argmax(axis=1))
    print(f"  RF accuracy: {rf_acc:.4f}")

    # ── FastText embeddings ──────────────────────────────────────────────────
    print("\n[4/6] Training FastText embeddings on corpus...")
    EMBED_DIM       = 100
    train_tokens    = preprocess_corpus(train_texts)
    test_tokens     = preprocess_corpus(test_texts)
    all_tokens      = train_tokens + test_tokens   # unsupervised — use all text
    ft_model        = train_fasttext(all_tokens, embed_dim=EMBED_DIM)

    vocab           = build_vocab(train_tokens)
    embed_matrix    = build_embedding_matrix(ft_model, vocab, EMBED_DIM)

    # ── Train TextCNN ────────────────────────────────────────────────────────
    print("\n[5/6] Training TextCNN...")
    cnn_model, test_dl = train_textcnn(
        train_tokens, train_labels, test_tokens, test_labels,
        vocab, embed_matrix, num_classes, EMBED_DIM,
        epochs=10,
    )
    cnn_proba_test = predict_proba_cnn(cnn_model, test_dl)
    cnn_acc = accuracy_score(test_labels, cnn_proba_test.argmax(axis=1))
    print(f"  TextCNN accuracy: {cnn_acc:.4f}")

    # ── Ensemble ─────────────────────────────────────────────────────────────
    print("\n[6/6] Ensemble (soft voting)...")
    ensemble_preds = ensemble_predict([lr_proba_test, rf_proba_test, cnn_proba_test])
    ens_acc = accuracy_score(test_labels, ensemble_preds)

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Logistic Regression : {lr_acc:.4f}")
    print(f"  Random Forest       : {rf_acc:.4f}")
    print(f"  TextCNN + FastText  : {cnn_acc:.4f}")
    print(f"  Ensemble (avg)      : {ens_acc:.4f}")
    print("=" * 60)

    print("\nEnsemble — Detailed Classification Report:")
    print(classification_report(test_labels, ensemble_preds, target_names=class_names))


if __name__ == "__main__":
    main()
