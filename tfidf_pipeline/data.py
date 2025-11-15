# data.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# JSON loaders (support both schemas)
# -----------------------------
def load_train_json(path: str) -> Tuple[List[str], List[int]]:
    """
    Supports:
      1) List-of-rows:
         [{"reviews": "...", "sentiments": 1}, ...]
      2) Dict-of-arrays:
         {"reviews": [...], "sentiments": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Case 1: list of dict rows
    if isinstance(obj, list):
        reviews = [row["reviews"] for row in obj]
        labels = [int(row["sentiments"]) for row in obj]
    # Case 2: dict of arrays
    elif isinstance(obj, dict):
        reviews = obj["reviews"]
        labels = obj["sentiments"]
    else:
        raise ValueError(f"Unsupported train.json structure: {type(obj)}")

    assert len(reviews) == len(labels), "reviews/labels length mismatch"
    return reviews, labels

def load_test_json(path: str) -> List[str]:
    """
    Supports:
      1) List-of-rows: [{"reviews": "..."}, ...]
      2) Dict-of-arrays: {"reviews": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        reviews = [row["reviews"] for row in obj]
    elif isinstance(obj, dict):
        reviews = obj["reviews"]
    else:
        raise ValueError(f"Unsupported test.json structure: {type(obj)}")
    return reviews

# -----------------------------
# Simple cleaner (optional)
# -----------------------------
def basic_clean(s: str) -> str:
    return " ".join(s.lower().split())

def clean_corpus(texts: List[str]) -> List[str]:
    return [basic_clean(t) for t in texts]

# -----------------------------
# Splitting
# -----------------------------
def train_val_split(
    texts: List[str],
    labels: List[int],
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    x_tr, x_va, y_tr, y_va = train_test_split(
        texts, labels, test_size=val_size, random_state=seed, stratify=labels
    )
    return x_tr, x_va, y_tr, y_va

# -----------------------------
# Dataset for TF-IDF features
# -----------------------------
class TfidfDataset(Dataset):
    """
    Expects already-transformed feature matrix (scipy.sparse or np.ndarray)
    and integer labels (optional for test).
    """
    def __init__(self, X, y: Optional[List[int]] = None):
        self.X = X
        self.y = y
        self.has_labels = y is not None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"features_index": idx}  # index to fetch row lazily in collate
        if self.has_labels:
            item["labels"] = int(self.y[idx])
        return item

# -----------------------------
# Collate for sparse → dense (batch-wise)
# -----------------------------
def tfidf_collate_dense(batch, X_sparse):
    idxs = [b["features_index"] for b in batch]
    Xb = X_sparse[idxs]
    if hasattr(Xb, "toarray"):
        Xb = Xb.toarray()
    Xb = torch.from_numpy(np.asarray(Xb, dtype=np.float32))
    out = {"features": Xb}
    if "labels" in batch[0]:
        yb = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        out["labels"] = yb
    return out

def make_loader(X, y=None, batch_size=128, shuffle=False, num_workers=0):
    ds = TfidfDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: tfidf_collate_dense(b, X),
        pin_memory=True,
    )
