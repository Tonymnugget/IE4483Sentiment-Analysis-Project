# data_gemma.py
from __future__ import annotations
import json, math, random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

def load_train_json(path: str) -> Tuple[List[str], List[int]]:
    """
    Supports:
      - List of rows: [{"reviews": "...", "sentiments": 1}, ...]
      - Dict of arrays: {"reviews":[...], "sentiments":[...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        texts = [row["reviews"] for row in obj]
        labels = [int(row["sentiments"]) for row in obj]
    elif isinstance(obj, dict):
        texts, labels = obj["reviews"], obj["sentiments"]
    else:
        raise ValueError("Unsupported train.json structure")
    assert len(texts) == len(labels)
    return texts, labels

def load_test_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [row["reviews"] for row in obj]
    elif isinstance(obj, dict):
        return obj["reviews"]
    raise ValueError("Unsupported test.json structure")

def basic_clean(s: str) -> str:
    return " ".join(s.lower().split())

def clean_corpus(texts: List[str]) -> List[str]:
    return [basic_clean(t) for t in texts]

def train_val_split(
    texts: List[str], labels: List[int], val_size: float = 0.15, seed: int = 42
) -> Tuple[List[str], List[str], List[int], List[int]]:
    x_tr, x_va, y_tr, y_va = train_test_split(
        texts, labels, test_size=val_size, random_state=seed, stratify=labels
    )
    return x_tr, x_va, y_tr, y_va

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        self.texts = texts
        self.labels = labels
        self.has_labels = labels is not None

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.has_labels:
            item["labels"] = int(self.labels[idx])
        return item

def make_loader(
    texts: List[str],
    labels: Optional[List[int]],
    tokenizer,
    max_len: int = 256,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
):
    ds = TextDataset(texts, labels)
    def collate(batch):
        texts_b = [b["text"] for b in batch]
        enc = tokenizer(
            texts_b, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate, pin_memory=True)
