# features_tfidf.py
from __future__ import annotations
import pickle
from typing import Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeaturizer:
    def __init__(
        self,
        ngram_max: int = 2,
        max_features: int = 30000,
        min_df: int = 2,
        max_df: float = 0.9,
        sublinear_tf: bool = True,
        lowercase: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, ngram_max),
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            lowercase=lowercase,
            dtype=np.float32,
        )

    @property
    def vocab_size(self) -> int:
        return 0 if self.vectorizer.vocabulary_ is None else len(self.vectorizer.vocabulary_)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        return X.toarray().astype(np.float32)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return X.toarray().astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @staticmethod
    def load(path: str) -> "TfidfFeaturizer":
        with open(path, "rb") as f:
            vec = pickle.load(f)
        obj = TfidfFeaturizer()
        obj.vectorizer = vec
        return obj
