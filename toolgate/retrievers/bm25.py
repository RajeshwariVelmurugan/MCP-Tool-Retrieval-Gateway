"""Okapi BM25, implemented from scratch (numpy only) so the project has no extra dependency."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..text import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        # BM25+ style idf (always positive)
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.n, dtype=np.float32)
        if not q:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(q):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.tf], dtype=np.float32)
            out += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return out

    def search(self, query: str, k: int):
        s = self.scores(query)
        k = min(k, self.n)
        idx = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in idx]
