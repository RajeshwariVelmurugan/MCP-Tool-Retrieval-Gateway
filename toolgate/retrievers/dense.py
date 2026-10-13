"""Dense retrieval with pluggable embedders.

Embedder specs (set TOOLGATE_EMBEDDER):
  hash                         offline, deterministic char-n-gram TF-IDF hashing. NOT semantic: it exists so the whole
                               pipeline runs with zero downloads. Use a real model for real results.
  st:<model>                   sentence-transformers, e.g. st:BAAI/bge-small-en-v1.5   (pip install sentence-transformers)
  fastembed:<model>            ONNX runtime, lighter than torch, e.g. fastembed:BAAI/bge-small-en-v1.5  (pip install fastembed)
"""
from __future__ import annotations

import hashlib
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from ..text import tokenize


class Embedder(Protocol):
    spec: str

    def fit(self, docs: list[str]) -> None: ...
    def encode_docs(self, docs: list[str]) -> np.ndarray: ...
    def encode_query(self, query: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class HashEmbedder:
    spec = "hash"

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.idf: dict[int, float] = {}
        self.default_idf = 1.0

    def _features(self, text: str) -> Counter:
        toks = tokenize(text)
        feats: Counter = Counter()
        for t in toks:
            feats[("w", t)] += 1
        for a, b in zip(toks, toks[1:]):
            feats[("b", f"{a} {b}")] += 1
        for t in toks:
            padded = f"#{t}#"
            for n in (3, 4, 5):
                for i in range(len(padded) - n + 1):
                    feats[("c", padded[i : i + n])] += 1
        return feats

    @staticmethod
    def _bucket(feat, dim: int) -> int:
        return zlib.crc32(f"{feat[0]}:{feat[1]}".encode()) % dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, c in self._features(text).items():
            b = self._bucket(feat, self.dim)
            v[b] += (1 + math.log(c)) * self.idf.get(b, self.default_idf)
        return v

    def fit(self, docs: list[str]) -> None:
        df: Counter = Counter()
        for d in docs:
            df.update({self._bucket(f, self.dim) for f in self._features(d)})
        n = max(len(docs), 1)
        self.idf = {b: math.log(1 + n / c) for b, c in df.items()}
        self.default_idf = math.log(1 + n)

    def encode_docs(self, docs: list[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(d) for d in docs]))

    def encode_query(self, query: str) -> np.ndarray:
        return _l2(self._vec(query))


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # lazy import: optional dependency

        self.spec = f"st:{model}"
        self.model = SentenceTransformer(model)
        # BGE-style query instruction improves retrieval for bge-*-en models
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model and "m3" not in model else ""

    def fit(self, docs):  # no-op
        pass

    def encode_docs(self, docs):
        return np.asarray(self.model.encode(docs, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def encode_query(self, query):
        return np.asarray(self.model.encode([self.query_prefix + query], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)


class FastEmbedEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # lazy import: optional dependency

        self.spec = f"fastembed:{model}"
        self.model = TextEmbedding(model_name=model)

    def fit(self, docs):
        pass

    def encode_docs(self, docs):
        return _l2(np.array(list(self.model.embed(docs)), dtype=np.float32))

    def encode_query(self, query):
        return _l2(np.array(list(self.model.query_embed(query)), dtype=np.float32))[0]


def build_embedder(spec: str) -> Embedder:
    if spec == "hash":
        return HashEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec.startswith("fastembed:"):
        return FastEmbedEmbedder(spec[10:])
    raise ValueError(f"unknown embedder spec: {spec!r} (use hash | st:<model> | fastembed:<model>)")


class DenseRetriever:
    name = "dense"

    def __init__(self, docs: list[str], embedder: Embedder, cache_dir: Path | None = None):
        self.embedder = embedder
        embedder.fit(docs)
        self.matrix = self._load_or_encode(docs, cache_dir)

    def _load_or_encode(self, docs, cache_dir):
        if cache_dir is None or self.embedder.spec == "hash":  # hash embedder is fast; no cache needed
            return self.embedder.encode_docs(docs)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1((self.embedder.spec + "\n" + "\n".join(docs)).encode()).hexdigest()[:16]
        f = cache_dir / f"emb_{key}.npy"
        if f.exists():
            return np.load(f)
        m = self.embedder.encode_docs(docs)
        np.save(f, m)
        return m

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.encode_query(query)

    def search(self, query: str, k: int):
        s = self.scores(query)
        idx = np.argsort(-s, kind="stable")[: min(k, len(s))]
        return [(int(i), float(s[i])) for i in idx]
