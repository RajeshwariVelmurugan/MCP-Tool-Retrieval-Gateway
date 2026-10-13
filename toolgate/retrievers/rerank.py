"""Second-stage rerankers.

Specs (set TOOLGATE_RERANKER):
  none                  skip reranking
  heuristic             offline, cheap feature-based reranker (query-term coverage, name match, phrase match).
                        A stand-in so the pipeline runs without downloads, NOT a learned model.
  st:<model>            sentence-transformers CrossEncoder, e.g. st:BAAI/bge-reranker-base
  fastembed:<model>     fastembed TextCrossEncoder, e.g. fastembed:Xenova/ms-marco-MiniLM-L-6-v2
"""
from __future__ import annotations

from typing import Protocol

from ..text import split_identifier, tokenize


class Reranker(Protocol):
    spec: str

    def score(self, query: str, docs: list[str], names: list[str]) -> list[float]: ...


class HeuristicReranker:
    spec = "heuristic"

    def score(self, query: str, docs: list[str], names: list[str]) -> list[float]:
        q = tokenize(query)
        if not q:
            return [0.0] * len(docs)
        qset = set(q)
        qbi = set(zip(q, q[1:]))
        out = []
        for doc, name in zip(docs, names):
            d = tokenize(doc)
            dset = set(d)
            n = set(tokenize(split_identifier(name)))
            coverage = len(qset & dset) / len(qset)           # how much of the query the doc explains
            name_cov = len(qset & n) / max(len(n), 1)         # query hits the tool *name*
            bigrams = len(qbi & set(zip(d, d[1:]))) / max(len(qbi), 1)
            out.append(1.0 * coverage + 1.5 * name_cov + 0.7 * bigrams)
        return out


class SentenceTransformerCrossEncoder:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder  # lazy import

        self.spec = f"st:{model}"
        self.model = CrossEncoder(model)

    def score(self, query, docs, names):
        return [float(s) for s in self.model.predict([(query, d) for d in docs], show_progress_bar=False)]


class FastEmbedCrossEncoder:
    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy import

        self.spec = f"fastembed:{model}"
        self.model = TextCrossEncoder(model_name=model)

    def score(self, query, docs, names):
        return [float(s) for s in self.model.rerank(query, docs)]


def build_reranker(spec: str) -> Reranker | None:
    if spec in ("none", "", None):
        return None
    if spec == "heuristic":
        return HeuristicReranker()
    if spec.startswith("st:"):
        return SentenceTransformerCrossEncoder(spec[3:])
    if spec.startswith("fastembed:"):
        return FastEmbedCrossEncoder(spec[10:])
    raise ValueError(f"unknown reranker spec: {spec!r}")
