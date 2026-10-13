"""The retrieval engine: builds indexes over a tool corpus and serves every retrieval method by name."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .corpus import Tool, load_tools
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever, build_embedder
from .retrievers.hybrid import rrf_fuse
from .retrievers.rerank import build_reranker
from .text import tool_tokens

METHOD_LABELS = {
    "all_tools": "All tools in context (baseline)",
    "bm25": "BM25",
    "dense": "Embedding only",
    "hybrid": "Hybrid (BM25 + embedding, RRF)",
    "dense_rerank": "Embedding + reranker",
    "hybrid_rerank": "Hybrid + reranker",
}
RERANK_METHODS = {"dense_rerank", "hybrid_rerank"}


@dataclass
class SearchResult:
    method: str
    query: str
    k: int
    hits: list[dict]
    latency_ms: float
    tokens_returned: int
    tokens_all: int

    def to_dict(self) -> dict:
        return {
            "method": self.method, "label": METHOD_LABELS.get(self.method, self.method), "query": self.query,
            "k": self.k, "hits": self.hits, "latency_ms": round(self.latency_ms, 3),
            "tokens_returned": self.tokens_returned, "tokens_all": self.tokens_all,
            "tokens_saved_pct": round(100 * (1 - self.tokens_returned / max(self.tokens_all, 1)), 1),
        }


class Engine:
    def __init__(self, tools: list[Tool], settings: Settings | None = None, *, doc_mode: str = "full",
                 embedder: str | None = None, reranker: str | None = None, rerank_depth: int | None = None,
                 use_cache: bool = True):
        self.settings = settings or get_settings()
        self.tools = tools
        self.doc_mode = doc_mode
        self.rerank_depth = rerank_depth or self.settings.rerank_depth
        self.docs = [t.doc(doc_mode) for t in tools]
        self.names = [t.name for t in tools]
        self.tok = [tool_tokens(t.api_definition()) for t in tools]
        self.tokens_all = sum(self.tok)
        self.id_to_idx = {t.id: i for i, t in enumerate(tools)}

        self.bm25 = BM25Retriever(self.docs)
        self.embedder_spec = embedder or self.settings.embedder
        self.dense = DenseRetriever(self.docs, build_embedder(self.embedder_spec),
                                    self.settings.cache_dir if use_cache else None)
        self.reranker_spec = reranker if reranker is not None else self.settings.reranker
        self.reranker = build_reranker(self.reranker_spec)

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_path(cls, path: Path | None = None, **kw) -> "Engine":
        s = kw.pop("settings", None) or get_settings()
        return cls(load_tools(path or s.corpus_path), s, **kw)

    def subset(self, tool_ids: list[str]) -> "Engine":
        """Engine over a subset of the pool (used for pool-size scaling experiments)."""
        keep = [self.tools[self.id_to_idx[i]] for i in tool_ids]
        return Engine(keep, self.settings, doc_mode=self.doc_mode, embedder=self.embedder_spec,
                      reranker=self.reranker_spec, rerank_depth=self.rerank_depth, use_cache=False)

    # -- API ------------------------------------------------------------------
    def methods(self) -> list[str]:
        m = ["all_tools", "bm25", "dense", "hybrid"]
        if self.reranker is not None:
            m += ["dense_rerank", "hybrid_rerank"]
        return m

    def info(self) -> dict:
        return {"tools": len(self.tools), "servers": len({t.server for t in self.tools}),
                "embedder": self.embedder_spec, "reranker": self.reranker_spec if self.reranker else "none",
                "rerank_depth": self.rerank_depth, "doc_mode": self.doc_mode, "tokens_all_tools": self.tokens_all,
                "methods": self.methods(), "method_labels": METHOD_LABELS}

    def _ranked(self, query: str, method: str, k: int) -> list[tuple[int, float]]:
        if method == "all_tools":
            return [(i, 0.0) for i in range(len(self.tools))]
        depth = max(k, self.rerank_depth) if method in RERANK_METHODS else k
        depth = min(depth, len(self.tools))
        if method == "bm25":
            return self.bm25.search(query, k)
        if method == "dense":
            return self.dense.search(query, k)
        if method in ("hybrid", "hybrid_rerank"):
            first = rrf_fuse([self.bm25.search(query, depth * 2), self.dense.search(query, depth * 2)])[:depth]
        elif method == "dense_rerank":
            first = self.dense.search(query, depth)
        else:
            raise ValueError(f"unknown method {method!r}; available: {self.methods()}")
        if method == "hybrid":
            return first[:k]
        if self.reranker is None:
            raise ValueError("this method needs a reranker; set TOOLGATE_RERANKER")
        idxs = [i for i, _ in first]
        scores = self.reranker.score(query, [self.docs[i] for i in idxs], [self.names[i] for i in idxs])
        order = sorted(zip(idxs, scores), key=lambda x: -x[1])
        return order[:k]

    def search(self, query: str, method: str = "hybrid_rerank", k: int = 5) -> SearchResult:
        t0 = time.perf_counter()
        ranked = self._ranked(query, method, k)
        latency = (time.perf_counter() - t0) * 1000
        top = ranked if method == "all_tools" else ranked[:k]
        hits = []
        for rank, (i, score) in enumerate(top, start=1):
            t = self.tools[i]
            hits.append({"rank": rank, "id": t.id, "name": t.name, "server": t.server,
                         "score": round(float(score), 4), "description": t.description})
        tokens_returned = sum(self.tok[self.id_to_idx[h["id"]]] for h in hits)
        return SearchResult(method, query, k, hits, latency, tokens_returned, self.tokens_all)

    def definitions(self, tool_ids: list[str]) -> list[dict]:
        return [self.tools[self.id_to_idx[i]].api_definition() for i in tool_ids]
