from __future__ import annotations


def rrf_fuse(rankings: list[list[tuple[int, float]]], weights: list[float] | None = None, k: int = 60):
    """Reciprocal Rank Fusion. Score-scale free, which is why it is the default way to mix BM25 and embeddings."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])
