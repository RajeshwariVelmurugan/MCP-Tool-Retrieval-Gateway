"""Retrieval metrics. `gold_mode`:
  all : every gold tool is required (multi-tool tasks)  -> recall = |found| / |gold|, full_hit = all found
  any : one of the listed tools is enough (vague queries) -> recall = 1 if any found, full_hit = same
"""
from __future__ import annotations

import math
import statistics


def _ranks(ranked_ids: list[str], gold: list[str]) -> dict[str, int | None]:
    pos = {tid: r for r, tid in enumerate(ranked_ids, start=1)}
    return {g: pos.get(g) for g in gold}


def recall_at_k(ranked_ids, gold, k, mode="all") -> float:
    found = [g for g, r in _ranks(ranked_ids, gold).items() if r is not None and r <= k]
    if mode == "any":
        return 1.0 if found else 0.0
    return len(found) / len(gold)


def full_hit_at_k(ranked_ids, gold, k, mode="all") -> float:
    """1.0 if the agent would have everything it needs in the top-k."""
    found = [g for g, r in _ranks(ranked_ids, gold).items() if r is not None and r <= k]
    return 1.0 if (found if mode == "any" else len(found) == len(gold)) else 0.0


def mrr(ranked_ids, gold, mode="all") -> float:
    ranks = [r for r in _ranks(ranked_ids, gold).values() if r is not None]
    if not ranks:
        return 0.0
    if mode == "any":
        return 1.0 / min(ranks)
    # "all": reciprocal rank of the *last* required tool (you need all of them)
    if len(ranks) < len(gold):
        return 0.0
    return 1.0 / max(ranks)


def ndcg_at_k(ranked_ids, gold, k, mode="all") -> float:
    gs = set(gold)
    dcg = sum(1.0 / math.log2(r + 1) for r, tid in enumerate(ranked_ids[:k], start=1) if tid in gs)
    n_ideal = 1 if mode == "any" else min(len(gs), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_ideal + 1))
    return min(dcg / idcg, 1.0) if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0
