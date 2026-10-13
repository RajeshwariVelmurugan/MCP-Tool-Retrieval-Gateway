"""Offline retrieval evaluation: metrics, latency, token savings, failure analysis, ablations."""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..corpus import Query
from ..engine import METHOD_LABELS, Engine
from ..text import tokenize
from . import metrics as M

DEFAULT_KS = (1, 3, 5, 10)


def _filter(queries: list[Query], split: str) -> list[Query]:
    return queries if split == "all" else [q for q in queries if q.split == split]


def _bucket(engine: Engine, q: Query, ranked_ids: list[str], k: int) -> tuple[str, dict]:
    """Heuristic failure taxonomy. Treat it as a starting point and read the actual misses by hand."""
    ranks = {g: (ranked_ids.index(g) + 1 if g in ranked_ids else None) for g in q.gold}
    found = [g for g, r in ranks.items() if r is not None and r <= k]
    missing = [g for g in q.gold if g not in found] if q.gold_mode == "all" else list(q.gold)
    if q.gold_mode == "all" and len(q.gold) > 1 and found:
        return "multi_tool_partial", ranks
    qtok = set(tokenize(q.query))
    gold_docs = [set(tokenize(engine.tools[engine.id_to_idx[g]].doc("full"))) for g in missing]
    if all(not (qtok & d) for d in gold_docs):
        return "vocab_mismatch", ranks
    gold_names = [set(tokenize(engine.tools[engine.id_to_idx[g]].name)) for g in missing]
    gold_servers = {engine.tools[engine.id_to_idx[g]].server for g in missing}
    for tid in ranked_ids[:k]:
        if tid in q.gold:
            continue
        t = engine.tools[engine.id_to_idx[tid]]
        if t.server in gold_servers or any(set(tokenize(t.name)) & gn for gn in gold_names):
            return "sibling_confusion", ranks
    if any(r is not None for r in ranks.values()):
        return "ranked_below_k", ranks
    return "not_retrieved", ranks


def run_eval(engine: Engine, queries: list[Query], *, methods: list[str] | None = None, ks=DEFAULT_KS,
             split: str = "all", failure_k: int = 5, max_examples: int = 30) -> dict:
    qs = _filter(queries, split)
    if not qs:
        raise ValueError(f"no queries for split={split!r}")
    methods = methods or engine.methods()
    ks = sorted(set(ks))
    kmax = max(max(ks), 20)
    all_ids = [t.id for t in engine.tools]
    out = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": split,
        "n_queries": len(qs), "n_tools": len(engine.tools), "ks": ks, "failure_k": failure_k,
        "embedder": engine.embedder_spec, "reranker": engine.reranker_spec if engine.reranker else "none",
        "rerank_depth": engine.rerank_depth, "doc_mode": engine.doc_mode, "tokens_all_tools": engine.tokens_all,
        "token_estimator": "chars/4 (set TOOLGATE_TOKENIZER=tiktoken for tiktoken counts)",
    }, "methods": {}, "failures": {}}

    for method in methods:
        # warm-up so first-call overhead (lazy init, caches) doesn't pollute P95
        engine.search(qs[0].query, method, kmax)
        per_q, lat, tok = [], [], []
        for q in qs:
            res = engine.search(q.query, method, kmax)
            lat.append(res.latency_ms)
            ranked = [h["id"] for h in res.hits]
            row = {"q": q, "ranked": ranked}
            for k in ks:
                kk = len(all_ids) if method == "all_tools" else k
                row[f"recall@{k}"] = M.recall_at_k(ranked, q.gold, kk, q.gold_mode)
                row[f"full_hit@{k}"] = M.full_hit_at_k(ranked, q.gold, kk, q.gold_mode)
                row[f"ndcg@{k}"] = M.ndcg_at_k(ranked, q.gold, kk, q.gold_mode)
            row["mrr"] = M.mrr(ranked, q.gold, q.gold_mode)
            row["fh_report"] = M.full_hit_at_k(ranked, q.gold, len(all_ids) if method == "all_tools" else failure_k, q.gold_mode)
            # tokens that would be sent to the LLM at the *reporting* k (failure_k)
            if method == "all_tools":
                row["tokens"] = engine.tokens_all
            else:
                row["tokens"] = sum(engine.tok[engine.id_to_idx[i]] for i in ranked[:failure_k])
            tok.append(row["tokens"])
            per_q.append(row)

        metrics = {f"{m}@{k}": round(M.mean(r[f"{m}@{k}"] for r in per_q), 4) for k in ks for m in ("recall", "full_hit", "ndcg")}
        metrics["mrr"] = round(M.mean(r["mrr"] for r in per_q), 4)
        if method == "all_tools":  # ranking metrics are meaningless when everything is in context
            metrics["mrr"] = None
            for k in ks:
                metrics[f"ndcg@{k}"] = None
        by_type = {}
        for typ in sorted({r["q"].type for r in per_q}):
            rows = [r for r in per_q if r["q"].type == typ]
            by_type[typ] = {"n": len(rows), f"full_hit@{failure_k}": round(M.mean(r["fh_report"] for r in rows), 4),
                            "mrr": round(M.mean(r["mrr"] for r in rows), 4)}
        avg_tok = M.mean(tok)
        out["methods"][method] = {
            "label": METHOD_LABELS.get(method, method), "metrics": metrics, "by_type": by_type,
            "latency_p50_ms": round(M.percentile(lat, 50), 3), "latency_p95_ms": round(M.percentile(lat, 95), 3),
            "tokens_avg": round(avg_tok, 1),
            "tokens_saved_pct": round(100 * (1 - avg_tok / max(engine.tokens_all, 1)), 1),
        }

        if method != "all_tools":
            buckets: Counter = Counter()
            examples = []
            for r in per_q:
                q = r["q"]
                if r["fh_report"]:
                    continue
                b, ranks = _bucket(engine, q, r["ranked"], failure_k)
                buckets[b] += 1
                if len(examples) < max_examples:
                    examples.append({"id": q.id, "type": q.type, "query": q.query, "gold": q.gold, "gold_mode": q.gold_mode,
                                     "gold_ranks": ranks, "bucket": b, "top": r["ranked"][:failure_k]})
            out["failures"][method] = {"n_failures": sum(buckets.values()), "buckets": dict(buckets), "examples": examples}
    return out


def run_ablations(engine: Engine, queries: list[Query], *, split: str = "all", k: int = 5, seed: int = 7,
                  pool_sizes=(25, 50, 100, 200, 500)) -> dict:
    """Doc-enrichment, rerank-depth and pool-size ablations for the main methods."""
    qs = _filter(queries, split)
    methods = [m for m in ("bm25", "hybrid", "hybrid_rerank") if m in engine.methods()]

    def score(eng: Engine) -> dict:
        res = {}
        for m in methods:
            rows = []
            for q in qs:
                ranked = [h["id"] for h in eng.search(q.query, m, k).hits]
                rows.append((M.full_hit_at_k(ranked, q.gold, k, q.gold_mode), M.mrr(ranked, q.gold, q.gold_mode)))
            res[m] = {f"full_hit@{k}": round(M.mean(r[0] for r in rows), 4), "mrr": round(M.mean(r[1] for r in rows), 4)}
        return res

    ab: dict = {"k": k, "n_queries": len(qs), "doc_mode": [], "rerank_depth": [], "pool_size": []}
    for mode in ("name_desc", "full"):
        eng = Engine(engine.tools, engine.settings, doc_mode=mode, embedder=engine.embedder_spec,
                     reranker=engine.reranker_spec, rerank_depth=engine.rerank_depth, use_cache=False)
        ab["doc_mode"].append({"setting": mode, "results": score(eng)})
    if "hybrid_rerank" in methods:
        for depth in (5, 10, 20, 30, 50):
            eng = Engine(engine.tools, engine.settings, doc_mode=engine.doc_mode, embedder=engine.embedder_spec,
                         reranker=engine.reranker_spec, rerank_depth=depth, use_cache=False)
            ab["rerank_depth"].append({"setting": depth, "results": {"hybrid_rerank": score(eng)["hybrid_rerank"]}})
    gold_ids = sorted({g for q in qs for g in q.gold})
    others = [t.id for t in engine.tools if t.id not in set(gold_ids)]
    rng = random.Random(seed)
    for n in pool_sizes:
        if n > len(engine.tools) or n < len(gold_ids):
            continue
        pool = gold_ids + rng.sample(others, n - len(gold_ids))
        ab["pool_size"].append({"setting": n, "results": score(engine.subset(pool))})
    if len(engine.tools) not in [p["setting"] for p in ab["pool_size"]]:
        ab["pool_size"].append({"setting": len(engine.tools), "results": score(engine)})
    return ab


def save_results(results: dict, results_dir: Path, name: str = "latest.json") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / name
    p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    stamp = results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stamp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return p


def format_table(results: dict, k: int = 5) -> str:
    rows = [f"{'method':34} {'recall@'+str(k):>9} {'fullhit@'+str(k):>10} {'MRR':>6} {'nDCG@'+str(k):>8} {'tok':>7} {'saved%':>7} {'p50ms':>7} {'p95ms':>7}"]
    for m, r in results["methods"].items():
        mt = r["metrics"]
        f = lambda v, w: (f"{v:{w}.3f}" if v is not None else f"{'-':>{w}}")
        rows.append(f"{r['label'][:34]:34} {f(mt.get(f'recall@{k}'), 9)} {f(mt.get(f'full_hit@{k}'), 10)} {f(mt['mrr'], 6)} "
                    f"{f(mt.get(f'ndcg@{k}'), 8)} {r['tokens_avg']:7.0f} {r['tokens_saved_pct']:7.1f} {r['latency_p50_ms']:7.2f} {r['latency_p95_ms']:7.2f}")
    return "\n".join(rows)


def main(argv=None):  # python -m toolgate.evaluation.runner
    import argparse

    from ..config import get_settings
    from ..corpus import load_queries

    ap = argparse.ArgumentParser(description="Run the offline retrieval evaluation")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ablations", action="store_true")
    a = ap.parse_args(argv)
    s = get_settings()
    t0 = time.time()
    eng = Engine.from_path(settings=s)
    qs = load_queries(s.queries_path)
    res = run_eval(eng, qs, split=a.split, failure_k=a.k)
    if a.ablations:
        res["ablations"] = run_ablations(eng, qs, split=a.split, k=a.k)
    p = save_results(res, s.results_dir)
    print(format_table(res, a.k))
    print(f"\nembedder={res['meta']['embedder']} reranker={res['meta']['reranker']} tools={res['meta']['n_tools']} "
          f"queries={res['meta']['n_queries']} split={a.split}  ({time.time() - t0:.1f}s)  -> {p}")


if __name__ == "__main__":
    main()
