"""End-to-end tool-selection evaluation against a real LLM, including Anthropic's native tool search.

    python -m toolgate e2e --dry-run                         # no API calls: show payload sizes per arm
    python -m toolgate e2e --split test --limit 30           # real run (needs ANTHROPIC_API_KEY + `pip install anthropic`)

Arms (same system prompt, same queries, same model):
  all_tools      every tool definition sent in the request (the baseline you are trying to beat)
  retrieval      only the top-k tools from OUR retriever (--method, --k) are sent
  native_bm25    all tools sent with defer_loading=true + Anthropic's BM25 tool-search tool (server side)
  native_regex   same, with the regex tool-search tool

What is measured per query: did the model call the gold tool(s) (selection success), input tokens billed, wall latency.
It does NOT execute tools or grade argument values; extend `judge()` if you add gold arguments.

Verify against the current docs before reporting numbers (tool type strings and SDK support change):
https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
Responses are cached in data/cache/e2e.jsonl so re-runs are free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from ..config import get_settings
from ..corpus import Query, load_queries, sanitize_name
from ..engine import Engine
from ..text import estimate_tokens
from . import metrics as M

SYSTEM = ("You are an assistant that completes the user's request by calling tools. Always call the single most "
          "appropriate tool (or several, if the task clearly needs several). If a required argument is missing, "
          "make a reasonable assumption instead of asking a question. Do not answer in prose before calling a tool.")

NATIVE_TOOLS = {
    "native_bm25": {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
    "native_regex": {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
}


def build_request(arm: str, engine: Engine, query: str, *, method: str, k: int, model: str) -> tuple[dict, list[str]]:
    """Returns (request kwargs for client.messages.create, ids of the tools that were exposed up-front)."""
    all_ids = [t.id for t in engine.tools]
    if arm == "all_tools":
        tools, exposed = engine.definitions(all_ids), all_ids
    elif arm == "retrieval":
        exposed = [h["id"] for h in engine.search(query, method, k).hits]
        tools = engine.definitions(exposed)
    elif arm in NATIVE_TOOLS:
        tools = [dict(NATIVE_TOOLS[arm])] + [{**d, "defer_loading": True} for d in engine.definitions(all_ids)]
        exposed = []
    else:
        raise ValueError(f"unknown arm {arm!r}")
    req = {"model": model, "max_tokens": 1024, "system": SYSTEM, "tools": tools,
           "messages": [{"role": "user", "content": query}]}
    return req, exposed


def judge(q: Query, called_ids: list[str]) -> dict:
    gold = set(q.gold)
    called = set(called_ids)
    ok = bool(gold & called) if q.gold_mode == "any" else gold <= called
    return {"success": ok, "called": sorted(called), "extra_calls": len(called - gold)}


def _call(client, req: dict, arm: str, max_rounds: int = 3):
    """Run one request; follow `pause_turn` for server-side tool search. Returns (called api names, input_tokens)."""
    called, tokens, messages = [], 0, list(req["messages"])
    for _ in range(max_rounds):
        resp = client.messages.create(**{**req, "messages": messages})
        tokens += resp.usage.input_tokens + (getattr(resp.usage, "cache_read_input_tokens", 0) or 0)
        called += [b.name for b in resp.content if b.type == "tool_use"]
        if called or resp.stop_reason != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": [b.model_dump() for b in resp.content]}]
    return called, tokens


def main(argv=None) -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="all_tools,retrieval,native_bm25")
    ap.add_argument("--method", default="hybrid_rerank")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="test", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (cheap smoke run)")
    ap.add_argument("--model", default=os.environ.get("TOOLGATE_E2E_MODEL", "claude-haiku-5-5"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    engine = Engine.from_path(settings=s)
    qs = [q for q in load_queries(s.queries_path) if a.split == "all" or q.split == a.split]
    if a.limit:
        qs = qs[: a.limit]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if a.method not in engine.methods():
        raise SystemExit(f"method {a.method!r} unavailable; choose from {engine.methods()}")

    if a.dry_run:
        print(f"model={a.model} queries={len(qs)} tools={len(engine.tools)}\n")
        for arm in arms:
            req, exposed = build_request(arm, engine, qs[0].query, method=a.method, k=a.k, model=a.model)
            size = estimate_tokens(json.dumps(req["tools"]))
            note = "deferred: only what search returns enters the context" if arm in NATIVE_TOOLS else f"{len(exposed)} loaded up-front"
            print(f"{arm:14} tools in request: {len(req['tools']):4d}  ~{size:6d} tokens of definitions ({note})")
        print("\n(dry run: no API calls made)")
        return 0

    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic   (and set ANTHROPIC_API_KEY)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (put it in .env)")
    client = anthropic.Anthropic()

    cache_file = s.cache_dir / "e2e.jsonl"
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r

    api_to_id = {sanitize_name(t.id): t.id for t in engine.tools}
    report = {"meta": {"model": a.model, "split": a.split, "n_queries": len(qs), "k": a.k, "method": a.method,
                       "n_tools": len(engine.tools)}, "arms": {}}
    for arm in arms:
        rows = []
        for q in qs:
            req, exposed = build_request(arm, engine, q.query, method=a.method, k=a.k, model=a.model)
            key = hashlib.sha1(json.dumps([a.model, arm, q.query, [t.get("name") for t in req["tools"]]]).encode()).hexdigest()
            if key in cache:
                rec = cache[key]
            else:
                t0 = time.perf_counter()
                try:
                    names, tokens = _call(client, req, arm)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {arm} {q.id}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                rec = {"key": key, "arm": arm, "qid": q.id, "called": names, "input_tokens": tokens,
                       "latency_s": time.perf_counter() - t0}
                with open(cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            called_ids = [api_to_id.get(n, n) for n in rec["called"]]
            rows.append({**judge(q, called_ids), "qid": q.id, "type": q.type, "input_tokens": rec["input_tokens"],
                         "latency_s": rec["latency_s"], "retrieval_hit": (set(q.gold) <= set(exposed)) if arm == "retrieval" else None})
        if rows:
            report["arms"][arm] = {
                "n": len(rows), "selection_success": round(M.mean(r["success"] for r in rows), 4),
                "avg_input_tokens": round(M.mean(r["input_tokens"] for r in rows), 1),
                "latency_p50_s": round(M.percentile([r["latency_s"] for r in rows], 50), 2),
                "latency_p95_s": round(M.percentile([r["latency_s"] for r in rows], 95), 2),
                "rows": rows}
        print(f"{arm:14} success={report['arms'].get(arm, {}).get('selection_success')} "
              f"avg_input_tokens={report['arms'].get(arm, {}).get('avg_input_tokens')}")
    out = s.results_dir / "e2e_latest.json"
    s.results_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
