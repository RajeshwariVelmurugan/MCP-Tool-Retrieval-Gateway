"""HTTP API + dashboard (Starlette). Run:  python -m toolgate serve"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .config import ROOT, get_settings
from .corpus import load_queries
from .engine import Engine
from .evaluation.runner import DEFAULT_KS, run_ablations, run_eval, save_results
from .text import tool_tokens

WEB = ROOT / "web"
settings = get_settings()
_lock = threading.Lock()
_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = Engine.from_path(settings=settings)
        return _engine


def _corpus_source() -> str:
    try:
        return json.loads(settings.corpus_path.read_text(encoding="utf-8")).get("meta", {}).get("source", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


async def _body(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "body must be JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, "body must be a JSON object")
    return data


async def health(_: Request):
    return JSONResponse({"ok": True})


async def config(_: Request):
    e = await run_in_threadpool(engine)
    return JSONResponse({**e.info(), "corpus_path": str(settings.corpus_path), "queries_path": str(settings.queries_path),
                         "default_k": settings.default_k, "root": str(ROOT), "python": sys.executable,
                         "corpus_source": _corpus_source()})


async def list_tools(request: Request):
    e = await run_in_threadpool(engine)
    server = request.query_params.get("server")
    q = (request.query_params.get("q") or "").lower()
    rows = []
    for t, tok in zip(e.tools, e.tok):
        if server and t.server != server:
            continue
        if q and q not in (t.id + " " + t.description).lower():
            continue
        rows.append({"id": t.id, "server": t.server, "name": t.name, "description": t.description,
                     "params": list((t.input_schema or {}).get("properties", {}).keys()), "tokens": tok})
    servers: dict[str, int] = {}
    for t in e.tools:
        servers[t.server] = servers.get(t.server, 0) + 1
    return JSONResponse({"tools": rows, "servers": servers, "total": len(e.tools)})


async def list_queries(_: Request):
    qs = load_queries(settings.queries_path)
    return JSONResponse({"queries": [asdict(q) for q in qs]})


async def search(request: Request):
    data = await _body(request)
    query = str(data.get("query", "")).strip()
    if not query:
        raise HTTPException(400, "query is required")
    k = max(1, min(int(data.get("k", settings.default_k)), 50))
    e = await run_in_threadpool(engine)
    methods = data.get("methods") or ["hybrid_rerank"]
    bad = [m for m in methods if m not in e.methods()]
    if bad:
        raise HTTPException(400, f"unknown method(s) {bad}; available: {e.methods()}")
    results = await run_in_threadpool(lambda: [e.search(query, m, k).to_dict() for m in methods])
    gold = None
    for q in load_queries(settings.queries_path):
        if q.query.strip().lower() == query.lower():
            gold = {"gold": q.gold, "gold_mode": q.gold_mode, "type": q.type}
            break
    return JSONResponse({"query": query, "results": results, "known_query": gold})


async def eval_run(request: Request):
    data = await _body(request)
    e = await run_in_threadpool(engine)
    split = data.get("split", "all")
    if split not in ("all", "dev", "test"):
        raise HTTPException(400, "split must be all, dev or test")
    ks = tuple(int(k) for k in data.get("ks", DEFAULT_KS))
    k = int(data.get("k", 5))
    if k not in ks:
        ks = tuple(sorted(set(ks) | {k}))
    methods = data.get("methods") or None
    qs = load_queries(settings.queries_path)

    def work():
        res = run_eval(e, qs, methods=methods, ks=ks, split=split, failure_k=k)
        if data.get("ablations"):
            res["ablations"] = run_ablations(e, qs, split=split, k=k)
        save_results(res, settings.results_dir)
        return res

    try:
        res = await run_in_threadpool(work)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return JSONResponse(res)


async def eval_latest(_: Request):
    p = settings.results_dir / "latest.json"
    if not p.exists():
        return JSONResponse({"empty": True})
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


async def reload(_: Request):
    global _engine
    with _lock:
        _engine = None
    e = await run_in_threadpool(engine)
    return JSONResponse(e.info())


async def index(_: Request):
    return FileResponse(WEB / "index.html")


async def http_exception(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


async def server_error(_: Request, exc: Exception):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/health", health),
        Route("/api/config", config),
        Route("/api/tools", list_tools),
        Route("/api/queries", list_queries),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/eval/run", eval_run, methods=["POST"]),
        Route("/api/eval/latest", eval_latest),
        Route("/api/reload", reload, methods=["POST"]),
        Mount("/static", StaticFiles(directory=WEB), name="static"),
    ],
    exception_handlers={HTTPException: http_exception, Exception: server_error},
)


def main(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    host, port = host or settings.host, port or settings.port
    print(f"toolgate dashboard -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
