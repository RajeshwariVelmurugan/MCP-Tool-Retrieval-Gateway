"""MCP gateway: an MCP server that sits in front of many MCP servers.

Instead of loading hundreds of tool definitions into the agent's context, the agent sees TWO tools:

  search_tools(query, k)           -> the k most relevant tool definitions (from the retrieval engine)
  call_tool(tool_id, arguments)    -> forwards the call to the real downstream MCP server

Run (stdio, for Antigravity / Claude Desktop / any MCP client):   python -m toolgate.gateway
Run (streamable HTTP):                                             python -m toolgate.gateway --http --port 8001

Downstream servers come from servers.json; the searchable corpus comes from data/tools.json (run
`python -m toolgate.collect` first so both agree). Set TOOLGATE_GATEWAY_LOG=0 to disable the query log.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .config import get_settings
from .engine import Engine
from .mcp_util import connect, load_servers

INSTRUCTIONS = (
    "This gateway fronts many MCP servers. Do not guess tool names. First call search_tools with a short natural-language "
    "description of what you need to do; it returns the best matching tools with their input schemas. Then call call_tool "
    "with the chosen tool_id and arguments that match that tool's input_schema. If the results do not fit, search again "
    "with different wording."
)

mcp = MCPServer("toolgate", instructions=INSTRUCTIONS)

_settings = get_settings()
_engine: Engine | None = None
_downstream: dict[str, "Downstream"] = {}


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine.from_path(settings=_settings)
    return _engine


def _method(engine: Engine) -> str:
    wanted = os.environ.get("TOOLGATE_GATEWAY_METHOD", "hybrid_rerank")
    return wanted if wanted in engine.methods() else "hybrid"


class Downstream:
    """One persistent stdio connection to a downstream server, owned by a single background task
    (the MCP stdio client must be entered and exited from the same task)."""

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def _run(self):
        try:
            async with connect(self.cfg, timeout=120) as client:
                while True:
                    item = await self._queue.get()
                    if item is None:
                        return
                    tool, args, fut = item
                    try:
                        fut.set_result(await client.call_tool(tool, args))
                    except Exception as e:  # noqa: BLE001
                        fut.set_exception(e)
        except Exception as e:  # noqa: BLE001 - connection failed: fail all waiting calls
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item:
                    item[2].set_exception(e)
            self._task = None

    async def call(self, tool: str, args: dict):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((tool, args, fut))
        return await fut


def _log(entry: dict) -> None:
    if os.environ.get("TOOLGATE_GATEWAY_LOG", "1") == "0":
        return
    try:
        _settings.results_dir.mkdir(parents=True, exist_ok=True)
        with open(_settings.results_dir / "gateway_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **entry}) + "\n")
    except OSError:
        pass


@mcp.tool()
async def search_tools(query: str, k: int = 5) -> str:
    """Find the tools that can do a task. Describe what you want to do in plain words; returns the top-k tool
    definitions (id, server, description, input_schema). Use the returned `id` with call_tool."""
    engine = get_engine()
    method = _method(engine)
    k = max(1, min(int(k), 20))
    res = engine.search(query, method, k)
    ids = [h["id"] for h in res.hits]
    _log({"event": "search", "query": query, "method": method, "k": k, "returned": ids, "latency_ms": res.latency_ms})
    out = []
    for h in res.hits:
        t = engine.tools[engine.id_to_idx[h["id"]]]
        out.append({"id": t.id, "server": t.server, "description": t.description, "input_schema": t.input_schema})
    return json.dumps({"query": query, "method": method, "tools": out}, ensure_ascii=False)


@mcp.tool()
async def call_tool(tool_id: str, arguments: dict | None = None) -> str:
    """Call a tool found with search_tools. `tool_id` is the `id` field from search_tools; `arguments` must match
    that tool's input_schema."""
    engine = get_engine()
    if tool_id not in engine.id_to_idx:
        return f"ERROR: unknown tool_id {tool_id!r}. Call search_tools first and use the `id` it returns."
    tool = engine.tools[engine.id_to_idx[tool_id]]
    servers = load_servers(_settings.servers_path, include_disabled=True)
    if tool.server not in servers:
        return f"ERROR: server {tool.server!r} is not configured in {_settings.servers_path.name}."
    ds = _downstream.setdefault(tool.server, Downstream(tool.server, servers[tool.server]))
    t0 = time.perf_counter()
    try:
        result = await ds.call(tool.name, arguments or {})
    except Exception as e:  # noqa: BLE001
        _log({"event": "call_error", "tool": tool_id, "error": str(e)[:300]})
        return f"ERROR calling {tool_id}: {type(e).__name__}: {e}"
    _log({"event": "call", "tool": tool_id, "ms": (time.perf_counter() - t0) * 1000, "is_error": bool(result.is_error)})
    parts = []
    for c in result.content:
        parts.append(getattr(c, "text", None) or f"[{getattr(c, 'type', 'content')} content]")
    text = "\n".join(parts)
    return ("ERROR: " + text) if result.is_error else text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args(argv)
    if a.http:
        mcp.run("streamable-http", host=a.host, port=a.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
