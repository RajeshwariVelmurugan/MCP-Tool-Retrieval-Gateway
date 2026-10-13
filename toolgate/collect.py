"""Collect REAL tool definitions from MCP servers into a corpus file.

    python -m toolgate.collect                       # all enabled servers in servers.json -> data/tools.json
    python -m toolgate.collect --servers my.json --out data/tools.json --merge

Every server is started over stdio, `tools/list` is called (with pagination) and the result is stored as
{id, server, name, description, input_schema}. Servers that fail are reported and skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .corpus import sanitize_name
from .mcp_util import connect, load_servers


async def collect_one(name: str, cfg: dict, timeout: float) -> list[dict]:
    async with connect(cfg, timeout) as client:
        tools, cursor = [], None
        while True:
            res = await client.list_tools(cursor=cursor)
            tools.extend(res.tools)
            cursor = res.next_cursor
            if not cursor:
                break
    return [{"id": sanitize_name(f"{name}__{t.name}"), "server": name, "name": t.name,
             "description": t.description or "", "input_schema": t.input_schema or {}} for t in tools]


async def collect_all(servers: dict[str, dict], timeout: float = 90.0) -> tuple[list[dict], dict[str, str]]:
    tools, errors = [], {}
    for name, cfg in servers.items():
        try:
            got = await asyncio.wait_for(collect_one(name, cfg, timeout), timeout + 15)
            print(f"  ok   {name:24} {len(got):3d} tools")
            tools.extend(got)
        except BaseException as e:  # noqa: BLE001 - report and keep going
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            errors[name] = f"{type(e).__name__}: {e}"[:300]
            print(f"  FAIL {name:24} {errors[name]}")
    return tools, errors


def main(argv=None) -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servers", type=Path, default=s.servers_path)
    ap.add_argument("--out", type=Path, default=s.corpus_path)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--merge", action="store_true", help="keep tools already in --out whose server was not re-collected")
    a = ap.parse_args(argv)

    servers = load_servers(a.servers)
    print(f"Collecting from {len(servers)} server(s) listed in {a.servers}")
    tools, errors = asyncio.run(collect_all(servers, a.timeout))
    if a.merge and a.out.exists():
        old = json.loads(a.out.read_text(encoding="utf-8")).get("tools", [])
        fresh = {t["server"] for t in tools}
        tools = [t for t in old if t["server"] not in fresh] + tools
    if not tools:
        print("No tools collected; not writing anything.", file=sys.stderr)
        return 1
    ids = [t["id"] for t in tools]
    assert len(ids) == len(set(ids)), "duplicate tool ids"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": {
        "source": "collected", "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "servers": sorted({t["server"] for t in tools}), "tool_count": len(tools), "errors": errors},
        "tools": tools}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(tools)} tools from {len({t['server'] for t in tools})} servers -> {a.out}")
    if errors:
        print(f"{len(errors)} server(s) failed (see messages above).")
    print("IMPORTANT: your gold queries must reference the new tool ids. See data/queries.json and README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
