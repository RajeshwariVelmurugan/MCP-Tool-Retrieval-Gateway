"""Helpers for talking to downstream MCP servers over stdio (used by the collector and the gateway)."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from mcp import Client, StdioServerParameters

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str) -> str:
    return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_servers(path: Path, include_disabled: bool = False) -> dict[str, dict]:
    """servers.json accepts {"servers": {...}} or the Claude-Desktop style {"mcpServers": {...}}.

    Each entry: {"command": "npx", "args": [...], "env": {...}, "enabled": true}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    servers = data.get("servers") or data.get("mcpServers") or {}
    return {n: c for n, c in servers.items() if include_disabled or c.get("enabled", True)}


def server_params(cfg: dict) -> StdioServerParameters:
    command = cfg["command"]
    resolved = shutil.which(command) or command   # on Windows this finds npx.cmd / uvx.exe
    env = {k: _expand(str(v)) for k, v in (cfg.get("env") or {}).items()} or None
    args = [_expand(str(a)) for a in cfg.get("args", [])]
    return StdioServerParameters(command=resolved, args=args, env=env, cwd=cfg.get("cwd"))


def connect(cfg: dict, timeout: float = 60.0) -> Client:
    return Client(server_params(cfg), read_timeout_seconds=timeout)
