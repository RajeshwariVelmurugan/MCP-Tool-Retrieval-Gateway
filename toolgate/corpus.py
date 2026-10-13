"""Tool corpus + gold query loading."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .text import split_identifier


@dataclass
class Tool:
    id: str                      # unique, API-safe: "<server>__<name>"
    server: str
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    def param_lines(self) -> list[str]:
        props = (self.input_schema or {}).get("properties", {}) or {}
        lines = []
        for pname, spec in props.items():
            desc = spec.get("description", "") if isinstance(spec, dict) else ""
            lines.append(f"{pname} {desc}".strip())
        return lines

    def doc(self, mode: str = "full") -> str:
        """Text that gets indexed.

        name_desc : tool name + description only
        full      : + server name + parameter names/descriptions   (the "enriched" document)
        """
        name = split_identifier(self.name)
        if mode == "name_desc":
            return f"{name}. {self.description}"
        server = split_identifier(self.server)
        return f"{name}. {self.description} Server: {server}. Parameters: {'; '.join(self.param_lines())}"

    def api_definition(self) -> dict:
        """The tool as you would put it in an LLM `tools=[...]` array."""
        return {
            "name": sanitize_name(self.id),
            "description": self.description,
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


def load_tools(path: Path) -> list[Tool]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["tools"] if isinstance(data, dict) else data
    tools, seen = [], set()
    for t in raw:
        server = t.get("server", "unknown")
        name = t["name"]
        tid = t.get("id") or sanitize_name(f"{server}__{name}")
        if tid in seen:
            raise ValueError(f"duplicate tool id in corpus: {tid}")
        seen.add(tid)
        tools.append(Tool(id=tid, server=server, name=name, description=t.get("description", "") or "",
                          input_schema=t.get("input_schema") or t.get("inputSchema") or {}))
    return tools


@dataclass
class Query:
    id: str
    query: str
    gold: list[str]
    gold_mode: str = "all"       # "all": every gold tool is required; "any": one of them is enough
    type: str = "direct"
    split: str = "dev"


def load_queries(path: Path) -> list[Query]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data["queries"] if isinstance(data, dict) else data
    return [Query(id=q["id"], query=q["query"], gold=list(q["gold"]), gold_mode=q.get("gold_mode", "all"),
                  type=q.get("type", "direct"), split=q.get("split", "dev")) for q in raw]
