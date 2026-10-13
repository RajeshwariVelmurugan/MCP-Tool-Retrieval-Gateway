"""Central configuration. Everything is overridable with environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no python-dotenv dependency). Real env vars win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    corpus_path: Path
    queries_path: Path
    results_dir: Path
    cache_dir: Path
    servers_path: Path
    embedder: str          # "hash" | "st:<model>" | "fastembed:<model>"
    reranker: str          # "none" | "heuristic" | "st:<model>" | "fastembed:<model>"
    rerank_depth: int      # how many first-stage candidates the reranker sees
    default_k: int
    host: str
    port: int


def get_settings() -> Settings:
    return Settings(
        corpus_path=_path("TOOLGATE_CORPUS", "data/tools.json"),
        queries_path=_path("TOOLGATE_QUERIES", "data/queries.json"),
        results_dir=_path("TOOLGATE_RESULTS_DIR", "data/results"),
        cache_dir=_path("TOOLGATE_CACHE_DIR", "data/cache"),
        servers_path=_path("TOOLGATE_SERVERS", "servers.json"),
        embedder=os.environ.get("TOOLGATE_EMBEDDER", "hash"),
        reranker=os.environ.get("TOOLGATE_RERANKER", "heuristic"),
        rerank_depth=int(os.environ.get("TOOLGATE_RERANK_DEPTH", "30")),
        default_k=int(os.environ.get("TOOLGATE_K", "5")),
        host=os.environ.get("TOOLGATE_HOST", "127.0.0.1"),
        port=int(os.environ.get("TOOLGATE_PORT", "8000")),
    )
