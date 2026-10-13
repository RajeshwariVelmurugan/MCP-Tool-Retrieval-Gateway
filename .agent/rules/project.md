# AGENTS.md: project context for AI coding agents

**Project:** toolgate, an MCP tool-retrieval gateway plus the evaluation harness that proves it works.
Goal: when an agent has 100-500+ MCP tools, send only the relevant few to the LLM, and *measure* the effect
(tool-selection accuracy, tokens, latency) against all-tools and Anthropic's native tool search.

## Layout
- `toolgate/engine.py`: builds BM25 / dense / hybrid(RRF) / +reranker over a tool corpus; one `search(query, method, k)` entry point.
- `toolgate/retrievers/`: `bm25.py`, `dense.py` (embedder specs), `hybrid.py` (RRF), `rerank.py` (reranker specs).
- `toolgate/evaluation/`: `metrics.py`, `runner.py` (offline eval, failure buckets, ablations), `e2e.py` (LLM eval, native tool search arm).
- `toolgate/server.py`: Starlette API + static dashboard in `web/` (vanilla JS, no build step).
- `toolgate/gateway.py`: the MCP gateway (`search_tools`, `call_tool`). `toolgate/collect.py`: dumps real `tools/list` into `data/tools.json`.
- `data/tools.json`, `data/queries.json`: corpus and gold queries (ids are `<server>__<tool>`).

## Commands
- `python -m toolgate serve` · `python -m toolgate eval --ablations` · `python -m toolgate collect` · `python -m toolgate gateway`
- Tests: `python -m unittest discover -s tests -v`

## Rules (important)
1. **Never invent or hard-code result numbers.** Report only what `eval`/`e2e` actually produced.
2. Tune only on `split == "dev"`; report on `"test"`. Do not move queries between splits after seeing results.
3. Gold labels must be reviewed by a human. Do not auto-label from model output.
4. The offline defaults (`hash` embedder, `heuristic` reranker) are stand-ins, not real models. Say so when reporting results.
5. Keep the core dependency-light (`mcp`, `numpy`). Heavy libraries go behind optional extras in `requirements-ml.txt`.
6. After changing retrieval code, run the tests and `python -m toolgate eval`.
