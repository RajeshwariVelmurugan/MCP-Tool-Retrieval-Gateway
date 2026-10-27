# toolgate: MCP tool retrieval gateway + evaluation harness

When an agent has hundreds of MCP tools, putting every tool definition into the prompt wastes tokens and makes the model pick the wrong tool. **toolgate** retrieves only the relevant tools and *measures* the result.

```
agent ──► toolgate gateway (2 tools: search_tools, call_tool)
              ├─ retrieval engine: BM25 · embeddings · hybrid (RRF) · reranker
              └─ forwards real calls to your MCP servers (servers.json)

dashboard + eval harness: compare all-tools vs BM25 vs embeddings vs hybrid vs +reranker (vs Anthropic native tool search)
```

## What is in the box

| Part | Where | Status |
|---|---|---|
| Retrieval engine (BM25, dense, hybrid RRF, reranker) | `toolgate/engine.py`, `toolgate/retrievers/` | tested |
| Offline eval: Recall@k, Full-hit@k, MRR, nDCG, tokens, P50/P95 latency, failure buckets, ablations | `toolgate/evaluation/runner.py` | tested |
| Dashboard (playground, evaluation, tool browser, gateway config) | `web/`, `toolgate/server.py` | tested in headless Chromium |
| MCP gateway (`search_tools` / `call_tool`) | `toolgate/gateway.py` | tested end-to-end over stdio with a real downstream server |
| Real-tool collector (`tools/list` from your servers → corpus) | `toolgate/collect.py` | tested (demo server + the Playwright MCP server) |
| LLM end-to-end eval incl. Anthropic native tool search arm | `toolgate/evaluation/e2e.py` | request building tested (`--dry-run`); **live API run not tested**, verify against the docs |
| Real embedding / reranker backends (fastembed, sentence-transformers) | `toolgate/retrievers/dense.py`, `rerank.py` | **not tested** (needs model downloads), code written to the libraries' documented APIs |

### Read this before you trust any number
* Defaults are **offline stand-ins**: `hash` embedder (char n-gram TF-IDF) and `heuristic` reranker. They need no downloads so everything runs anywhere, but they are *not* semantic models. Switch to real models (below) before drawing conclusions.
* `data/tools.json` is a **starter corpus (139 tools)** approximated from the public definitions of popular MCP servers. `data/queries.json` has **86 hand-written starter queries**. Both exist so the pipeline runs on day one. For a resume-grade result you must collect real tools and write/verify your own queries (plan below).
* Token counts use a `chars/4` estimate unless you set `TOOLGATE_TOKENIZER=tiktoken`; the e2e script reports the API's real input-token counts.

## Quick start

Requires Python 3.10+. Node.js is only needed if you collect tools from `npx`-based servers.

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
#   py -m venv .venv ; .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env

python -m toolgate serve        # dashboard → http://127.0.0.1:8000
python -m toolgate eval --ablations   # same evaluation in the terminal
python -m unittest discover -s tests -v
```

## Using it in Antigravity

1. **File → Open Folder** and open this folder. Antigravity is a VS Code-style IDE, so the bundled `.vscode/launch.json` and `tasks.json` give you Run/Debug entries (Dashboard, Offline eval, Gateway, Tests).
2. Create the venv and install requirements (above), then select `.venv` as the Python interpreter.
3. `AGENTS.md` and `.agent/` contain project rules and two workflows for the IDE's agent. I wrote these from memory of how Antigravity reads workspace rules; if your version looks elsewhere, move the files. Nothing in the project depends on them.
4. **Use the gateway as an MCP server in Antigravity:** open the dashboard's **Gateway** tab, copy the generated config (it already contains your absolute paths) and paste it into the IDE's MCP server configuration (the menu name varies by version; look for "MCP servers" / "raw config" in the agent panel). Restart the connection: the agent should now see exactly two tools. See `antigravity_mcp_config.example.json` for the shape.

## Make it real 

1. **Collect real tools (target 200+).** Edit `servers.json` (enable servers you can run, put API keys in `.env`), then:
   ```bash
   python -m toolgate collect            # writes data/tools.json from real tools/list calls
   ```
   Pick servers with overlapping tools (browsers, file access, search) so retrieval is genuinely hard. Collected tool ids look like `<server>__<tool>`.
2. **Rewrite `data/queries.json` for your corpus.** ≥300 queries: single-tool, multi-tool (`gold_mode: "all"`), vague (`"any"`), paraphrased, adversarial (near-duplicate tools). Review every gold label by hand. Keep `dev`/`test` splits fixed; tune on dev, report on test. The gateway logs real agent queries to `data/results/gateway_log.jsonl`, which is a good source of realistic wording.
3. **Use real models.** `pip install -r requirements-ml.txt`, then in `.env`:
   ```
   TOOLGATE_EMBEDDER=fastembed:BAAI/bge-small-en-v1.5
   TOOLGATE_RERANKER=fastembed:Xenova/ms-marco-MiniLM-L-6-v2
   ```
   (or `st:BAAI/bge-small-en-v1.5` / `st:BAAI/bge-reranker-base` with sentence-transformers). First run downloads the models; embeddings are cached in `data/cache/`.
4. **Run the end-to-end comparison** with a real LLM and Anthropic's native tool search:
   ```bash
   pip install anthropic          # key in .env as ANTHROPIC_API_KEY
   python -m toolgate e2e --dry-run                  # payload sizes, no API calls
   python -m toolgate e2e --split test --limit 30    # cheap smoke run, results are cached
   ```
   Arms: `all_tools`, `retrieval` (your top-k), `native_bm25`, `native_regex`. Check the current tool-search docs first: tool type names and SDK support change.
5. **Read the failures.** The dashboard's failure analysis buckets misses (`vocab_mismatch`, `sibling_confusion`, `multi_tool_partial`, `ranked_below_k`, `not_retrieved`). The buckets are heuristics; open the examples and write your own taxonomy. This is what interviewers care about.
6. **Report honestly.** Resume bullet only after you have your own numbers, e.g. *"Built an MCP tool-retrieval gateway over N tools; hybrid retrieval + reranking cut prompt tokens by X% and changed tool-selection accuracy from Y% to Z% vs. all-tools, P95 retrieval latency N ms."* If native tool search ties you, say so and explain when yours differs (cross-client, latency, control, cost).

## Metrics

* **Recall@k**: fraction of required tools in the top-k (for `any` queries: 1 if any acceptable tool is present).
* **Full-hit@k**: share of queries where *everything needed* is in the top-k (the number that matters for agents).
* **MRR / nDCG@k**: ranking quality. Reported as `–` for the all-tools baseline (ranking is meaningless there).
* **Tokens**: average token cost of the tool definitions sent at the report k, and % saved vs all tools.
* **Latency**: P50/P95 of retrieval (including reranking), after a warm-up call.
* **Ablations**: document enrichment (name+description vs +server+params), reranker depth, tool pool size (gold tools are always in the pool; distractors are random, seed 7).

## API

`GET /api/config` · `GET /api/tools?server=&q=` · `GET /api/queries` · `POST /api/search {query,k,methods[]}` · `POST /api/eval/run {split,k,ablations}` · `GET /api/eval/latest` · `POST /api/reload`

## Layout

```
toolgate/            engine, retrievers/, evaluation/, server.py, gateway.py, collect.py
web/                 index.html, app.js, style.css   (no build step)
data/                tools.json, queries.json, results/, cache/
tests/               unit + API + gateway end-to-end tests
servers.json         downstream MCP servers
```

## Known limitations

* The `hash` embedder cannot bridge vocabulary gaps ("flip this string" vs "reverse text"). That is by design and exactly the failure real embeddings fix; expect the vocab-mismatch bucket to shrink a lot with a real model.
* The gateway forwards calls over stdio and keeps one persistent connection per downstream server; no auth/OAuth flows for remote servers yet.
* `e2e` judges which tool the model *selects*; it does not execute tools or grade argument values.
* Failure buckets and token counts are heuristics/estimates, labelled as such in the UI.
