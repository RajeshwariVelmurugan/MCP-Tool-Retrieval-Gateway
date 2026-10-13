---
description: Run the offline retrieval evaluation and summarize it honestly
---
1. Run `python -m toolgate eval --split test --ablations`.
2. Read `data/results/latest.json`.
3. Summarize: table of methods (full-hit@5, MRR, tokens saved, P95 latency), the biggest failure bucket with 3 concrete examples, and which ablation moved the numbers most.
4. State which embedder/reranker were used. If they were `hash`/`heuristic`, say the numbers are pipeline checks only.
