---
description: Add a new retrieval method to the comparison
---
1. Implement a class with `search(query, k) -> list[(tool_index, score)]` in `toolgate/retrievers/`.
2. Register a new method name in `toolgate/engine.py` (`METHOD_LABELS`, `methods()`, `_ranked()`).
3. Add a test in `tests/test_retrieval.py`.
4. Run `python -m toolgate eval` and confirm the new method appears in the dashboard table.
