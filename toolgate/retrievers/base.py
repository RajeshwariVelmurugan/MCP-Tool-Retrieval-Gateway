from __future__ import annotations

from typing import Protocol

Hit = tuple[int, float]  # (index into the tool list, score)


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]:
        """Return up to k (tool_index, score) pairs, best first."""
        ...
