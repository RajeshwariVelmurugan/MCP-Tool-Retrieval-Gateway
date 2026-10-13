"""Text helpers shared by retrievers and the evaluation code."""
from __future__ import annotations

import json
import re

STOPWORDS = frozenset("""
a an and are as at be but by for from has have i if in into is it its me my of on or our so that the their
them then there these they this to up us was we what when where which who will with you your can could do does
did should would about all any some also just how need want like get let please using use
""".split())

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")


def split_identifier(s: str) -> str:
    """read_text_file -> 'read text file', createPullRequest -> 'create Pull Request'."""
    return _CAMEL.sub(" ", s).replace("_", " ").replace("-", " ").replace(".", " ")


def stem(tok: str) -> str:
    """Very small suffix stripper. Deliberately simple and dependency-free."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    for suf, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: len(tok) - len(suf)] + repl
    return tok


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    text = split_identifier(text).lower()
    toks = _WORD.findall(text)
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Replace with tiktoken / the API's usage counts for real numbers.

    Override at runtime with TOOLGATE_TOKENIZER=tiktoken (requires `pip install tiktoken`).
    """
    global _TIKTOKEN
    if _TIKTOKEN is None:
        import os

        _TIKTOKEN = False
        if os.environ.get("TOOLGATE_TOKENIZER") == "tiktoken":
            try:
                import tiktoken

                _TIKTOKEN = tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover
                _TIKTOKEN = False
    if _TIKTOKEN:
        return len(_TIKTOKEN.encode(text))
    return max(1, round(len(text) / 4))


_TIKTOKEN = None


def tool_tokens(definition: dict) -> int:
    """Token cost of one tool definition as it would appear in an LLM request."""
    return estimate_tokens(json.dumps(definition, separators=(",", ":")))
