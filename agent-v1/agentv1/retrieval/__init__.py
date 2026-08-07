"""Retrieval layer.

``search`` is the only entry point anything outside this package should need.
It is deliberately the *only* way to reach the vector store from the agent:
every other path would have to rebuild the ``training_safe`` floor, and a
safety property that has to be rebuilt at each call site is a safety property
that will eventually be forgotten at one of them.

Exports are resolved lazily (PEP 562) for two reasons: importing this package
should not drag in torch and transformers before anything has asked for a
reranker, and eager submodule imports make ``python -m
agentv1.retrieval.<module>`` emit a double-import warning on every self-check.
"""

from typing import TYPE_CHECKING

_EXPORTS = {
    "search": "pipeline",
    "SearchResult": "pipeline",
    "RetrievedUnit": "pipeline",
    "normalize_query": "pipeline",
    "unit_text": "pipeline",
    "apply_supersession": "supersede",
    "SupersessionResult": "supersede",
    "Drop": "supersede",
    "get_reranker": "rerank",
    "Reranker": "rerank",
    "RerankScore": "rerank",
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:  # so type checkers and editors still resolve the names
    from .pipeline import RetrievedUnit, SearchResult, normalize_query, search, unit_text
    from .rerank import Reranker, RerankScore, get_reranker
    from .supersede import Drop, SupersessionResult, apply_supersession


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
