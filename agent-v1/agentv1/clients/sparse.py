"""Deterministic BM25 sparse vectors for Qdrant's hybrid search.

Hand-rolled rather than taken from ``fastembed`` for one reason: the whole
value of the lexical half on this corpus is the marker tokens from
``text/normalize.py`` (``stage_1_plus``, ``uniconnect_plus``, ``dtc_P0420``).
Every off-the-shelf tokenizer strips the ``+`` that distinguishes a Stage 1
tune from a Stage 1+ tune, which is precisely the distinction we need. Owning
the analyzer makes index/query symmetry provable instead of hopeful.

Term ids are ``crc32`` of the term, not Python's ``hash`` -- string hashing is
salted per process, so a ``hash``-derived index would silently stop matching
across a restart.

The IDF table is corpus statistics and must be identical at index and query
time, so it is persisted next to the build and versioned with it.
"""

from __future__ import annotations

import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..text.normalize import lexical_terms

K1 = 1.2
B = 0.75
ANALYZER_VERSION = 1


def term_id(term: str) -> int:
    """Stable 32-bit id. Qdrant sparse indices must be non-negative."""
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

    def as_qdrant(self) -> dict:
        return {"indices": self.indices, "values": self.values}


class Bm25Encoder:
    """Fit document frequencies once, then encode documents and queries.

    ``fit`` is a full pass over the corpus; for ~65k units that is seconds.
    """

    def __init__(self) -> None:
        self.doc_freq: dict[str, int] = {}
        self.n_docs: int = 0
        self.avg_len: float = 0.0
        self.analyzer_version: int = ANALYZER_VERSION

    # -- fitting -------------------------------------------------------------
    def fit(self, corpus: Iterable[str]) -> "Bm25Encoder":
        df: Counter[str] = Counter()
        total_len = 0
        n = 0
        for text in corpus:
            terms = lexical_terms(text)
            total_len += len(terms)
            n += 1
            df.update(set(terms))
        self.doc_freq = dict(df)
        self.n_docs = n
        self.avg_len = (total_len / n) if n else 0.0
        return self

    def _idf(self, term: str) -> float:
        # Lucene-style BM25 IDF: always positive, so a term present in almost
        # every document contributes ~0 rather than a negative score that would
        # actively push matching documents down.
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    # -- encoding ------------------------------------------------------------
    def encode_document(self, text: str) -> SparseVector:
        terms = lexical_terms(text)
        if not terms:
            return SparseVector([], [])
        tf = Counter(terms)
        dl = len(terms)
        denom_len = K1 * (1 - B + B * (dl / self.avg_len if self.avg_len else 1.0))
        weights: dict[int, float] = {}
        for term, freq in tf.items():
            score = self._idf(term) * (freq * (K1 + 1)) / (freq + denom_len)
            tid = term_id(term)
            # Two distinct terms can collide in 31 bits; keep the stronger.
            if score > weights.get(tid, 0.0):
                weights[tid] = score
        items = sorted(weights.items())
        return SparseVector([i for i, _ in items], [round(v, 6) for _, v in items])

    def encode_query(self, text: str) -> SparseVector:
        """Query side carries IDF only.

        No term-frequency saturation: a query is short, and repeating a word in
        a question is not evidence of relevance the way repeating it in a
        document is.
        """
        terms = lexical_terms(text)
        if not terms:
            return SparseVector([], [])
        weights: dict[int, float] = {}
        for term in set(terms):
            tid = term_id(term)
            weights[tid] = max(weights.get(tid, 0.0), self._idf(term))
        items = sorted(weights.items())
        return SparseVector([i for i, _ in items], [round(v, 6) for _, v in items])

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "analyzer_version": self.analyzer_version,
                    "n_docs": self.n_docs,
                    "avg_len": self.avg_len,
                    "doc_freq": self.doc_freq,
                },
                separators=(",", ":"),
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Bm25Encoder":
        blob = json.loads(Path(path).read_text())
        enc = cls()
        enc.doc_freq = blob["doc_freq"]
        enc.n_docs = blob["n_docs"]
        enc.avg_len = blob["avg_len"]
        enc.analyzer_version = blob.get("analyzer_version", 0)
        if enc.analyzer_version != ANALYZER_VERSION:
            raise RuntimeError(
                f"BM25 statistics were fitted with analyzer v{enc.analyzer_version} "
                f"but this code is v{ANALYZER_VERSION}. The tokenizer changed, so "
                f"index and query terms no longer agree -- rebuild the index."
            )
        return enc


def fuse_rrf(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Reciprocal-rank fusion across ranked id lists.

    Preferred over score normalisation because dense cosine and BM25 live on
    incomparable scales, and any normalisation that makes them comparable is a
    tuning knob nobody will ever revisit.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return scores
