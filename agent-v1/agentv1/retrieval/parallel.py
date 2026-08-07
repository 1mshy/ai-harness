"""Encode the query once, fan out across collections concurrently.

The version this replaces (``ParallelRetriever``, dead code with zero callers,
and ``GuaranteedIndexQueryEngine`` beside it) got two things wrong that are
worth stating so they do not come back:

* It ran a **full LLM synthesis per guaranteed collection** and then read only
  ``source_nodes``, discarding every generated token. N+1 paid generations per
  query, and the reason streaming had to be collapsed. **Nothing in this module
  calls an LLM.** It moves vectors and returns hits.
* It called the *document* embedding path for a query, skipping the Qwen3
  instruct prefix. Here the query is encoded exactly once, through
  ``Embedder.embed_query``, and the same vectors are reused for every
  collection -- which is also why the fan-out is worth having: the expensive
  part is the encode, and it happens before the fan-out, not inside it.

Per-collection capability is probed, not assumed. Two facts about the live
cluster force this:

* Five of the fifteen existing collections have named ``dense``/``sparse``
  vectors; the rest are legacy single unnamed vectors, so passing ``using=``
  to them is an error.
* The legacy collections' sparse vectors were built by a *different* analyzer.
  Measured 2026-08-05: their term ids top out at 29,206 (a fastembed
  vocabulary index), while ``clients/sparse.term_id`` emits 31-bit CRC32
  values. The two spaces are disjoint, so sending our sparse query vector at a
  legacy collection contributes nothing but latency. Sparse is used only where
  this project wrote the index.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from qdrant_client import models as qm

from .. import config
from ..clients.embeddings import Embedder, get_embedder
from ..clients.qdrant import (
    CollectionMissing,
    Hit,
    dense_search,
    get_client,
    has_named_vectors,
    hybrid_search,
    open_collection,
    resolve_alias,
)
from ..clients.sparse import Bm25Encoder, SparseVector

# Ceiling on Qdrant requests in flight across *all* concurrent searches, not
# just one. Measured 2026-08-06 against the local cluster through the shared
# QdrantClient: 8 concurrent requests 14 ms wall, 32 concurrent 49 ms, 56
# concurrent raised `httpcore.ReadError: [Errno 9] Bad file descriptor` from
# the connection pool -- and because QDRANT_TIMEOUT is 60 s, the sibling
# requests on that broken connection stalled for a full minute. Seven routed
# collections times eight concurrent users is 56, so this is not a synthetic
# limit; it is the load a second user creates. The semaphore is a queue in
# front of the pool rather than a larger pool because the server answers each
# request in ~10 ms: waiting 20 ms for a slot is strictly better than losing a
# connection.
MAX_INFLIGHT_QUERIES = int(os.environ.get("RETRIEVAL_MAX_INFLIGHT", "16"))
_inflight = threading.BoundedSemaphore(MAX_INFLIGHT_QUERIES)

# Wall-clock budget for the whole fan-out. `config.QDRANT_TIMEOUT` is 60 s,
# which is a sane ceiling for a build and an outage for a synchronous
# user-facing call: measured, a single stalled connection turned an 8-user
# burst into a 62 s request while the other six collections had answered in
# 110 ms. Past the deadline the straggler is abandoned, its collection is
# reported failed, and the answer is assembled from what did arrive. The
# abandoned thread is not killed -- it cannot be -- it just finishes into a
# result nobody reads, which is why the pool below is shared and generously
# sized rather than created per request.
FANOUT_DEADLINE = float(os.environ.get("RETRIEVAL_FANOUT_DEADLINE", "5.0"))
FANOUT_POOL_SIZE = int(os.environ.get("RETRIEVAL_FANOUT_POOL", "48"))

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    """One shared pool for the whole process.

    A per-request pool would have to be shut down at the end of the request,
    and shutting one down waits for its threads -- which is precisely the stall
    the deadline exists to avoid. Pool size exceeds MAX_INFLIGHT_QUERIES on
    purpose: threads are cheap and blocked on sockets, and the semaphore, not
    the pool, is what bounds load on Qdrant.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(
                    max_workers=FANOUT_POOL_SIZE, thread_name_prefix="fanout"
                )
    return _pool

# Capability probes are cached: a hit is a network round trip and the answer
# changes only on a build. Missing collections get a much shorter TTL because
# `kb_units` does not exist yet and the process must notice when it appears
# without a restart.
_PROFILE_TTL_PRESENT = float(os.environ.get("RETRIEVAL_PROFILE_TTL", "300"))
_PROFILE_TTL_MISSING = float(os.environ.get("RETRIEVAL_PROFILE_TTL_MISSING", "20"))


@dataclass(frozen=True)
class CollectionProfile:
    """What a collection can actually answer, probed from the live cluster."""

    name: str  # the alias or name asked for
    physical: str | None  # what it resolves to, None when absent
    exists: bool
    named_vectors: bool
    sparse_compatible: bool
    gate_bearing: bool  # carries training_safe / internal_only / status
    error: str | None = None


_profiles: dict[str, tuple[float, CollectionProfile]] = {}
_profile_lock = threading.Lock()


def _is_ours(name: str, physical: str | None) -> bool:
    """True for collections this project built.

    Only those carry the ``kb_units`` payload contract -- the gate fields and a
    BM25 index in our term space. Membership is decided from the pinned alias
    list rather than from a name pattern, so a stray collection that happens to
    match a prefix cannot claim to be gate-bearing.
    """
    if name in config.NEW_ALIASES:
        return True
    if physical is None:
        return False
    return any(physical.startswith(f"{alias}_") for alias in config.NEW_ALIASES)


def profile(name: str, *, refresh: bool = False) -> CollectionProfile:
    now = time.time()
    if not refresh:
        cached = _profiles.get(name)
        if cached and cached[0] > now:
            return cached[1]
    try:
        # A cold probe is three round trips per collection; on a cold cache
        # with several concurrent searches that is enough on its own to reach
        # the pool limit, so it queues behind the same gate as a search.
        with _inflight:
            physical = open_collection(name)
            named = has_named_vectors(physical)
            indexed = set(get_client().get_collection(physical).payload_schema or {})
        ours = _is_ours(name, physical)
        # `training_safe` present as an *indexed* payload field is independent
        # evidence that the gate contract holds, and it can only strengthen the
        # floor -- never weaken what the alias list already asserts.
        prof = CollectionProfile(
            name=name,
            physical=physical,
            exists=True,
            named_vectors=named,
            sparse_compatible=ours and named,
            gate_bearing=ours or "training_safe" in indexed,
        )
        ttl = _PROFILE_TTL_PRESENT
    except CollectionMissing as exc:
        prof = CollectionProfile(
            name=name,
            physical=resolve_alias(name),
            exists=False,
            named_vectors=False,
            sparse_compatible=False,
            # An absent collection is treated as gate-bearing so that a filter
            # built for it is never accidentally the permissive variant.
            gate_bearing=True,
            error=str(exc),
        )
        ttl = _PROFILE_TTL_MISSING
    with _profile_lock:
        _profiles[name] = (now + ttl, prof)
    return prof


def clear_profile_cache() -> None:
    with _profile_lock:
        _profiles.clear()


# --- query encoding ----------------------------------------------------------


@dataclass(frozen=True)
class QueryVectors:
    """The single encode of a query, reused by every collection in the fan-out.

    ``sparse`` here is the fallback vector. IDF is *corpus* statistics and the
    index builder fits one table per collection (``data/bm25/<collection>.json``,
    measured: call_residual 9,329 terms over 5,339 docs, platform_stages 1,050
    over 410), so the correct query-side weighting differs per collection.
    Re-encoding sparse per collection costs microseconds of pure Python over a
    dozen terms; the dense embed, which is the part worth not repeating, still
    happens exactly once.
    """

    text: str
    dense: list[float]
    sparse: SparseVector
    encode_ms: float = 0.0
    sparse_source: str = "fitted"  # or "uniform_idf" -- see _bm25()


# Where the index builder writes fitted statistics, under both the alias name
# and the physical generation name.
BM25_DIR = Path(os.environ.get("BM25_STATS_DIR", str(config.DATA_DIR / "bm25")))
# Global fallbacks, for a build that writes one table for everything.
_BM25_GLOBAL = [
    os.environ.get("BM25_STATS_PATH", ""),
    str(config.DATA_DIR / f"bm25_{config.QWEN_SIZE}.json"),
    str(config.DATA_DIR / "bm25.json"),
]

_bm25_cache: tuple[float, Bm25Encoder, str] | None = None
_bm25_per_collection: dict[str, tuple[float, Bm25Encoder, str]] = {}
_bm25_lock = threading.Lock()


def _load_first(paths: Sequence[str]) -> tuple[Bm25Encoder, str] | None:
    for cand in paths:
        if cand and Path(cand).exists():
            # Bm25Encoder.load raises when the analyzer version moved, which is
            # the right outcome: mismatched tokenizers mean index and query
            # terms silently stop agreeing, and a loud failure is cheaper than
            # a quiet recall collapse.
            return Bm25Encoder.load(Path(cand)), cand
    return None


def _bm25() -> tuple[Bm25Encoder, str]:
    """The fallback query-side BM25 encoder, and where its statistics came from.

    When no fitted statistics exist the fallback is an *unfitted* encoder, not
    a disabled sparse half. An unfitted encoder assigns every term the same
    IDF, which loses term weighting but keeps the thing the lexical half exists
    for: ``stage_1_plus`` either appears in the document's term set or it does
    not, and that match survives a uniform IDF intact. Silently dropping the
    sparse half instead would lose the ``Stage 1`` / ``Stage 1+`` distinction
    entirely, which is the failure the whole hybrid design targets.

    Re-checked on a TTL so the encoder picks up a build without a restart.
    """
    global _bm25_cache
    now = time.time()
    if _bm25_cache and _bm25_cache[0] > now:
        return _bm25_cache[1], _bm25_cache[2]
    found = _load_first(_BM25_GLOBAL)
    enc, source, ttl = (
        found[0] if found else Bm25Encoder(),
        found[1] if found else "uniform_idf",
        _PROFILE_TTL_PRESENT if found else _PROFILE_TTL_MISSING,
    )
    with _bm25_lock:
        _bm25_cache = (now + ttl, enc, source)
    return enc, source


def bm25_for(prof: CollectionProfile) -> tuple[Bm25Encoder, str]:
    """Statistics fitted for this collection, else the global fallback.

    The physical generation name is tried first: an alias-named file is
    whatever the last build left behind, while ``<alias>_<size>__<gen>.json``
    is provably the table the points in that collection were encoded with.
    """
    now = time.time()
    cached = _bm25_per_collection.get(prof.name)
    if cached and cached[0] > now:
        return cached[1], cached[2]
    found = _load_first(
        [
            str(BM25_DIR / f"{prof.physical}.json") if prof.physical else "",
            str(BM25_DIR / f"{prof.name}.json"),
        ]
    )
    if found:
        enc, source, ttl = found[0], found[1], _PROFILE_TTL_PRESENT
    else:
        enc, source, ttl = (*_bm25(), _PROFILE_TTL_MISSING)
    with _bm25_lock:
        _bm25_per_collection[prof.name] = (now + ttl, enc, source)
    return enc, source


def clear_bm25_cache() -> None:
    global _bm25_cache
    with _bm25_lock:
        _bm25_cache = None
        _bm25_per_collection.clear()


def encode_query(
    text: str, *, embedder: Embedder | None = None, encoder: Bm25Encoder | None = None
) -> QueryVectors:
    """One dense embed (instruct prefix applied) plus the fallback sparse encode."""
    t0 = time.perf_counter()
    emb = embedder or get_embedder()
    dense = emb.embed_query(text)
    if encoder is not None:
        sparse, source = encoder.encode_query(text), "caller"
    else:
        enc, source = _bm25()
        sparse = enc.encode_query(text)
    return QueryVectors(
        text=text,
        dense=[float(x) for x in dense],
        sparse=sparse,
        encode_ms=(time.perf_counter() - t0) * 1000.0,
        sparse_source=source,
    )


# --- fan-out -----------------------------------------------------------------


@dataclass
class CollectionResult:
    collection: str
    profile: CollectionProfile
    hits: list[Hit] = field(default_factory=list)
    mode: str = "skipped"  # hybrid | dense | skipped
    ok: bool = False
    # Set explicitly rather than inferred from `profile.exists`: a collection
    # that blew the deadline has no probed profile either, and reporting a
    # stalled collection as "not built yet" would send whoever is debugging it
    # to the wrong pipeline.
    status: str = "failed"  # ok | missing | failed
    error: str | None = None
    elapsed_ms: float = 0.0
    sparse_source: str = ""
    attempts: int = 0


FilterFor = Callable[[CollectionProfile], "qm.Filter | None"]


_UNKNOWN = CollectionProfile("", None, False, False, False, True, "not probed")

# Transport-level symptoms of a connection the shared httpx pool handed out
# after something else closed it. Retrying these once costs ~10 ms and turns a
# degraded answer into a correct one. Deliberately *not* retried: a timeout
# (the 60 s budget is already spent, and a retry doubles a stall into a
# user-visible outage) and an UnexpectedResponse (the server answered -- a 404
# for a collection that was rebuilt under us is real, and hammering it does
# not make it exist).
_RETRYABLE = (
    "bad file descriptor",
    "connection reset",
    "server disconnected",
    "readerror",
    "writeerror",
    "connecterror",
    "remoteprotocolerror",
)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, CollectionMissing):
        return False
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "timed out" in msg or "timeout" in msg or "unexpectedresponse" in msg:
        return False
    return any(h in msg for h in _RETRYABLE)


def _search_one(
    name: str, vectors: QueryVectors, limit: int, filter_for: FilterFor
) -> CollectionResult:
    # The whole body is inside the try, capability probe included. Measured
    # 2026-08-06: eight concurrent searches across seven collections through
    # the shared QdrantClient produced a transient
    # `ResponseHandlingException: [Errno 9] Bad file descriptor` from the
    # connection pool. With the probe outside the guard that killed the entire
    # request; inside it, one collection reports failed and the other six still
    # answer. A fan-out whose failure mode is all-or-nothing is worse than no
    # fan-out.
    res = CollectionResult(collection=name, profile=_UNKNOWN)
    t0 = time.perf_counter()
    try:
        prof = profile(name)
        res.profile = prof
        if not prof.exists:
            # Degrade, do not fail. `kb_units` is built by a separate pipeline
            # and its absence must not take the whole retrieval path down; the
            # caller sees it in SearchResult.collections_missing.
            res.error = prof.error or "missing"
            res.status = "missing"
            return res  # elapsed_ms is still set by the finally below
        qfilter = filter_for(prof)
        sparse, source = vectors.sparse, vectors.sparse_source
        if prof.sparse_compatible:
            # Per-collection IDF. Cheap: a dozen terms of pure Python against
            # an already-loaded table.
            encoder, source = bm25_for(prof)
            sparse = encoder.encode_query(vectors.text)
        if prof.sparse_compatible and sparse.indices:
            # sparse_source is only reported for collections that actually used
            # it -- a sparse source on a dense-only search is a misleading
            # trace line.
            res.sparse_source = source
            res.mode = "hybrid"

            def run():
                return hybrid_search(
                    prof.physical or name,
                    dense=vectors.dense,
                    sparse_indices=sparse.indices,
                    sparse_values=sparse.values,
                    limit=limit,
                    query_filter=qfilter,
                )
        else:
            res.mode = "dense"

            def run():
                return dense_search(
                    prof.physical or name,
                    dense=vectors.dense,
                    limit=limit,
                    query_filter=qfilter,
                    vector_name=config.DENSE_VECTOR if prof.named_vectors else None,
                )

        for attempt in range(2):
            res.attempts = attempt + 1
            try:
                with _inflight:
                    res.hits = run()
                break
            except Exception as exc:
                if attempt == 0 and _is_retryable(exc):
                    continue
                raise
        res.ok = True
        res.status = "ok"
    except Exception as exc:  # one bad collection must not empty the result set
        res.error = f"{type(exc).__name__}: {exc}"
    finally:
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return res


def fan_out(
    collections: Sequence[str],
    vectors: QueryVectors,
    *,
    limit: int,
    filter_for: FilterFor,
    deadline: float | None = None,
) -> list[CollectionResult]:
    """Search every collection concurrently with the already-encoded query.

    ``filter_for`` is a callable rather than a filter so the server-side floor
    can be built per collection: the strict form where the gate fields exist,
    the defensive form where they do not.

    Returns in the order requested, always one result per collection. A
    collection that did not answer inside ``deadline`` comes back with
    ``ok=False`` rather than holding up the ones that did.
    """
    names = list(dict.fromkeys(collections))
    if not names:
        return []
    budget = FANOUT_DEADLINE if deadline is None else deadline
    pool = _executor()
    futures = {
        pool.submit(_search_one, n, vectors, limit, filter_for): n for n in names
    }
    wait(list(futures), timeout=budget)
    out: dict[str, CollectionResult] = {}
    for fut, name in futures.items():
        if fut.done():
            try:
                out[name] = fut.result()
            except Exception as exc:  # _search_one already guards; belt and braces
                out[name] = CollectionResult(
                    collection=name, profile=_UNKNOWN, error=f"{type(exc).__name__}: {exc}"
                )
        else:
            out[name] = CollectionResult(
                collection=name,
                profile=_UNKNOWN,
                error=f"deadline exceeded after {budget:.1f}s",
                elapsed_ms=budget * 1000.0,
            )
    return [out[n] for n in names]


if __name__ == "__main__":  # self-check against the live cluster
    print(f"qdrant={config.QDRANT_URL}")
    for name in [
        config.ALIAS_KB_UNITS,
        *config.LEGACY_COLLECTIONS,
    ]:
        p = profile(name)
        print(
            f"  {name:36s} exists={p.exists!s:5s} named={p.named_vectors!s:5s} "
            f"sparse_ok={p.sparse_compatible!s:5s} gated={p.gate_bearing!s:5s} "
            f"-> {p.physical}"
        )

    assert profile(config.ALIAS_KB_UNITS).gate_bearing, (
        "an absent collection must default to the strict floor"
    )

    qv = encode_query("does stage 1+ need a downpipe on a mk7 gti")
    print(f"  encode: {qv.encode_ms:.0f} ms  dense_dim={len(qv.dense)} "
          f"fallback_sparse_terms={len(qv.sparse.indices)} source={qv.sparse_source}")
    assert len(qv.dense) == config.EMBED_DIM
    assert qv.sparse.indices, "sparse encode produced no terms"

    live = [*config.NEW_ALIASES, *config.LEGACY_COLLECTIONS]
    t0 = time.perf_counter()
    results = fan_out(live, qv, limit=30, filter_for=lambda p: None)
    wall = (time.perf_counter() - t0) * 1000.0
    serial = sum(r.elapsed_ms for r in results)
    for r in results:
        src = Path(r.sparse_source).name if r.sparse_source else ""
        print(f"  {r.collection:36s} {r.mode:7s} ok={r.ok!s:5s} "
              f"hits={len(r.hits):3d} {r.elapsed_ms:7.1f} ms  {src}{r.error or ''}")
    print(f"  wall {wall:.0f} ms vs {serial:.0f} ms serial "
          f"({len(results)} collections)")
    assert any(r.ok and r.hits for r in results), "no live collection answered"
    assert wall < serial, "fan-out was not concurrent"
    # Every collection this project built must be searched hybrid, with its own
    # fitted statistics -- never the uniform-IDF fallback.
    for r in results:
        if r.ok and r.profile.sparse_compatible:
            assert r.mode == "hybrid", (r.collection, r.mode)
            assert r.sparse_source.startswith(str(BM25_DIR)), (
                f"{r.collection} fell back to {r.sparse_source}"
            )
    print("parallel.py self-check OK")
