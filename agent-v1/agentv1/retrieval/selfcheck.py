"""End-to-end proofs for the retrieval layer.

    .venv/bin/python -m agentv1.retrieval.selfcheck

Two halves.

**Fixture half.** ``kb_units`` does not exist yet -- the Phase 4a pipeline
builds it -- so the properties that only exist on gate-bearing collections
(the ``training_safe`` floor, supersession, ``internal_process`` exclusion,
revocation, occurrence boundedness) are proved against a fixture collection
built in an in-memory Qdrant. It is a real collection with real Qwen3 vectors,
real BM25 sparse vectors and the real cross-encoder; only the storage is
ephemeral. Nothing here touches the live cluster, which is exactly the point:
a test that proves the safety floor must not need write access to prove it.

**Live half.** Latency and real retrieval output against the populated legacy
collections on ``localhost:6333``.
"""

from __future__ import annotations

import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .. import config
from ..clients import qdrant as qc
from ..clients.embeddings import get_embedder
from ..clients.sparse import Bm25Encoder
from . import parallel, pipeline
from .rerank import get_reranker

# Named as a physical generation of the kb_units alias so the pipeline treats
# it as gate-bearing by exactly the rule production uses -- membership of
# config.NEW_ALIASES -- rather than by a test-only branch.
FIXTURE = f"{config.ALIAS_KB_UNITS}_{config.QWEN_SIZE}__selfcheck"

Q_DSE = "uniconnect cable not detected, windows flashing software driver signature"

UNITS = [
    {
        "unit_id": "u_call_dse",
        "kind": "troubleshooting",
        "title": "UniCONNECT+ cable not detected in Windows",
        "question": "The flashing software does not see my UniCONNECT+ cable. What do I do?",
        "answer": (
            "Disable Windows Driver Signature Enforcement, reinstall the "
            "UniCONNECT+ drivers and reconnect the cable."
        ),
        "hypothetical_questions": ["windows does not detect uniconnect cable"],
        "evidence": "single_call",
        "occurrences": 12,
        "training_safe": True,
        "status": "active",
        "language": "en",
    },
    {
        "unit_id": "u_case_secureboot",
        "kind": "troubleshooting",
        "title": "Driver signature workaround fails under Secure Boot",
        "question": "Driver Signature Enforcement is already disabled and the cable is still not detected.",
        "answer": (
            "Disabling Driver Signature Enforcement does not take effect while "
            "Secure Boot is enabled in the BIOS. Turn Secure Boot off first, "
            "then reinstall the UniCONNECT+ drivers."
        ),
        "hypothetical_questions": ["uniconnect cable still not detected secure boot"],
        "evidence": "multi_call_case",
        "outranks_call_units": True,
        "superseded_unit_ids": ["u_call_dse"],
        "occurrences": 1,
        "training_safe": True,
        "status": "active",
        "language": "en",
    },
    {
        # Maximally on-topic and NOT training_safe. If the floor ever leaks,
        # this is the unit that arrives first.
        "unit_id": "u_unsafe",
        "kind": "troubleshooting",
        "title": "UniCONNECT+ cable not detected by the Windows flashing software",
        "question": "uniconnect cable not detected windows flashing software driver signature",
        "answer": "Withheld: this unit failed the training_safe screen.",
        "occurrences": 40,
        "training_safe": False,
        "status": "active",
        "language": "en",
    },
    {
        "unit_id": "u_internal",
        "kind": "internal_process",
        "internal_only": True,
        "title": "Internal: escalating an undetected UniCONNECT+ cable",
        "question": "How does an agent escalate a cable that windows will not detect?",
        "answer": "Raise a tier-2 ticket and attach the driver signature logs.",
        "occurrences": 3,
        "training_safe": True,
        "status": "active",
        "language": "en",
    },
    {
        "unit_id": "u_revoked",
        "kind": "troubleshooting",
        "title": "UniCONNECT+ cable not detected -- revoked guidance",
        "question": "uniconnect cable not detected windows flashing software",
        "answer": "This guidance was revoked by the kb_revocations ledger.",
        "occurrences": 9,
        "training_safe": True,
        "status": "revoked",
        "language": "en",
    },
    {
        "unit_id": "u_singleton",
        "kind": "procedure",
        "title": "Remote cable reset for a bricked UniCONNECT+",
        "question": "How do I run a remote reset on a UniCONNECT+ cable?",
        "answer": "Book a remote reset session; the technician resets the cable over TeamViewer.",
        "occurrences": 1,
        "training_safe": True,
        "status": "active",
        "language": "en",
    },
    {
        "unit_id": "u_popular",
        "kind": "policy",
        "title": "Shipping times for UniGear apparel",
        "question": "How long does apparel shipping take?",
        "answer": "UniGear apparel ships in three to five business days.",
        "occurrences": 400,
        "training_safe": True,
        "status": "active",
        "language": "en",
    },
    {
        "unit_id": "u_price_stale",
        "kind": "pricing",
        "title": "UniFlex credit pricing",
        "question": "What does a UniFlex credit cost?",
        "answer": "A UniFlex credit is $500.",
        "occurrences": 6,
        "training_safe": True,
        "time_sensitive": True,
        "updated_at": "2024-02-01T00:00:00Z",
        "status": "active",
        "language": "en",
    },
]


def _point_texts(unit: dict) -> list[tuple[str, str]]:
    """The two points per unit from the contract: answer role and query role."""
    answer = "\n".join(
        str(unit[f])
        for f in ("title", "question", "answer", "conditions")
        if unit.get(f)
    )
    query = " ".join([unit.get("question", "")] + list(unit.get("hypothetical_questions", [])))
    return [("answer", answer), ("query", query or unit.get("question", ""))]


@contextmanager
def memory_cluster():
    """Swap the shared Qdrant client for an in-memory one, then put it back.

    The retrieval layer holds a process-global client by design; the honest way
    to test against a different cluster is to swap it and restore it, not to
    thread a client parameter through every call site so that tests can reach
    it.
    """
    saved_client = qc._client
    qc._client = QdrantClient(":memory:")
    parallel.clear_profile_cache()
    parallel.clear_bm25_cache()
    try:
        yield qc._client
    finally:
        # The BM25 caches are cleared on the way out as well as the way in: the
        # fixture fits its own encoder over eight documents, and leaving that
        # cached would have the live half querying the real cluster with eight
        # documents' worth of IDF.
        qc._client = saved_client
        parallel.clear_profile_cache()
        parallel.clear_bm25_cache()


def build_fixture(client: QdrantClient) -> None:
    emb = get_embedder()
    texts, meta = [], []
    for unit in UNITS:
        for role, text in _point_texts(unit):
            texts.append(text)
            meta.append((unit, role, text))
    dense = emb.embed_documents(texts)

    encoder = Bm25Encoder().fit(texts)
    # Point the query encoder at the same statistics: index/query BM25 symmetry
    # is the whole reason the analyzer is owned in-repo. The fixture has no
    # file under data/bm25/, so it resolves through the global fallback.
    parallel.clear_bm25_cache()
    parallel._bm25_cache = (time.time() + 3600, encoder, "selfcheck")

    qc.create_hybrid_collection(FIXTURE, recreate=True)
    points = []
    for i, (unit, role, text) in enumerate(meta):
        sparse = encoder.encode_document(text)
        payload = dict(unit)
        payload["point_role"] = role
        points.append(
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{unit['unit_id']}/{role}")),
                vector={
                    config.DENSE_VECTOR: [float(x) for x in dense[i]],
                    config.SPARSE_VECTOR: qm.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    ),
                },
                payload=payload,
            )
        )
    qc.upsert(FIXTURE, points, wait=True)


def _ids(res) -> list[str]:
    return [u.unit_id for u in res.units]


def run_fixture_checks() -> None:
    with memory_cluster() as client:
        build_fixture(client)
        prof = parallel.profile(FIXTURE)
        assert prof.exists and prof.gate_bearing and prof.sparse_compatible, prof
        print(f"fixture {FIXTURE}: {qc.points_count(FIXTURE)} points, "
              f"gate_bearing={prof.gate_bearing} sparse={prof.sparse_compatible}")

        vecs = parallel.encode_query(Q_DSE)

        # --- 1. Without the floor, the unsafe unit wins -------------------
        raw = qc.hybrid_search(
            FIXTURE,
            dense=vecs.dense,
            sparse_indices=vecs.sparse.indices,
            sparse_values=vecs.sparse.values,
            limit=10,
        )
        raw_ids = [h.payload["unit_id"] for h in raw]
        assert "u_unsafe" in raw_ids, raw_ids
        print(f"\n1 unfiltered hybrid (no floor): {raw_ids[:5]}")
        print("  -> u_unsafe is retrievable and ranks "
              f"#{raw_ids.index('u_unsafe') + 1}. The floor is the only thing "
              "stopping it.")

        # --- 2. The floor holds against a hostile caller filter -----------
        hostile = qm.Filter(
            must=[qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=False))]
        )
        res = pipeline.search(
            Q_DSE, collections=[FIXTURE], extra_filter=hostile, top_k=8
        )
        assert "u_unsafe" not in _ids(res), _ids(res)
        assert res.units == [], _ids(res)
        assert res.trace["caller_filter_reserved_keys"] == ["training_safe"]
        print("\n2 caller passes extra_filter training_safe=False:")
        print(f"  units={_ids(res)}  (floor AND caller filter is unsatisfiable)")
        print(f"  trace.caller_filter_reserved_keys="
              f"{res.trace['caller_filter_reserved_keys']}")
        print(f"  trace.training_safe_floor={res.trace['training_safe_floor']!r}")

        # A caller filter that is merely wrong, not hostile, also cannot lift
        # the floor: the unsafe unit stays gone.
        wide = qm.Filter(
            should=[
                qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=False)),
                qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=True)),
            ]
        )
        res = pipeline.search(Q_DSE, collections=[FIXTURE], extra_filter=wide, top_k=8)
        assert "u_unsafe" not in _ids(res), _ids(res)
        print(f"  caller filter (safe OR unsafe): units={_ids(res)} -- still no u_unsafe")

        # --- 3. Default path: unsafe, internal and revoked all excluded ---
        res = pipeline.search(Q_DSE, collections=[FIXTURE], top_k=8)
        got = _ids(res)
        assert "u_unsafe" not in got and "u_revoked" not in got and "u_internal" not in got
        print(f"\n3 default search: {got}")
        print("  excluded: u_unsafe (training_safe=False), u_revoked (status), "
              "u_internal (internal_process)")

        # --- 4. Supersession hard-drops the call unit ---------------------
        assert "u_case_secureboot" in got, got
        assert "u_call_dse" not in got, got
        drops = [(d.unit_id, d.superseded_by) for d in res.dropped_superseded]
        assert ("u_call_dse", "u_case_secureboot") in drops, drops
        print(f"\n4 supersession: dropped={drops}")
        print("  the call unit that says 'disable Driver Signature Enforcement' is gone; "
              "the case unit that says it fails under Secure Boot survived.")
        for u in res.units[:3]:
            print(f"    {u.final_score:.4f} rr={u.rerank_score:+7.3f} {u.unit_id}")

        # --- 5. internal_process only with allow_internal -----------------
        res_i = pipeline.search(
            Q_DSE, collections=[FIXTURE], top_k=8, allow_internal=True
        )
        assert "u_internal" in _ids(res_i), _ids(res_i)
        print(f"\n5 allow_internal=True: {_ids(res_i)}")

        # --- 6. Dedupe: two points per unit, one result -------------------
        counts = {u.unit_id: len(u.point_ids) for u in res.units}
        assert len(_ids(res)) == len(set(_ids(res))), _ids(res)
        assert any(v == 2 for v in counts.values()), counts
        print(f"\n6 dedupe on unit_id: points collapsed per unit = {counts}")

        # --- 7. Occurrence is a bounded boost, not a sort key -------------
        res_o = pipeline.search(
            "how long does apparel shipping take", collections=[FIXTURE], top_k=8
        )
        by_id = {u.unit_id: u for u in res_o.units}
        pop = by_id["u_popular"]
        assert pop.adjustments["occurrence_boost"] <= pipeline.OCC_BOOST_MAX
        assert abs(pop.adjustments["applied"]) <= pipeline.ADJUSTMENT_CAP
        print(f"\n7 occurrence: u_popular occ=400 boost="
              f"{pop.adjustments['occurrence_boost']:.4f} "
              f"(cap {pipeline.OCC_BOOST_MAX}), relevance={pop.relevance:.4f} "
              f"final={pop.final_score:.4f}")
        # On an unrelated query, 400 occurrences must not buy rank 1.
        res_s = pipeline.search(
            "how do I run a remote reset on my cable", collections=[FIXTURE], top_k=8
        )
        assert _ids(res_s)[0] == "u_singleton", _ids(res_s)
        top, popular = res_s.units[0], {u.unit_id: u for u in res_s.units}["u_popular"]
        print(f"  singleton (occ=1) rr={top.rerank_score:+.3f} final={top.final_score:.4f} "
              f"beats occ=400 rr={popular.rerank_score:+.3f} "
              f"final={popular.final_score:.4f}")

        # --- 8. Decay only where time_sensitive ---------------------------
        res_p = pipeline.search(
            "what does a uniflex credit cost", collections=[FIXTURE], top_k=8
        )
        stale = {u.unit_id: u for u in res_p.units}["u_price_stale"]
        assert stale.adjustments["time_decay"] < 0
        others = [u for u in res_p.units if not u.payload.get("time_sensitive")]
        assert all(u.adjustments["time_decay"] == 0.0 for u in others)
        print(f"\n8 decay: u_price_stale (time_sensitive, 2024) "
              f"{stale.adjustments['time_decay']:+.4f}; "
              f"{len(others)} non-sensitive units decayed by 0.0000")

        # --- 9. A missing collection degrades, it does not fail -----------
        res_m = pipeline.search(
            Q_DSE, collections=[FIXTURE, config.ALIAS_CASE_NARRATIVES], top_k=5
        )
        assert res_m.collections_missing == [config.ALIAS_CASE_NARRATIVES]
        assert res_m.units
        print(f"\n9 degrade: missing={res_m.collections_missing} "
              f"searched={res_m.collections_searched} units={len(res_m.units)}")


def run_live_checks() -> None:
    print("\n" + "=" * 72)
    print(f"LIVE: {config.QDRANT_URL}")
    parallel.clear_profile_cache()

    rr = get_reranker()
    print(f"reranker cold start: {rr.warmup():.2f}s ({rr.device}/{rr.dtype})")
    parallel.encode_query("warmup")  # pay the embedder load before timing

    q = "my uniconnect cable is not recognised by the flashing software on windows"
    res = pipeline.search(q, persona="support", top_k=5)
    print(f"\nquery: {q!r}")
    print(f"routed={res.trace['routed']}")
    print(f"missing={res.collections_missing} failed={res.collections_failed}")
    print(f"modes={res.trace['modes']}")
    print(f"alias_expansions={res.trace['alias_expansions']} "
          f"markers={res.trace['marker_tokens']}")
    for coll, src in res.trace["sparse_source"].items():
        print(f"  bm25[{coll}] = {src}")
    print(f"candidates={res.trace['candidates']} -> reranked "
          f"{res.trace['rerank_candidates']} -> {len(res.units)}")
    for u in res.units:
        print(f"  {u.final_score:.4f} rr={u.rerank_score:+7.3f} "
              f"[{','.join(c.replace('unitronic_', '').replace('_0_6b', '') for c in u.collections)}]")
        print(f"      {u.text[:150].replace(chr(10), ' | ')!r}")

    # Steady-state latency: the shape a request actually has.
    print("\nsteady-state latency (5 runs, support route, top_k=5):")
    runs = [pipeline.search(q, persona="support", top_k=5).timings_ms for _ in range(5)]
    for stage in ("encode", "fanout", "supersede", "rerank", "total"):
        vals = [r[stage] for r in runs if stage in r]
        if vals:
            print(f"  {stage:10s} median {statistics.median(vals):7.1f} ms  "
                  f"min {min(vals):7.1f}  max {max(vals):7.1f}")

    # Concurrency: the reranker is one shared model behind a lock and the
    # fan-out is a thread pool. Eight simultaneous searches must all succeed
    # and must all agree, or the "lazy, thread-safe" claim is decoration.
    with ThreadPoolExecutor(max_workers=8) as pool:
        t0 = time.perf_counter()
        concurrent = [
            f.result()
            for f in [pool.submit(pipeline.search, q, persona="support", top_k=5)
                      for _ in range(8)]
        ]
        wall = (time.perf_counter() - t0) * 1000.0
    # Agreement is asserted only across runs that saw the same collections.
    # Under 8x7 concurrent requests the shared Qdrant connection pool does
    # occasionally drop one, and the designed response to that is a degraded
    # result, not a failed request -- so a strict equality assertion here would
    # be asserting that the degradation never happens, which is not true and
    # not what we want.
    groups: dict[tuple, set[tuple]] = {}
    for r in concurrent:
        assert r.units, "a concurrent search returned nothing"
        groups.setdefault(tuple(r.collections_searched), set()).add(
            tuple(u.unit_id for u in r.units)
        )
    degraded = [r for r in concurrent if r.collections_failed]
    for cols, id_sets in groups.items():
        assert len(id_sets) == 1, (cols, id_sets)
    print(f"\nconcurrency: 8 simultaneous searches in {wall:.0f} ms wall, "
          f"{len(groups)} distinct collection set(s), deterministic within each; "
          f"{len(degraded)} degraded")
    for r in degraded:
        print(f"  degraded run: failed={r.collections_failed} "
              f"({list(r.trace['collection_errors'].values())})")

    # Reranker latency across the document-length range, measured on real
    # retrieved text -- a synthetic string of repeated characters tokenises
    # nothing like prose and understates the cost by 3x.
    qv = parallel.encode_query(q)
    fan = parallel.fan_out(
        [config.LEGACY_COLLECTIONS[0]], qv, limit=30, filter_for=lambda p: None
    )
    corpus = [pipeline.unit_text(h.payload) for h in fan[0].hits]
    tok = rr._tok
    print("\nreranker, 30 real pairs (median of 7):")
    for label, cut in [
        ("kb_unit median (336 chars)", 336),
        ("kb_unit p95 (591 chars)", 591),
        ("legacy page (full)", 10**6),
    ]:
        docs = [d[:cut] for d in corpus]
        ntok = statistics.median(len(tok(q, d)["input_ids"]) for d in docs)
        for _ in range(2):
            rr.score(q, docs)
        s = []
        for _ in range(7):
            t0 = time.perf_counter()
            rr.score(q, docs)
            s.append((time.perf_counter() - t0) * 1000)
        print(f"  {label:28s} {ntok:4.0f} tok/pair  "
              f"median {statistics.median(s):7.1f} ms  min {min(s):7.1f}")


if __name__ == "__main__":
    run_fixture_checks()
    run_live_checks()
    print("\nselfcheck OK")
