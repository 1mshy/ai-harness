"""Runnable self-check for the Phase 4b projection.

``python -m agentv1.index.selfcheck`` — no pytest required, and it runs against
the live systems rather than fixtures, because the failures this guards against
(a payload key that leaks, an IDF table that disagrees with itself, a delete
filter that matches nothing) are all failures of contact with reality.

The centrepiece is :func:`check_allowlist`. It takes a knowledge unit doctored
with exactly the four identifiers AGENT_PLAN.md §3.1 found live —
``file_name``, ``agent_name``, a phone number and ``thread_path`` — placed both
as top-level keys *and* smuggled inside permitted text, and asserts that none of
them can reach a payload by any route.
"""

from __future__ import annotations

import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from typing import Callable

from qdrant_client import models as qm

from .. import config
from ..clients import qdrant as q
from ..clients.mongo import kb_db, put_state, source_db
from ..clients.sparse import Bm25Encoder
from . import build_kb_index as B
from . import payload as P
from . import revoke as R

# The live leak, verbatim from a real 3CX recording filename plus the Instagram
# handle shape found on 100% of sampled points in the sibling collection.
LEAKED_FILENAME = "[Coles, Zoe]_124-8012307610_20250603202223(152).wav"
LEAKED_HANDLE = "inbox/some_customer_handle"
LEAKED_PHONE = "801-230-7610"
LEAKED_AGENT = "Zoe"
# A courtesy title plus a surname carries no digits and no filename, so nothing
# else in the scrubber catches it. One reached a live call_residual payload
# ("Mr. Jelinski"), sourced from the LLM-written `problem` field.
LEAKED_SURNAME = "Mr. Jelinski"

DOCTORED_UNIT = {
    "unit_id": "u_0123456789abcdef",
    "point_role": "answer",
    "doc_type": "kb_unit",
    "kind": "troubleshooting",
    "title": "Stage 1+ flash fails on a cold ECU",
    "question": "Why does my Stage 1+ flash fail?",
    "answer": (
        "Let the ECU cool, then retry. Called back from "
        f"{LEAKED_PHONE} about {LEAKED_FILENAME}. Follow up with {LEAKED_SURNAME}."
    ),
    "conditions": f"Reported via {LEAKED_HANDLE}",
    "hypothetical_questions": ["flash fails", f"see {LEAKED_FILENAME}"],
    "source_ids": ["6a6bbf972bd05de65c37af0b"],
    "training_safe": True,
    # the four keys the replaced collection actually published
    "file_name": LEAKED_FILENAME,
    "agent_name": LEAKED_AGENT,
    "caller_area_code": "801",
    "caller_phone_number": "8012307610",
    "thread_path": LEAKED_HANDLE,
    "phone": LEAKED_PHONE,
    # and the second copy, which is what defeated the last fix
    "_node_content": json.dumps({"metadata": {"file_name": LEAKED_FILENAME}}),
    "customer_match": {"name": "Gary Vlahakis", "phone": "2133933153"},
}

FORBIDDEN_SUBSTRINGS = (
    "8012307610", "801-230-7610", "801 230 7610", ".wav",
    "inbox/", "Coles", "Zoe", "Vlahakis", "2133933153", "Jelinski",
)


def _flatten(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out += _flatten(item)
        return out
    return [str(value)]


# --- checks ------------------------------------------------------------------
def check_allowlist() -> None:
    out = P.project(DOCTORED_UNIT)

    for key in ("file_name", "agent_name", "caller_area_code", "caller_phone_number",
                "thread_path", "phone", "_node_content", "customer_match"):
        assert key not in out, f"{key} survived the allowlist"

    blob = json.dumps(out)
    for needle in FORBIDDEN_SUBSTRINGS:
        assert needle not in blob, f"{needle!r} survived into {blob}"

    # The unit is still *useful* afterwards -- an allowlist that empties the
    # payload is not a fix, it is a regression with good press.
    assert out["title"].startswith("Stage 1+"), out
    assert "cool" in out["answer"]
    assert out["training_safe"] is True
    assert out["unit_id"] == "u_0123456789abcdef"
    assert out["source_ids"] == ["6a6bbf972bd05de65c37af0b"]
    print(f"  allowlist: {len(out)} keys published, {sorted(out)}")


def check_no_nested_containers() -> None:
    for bad in ({"title": {"a": 1}}, {"products_applicable": [{"sku": "x"}]}):
        try:
            P.project(bad)
        except P.PayloadLeak:
            continue
        raise AssertionError(f"nested container accepted: {bad}")
    print("  nested dict / list-of-dict on a permitted key: refused")


def check_opaque_keys_survive() -> None:
    """A 24-digit ObjectId must not be mangled by the digit-run scrubber.

    It happens roughly once in 250k ids, and the failure mode is a Law 25
    deletion request that silently matches nothing.
    """
    oid = "1" * 24
    out = P.project({"unit_id": "c_" + "9" * 16, "source_ids": [oid], "training_safe": True})
    assert out["source_ids"] == [oid], out
    assert out["unit_id"] == "c_" + "9" * 16
    # ...while the same digits inside free text are redacted.
    scrubbed = P.project({"answer": f"call {oid} back", "training_safe": True})
    assert oid not in scrubbed["answer"], scrubbed
    print(f"  opaque ids preserved ({oid[:8]}...), same digits in text redacted")


def check_point_ids() -> None:
    a = P.point_uuid("u_abc", "answer")
    b = P.point_uuid("u_abc", "query")
    assert a != b
    assert a == str(uuid.uuid5(uuid.NAMESPACE_URL, "u_abc:answer"))
    assert P.point_uuid("u_abc", "answer") == a, "point ids must be deterministic"
    assert P.unit_id_for_call("6a6bbf972bd05de65c37af0b").startswith("c_")
    assert P.unit_id_for_platform_stage(1, 48) == P.unit_id_for_platform_stage(1, 48)
    print(f"  point ids deterministic: u_abc:answer -> {a}")


def check_text_composition() -> None:
    ans = P.answer_text(DOCTORED_UNIT)
    qry = P.query_text(DOCTORED_UNIT)
    assert "Stage 1+ flash fails" in ans
    assert "Why does my Stage 1+ flash fail?" in qry
    assert "flash fails" in qry, "hypothetical_questions must reach the query point"
    # The alias table maps "flash" -> "ecu tune"; the query point should be
    # findable from either side of that mapping.
    assert "ecu tune" in qry.lower() or "flash" in qry.lower()
    # The hypothetical questions carry the leaked filename too; the query point
    # is a payload like any other and must be scrubbed on the way out.
    body = P.project({"unit_id": "u_x", "point_role": "query", "text": qry, "training_safe": True})
    assert ".wav" not in body["text"] and "Coles" not in body["text"], body
    print(f"  answer_text {len(ans)} chars, query_text {len(qry)} chars, "
          f"query payload clean")


def check_bm25_symmetry() -> None:
    """Index-time and query-time IDF must come from the same fit."""
    corpus = ["stage 1+ tune for golf r", "stage 1 tune for golf r", "uniconnect plus module"]
    enc = Bm25Encoder().fit(corpus)
    path = B.bm25_stats_path("selfcheck_tmp")
    B.BM25_DIR.mkdir(parents=True, exist_ok=True)
    enc.save(path)
    reloaded = Bm25Encoder.load(path)
    path.unlink()
    a = enc.encode_query("stage 1+ golf r")
    b = reloaded.encode_query("stage 1+ golf r")
    assert a.indices == b.indices and a.values == b.values, "IDF did not round-trip"
    doc = enc.encode_document("stage 1+ tune for golf r")
    plus_term = {i: v for i, v in zip(doc.indices, doc.values)}
    from ..clients.sparse import term_id

    assert term_id("stage_1_plus") in plus_term, "the + marker token did not survive"
    assert term_id("stage_1") not in plus_term, "Stage 1+ collapsed into Stage 1"
    print("  BM25 round-trips and keeps stage_1_plus distinct from stage_1")


def check_kb_unit_projector() -> None:
    """Exercise the kb_units projector while ``kb_units`` is still empty.

    Phase 4a has not published a generation yet, so the projector with the most
    schema surface would otherwise be the only one never run. The document below
    is CONTRACT.md's ``kb_units`` shape with the PII fields the source documents
    carry bolted back on, because that is what a schema drift would look like.
    """
    doc = {
        "unit_id": "u_abcdef0123456789",
        "gen": "a1b2c3d4e5f6",
        "kb_version": 1, "merge_version": 1,
        "kind": "compatibility",
        "title": "Stage 2 availability on 2.5TFSI EVO",
        "question": "Is Stage 2 available for a 2021 TTRS?",
        "answer": "Yes. Stage 2 requires the downpipe; Stage 1+ does not.",
        "conditions": "2021 TTRS, 2.5TFSI EVO",
        "vehicles_applicable": ["2021 Audi TTRS"],
        "products_applicable": ["Stage 2 ECU tune"],
        "hypothetical_questions": ["can I run stage 2 on my ttrs", "do I need a downpipe"],
        "evidence": "multi_call_case", "outranks_call_units": True,
        "superseded_unit_ids": ["u_1111111111111111"],
        "case_id": "81e5f6a793a43afd",
        "confidence": "high", "occurrences": 7,
        "source_ids": ["6a6bbf972bd05de65c37af0b", "6a6bc0ce2bd05de65c37af10"],
        "cluster_id": "cl_7", "merge_action": "merged",
        "training_safe": True, "time_sensitive": False, "emissions_risk": False,
        "safety_gated": False, "dealer_pricing": False, "internal_only": False,
        "contains_price": False,
        "department": "sales", "language": "en", "technical_category": "compatibility",
        "created_at": "2026-08-05T00:00:00Z", "updated_at": "2026-08-05T00:00:00Z",
        "status": "active",
        # drift that must not survive
        "file_name": LEAKED_FILENAME, "agent_name": LEAKED_AGENT,
        "supersedes_calls": [LEAKED_FILENAME],
    }
    records = list(B.kb_records_for(doc))
    roles = [r.role for r in records]
    assert roles == ["answer", "query"], roles
    assert len({r.point_id for r in records}) == 2, "answer and query collided"
    for rec in records:
        body = P.project(rec.candidate)
        P.assert_no_leak(body)
        for needle in FORBIDDEN_SUBSTRINGS:
            assert needle not in json.dumps(body), (needle, body)
        assert body["source_ids"] == doc["source_ids"]
        assert body["superseded_unit_ids"] == ["u_1111111111111111"]
    answer, query = records
    assert "downpipe" in answer.text
    assert "can I run stage 2 on my ttrs" in query.text
    print(f"  kb_units projector: 2 points, roles {roles}, "
          f"answer {len(answer.text)} chars / query {len(query.text)} chars")


def check_live_projections() -> None:
    """Every projector must produce a payload that passes the leak assertion."""
    for proj in B.PROJECTIONS:
        fp = proj.fingerprint()
        records = list(proj.records(25))
        if not records:
            print(f"  {proj.name:<16} source n={fp['n']} -> no records ({proj.empty_hint[:60]})")
            continue
        for rec in records:
            body = P.project(rec.candidate)
            P.assert_no_leak(body)
            assert body.get("unit_id"), rec
            assert body.get("text"), rec
            assert "training_safe" in body
        print(f"  {proj.name:<16} source n={fp['n']:<6} sampled {len(records)} records, "
              f"payload keys {len(P.project(records[0].candidate))}")


def check_no_pii_in_live_payloads() -> None:
    """Re-run the leak scan over the four live aliases, if they exist."""
    client = q.get_client()
    for alias in config.NEW_ALIASES:
        try:
            target = q.open_collection(alias)
        except q.CollectionMissing:
            print(f"  {alias:<28} not built yet")
            continue
        points, _ = client.scroll(collection_name=target, limit=500, with_payload=True)
        for pt in points:
            P.assert_no_leak(pt.payload or {})
            blob = " ".join(_flatten(list((pt.payload or {}).values())))
            assert ".wav" not in blob, f"{alias} point {pt.id} carries a recording filename"
            assert "inbox/" not in blob, f"{alias} point {pt.id} carries a handle"
        print(f"  {alias:<28} {len(points)} sampled points clean "
              f"({q.points_count(target)} total)")


def check_build_lease() -> None:
    """A second build of the same alias must refuse rather than race.

    Observed during bring-up: two concurrent ``--refresh`` runs of the same
    alias, and the loser's swap deleted the winner's collection, leaving a
    300-point index behind an alias that reported success.
    """
    alias = "agentv1_selfcheck_lease"
    key = B._progress_key(alias)
    proj = B.Projection(
        name="lease_probe", alias=alias, describe="lease probe",
        fingerprint=lambda: {"n": 1, "latest": None},
        records=lambda _limit: iter(()),
    )
    put_state(key, {
        "status": "building", "gen": "abc123456789", "committed": 7,
        "owner": "someone-else:1", "heartbeat": datetime.now(timezone.utc).isoformat(),
    })
    try:
        result = B.build(proj, log=lambda *_: None)
        assert result["status"] == "locked", result
        assert result["lease_holder"] == "someone-else:1", result
        # ...and a stale lease is ignored, so a killed build cannot wedge it.
        put_state(key, {"heartbeat": "2020-01-01T00:00:00+00:00"})
        stale = B.build(proj, log=lambda *_: None)
        assert stale["status"] != "locked", stale
        print(f"  build lease: live lease refused ({result['status']}), "
              f"stale lease ignored ({stale['status']})")
    finally:
        kb_db()[config.COLL_KB_STATE].delete_one({"key": key})


def check_lease_claimed_before_writing() -> None:
    """The lease must be claimed *before* the first mutating call, not after.

    The top-of-``build`` check runs before ``proj.records(...)`` is materialised,
    which for the residual is a multi-minute Mongo scan. A second build starting
    inside that window has already passed the same check, so a claim written
    only after ``create_hybrid_collection`` still leaves both builds creating a
    generation and racing on the swap -- the failure that truncated
    case_narratives to 300 points during bring-up. The projector below plants a
    foreign lease *while it is being iterated*, which is precisely that window.
    """
    alias = "agentv1_selfcheck_lease2"
    key = B._progress_key(alias)

    def _records(_limit):
        put_state(key, {
            "status": "building", "gen": "gtest00000000", "committed": 0,
            "owner": "racer:2", "heartbeat": datetime.now(timezone.utc).isoformat(),
        })
        uid = "c_" + "0" * 16
        yield B.Record(uid, "answer", "lease race probe",
                       {"unit_id": uid, "text": "lease race probe", "training_safe": True})

    proj = B.Projection(
        name="lease_race", alias=alias, describe="lease race probe",
        fingerprint=lambda: {"n": 1, "latest": None}, records=_records,
    )
    try:
        result = B.build(proj, log=lambda *_: None)
        assert result["status"] == "locked", result
        assert result["lease_holder"] == "racer:2", result
        leftovers = [
            c.name for c in q.get_client().get_collections().collections
            if c.name.startswith(alias)
        ]
        assert not leftovers, f"a collection was created despite a live lease: {leftovers}"
        print("  lease claimed before the first write: race mid-projection refused, "
              "nothing created")
    finally:
        kb_db()[config.COLL_KB_STATE].delete_one({"key": key})
        for c in q.get_client().get_collections().collections:
            if c.name.startswith(alias):
                q.get_client().delete_collection(c.name)


def check_revocation_cycle() -> None:
    """End-to-end delete on a throwaway collection we own outright.

    Deliberately not run against a live alias: this repo's rule is that only
    ``ops/`` mutates existing collections, and it snapshots first.
    """
    name = f"agentv1_selfcheck__{uuid.uuid4().hex[:8]}"
    q.create_hybrid_collection(name)
    try:
        unit_ids = ["c_aaaaaaaaaaaaaaaa", "c_bbbbbbbbbbbbbbbb", "n_cccccccccccccccc"]
        points = [
            qm.PointStruct(
                id=P.point_uuid(uid, "answer"),
                vector={config.DENSE_VECTOR: [0.1] * config.EMBED_DIM},
                payload=P.project({"unit_id": uid, "text": "selfcheck", "training_safe": True}),
            )
            for uid in unit_ids
        ]
        q.upsert(name, points, wait=True)
        assert q.points_count(name) == 3, q.points_count(name)
        deleted = q.delete_by_unit_ids(name, unit_ids[:1])
        assert deleted == 1
        assert q.points_count(name) == 2, q.points_count(name)
        # Idempotent: applying the same revocation twice removes nothing more.
        q.delete_by_unit_ids(name, unit_ids[:1])
        assert q.points_count(name) == 2
        grouped = R._aliases_for(unit_ids, None)
        assert grouped[config.ALIAS_CALL_RESIDUAL] == unit_ids[:2], grouped
        assert grouped[config.ALIAS_CASE_NARRATIVES] == unit_ids[2:], grouped
        print(f"  revoke: 3 points -> deleted 1 -> {q.points_count(name)} left, "
              f"prefix routing {[(k.split('unitronic_')[-1], len(v)) for k, v in grouped.items()]}")
    finally:
        q.get_client().delete_collection(name)


def check_revocation_resolution() -> None:
    """A real source _id must resolve to the unit ids that publish it."""
    doc = source_db()[config.COLL_ANALYSIS].find_one(B._RESIDUAL_FILTER, {"_id": 1})
    if not doc:
        print("  revoke resolution: no residual document available")
        return
    sid = str(doc["_id"])
    ids = R.unit_ids_for_source(sid)
    assert P.unit_id_for_call(sid) in ids, ids
    counts = R.sweep(dry_run=True, log=lambda *_: None)
    print(f"  revoke resolution: source {sid} -> {sorted(ids)}; "
          f"a sweep today would record {sum(counts.values())} entries {counts}")


CHECKS: list[Callable[[], None]] = [
    check_allowlist,
    check_no_nested_containers,
    check_opaque_keys_survive,
    check_point_ids,
    check_text_composition,
    check_bm25_symmetry,
    check_kb_unit_projector,
    check_live_projections,
    check_no_pii_in_live_payloads,
    check_build_lease,
    check_lease_claimed_before_writing,
    check_revocation_cycle,
    check_revocation_resolution,
]


def main() -> int:
    failures = 0
    for check in CHECKS:
        print(f"{check.__name__}:")
        try:
            check()
        except Exception:
            failures += 1
            traceback.print_exc()
            print("  FAILED")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
