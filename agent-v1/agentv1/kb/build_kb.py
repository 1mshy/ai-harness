"""Phase 4a -- build ``transcribing.kb_units``, the canonical knowledge base.

Deliberately depends on nothing but stdlib and pymongo (plus the LLM client for
the merge pass). It owns *identity*: gates, dedup, merge, supersession,
tombstones, occurrences. It never touches Qdrant. Phase 4b
(``index/build_kb_index.py``) owns the projection and never runs an LLM. That
split means a Qdrant rebuild never re-runs a 4,000-call merge, and a merge
re-run never needs a vector store to be up.

Generation swap, in the order this repo's pipeline already uses everywhere
else: write the new generation, delete the old generation, stamp state last.
A crash before the stamp leaves two generations present and the state pointing
at the old one, which the next run cleans up. The reverse order would leave the
state pointing at a generation that had already been deleted.

    python -m agentv1.kb.build_kb --refresh
    python -m agentv1.kb.build_kb --limit 2000 --no-merge   # fast dry run
    python -m agentv1.kb.build_kb --revoke                  # apply the ledger
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

from pymongo import UpdateOne

from .. import config
from ..clients.llm import get_llm
from ..clients.mongo import ensure_kb_indexes, get_state, kb_db, put_state, source_db
from . import dedup, extract, merge, supersession

STATE_KEY = "kb_units_build"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _generation(units: list) -> str:
    """Generation id from the input set, so an identical corpus is idempotent."""
    h = hashlib.blake2b(digest_size=6)
    h.update(f"{config.KB_VERSION}:{config.MERGE_VERSION}:{len(units)}".encode())
    for u in sorted(units, key=lambda x: x.unit_id)[:5000]:
        h.update(u.unit_id.encode())
    return h.hexdigest()


def _canonical_id(members: list, index: int) -> str:
    """Canonical id keyed on the *lowest* member unit id, so a canonical unit
    keeps its identity across rebuilds as long as its founding member survives.
    """
    anchor = min(m.unit_id for m in members)
    digest = hashlib.blake2b(f"{anchor}:{index}".encode(), digest_size=8).hexdigest()
    return f"u_{digest}"


def build(
    *,
    limit: int | None = None,
    do_merge: bool = True,
    dry_run: bool = False,
    merge_workers: int | None = None,
) -> dict:
    started = time.time()
    extract.reset_drops()

    print("[1/6] extracting + gating", flush=True)
    raw = list(extract.iter_all_units(limit=limit))
    print(f"      {len(raw):,} units survived gates; drops={dict(extract.DROPS)}", flush=True)
    if not raw:
        raise SystemExit("no units survived the gates -- refusing to build an empty KB")

    print("[2/6] clustering", flush=True)
    clusters, cstats = dedup.cluster_units(raw)
    print(
        f"      {cstats.n_units:,} -> {cstats.n_clusters:,} clusters "
        f"({cstats.singletons:,} singletons, {cstats.pairs:,} pairs, "
        f"{cstats.triple_plus:,} of size>=3)",
        flush=True,
    )

    mergeable = [c for c in clusters if len(c) > 1]
    print(f"[3/6] merging {len(mergeable):,} clusters" + ("" if do_merge else " (SKIPPED)"), flush=True)

    outcomes: dict[int, merge.MergeOutcome] = {}
    if do_merge and mergeable:
        llm = get_llm()
        done = [0]
        # `done[0] += 1` from a thread pool is not atomic, so a bare
        # `% 250 == 0` check can skip the exact value and print nothing at all
        # for a two-hour run. Guard the counter.
        counter_lock = __import__("threading").Lock()

        def run(cluster_index: int):
            outcome = merge.merge_cluster(raw, clusters[cluster_index])
            with counter_lock:
                done[0] += 1
                n = done[0]
            if n % 100 == 0:
                pct = 100 * n / len(mergeable)
                print(f"      merged {n:,}/{len(mergeable):,} ({pct:.0f}%)", flush=True)
            return cluster_index, outcome

        indices = [i for i, c in enumerate(clusters) if len(c) > 1]
        for result in llm.map_concurrent(indices, run, workers=merge_workers):
            if result:
                outcomes[result[0]] = result[1]

    print("[4/6] assembling canonical units", flush=True)
    canonical: list[dict] = []
    canonical_of: dict[str, str] = {}
    action_counts: dict[str, int] = {}

    for ci, cluster in enumerate(clusters):
        members = [raw[i] for i in cluster]
        outcome = outcomes.get(ci)
        if outcome is None:
            outcome = merge.MergeOutcome(
                units=[
                    merge.MergedUnit(
                        title=members[0].title, question=members[0].question,
                        answer=members[0].answer, conditions=members[0].conditions,
                        kind=members[0].kind,
                        hypothetical_questions=list(members[0].hypothetical_questions),
                        confidence=members[0].confidence,
                        source_indices=list(range(len(members))),
                    )
                ],
                action="verbatim" if len(members) == 1 else "unmerged",
            )
        action_counts[outcome.action] = action_counts.get(outcome.action, 0) + 1

        for oi, mu in enumerate(outcome.units):
            picked = [members[i] for i in mu.source_indices if 0 <= i < len(members)] or members
            unit_id = _canonical_id(picked, oi)
            for m in picked:
                canonical_of[m.unit_id] = unit_id

            # Union the evidence across members. A case unit anywhere in the
            # cluster promotes the whole canonical unit -- it was built from an
            # observed outcome, and that property should not be lost because a
            # single-call restatement happened to sort first.
            is_case = any(m.evidence == "multi_call_case" for m in picked)
            hqs: list[str] = list(mu.hypothetical_questions)
            for m in picked:
                hqs.extend(m.hypothetical_questions)
            seen: dict[str, None] = {}
            for q in hqs:
                q = q.strip()
                if q:
                    seen.setdefault(q, None)

            labels: dict[str, bool] = {}
            for key in (
                "emissions_risk", "safety_gated", "dealer_pricing", "contains_price",
                "internal_only", "time_sensitive", "stage_claim", "agent_uncertain",
            ):
                labels[key] = any(bool(m.labels.get(key)) for m in picked)

            langs = [m.language for m in picked if m.language]
            depts = [m.department for m in picked if m.department]
            cats = [m.technical_category for m in picked if m.technical_category]

            canonical.append(
                {
                    "unit_id": unit_id,
                    "kind": mu.kind or picked[0].kind,
                    "title": mu.title,
                    "question": mu.question,
                    "answer": mu.answer,
                    "conditions": mu.conditions,
                    "vehicles_applicable": sorted(
                        {v for m in picked for v in m.vehicles_applicable}
                    ),
                    "products_applicable": sorted(
                        {p for m in picked for p in m.products_applicable}
                    ),
                    "hypothetical_questions": list(seen)[:24],
                    "evidence": "multi_call_case" if is_case else "single_call",
                    "outranks_call_units": is_case,
                    "superseded_unit_ids": [],  # filled in step 5
                    "case_id": next((m.case_id for m in picked if m.case_id), None),
                    "confidence": mu.confidence or picked[0].confidence,
                    # Occurrence is recorded, then used only as a BOUNDED BOOST
                    # at retrieval. Making it a sort key would turn the index
                    # into an FAQ of things everyone already knows and bury the
                    # singletons, which are the reason this corpus exists.
                    "occurrences": len(picked),
                    "source_ids": sorted({m.source_id for m in picked}),
                    "cluster_id": f"c_{ci}",
                    "merge_action": outcome.action,
                    "split_reason": outcome.split_reason or None,
                    "training_safe": True,  # everything here already passed the gate
                    "department": max(set(depts), key=depts.count) if depts else None,
                    "language": max(set(langs), key=langs.count) if langs else "en",
                    "technical_category": max(set(cats), key=cats.count) if cats else None,
                    "observed_at": max((m.call_ts for m in picked if m.call_ts), default=None),
                    **labels,
                }
            )

    print(f"      {len(canonical):,} canonical units; actions={action_counts}", flush=True)

    print("[5/6] remapping supersession off WAV filenames", flush=True)
    edges, sstats = supersession.resolve(raw, canonical_of)
    by_id = {u["unit_id"]: u for u in canonical}
    for target, superseded in edges.items():
        if target in by_id:
            by_id[target]["superseded_unit_ids"] = superseded
    print(f"      {json.dumps(sstats.as_dict())}", flush=True)

    gen = _generation(raw)
    stamp = {
        "gen": gen,
        "kb_version": config.KB_VERSION,
        "merge_version": config.MERGE_VERSION,
        "status": "active",
        "created_at": _now(),
        "updated_at": _now(),
    }
    for u in canonical:
        u.update(stamp)

    summary = {
        "gen": gen,
        "raw_units": len(raw),
        "canonical_units": len(canonical),
        "clusters": cstats.as_dict(),
        "gate_drops": dict(extract.DROPS),
        "merge_actions": action_counts,
        "supersession": sstats.as_dict(),
        "elapsed_s": round(time.time() - started, 1),
        "settings": config.SETTINGS.as_dict(),
    }

    if dry_run:
        print("[6/6] dry run -- nothing written", flush=True)
        return summary

    print(f"[6/6] writing generation {gen}", flush=True)
    ensure_kb_indexes()
    coll = kb_db()[config.COLL_KB_UNITS]

    ops = [
        UpdateOne({"unit_id": u["unit_id"]}, {"$set": u}, upsert=True) for u in canonical
    ]
    written = 0
    for i in range(0, len(ops), 1000):
        res = coll.bulk_write(ops[i : i + 1000], ordered=False)
        written += res.upserted_count + res.modified_count
    # Delete the previous generation only after the new one is fully written.
    removed = coll.delete_many({"gen": {"$ne": gen}}).deleted_count
    put_state(STATE_KEY, {**summary, "stamped_at": _now()})

    summary["written"] = written
    summary["removed_old_generation"] = removed
    print(f"      wrote {written:,}, removed {removed:,} from prior generations", flush=True)
    return summary


def revoke() -> dict:
    """Apply the ``kb_revocations`` ledger.

    Today ``training_safe`` is enforced at ingestion only, and re-screening
    does not un-embed anything -- a re-screen that flips calls to unsafe means
    rebuilding the collection. Because identity lives in Mongo, this is instead
    a targeted status flip plus a filtered Qdrant delete (``index/revoke.py``).
    The same mechanism serves a Law 25 / PIPEDA deletion request.
    """
    db = kb_db()
    ledger = db[config.COLL_KB_REVOCATIONS]
    units = db[config.COLL_KB_UNITS]

    pending = list(ledger.find({"applied": {"$ne": True}}))
    if not pending:
        return {"pending": 0, "revoked": 0}

    direct = [r["unit_id"] for r in pending if r.get("unit_id")]
    by_source = [r["source_id"] for r in pending if r.get("source_id")]

    targets: set[str] = set(direct)
    if by_source:
        for u in units.find({"source_ids": {"$in": by_source}}, {"unit_id": 1}):
            targets.add(u["unit_id"])

    if targets:
        units.update_many(
            {"unit_id": {"$in": sorted(targets)}},
            {"$set": {"status": "revoked", "training_safe": False, "updated_at": _now()}},
        )
    ledger.update_many(
        {"_id": {"$in": [r["_id"] for r in pending]}},
        {"$set": {"applied": True, "applied_at": _now()}},
    )
    return {"pending": len(pending), "revoked": len(targets), "unit_ids": sorted(targets)[:20]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_kb", description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="full rebuild (default)")
    ap.add_argument("--limit", type=int, default=None, help="cap source documents")
    ap.add_argument("--no-merge", action="store_true", help="skip the LLM merge pass")
    ap.add_argument("--dry-run", action="store_true", help="compute but write nothing")
    ap.add_argument("--workers", type=int, default=None, help="merge concurrency")
    ap.add_argument("--revoke", action="store_true", help="apply the revocation ledger")
    ap.add_argument("--status", action="store_true", help="show the last build stamp")
    args = ap.parse_args(argv)

    if args.status:
        state = get_state(STATE_KEY)
        print(json.dumps(state, indent=2, default=str) if state else "no build recorded")
        return 0

    if args.revoke:
        print(json.dumps(revoke(), indent=2))
        return 0

    summary = build(
        limit=args.limit,
        do_merge=not args.no_merge,
        dry_run=args.dry_run,
        merge_workers=args.workers,
    )
    print("\n" + json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
