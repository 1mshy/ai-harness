"""Targeted un-publishing, driven by the ``kb_revocations`` ledger.

Today ``training_safe`` is enforced at **ingestion only**. Re-screening a call
and flipping it to unsafe updates Mongo and nothing else: the vector store keeps
serving the point, because nothing in it knows which source document a point
came from. The only remedy available is rebuilding the entire collection, which
is why nobody does it promptly.

Phase 4a fixes that upstream by giving every unit a stable ``unit_id`` and a
``source_ids`` list, and Phase 4b publishes both. So un-publishing becomes a
filtered delete over four collections — minutes, not a re-embed — via
``clients.qdrant.delete_by_unit_ids``.

The same mechanism serves a Law 25 / PIPEDA deletion request. A request naming
one caller resolves to their call documents, their documents resolve to unit
ids, and the unit ids resolve to points. Against a vector store whose payloads
carry no stable source id that request is simply unservable, so this pays for
itself twice.

Ledger shape (``transcribing.kb_revocations``, owned by this project)::

    {
      "unit_id":   "u_...|c_...|n_...|p_..."   # either this
      "source_id": "<mongo _id str>",          # ...or this, or both
      "reason":    "training_safe|law25|manual|emissions|safety",
      "scope":     ["unitronic_kb_units", ...] | null,   # null = every alias
      "requested_at": iso, "requested_by": str, "note": str,
      "applied": bool, "applied_at": iso,
      "resolved_unit_ids": [...], "deleted": {alias: n}
    }

An entry is idempotent: applying it twice deletes nothing the second time, and
``applied`` is stamped only after the deletes return.

CLI::

    python -m agentv1.index.revoke --sweep --dry-run     # what a re-screen implies
    python -m agentv1.index.revoke --sweep --apply
    python -m agentv1.index.revoke --source-id 6a6b... --reason law25 --apply
    python -m agentv1.index.revoke --unit-id u_0123456789abcdef --apply
    python -m agentv1.index.revoke --status
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from bson import ObjectId

from .. import config
from ..clients import qdrant as q
from ..clients.mongo import kb_db, source_db
from .payload import unit_id_for_call, unit_id_for_case

# Which alias each unit-id prefix can possibly live in. Deleting by unit_id is
# cheap, but it is a filtered scan per collection, so a Law 25 request over four
# collections should not pay for three scans it can rule out from the prefix.
_PREFIX_ALIASES: dict[str, tuple[str, ...]] = {
    "u": (config.ALIAS_KB_UNITS,),
    "n": (config.ALIAS_CASE_NARRATIVES,),
    "c": (config.ALIAS_CALL_RESIDUAL,),
    "p": (config.ALIAS_PLATFORM_STAGES,),
}
ALL_ALIASES: tuple[str, ...] = tuple(config.NEW_ALIASES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ledger():
    return kb_db()[config.COLL_KB_REVOCATIONS]


# --- writing the ledger ------------------------------------------------------
def record_revocation(
    *,
    unit_id: str | None = None,
    source_id: str | None = None,
    reason: str = "manual",
    scope: Sequence[str] | None = None,
    requested_by: str = "cli",
    note: str = "",
) -> dict[str, Any]:
    """Append one revocation request. Does not delete anything by itself.

    Recording and applying are separate on purpose: the ledger is the audit
    trail a Law 25 request is answered with, and it has to survive the case
    where the delete fails halfway.
    """
    if not unit_id and not source_id:
        raise ValueError("a revocation needs a unit_id or a source_id")
    entry = {
        "unit_id": unit_id,
        "source_id": source_id,
        "reason": reason,
        "scope": list(scope) if scope else None,
        "requested_at": _now(),
        "requested_by": requested_by,
        "note": note,
        "applied": False,
    }
    entry["_id"] = _ledger().insert_one(entry).inserted_id
    return entry


def pending() -> list[dict[str, Any]]:
    return list(_ledger().find({"applied": {"$ne": True}}).sort("_id", 1))


# --- resolving a request to unit ids -----------------------------------------
def unit_ids_for_source(source_id: str) -> set[str]:
    """Every published unit id that a single source document feeds.

    Three routes, because the four collections derive identity differently:
    ``kb_units`` records ``source_ids`` explicitly, while the residual and case
    projections derive their ids from the Mongo ``_id`` and the ``case_id``
    deterministically -- so those need no lookup at all, which is what keeps a
    deletion request answerable even if ``kb_units`` has not been built yet.
    """
    ids = {unit_id_for_call(source_id)}
    for doc in kb_db()[config.COLL_KB_UNITS].find(
        {"source_ids": source_id}, {"unit_id": 1}
    ):
        ids.add(str(doc["unit_id"]))
    try:
        oid = ObjectId(source_id)
    except Exception:
        oid = None
    if oid is not None:
        case = source_db()[config.COLL_CASES].find_one({"_id": oid}, {"case_id": 1})
        if case:
            ids.add(unit_id_for_case(str(case.get("case_id") or source_id)))
    return ids


def resolve(entry: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    if entry.get("unit_id"):
        ids.add(str(entry["unit_id"]))
    if entry.get("source_id"):
        ids |= unit_ids_for_source(str(entry["source_id"]))
    return ids


def _aliases_for(unit_ids: Iterable[str], scope: Sequence[str] | None) -> dict[str, list[str]]:
    """Group unit ids by the alias that could hold them."""
    grouped: dict[str, list[str]] = {}
    for uid in unit_ids:
        prefix = uid.split("_", 1)[0]
        targets = _PREFIX_ALIASES.get(prefix, ALL_ALIASES)
        if scope:
            targets = tuple(a for a in targets if a in scope)
        for alias in targets:
            grouped.setdefault(alias, []).append(uid)
    return grouped


# --- applying ----------------------------------------------------------------
def apply_pending(*, dry_run: bool = False, log=print) -> dict[str, Any]:
    """Drain the ledger. Safe to run on every build."""
    entries = pending()
    report: dict[str, Any] = {
        "entries": len(entries), "unit_ids": 0, "deleted": 0,
        "by_alias": {}, "dry_run": dry_run,
    }
    if not entries:
        log("[revoke] ledger is empty; nothing to apply.")
        return report

    for entry in entries:
        unit_ids = sorted(resolve(entry))
        report["unit_ids"] += len(unit_ids)
        deleted: dict[str, int] = {}
        for alias, ids in _aliases_for(unit_ids, entry.get("scope")).items():
            try:
                target = q.open_collection(alias)
            except q.CollectionMissing:
                # Not built yet. The ledger entry stays unapplied so that the
                # next build's --revoke pass picks it up once the alias exists.
                log(f"[revoke] {alias} does not exist yet; deferring {len(ids)} ids")
                continue
            if dry_run:
                deleted[alias] = 0
                log(f"[revoke] DRY RUN would delete {len(ids)} unit ids from {target}")
                continue
            n = q.delete_by_unit_ids(target, ids)
            deleted[alias] = n
            report["by_alias"][alias] = report["by_alias"].get(alias, 0) + n
            report["deleted"] += n

        if not dry_run:
            # Tombstone in Mongo as well: a rebuild reads kb_units, so a unit
            # left `active` would come straight back on the next projection.
            if unit_ids:
                kb_db()[config.COLL_KB_UNITS].update_many(
                    {"unit_id": {"$in": unit_ids}},
                    {"$set": {"status": "revoked", "updated_at": _now()}},
                )
            _ledger().update_one(
                {"_id": entry["_id"]},
                {"$set": {
                    "applied": True, "applied_at": _now(),
                    "resolved_unit_ids": unit_ids, "deleted": deleted,
                }},
            )
        log(f"[revoke] {entry.get('reason')} {entry.get('unit_id') or entry.get('source_id')}"
            f" -> {len(unit_ids)} unit ids, deleted {deleted}")
    return report


# --- the re-screen sweep -----------------------------------------------------
def sweep(*, dry_run: bool = False, log=print) -> dict[str, int]:
    """Turn a re-screen into ledger entries.

    Three sources of newly-unsafe content, and they need different handling:

    * ``kb_units`` marked unsafe or already tombstoned but still published;
    * ``calls_analysis`` documents that flipped ``training_safe`` to false and
      whose residual card is therefore live under an id we can derive;
    * ``calls_cases`` likewise, plus any case that gained unscreened members.

    Sweeping only *records* -- applying is a separate, explicit step, because a
    sweep that both discovers and deletes has no dry-run worth the name.
    """
    added = {"kb_units": 0, "call_residual": 0, "case_narratives": 0}
    seen_units: set[str] = set()
    seen_sources: set[str] = set()
    for entry in _ledger().find({}, {"unit_id": 1, "source_id": 1}):
        if entry.get("unit_id"):
            seen_units.add(str(entry["unit_id"]))
        if entry.get("source_id"):
            seen_sources.add(str(entry["source_id"]))

    def _add(bucket: str, uid: str, sid: str | None, scope: list[str] | None) -> None:
        if uid in seen_units or (sid and sid in seen_sources):
            return
        seen_units.add(uid)
        if sid:
            seen_sources.add(sid)
        added[bucket] += 1
        if not dry_run:
            record_revocation(
                unit_id=uid, source_id=sid, reason="training_safe",
                scope=scope, requested_by="sweep",
            )

    for doc in kb_db()[config.COLL_KB_UNITS].find(
        {"$or": [{"training_safe": False}, {"status": "revoked"}]}, {"unit_id": 1}
    ):
        _add("kb_units", str(doc["unit_id"]), None, [config.ALIAS_KB_UNITS])

    for doc in source_db()[config.COLL_ANALYSIS].find({"training_safe": False}, {"_id": 1}):
        sid = str(doc["_id"])
        _add("call_residual", unit_id_for_call(sid), sid,
             [config.ALIAS_CALL_RESIDUAL, config.ALIAS_KB_UNITS])

    for doc in source_db()[config.COLL_CASES].find(
        {"$or": [{"training_safe": False}, {"unscreened_members": {"$gt": 0}}]},
        {"_id": 1, "case_id": 1},
    ):
        sid = str(doc["_id"])
        _add("case_narratives", unit_id_for_case(str(doc.get("case_id") or sid)), sid,
             [config.ALIAS_CASE_NARRATIVES, config.ALIAS_KB_UNITS])

    verb = "would record" if dry_run else "recorded"
    log(f"[revoke] sweep {verb} {sum(added.values())} new ledger entries: {added}")
    return added


def status(*, log=print) -> dict[str, Any]:
    ledger = _ledger()
    out = {
        "pending": ledger.count_documents({"applied": {"$ne": True}}),
        "applied": ledger.count_documents({"applied": True}),
        "collections": {},
    }
    for alias in ALL_ALIASES:
        try:
            out["collections"][alias] = q.points_count(alias)
        except q.CollectionMissing:
            out["collections"][alias] = None
    log(f"[revoke] ledger: {out['pending']} pending / {out['applied']} applied")
    for alias, n in out["collections"].items():
        log(f"[revoke]   {alias}: {'not built' if n is None else str(n) + ' points'}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="revoke",
        description="Un-publish knowledge from Qdrant via the kb_revocations ledger.",
    )
    ap.add_argument("--unit-id", action="append", default=None)
    ap.add_argument("--source-id", action="append", default=None,
                    help="Mongo _id of a source document (Law 25 deletion path)")
    ap.add_argument("--reason", default="manual")
    ap.add_argument("--note", default="")
    ap.add_argument("--scope", action="append", default=None, metavar="ALIAS")
    ap.add_argument("--sweep", action="store_true",
                    help="record entries for everything a re-screen flipped to unsafe")
    ap.add_argument("--apply", action="store_true", help="drain the pending ledger")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and report, delete nothing, stamp nothing")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    if not any([args.unit_id, args.source_id, args.sweep, args.apply, args.status]):
        ap.error("nothing to do: pass --unit-id/--source-id, --sweep, --apply or --status")

    # --dry-run means "write nothing", including to the ledger.
    if not args.dry_run:
        for uid in args.unit_id or []:
            record_revocation(unit_id=uid, reason=args.reason, scope=args.scope,
                              note=args.note, requested_by="cli")
        for sid in args.source_id or []:
            record_revocation(source_id=sid, reason=args.reason, scope=args.scope,
                              note=args.note, requested_by="cli")
    elif args.unit_id or args.source_id:
        for uid in (args.unit_id or []) + [
            u for sid in (args.source_id or []) for u in sorted(unit_ids_for_source(sid))
        ]:
            print(f"[revoke] DRY RUN would revoke unit id {uid}")
    if args.sweep:
        sweep(dry_run=args.dry_run)
    if args.apply or args.dry_run:
        apply_pending(dry_run=args.dry_run)
    if args.status:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
