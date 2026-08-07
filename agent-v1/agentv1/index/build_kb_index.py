"""Phase 4b — the Mongo to Qdrant projection. No LLM anywhere in this file.

AGENT_PLAN.md §5.1 splits the knowledge pipeline along the deployment boundary
that already exists: Phase 4a owns canonical unit *identity* in
``transcribing.kb_units`` (gates, dedup, LLM merge, tombstones) and this file
owns the *projection* into Qdrant. The split is the point — a Qdrant rebuild
must never re-run the merge, and a merge re-run must never need Qdrant. So
everything here is a pure function of what Mongo already holds: no generation,
no summarisation, no judgement calls, nothing that costs a GPU-second to redo.

Four projections, all built the same way::

    unitronic_kb_units        2 points/unit  <- kb_units          (Phase 4a)
    unitronic_case_narratives 1 point/case   <- calls_cases
    unitronic_call_residual   1 point/call   <- calls_analysis    (the residual)
    unitronic_platform_stages 1 point/card   <- tuning_platforms

**No chunking.** Median unit is 336 chars, p95 591, longest answer 1,165 — the
``SentenceSplitter(chunk_size=1024)`` in the stack this replaces never fired
once. Two points per canonical unit instead: an ANSWER point carrying what the
unit says, and a QUERY point carrying the shapes the question takes. Only
``kb_units`` gets the second point, because only it carries the offline HyDE
corpus (``hypothetical_questions``) that makes a query point worth its storage.

**The residual, not the summaries.** 29,420 of the 34,781 safe+useful calls
already have at least one knowledge unit, so indexing whole-call summaries
would duplicate the KB 6.5x for zero coverage gain. Only calls with *no*
knowledge unit are projected, and every one of them has both ``problem`` and
``solution`` populated (measured: 5,361 of 5,361).

**Hybrid from birth.** Adding a sparse vector to an existing Qdrant collection
means deleting and recreating it, so ``create_hybrid_collection`` is called on
a fresh generation and never on a live one.

**Alias-fronted.** Every build writes ``<alias>_<size>__<gen>`` and moves the
alias at the end. Rolling back a bad build is an alias flip measured in
seconds. The BM25 statistics are written next to the build under the physical
collection name and copied to the alias name on swap, because index-time and
query-time IDF must come from the same fit or the lexical half silently ranks
on the wrong numbers.

CLI::

    python -m agentv1.index.build_kb_index                     # build what changed
    python -m agentv1.index.build_kb_index --dry-run
    python -m agentv1.index.build_kb_index --refresh
    python -m agentv1.index.build_kb_index --redo-outdated
    python -m agentv1.index.build_kb_index --alias unitronic_call_residual --limit 200
    python -m agentv1.index.build_kb_index --revoke              # drain the ledger first
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from qdrant_client import models as qm

from .. import config
from ..clients import qdrant as q
from ..clients.embeddings import get_embedder
from ..clients.mongo import get_state, kb_db, put_state, source_db
from ..clients.sparse import Bm25Encoder
from . import payload as P

# Points per upsert round-trip. 512 x 1024 float32 is ~2 MB on the wire, which
# is comfortably under any gRPC/HTTP limit and small enough that a crash loses
# at most one chunk of work.
CHUNK = 512

BM25_DIR = config.DATA_DIR / "bm25"

# Two builds of the same alias racing is not a theoretical concern: it happened
# during bring-up and the loser's alias swap deleted the winner's collection,
# leaving a truncated index behind an alias that claimed to be complete. The
# progress document doubles as an advisory lease -- heartbeated per chunk, and
# considered abandoned after this long so a killed build does not wedge the
# pipeline forever.
BUILD_LEASE_STALE_SECONDS = float(os.environ.get("INDEX_LEASE_STALE_SECONDS", "1800"))
OWNER = f"{socket.gethostname()}:{os.getpid()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bm25_stats_path(name: str) -> Path:
    """Where the fitted BM25 statistics for ``name`` live.

    ``name`` is either a physical collection (written during the build) or an
    alias (copied there on swap). The query side loads the alias file, so a
    rollback that flips the alias back must also flip these statistics back --
    see :func:`_publish_bm25`.
    """
    return BM25_DIR / f"{name}.json"


# --- records -----------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    """One Qdrant point, before embedding.

    ``candidate`` is deliberately *not* a payload yet: it may carry anything the
    projector found convenient, and :func:`payload.project` decides what
    survives. Building the payload early would put the allowlist in four places.
    """

    unit_id: str
    role: str
    text: str
    candidate: dict[str, Any]

    @property
    def point_id(self) -> str:
        return P.point_uuid(self.unit_id, self.role)


@dataclass(frozen=True)
class Projection:
    name: str
    alias: str
    describe: str
    fingerprint: Callable[[], dict[str, Any]]
    records: Callable[[int | None], Iterator[Record]]
    empty_hint: str = ""


# --- shared derivations ------------------------------------------------------
def _truthy_list(value: Any) -> list[str]:
    if not value:
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float)) and str(v).strip()]


def _joined(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and str(p).strip())


def _emissions_risk(review: dict) -> bool:
    """Emissions is a behavioural gate, not a PII gate.

    ``training_safe`` says nothing about whether the agent should have said what
    they said. Phase 4a drops ``complied_improperly`` outright; what survives is
    flagged so the guardrail layer can refuse to elaborate on it.
    """
    handling = str(review.get("emissions_handling") or "")
    return bool(review.get("emissions_tampering_request")) or handling in (
        "refused_correctly",
        "complied_improperly",
    )


# --- projection 1: kb_units --------------------------------------------------
_KB_FILTER = {"status": "active", "training_safe": True}


def _kb_fingerprint() -> dict[str, Any]:
    coll = kb_db()[config.COLL_KB_UNITS]
    latest = list(coll.find(_KB_FILTER, {"updated_at": 1}).sort("updated_at", -1).limit(1))
    return {
        "n": coll.count_documents(_KB_FILTER),
        "latest": str(latest[0].get("updated_at")) if latest else None,
        "kb_version": config.KB_VERSION,
        "merge_version": config.MERGE_VERSION,
    }


def kb_records_for(doc: dict[str, Any]) -> Iterator[Record]:
    """Both points for one ``kb_units`` document.

    Split out of the cursor loop so the self-check can exercise it against a
    contract-shaped document while ``kb_units`` is still empty -- otherwise the
    projector with the most schema surface would be the only untested one.
    """
    unit_id = str(doc["unit_id"])
    base = {
        "unit_id": unit_id,
        "doc_type": "kb_unit",
        "kind": doc.get("kind"),
        "title": doc.get("title"),
        "question": doc.get("question"),
        "answer": doc.get("answer"),
        "conditions": doc.get("conditions"),
        "vehicles_applicable": _truthy_list(doc.get("vehicles_applicable")),
        "products_applicable": _truthy_list(doc.get("products_applicable")),
        "evidence": doc.get("evidence"),
        "confidence": doc.get("confidence"),
        "occurrences": int(doc.get("occurrences") or 1),
        "outranks_call_units": bool(doc.get("outranks_call_units")),
        "superseded_unit_ids": _truthy_list(doc.get("superseded_unit_ids")),
        "case_id": doc.get("case_id"),
        "cluster_id": doc.get("cluster_id"),
        "source_ids": _truthy_list(doc.get("source_ids")),
        "department": doc.get("department"),
        "language": doc.get("language"),
        "technical_category": doc.get("technical_category"),
        "training_safe": bool(doc.get("training_safe")),
        "time_sensitive": bool(doc.get("time_sensitive")),
        "emissions_risk": bool(doc.get("emissions_risk")),
        "safety_gated": bool(doc.get("safety_gated")),
        "dealer_pricing": bool(doc.get("dealer_pricing")),
        "internal_only": bool(doc.get("internal_only")),
        "contains_price": bool(doc.get("contains_price")),
        "kb_version": int(doc.get("kb_version") or config.KB_VERSION),
        "merge_version": int(doc.get("merge_version") or config.MERGE_VERSION),
        "updated_at": doc.get("updated_at"),
    }
    answer = P.answer_text(doc)
    if answer:
        yield Record(unit_id, "answer", answer, dict(base, point_role="answer", text=answer))
    query = P.query_text(doc)
    # A query point whose text is a copy of the question adds a duplicate
    # neighbour for no recall; only emit it when the HyDE corpus or the
    # alias expansion actually contributed something.
    if query and query != str(doc.get("question") or "").strip():
        yield Record(unit_id, "query", query, dict(base, point_role="query", text=query))


def _kb_records(limit: int | None) -> Iterator[Record]:
    coll = kb_db()[config.COLL_KB_UNITS]
    cursor = coll.find(_KB_FILTER).sort("unit_id", 1)
    if limit:
        cursor = cursor.limit(limit)
    for doc in cursor:
        yield from kb_records_for(doc)


# --- projection 2: case narratives -------------------------------------------
# Fail-closed on `unscreened_members`: a case whose member calls have not all
# been screened is not safe merely because the case document says so.
_CASE_FILTER = {
    "status": "done",
    "training_safe": True,
    "unscreened_members": 0,
    "issue": {"$nin": ["", None]},
}


def _case_fingerprint() -> dict[str, Any]:
    coll = source_db()[config.COLL_CASES]
    latest = list(coll.find(_CASE_FILTER, {"_id": 1}).sort("_id", -1).limit(1))
    return {
        "n": coll.count_documents(_CASE_FILTER),
        "latest": str(latest[0]["_id"]) if latest else None,
    }


def _case_records(limit: int | None) -> Iterator[Record]:
    coll = source_db()[config.COLL_CASES]
    cursor = coll.find(_CASE_FILTER).sort("_id", 1)
    if limit:
        cursor = cursor.limit(limit)
    for doc in cursor:
        case_id = str(doc.get("case_id") or doc["_id"])
        unit_id = P.unit_id_for_case(case_id)
        metrics = doc.get("case_metrics") or {}
        attempts = [
            f"{a.get('attempt')} -> {a.get('result')}"
            for a in (doc.get("attempts") or [])
            if isinstance(a, dict) and a.get("attempt")
        ]
        chronology = [
            str(step.get("what_happened"))
            for step in (doc.get("chronology") or [])
            if isinstance(step, dict) and step.get("what_happened")
        ]
        text = _joined(
            str(doc.get("case_label") or ""),
            str(doc.get("issue") or ""),
            str(doc.get("root_cause_detail") or ""),
            "\n".join(chronology),
            "\n".join(attempts),
            str(doc.get("what_finally_worked") or ""),
            str(doc.get("final_resolution") or ""),
        )
        if not text:
            continue
        # `members` / `sessions` / `supersedes_calls` are 3CX WAV filenames that
        # embed the caller's phone number. They are never carried here in any
        # form; `source_ids` is the Mongo _id, which is what a revocation or a
        # Law 25 deletion actually needs.
        candidate = {
            "unit_id": unit_id,
            "point_role": "answer",
            "doc_type": "case_narrative",
            "title": doc.get("case_label"),
            "question": doc.get("issue"),
            "answer": doc.get("what_finally_worked") or doc.get("final_resolution"),
            "conditions": doc.get("root_cause_detail"),
            "text": text,
            "kind": "troubleshooting",
            "evidence": "multi_call_case",
            "confidence": doc.get("link_confidence"),
            "occurrences": int(metrics.get("contacts") or 1),
            "issue_category": doc.get("issue_category"),
            "root_cause": doc.get("root_cause"),
            "resolution_status": doc.get("case_resolution_status"),
            "language": (_truthy_list(metrics.get("languages")) or ["en"])[0],
            "case_id": case_id,
            "source_ids": [str(doc["_id"])],
            "outranks_call_units": True,
            "training_safe": True,
            "time_sensitive": False,
            "emissions_risk": False,
            "safety_gated": False,
            "dealer_pricing": False,
            "internal_only": False,
            "contains_price": "$" in text,
            "updated_at": doc.get("connected_at"),
        }
        yield Record(unit_id, "answer", text, candidate)


# --- projection 3: the call residual -----------------------------------------
# Gates 1-4 of AGENT_PLAN.md §5.1, applied here because these calls never pass
# through Phase 4a: they are exactly the calls that produced no knowledge unit.
# The `.0` forms are deliberate -- `incorrect_statements` is an *array*, and
# `{review.incorrect_statements: true}` matches nothing while
# `{'review.incorrect_statements.0': {$exists: true}}` matches 535 documents.
_RESIDUAL_FILTER = {
    "training_safe": True,
    "useful_content": True,
    "knowledge_units.0": {"$exists": False},
    "problem": {"$nin": ["", None]},
    "solution": {"$nin": ["", None]},
    "review.emissions_handling": {"$ne": "complied_improperly"},
    "review.incorrect_statements.0": {"$exists": False},
    # `sensitivity.derived_text_clean == False` means the screener found a
    # credential spoken aloud that survived into `solution`/`solution_steps` --
    # the very fields projected below. Zero documents match today; the gate
    # exists because a re-screen can flip it and nothing else would catch it.
    "sensitivity.derived_text_clean": {"$ne": False},
}


def _residual_fingerprint() -> dict[str, Any]:
    coll = source_db()[config.COLL_ANALYSIS]
    latest = list(coll.find(_RESIDUAL_FILTER, {"_id": 1}).sort("_id", -1).limit(1))
    return {
        "n": coll.count_documents(_RESIDUAL_FILTER),
        "latest": str(latest[0]["_id"]) if latest else None,
    }


_RESIDUAL_FIELDS = {
    "_id": 1, "canonical_problem": 1, "call_reason": 1, "problem": 1, "solution": 1,
    "solution_steps": 1, "symptoms": 1, "outcome": 1, "summary": 1, "keywords": 1,
    "department": 1, "language": 1, "technical_category": 1, "caller_type": 1,
    "products_mentioned": 1, "part_numbers": 1, "error_codes_mentioned": 1,
    "tune_stages": 1, "vehicles": 1, "prices_discussed": 1, "dates_promised": 1,
    "promo_or_discount_mentioned": 1, "review": 1, "resolution_status": 1,
    "training_safe": 1, "updated_at": 1,
}


def _residual_records(limit: int | None) -> Iterator[Record]:
    coll = source_db()[config.COLL_ANALYSIS]
    cursor = coll.find(_RESIDUAL_FILTER, _RESIDUAL_FIELDS).sort("_id", 1)
    if limit:
        cursor = cursor.limit(limit)
    for doc in cursor:
        source_id = str(doc["_id"])
        unit_id = P.unit_id_for_call(source_id)
        review = doc.get("review") or {}
        steps = _truthy_list(doc.get("solution_steps"))
        problem = str(doc.get("canonical_problem") or doc.get("problem") or "")
        text = _joined(
            problem,
            str(doc.get("problem") or ""),
            "; ".join(_truthy_list(doc.get("symptoms"))),
            str(doc.get("solution") or ""),
            "\n".join(f"- {s}" for s in steps),
            str(doc.get("outcome") or ""),
        )
        if not text:
            continue
        prices = doc.get("prices_discussed") or []
        caller_type = str(doc.get("caller_type") or "")
        candidate = {
            "unit_id": unit_id,
            "point_role": "answer",
            "doc_type": "call_residual",
            "title": problem,
            "question": problem,
            "answer": doc.get("solution"),
            "conditions": "; ".join(_truthy_list(doc.get("symptoms"))),
            "text": text,
            "kind": "troubleshooting",
            "evidence": "single_call",
            # Single uncorroborated call, so never better than medium regardless
            # of how confident the transcript sounds.
            "confidence": "medium",
            "occurrences": 1,
            "language": doc.get("language") or "en",
            "department": doc.get("department"),
            "technical_category": doc.get("technical_category"),
            "resolution_status": doc.get("resolution_status"),
            "products_applicable": _truthy_list(doc.get("products_mentioned")),
            "vehicles_applicable": _truthy_list(doc.get("vehicles")),
            "part_numbers": _truthy_list(doc.get("part_numbers")),
            "error_codes": _truthy_list(doc.get("error_codes_mentioned")),
            "tune_stages": _truthy_list(doc.get("tune_stages")),
            "source_ids": [source_id],
            "outranks_call_units": False,
            "training_safe": True,
            "time_sensitive": bool(doc.get("promo_or_discount_mentioned"))
            or bool(doc.get("dates_promised")),
            "emissions_risk": _emissions_risk(review),
            "safety_gated": bool(review.get("safety_issue")),
            "dealer_pricing": bool(prices)
            and caller_type in ("dealer_installer", "distributor"),
            "internal_only": False,
            "contains_price": bool(prices),
            "requires_tool_confirmation": bool(prices),
            "updated_at": doc.get("updated_at"),
        }
        yield Record(unit_id, "answer", text, candidate)


# --- projection 4: platform x stage cards ------------------------------------
def _platform_fingerprint() -> dict[str, Any]:
    coll = source_db()[config.COLL_PLATFORMS]
    latest = list(coll.find({}, {"synced_at": 1}).sort("synced_at", -1).limit(1))
    n_cards = sum(len(p.get("stages") or []) for p in coll.find({}, {"stages.stage_id": 1}))
    return {
        "n": n_cards,
        "latest": str(latest[0].get("synced_at")) if latest else None,
    }


def _platform_records(limit: int | None) -> Iterator[Record]:
    coll = source_db()[config.COLL_PLATFORMS]
    emitted = 0
    for plat in coll.find({}).sort("_id", 1):
        platform_id = int(plat["_id"])
        makes = _truthy_list(plat.get("makes"))
        models = _truthy_list(plat.get("model_names"))
        years = "-".join(x for x in (str(plat.get("year_start") or ""), str(plat.get("year_end") or "")) if x)
        header = f"{plat.get('engine_name') or ''} {years} — {', '.join(makes)}".strip()
        for stage in plat.get("stages") or []:
            if limit and emitted >= limit:
                return
            stage_id = int(stage.get("stage_id") or -1)
            unit_id = P.unit_id_for_platform_stage(platform_id, stage_id)
            power = [
                f"{pf.get('reference')}: {pf.get('power_gain')} on {pf.get('fuel')}"
                for pf in (stage.get("power_figures") or [])
                if isinstance(pf, dict) and pf.get("power_gain")
            ]
            label = str(stage.get("label") or stage.get("raw") or "")
            released = bool(stage.get("released"))
            availability = (
                f"{label} is released for this platform."
                if released
                else f"{label} is NOT released for this platform."
            )
            text = _joined(
                header,
                f"Fits: {', '.join(models[:24])}" if models else "",
                f"Stage: {label} (family {stage.get('family')}, rank {stage.get('rank')})",
                availability,
                f"Files available: {stage.get('files_available')} of {stage.get('files_total')}",
                "\n".join(power[:12]),
                str(plat.get("description") or ""),
                f"Highest released stage on this platform: {plat.get('max_released_stage')}",
            )
            candidate = {
                "unit_id": unit_id,
                "point_role": "answer",
                "doc_type": "platform_stage",
                "title": f"{plat.get('engine_name')} — {label}",
                "question": f"Is {label} available for {', '.join(models[:6])}?",
                "answer": availability,
                "conditions": years,
                "text": text,
                "kind": "compatibility",
                "evidence": "single_call",
                "confidence": "high",
                "language": "en",
                "platform_id": platform_id,
                "platform_name": plat.get("engine_name"),
                "platform_code": plat.get("platform_code"),
                "makes": makes,
                "model_names": models,
                "year_start": str(plat.get("year_start") or ""),
                "year_end": str(plat.get("year_end") or ""),
                "stage_id": stage_id,
                "stage_label": label,
                "stage_family": stage.get("family"),
                "stage_number": stage.get("number"),
                "stage_plus": bool(stage.get("plus")),
                "stage_rank": float(stage.get("rank") or 0.0),
                "stage_released": released,
                "max_released_stage": plat.get("max_released_stage"),
                "synced_at": plat.get("synced_at"),
                "training_safe": True,
                # These cards exist so a customer question can *find* the
                # platform. They are a snapshot of a sync that is unscheduled
                # and Mac-pinned (measured 5.8 days stale), and non-negotiable 4
                # forbids quoting stage availability without a tool result in the
                # same turn -- so the card announces that about itself.
                "requires_tool_confirmation": True,
                "time_sensitive": True,
                "emissions_risk": False,
                "safety_gated": False,
                "dealer_pricing": False,
                "internal_only": False,
                "contains_price": False,
                "updated_at": plat.get("synced_at"),
            }
            emitted += 1
            yield Record(unit_id, "answer", text, candidate)


PROJECTIONS: list[Projection] = [
    Projection(
        name="kb_units",
        alias=config.ALIAS_KB_UNITS,
        describe="merged call + case knowledge units, two points each",
        fingerprint=_kb_fingerprint,
        records=_kb_records,
        empty_hint=(
            f"{config.COLL_KB_UNITS} holds no active training_safe units. Phase 4a "
            f"(dgx_pipeline/build_kb) has not published a generation yet -- run it "
            f"first. This is not an error; there is simply nothing to project."
        ),
    ),
    Projection(
        name="case_narratives",
        alias=config.ALIAS_CASE_NARRATIVES,
        describe="multi-call case narratives from calls_cases",
        fingerprint=_case_fingerprint,
        records=_case_records,
    ),
    Projection(
        name="call_residual",
        alias=config.ALIAS_CALL_RESIDUAL,
        describe="safe+useful calls that produced no knowledge unit",
        fingerprint=_residual_fingerprint,
        records=_residual_records,
    ),
    Projection(
        name="platform_stages",
        alias=config.ALIAS_PLATFORM_STAGES,
        describe="platform x stage discovery cards from tuning_platforms",
        fingerprint=_platform_fingerprint,
        records=_platform_records,
    ),
]


def projection_by_key(key: str) -> Projection | None:
    for proj in PROJECTIONS:
        if key in (proj.name, proj.alias):
            return proj
    return None


# --- build -------------------------------------------------------------------
def _state_key(alias: str) -> str:
    """Stamp of the last *successful* build. Written once, at the very end."""
    return f"index:{alias}"


def _progress_key(alias: str) -> str:
    """In-flight generation and commit cursor. A separate document on purpose.

    ``put_state`` is a ``$set`` merge, so writing progress into the same
    document as the live stamp overwrites ``status``/``gen``/``collection``
    the moment a rebuild starts -- and a rebuild that then dies leaves the state
    claiming a half-written generation while the alias is still correctly
    serving the previous one. Observed exactly that during bring-up. Two keys,
    and a crash can no longer make the record of what is live disagree with what
    is live.
    """
    return f"index:{alias}:progress"


def _build_stamp() -> dict[str, Any]:
    return {
        "index_version": config.INDEX_VERSION,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
        "qwen_size": config.QWEN_SIZE,
    }


class BuildLocked(RuntimeError):
    pass


def _lease_holder(progress: dict | None) -> str | None:
    """Owner of a live build lease on this alias, if anyone else holds one."""
    if not progress or progress.get("status") != "building":
        return None
    owner = progress.get("owner")
    if not owner or owner == OWNER:
        return None
    beat = progress.get("heartbeat") or progress.get("started_at")
    if not beat:
        return owner
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat)).total_seconds()
    except ValueError:
        return owner
    return owner if age < BUILD_LEASE_STALE_SECONDS else None


def _is_outdated(state: dict | None) -> bool:
    if not state:
        return True
    stamp = _build_stamp()
    return any(state.get(k) != v for k, v in stamp.items())


def _publish_bm25(physical: str, alias: str) -> None:
    """Copy, do not symlink.

    The alias-named file is what the query encoder loads. A copy means the old
    generation's statistics are still on disk under its own name, so an alias
    rollback is a second copy rather than a refit.
    """
    src = bm25_stats_path(physical)
    shutil.copyfile(src, bm25_stats_path(alias))


def _swap_alias(alias: str, physical: str) -> str | None:
    """``clients.qdrant.swap_alias`` with a fallback for a live bug in it.

    That function builds ``qm.AliasOperations(...)``, but in the installed
    qdrant-client ``AliasOperations`` is a ``typing.Union`` of
    ``CreateAliasOperation``/``DeleteAliasOperation``/``RenameAliasOperation``,
    not a model class -- instantiating it raises ``TypeError: Cannot instantiate
    typing.Union``. It is reported rather than patched here because
    ``clients/`` belongs to another owner; the fallback keeps a finished build
    from being stranded behind a naming detail, and disappears the moment the
    contract function starts working.
    """
    try:
        return q.swap_alias(alias, physical, drop_old=True)
    except TypeError:
        client = q.get_client()
        old = q.resolve_alias(alias)
        ops: list[Any] = []
        if old:
            ops.append(qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=alias)))
        ops.append(
            qm.CreateAliasOperation(
                create_alias=qm.CreateAlias(collection_name=physical, alias_name=alias)
            )
        )
        client.update_collection_aliases(change_aliases_operations=ops)
        # Only ever a previous generation of this same alias, created by this
        # builder. Nothing pre-existing is reachable from here.
        if old and old != physical:
            client.delete_collection(old)
        return old


def _points(
    records: Sequence[Record], dense, encoder: Bm25Encoder
) -> list[qm.PointStruct]:
    points = []
    for rec, vec in zip(records, dense):
        sparse = encoder.encode_document(rec.text)
        vectors: dict[str, Any] = {config.DENSE_VECTOR: [float(x) for x in vec]}
        if sparse.indices:
            vectors[config.SPARSE_VECTOR] = qm.SparseVector(
                indices=sparse.indices, values=sparse.values
            )
        points.append(
            qm.PointStruct(
                id=rec.point_id, vector=vectors, payload=P.project(rec.candidate)
            )
        )
    return points


def build(
    proj: Projection,
    *,
    refresh: bool = False,
    redo_outdated: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Project one collection. Idempotent, resumable, alias-fronted."""
    started = time.time()
    state = get_state(_state_key(proj.alias))
    progress = get_state(_progress_key(proj.alias))
    fingerprint = proj.fingerprint()
    result: dict[str, Any] = {
        "projection": proj.name,
        "alias": proj.alias,
        "fingerprint": fingerprint,
        "status": "pending",
    }

    if fingerprint["n"] == 0:
        log(f"[{proj.name}] {proj.empty_hint or 'no source documents match the gates.'}")
        result["status"] = "empty_source"
        result["points"] = 0
        return result

    outdated = _is_outdated(state)
    live = None
    try:
        live = q.open_collection(proj.alias)
    except q.CollectionMissing:
        live = None
    live_count = q.points_count(live) if live else 0

    holder = _lease_holder(progress)
    if holder and not dry_run:
        log(f"[{proj.name}] another build holds the lease ({holder}, generation "
            f"{progress.get('gen')}, {progress.get('committed')} records committed). "
            f"Refusing: two builds of one alias race on the swap, and the loser's "
            f"swap deletes the winner's collection.")
        result["status"] = "locked"
        result["points"] = live_count
        result["lease_holder"] = holder
        return result

    unchanged = (
        not refresh
        and not outdated
        and state is not None
        and state.get("status") == "live"
        and state.get("fingerprint") == fingerprint
        and live is not None
        and live_count > 0
    )
    would_skip = None
    if redo_outdated and not refresh and not outdated:
        would_skip = "skipped_current"
        reason = f"build stamp current ({config.EMBED_MODEL}); --redo-outdated skips it."
    elif unchanged:
        would_skip = "unchanged"
        reason = (f"source unchanged ({fingerprint['n']} docs) and {live} holds "
                  f"{live_count} points; nothing to do.")
    if would_skip:
        log(f"[{proj.name}] {reason}")
        # A dry run still projects and reports. "Nothing would happen" is the
        # answer least worth trusting without seeing the record count behind it.
        if not dry_run:
            result["status"] = would_skip
            result["points"] = live_count
            return result
        result["would_skip"] = would_skip

    log(f"[{proj.name}] projecting {fingerprint['n']} source documents "
        f"({proj.describe}){' [LIMIT %d]' % limit if limit else ''}")
    records = list(proj.records(limit))
    result["records"] = len(records)
    if not records:
        log(f"[{proj.name}] {proj.empty_hint or 'projection produced no records.'}")
        result["status"] = "empty_projection"
        result["points"] = 0
        return result

    if dry_run:
        sample = P.project(records[0].candidate)
        log(f"[{proj.name}] DRY RUN: {len(records)} points would be written to "
            f"{config.collection_for(proj.alias, '<gen>')}")
        log(f"[{proj.name}] sample payload keys: {sorted(sample)}")
        result["status"] = "dry_run"
        result["points"] = len(records)
        result["sample_payload"] = sample
        return result

    # Resume into the same generation when a previous run died mid-build over
    # the identical source. Point ids are deterministic, so re-upserting an
    # already-written chunk would be harmless -- but skipping it is free.
    resumable = (
        not refresh
        and progress is not None
        and progress.get("status") == "building"
        and progress.get("fingerprint") == fingerprint
        and not outdated
        and progress.get("gen")
        and q.collection_exists(config.collection_for(proj.alias, progress["gen"]))
    )
    gen = progress["gen"] if resumable else uuid.uuid4().hex[:12]
    # A leading letter keeps the generation out of the "bare long digit run"
    # shape that payload.scrub() redacts.
    gen = gen if gen[0].isalpha() else "g" + gen[1:]
    physical = config.collection_for(proj.alias, gen)
    committed = int(progress.get("committed", 0)) if resumable else 0
    if resumable:
        log(f"[{proj.name}] resuming generation {gen} at record {committed}")

    # Claim the lease *before* the first mutating call, and re-read it first.
    # The check at the top of this function happened before `proj.records(...)`
    # was materialised, which for the residual is a multi-minute Mongo scan --
    # long enough for a second build started inside that window to have passed
    # the same check and to be about to create its own generation. Claiming here
    # leaves one Mongo round-trip of exposure instead of the whole projection,
    # and everything that can corrupt a live alias (create, upsert, swap) is on
    # the far side of it.
    holder = _lease_holder(get_state(_progress_key(proj.alias)))
    if holder:
        log(f"[{proj.name}] another build took the lease while this one was "
            f"projecting ({holder}). Refusing before writing anything.")
        result["status"] = "locked"
        result["points"] = live_count
        result["lease_holder"] = holder
        return result
    put_state(
        _progress_key(proj.alias),
        {
            "status": "building", "gen": gen, "collection": physical,
            "fingerprint": fingerprint, "committed": committed,
            "records": len(records), "owner": OWNER,
            "started_at": _now(), "heartbeat": _now(), **_build_stamp(),
        },
    )

    q.create_hybrid_collection(physical)

    # IDF is corpus statistics: it must be fitted over the whole corpus even on
    # a resume, or the second half of the index would be scored on different
    # numbers than the first.
    encoder = Bm25Encoder().fit(rec.text for rec in records)
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    encoder.save(bm25_stats_path(physical))

    embedder = get_embedder()

    for start in range(committed, len(records), CHUNK):
        batch = records[start : start + CHUNK]
        dense = embedder.embed_documents([r.text for r in batch])
        q.upsert(physical, _points(batch, dense, encoder), wait=True)
        committed = start + len(batch)
        put_state(_progress_key(proj.alias), {"committed": committed, "heartbeat": _now()})
        log(f"[{proj.name}]   {committed}/{len(records)} points "
            f"({committed / max(time.time() - started, 1e-9):.0f}/s)")

    made = q.ensure_payload_indexes(physical)
    old = _swap_alias(proj.alias, physical)
    _publish_bm25(physical, proj.alias)
    count = q.points_count(proj.alias)
    put_state(
        _state_key(proj.alias),
        {
            "status": "live", "gen": gen, "collection": physical,
            "fingerprint": fingerprint, "committed": committed, "points": count,
            "replaced": old, "payload_indexes": made,
            "bm25_stats": str(bm25_stats_path(proj.alias)),
            "finished_at": _now(), "seconds": round(time.time() - started, 1),
            **_build_stamp(),
        },
    )
    # Cleared last: while this says "building", a resume is still on the table.
    put_state(
        _progress_key(proj.alias),
        {"status": "done", "gen": gen, "committed": committed,
         "owner": OWNER, "heartbeat": _now()},
    )
    log(f"[{proj.name}] live: alias {proj.alias} -> {physical}, {count} points, "
        f"{round(time.time() - started, 1)}s (replaced {old})")
    result.update(status="live", points=count, collection=physical, gen=gen, replaced=old)
    return result


def build_all(
    aliases: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    redo_outdated: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    log=print,
) -> list[dict[str, Any]]:
    if aliases:
        chosen = []
        for key in aliases:
            proj = projection_by_key(key)
            if proj is None:
                raise SystemExit(
                    f"unknown projection {key!r}; known: "
                    + ", ".join(p.name for p in PROJECTIONS)
                )
            chosen.append(proj)
    else:
        chosen = list(PROJECTIONS)
    return [
        build(p, refresh=refresh, redo_outdated=redo_outdated,
              dry_run=dry_run, limit=limit, log=log)
        for p in chosen
    ]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_kb_index",
        description="Phase 4b: project Mongo knowledge into Qdrant. No LLM calls.",
    )
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild everything into a fresh generation, ignoring state")
    ap.add_argument("--redo-outdated", action="store_true",
                    help="rebuild only where the stamped index_version/embed model differs")
    ap.add_argument("--dry-run", action="store_true",
                    help="project and count, but touch neither Qdrant nor the state stamp")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap records per projection (smoke tests)")
    ap.add_argument("--alias", action="append", default=None, metavar="NAME",
                    help="restrict to this projection (short name or alias); repeatable")
    ap.add_argument("--revoke", action="store_true",
                    help="drain kb_revocations against the live aliases before building")
    args = ap.parse_args(argv)

    if args.revoke:
        from . import revoke as revoke_mod

        report = revoke_mod.apply_pending(dry_run=args.dry_run)
        print(f"[revoke] {report['entries']} ledger entries, "
              f"{report['unit_ids']} unit ids, {report['deleted']} point deletions")

    results = build_all(
        args.alias,
        refresh=args.refresh,
        redo_outdated=args.redo_outdated,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print("\n=== summary ===")
    for r in results:
        print(f"  {r['projection']:<18} {r['status']:<16} points={r.get('points', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
