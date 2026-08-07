"""Extract raw knowledge units from ``calls_analysis`` and ``calls_cases``.

Call units and case units land in **one** stream and later one collection.
That is deliberate and is the single most important structural decision in
Phase 4a. A case unit's entire value is *outranking* a call unit, and the
router this feeds does per-collection selection -- so ~3,700 case units in
their own collection would essentially never be picked, destroying exactly the
property that makes them worth having. The schemas are near-identical; merging
them turns supersession into a rerank-time filter over one result set.

Unit identity is ``(source_id, ordinal)``, not a content hash. Dedup cluster
membership moves every time the analyzer adds a document, so a content-hash id
is unstable and idempotent upsert becomes impossible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..clients.mongo import source_db
from . import gates


def unit_id_for(source_id: str, ordinal: int) -> str:
    digest = hashlib.blake2b(f"{source_id}:{ordinal}".encode(), digest_size=8).hexdigest()
    return f"u_{digest}"


@dataclass
class RawUnit:
    unit_id: str
    source_id: str
    ordinal: int
    source_kind: str  # "call" | "case"

    kind: str = ""
    title: str = ""
    question: str = ""
    answer: str = ""
    conditions: str = ""
    vehicles_applicable: list[str] = field(default_factory=list)
    products_applicable: list[str] = field(default_factory=list)
    hypothetical_questions: list[str] = field(default_factory=list)
    confidence: str = "medium"
    confidence_reason: str = ""

    evidence: str = "single_call"
    outranks_call_units: bool = False
    supersedes_calls: list[str] = field(default_factory=list)  # WAV filenames -- PII
    case_id: str | None = None
    evidence_contacts: int = 0

    department: str | None = None
    language: str = "en"
    technical_category: str | None = None
    caller_type: str | None = None
    call_ts: str | None = None
    aliases: list[dict] = field(default_factory=list)

    labels: dict[str, Any] = field(default_factory=dict)

    def text_for_answer_point(self) -> str:
        parts = [self.title, self.question, self.answer]
        if self.conditions:
            parts.append(f"Conditions: {self.conditions}")
        if self.vehicles_applicable:
            parts.append("Vehicles: " + ", ".join(self.vehicles_applicable))
        if self.products_applicable:
            parts.append("Products: " + ", ".join(self.products_applicable))
        return "\n".join(p for p in parts if p)

    def text_for_query_point(self) -> str:
        """Question side. The ~186k hypothetical questions in this corpus were
        written offline *with the real answer in hand*, which is strictly
        better than generating a hypothetical document at query time -- so
        HyDE is done here, at index time, once, rather than per query."""
        parts = [self.question, self.title]
        parts.extend(self.hypothetical_questions)
        for alias in self.aliases:
            colloquial = (alias or {}).get("colloquial")
            if colloquial:
                parts.append(str(colloquial))
        return "\n".join(p for p in parts if p)


def _norm_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


def iter_call_units(
    *, limit: int | None = None, query: dict | None = None
) -> Iterator[tuple[RawUnit, dict]]:
    """Yield ``(unit, gate_stats)`` for every gated unit in ``calls_analysis``."""
    db = source_db()
    base = {"knowledge_units.0": {"$exists": True}}
    if query:
        base.update(query)
    projection = {
        "knowledge_units": 1, "review": 1, "training_safe": 1, "useful_content": 1,
        "status": 1, "department": 1, "language": 1, "technical_category": 1,
        "caller_type": 1, "call_ts": 1, "start_time": 1, "search_aliases": 1,
    }
    cursor = db.calls_analysis.find(base, projection)
    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        source_id = str(doc["_id"])
        verdict = gates.document_gate(doc, is_case=False)
        if not verdict.passed:
            _record_drop(verdict.reason, len(doc.get("knowledge_units") or []))
            continue
        aliases = doc.get("search_aliases") or []
        for ordinal, unit in enumerate(doc.get("knowledge_units") or []):
            labels = gates.derive_labels(unit, doc, is_case=False)
            ugate = gates.unit_gate(unit, labels)
            if not ugate.passed:
                _record_drop(ugate.reason, 1)
                continue
            yield (
                RawUnit(
                    unit_id=unit_id_for(source_id, ordinal),
                    source_id=source_id,
                    ordinal=ordinal,
                    source_kind="call",
                    kind=(unit.get("kind") or "").strip().lower(),
                    title=(unit.get("title") or "").strip(),
                    question=(unit.get("question") or "").strip(),
                    answer=(unit.get("answer") or "").strip(),
                    conditions=(unit.get("conditions") or "").strip(),
                    vehicles_applicable=_norm_list(unit.get("vehicles_applicable")),
                    products_applicable=_norm_list(unit.get("products_applicable")),
                    hypothetical_questions=_norm_list(unit.get("hypothetical_questions")),
                    confidence=(unit.get("confidence") or "medium").strip().lower(),
                    confidence_reason=(unit.get("confidence_reason") or "").strip(),
                    evidence="single_call",
                    outranks_call_units=False,
                    department=doc.get("department"),
                    language=(doc.get("language") or "en"),
                    technical_category=doc.get("technical_category"),
                    caller_type=doc.get("caller_type"),
                    call_ts=doc.get("call_ts") or doc.get("start_time"),
                    aliases=aliases if isinstance(aliases, list) else [],
                    labels=labels,
                ),
                {},
            )


def iter_case_units(
    *, limit: int | None = None, query: dict | None = None
) -> Iterator[tuple[RawUnit, dict]]:
    """Yield gated units from ``calls_cases``.

    ``supersedes_calls[]`` is carried through *unresolved* here and remapped in
    ``supersession.py``. It is a list of 3CX WAV filenames that embed the
    caller's phone number, so the supersession relation -- the entire point of
    case units -- is expressed in PII until that remap runs. Nothing may embed
    before it does.
    """
    db = source_db()
    base = {"case_knowledge_units.0": {"$exists": True}}
    if query:
        base.update(query)
    cursor = db.calls_cases.find(
        base,
        {
            "case_knowledge_units": 1, "case_id": 1, "training_safe": 1, "status": 1,
            "unscreened_members": 1, "withheld_members": 1, "issue_category": 1,
            "last_call_ts": 1, "what_finally_worked": 1, "review": 1,
        },
    )
    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        source_id = str(doc["_id"])
        verdict = gates.document_gate(doc, is_case=True)
        if not verdict.passed:
            _record_drop(verdict.reason, len(doc.get("case_knowledge_units") or []))
            continue
        for ordinal, unit in enumerate(doc.get("case_knowledge_units") or []):
            labels = gates.derive_labels(unit, doc, is_case=True)
            ugate = gates.unit_gate(unit, labels)
            if not ugate.passed:
                _record_drop(ugate.reason, 1)
                continue
            yield (
                RawUnit(
                    unit_id=unit_id_for(source_id, ordinal),
                    source_id=source_id,
                    ordinal=ordinal,
                    source_kind="case",
                    kind=(unit.get("kind") or "").strip().lower(),
                    title=(unit.get("title") or "").strip(),
                    question=(unit.get("question") or "").strip(),
                    answer=(unit.get("answer") or "").strip(),
                    conditions=(unit.get("conditions") or "").strip(),
                    hypothetical_questions=_norm_list(unit.get("hypothetical_questions")),
                    confidence=(unit.get("confidence") or "medium").strip().lower(),
                    confidence_reason=(unit.get("confidence_reason") or "").strip(),
                    evidence=(unit.get("evidence") or "multi_call_case"),
                    outranks_call_units=bool(unit.get("outranks_call_units", True)),
                    supersedes_calls=_norm_list(unit.get("supersedes_calls")),
                    case_id=doc.get("case_id"),
                    evidence_contacts=int(unit.get("evidence_contacts") or 0),
                    technical_category=doc.get("issue_category"),
                    call_ts=doc.get("last_call_ts"),
                    labels=labels,
                ),
                {},
            )


# Drop accounting is process-global so that the CLI can report exactly what the
# gates removed. A gate that silently drops half the corpus and a gate that
# works look identical without this.
DROPS: dict[str, int] = {}


def _record_drop(reason: str, n: int) -> None:
    DROPS[reason] = DROPS.get(reason, 0) + n


def reset_drops() -> None:
    DROPS.clear()


def iter_all_units(*, limit: int | None = None) -> Iterator[RawUnit]:
    for unit, _ in iter_call_units(limit=limit):
        yield unit
    for unit, _ in iter_case_units(limit=limit):
        yield unit
