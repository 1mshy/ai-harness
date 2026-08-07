"""LLM merge of a duplicate cluster into one or more canonical units.

The prompt must be allowed to answer *"these are two units, split them"*.
Without that escape hatch a $150 fee and a $300 fee get averaged into a
canonical unit that is wrong in a way no metric catches -- the merge looks
successful, the collection looks clean, and the agent quotes a price that never
existed.

**Recency does not arbitrate.** Price disagreement inside a topic is usually
context noise rather than drift: ``license transfer fee`` reads $150 across
2024/25/26, and the $300 outliers co-occur with *cable-reset* scenarios, which
is a different question wearing a similar title. So the instruction is
consensus-within-window, with recency as a tiebreak and only on units flagged
``time_sensitive`` -- 13% of the corpus. The other 87% should not decay at all:
a 2024 fix for a flashing error is still the fix.

Case units win by construction. A unit carrying ``evidence: multi_call_case``
was synthesised from a thread where the outcome was observed, including which
attempts *failed*. When it disagrees with a single-call unit, the single-call
unit is what somebody said on the phone once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..clients.llm import LLMPermanentError, get_llm

SYSTEM = """You consolidate duplicate knowledge units from a vehicle-performance-tuning company's support and sales calls.

You will be shown several units that a clustering pass believes describe the same fact. Your job is to emit the canonical version.

RULES
1. If the units genuinely describe ONE fact, emit exactly one merged unit.
2. If they describe TWO OR MORE DIFFERENT facts that merely share similar wording, SPLIT them and emit one unit per distinct fact. This is expected and correct. Never average conflicting specifics into a single unit - a fee of $150 and a fee of $300 for different services are two units, not one unit saying $225.
3. Prefer the value the MAJORITY of units agree on. Do not prefer the newest unit unless the units are marked time_sensitive and the majority is tied.
4. A unit marked evidence=multi_call_case outranks single-call units. It was built from a thread where the outcome was actually observed. If it disagrees, follow it.
5. Preserve specificity. If units differ only because one names a vehicle or condition the other omits, keep the condition in `conditions` rather than dropping it.
6. Never invent a fact that appears in none of the inputs. Never soften a refusal into a neutral statement.
7. Write the answer so it stands alone, without reference to "the agent" or "the call".

Reply with JSON only, no prose and no code fence:
{"units":[{"title":"...","question":"...","answer":"...","conditions":"...","kind":"...","hypothetical_questions":["..."],"confidence":"high|medium|low","source_indices":[0,2]}],"split_reason":"..."}

`source_indices` lists which input units (0-based) each output unit came from. Every input index must appear in exactly one output unit."""


@dataclass
class MergedUnit:
    title: str
    question: str
    answer: str
    conditions: str
    kind: str
    hypothetical_questions: list[str]
    confidence: str
    source_indices: list[int]


@dataclass
class MergeOutcome:
    units: list[MergedUnit]
    action: str  # "merged" | "split" | "verbatim" | "failed"
    split_reason: str = ""
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _render(units: Sequence, indices: Sequence[int]) -> str:
    lines = []
    for out_i, idx in enumerate(indices):
        u = units[idx]
        lines.append(
            f"--- unit {out_i} ---\n"
            f"kind: {u.kind}\n"
            f"evidence: {u.evidence}\n"
            f"confidence: {u.confidence}\n"
            f"time_sensitive: {bool(u.labels.get('time_sensitive'))}\n"
            f"observed_at: {u.call_ts or 'unknown'}\n"
            f"title: {u.title}\n"
            f"question: {u.question}\n"
            f"answer: {u.answer}\n"
            f"conditions: {u.conditions}"
        )
    return "\n".join(lines)


def _validate(payload: Any, n_inputs: int) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("units"), list):
        raise ValueError("expected {'units': [...]}")
    if not payload["units"]:
        raise ValueError("units[] was empty")
    seen: set[int] = set()
    for u in payload["units"]:
        if not isinstance(u, dict):
            raise ValueError("unit was not an object")
        if not (u.get("answer") or "").strip():
            raise ValueError("unit had an empty answer")
        if not (u.get("title") or "").strip():
            raise ValueError("unit had an empty title")
        idxs = u.get("source_indices") or []
        if not isinstance(idxs, list):
            raise ValueError("source_indices must be a list")
        for i in idxs:
            if not isinstance(i, int) or not 0 <= i < n_inputs:
                raise ValueError(f"source_index {i} out of range 0..{n_inputs - 1}")
            seen.add(i)
    # Every input must be accounted for -- a model that silently drops half the
    # cluster produces a canonical unit that looks fine and has quietly lost
    # evidence, and worse, orphans those source_ids from the revocation path.
    #
    # But do NOT spend a repair round-trip on it. Coverage is a bookkeeping
    # property, not a judgement, so the cheap deterministic repair is to attach
    # any unclaimed input to the first emitted unit. On a corpus this size the
    # retries were costing more wall clock than the merge itself.
    missing = sorted(set(range(n_inputs)) - seen)
    if missing:
        first = payload["units"][0]
        first["source_indices"] = sorted(set(first.get("source_indices") or []) | set(missing))
    return payload


def merge_cluster(units: Sequence, indices: Sequence[int]) -> MergeOutcome:
    """Merge one cluster. A cluster of size 1 never reaches the model."""
    if len(indices) == 1:
        u = units[indices[0]]
        return MergeOutcome(
            units=[
                MergedUnit(
                    title=u.title, question=u.question, answer=u.answer,
                    conditions=u.conditions, kind=u.kind,
                    hypothetical_questions=list(u.hypothetical_questions),
                    confidence=u.confidence, source_indices=[0],
                )
            ],
            action="verbatim",
        )

    # A handful of head topics survive answer-side refinement still oversized
    # (200 near-identical restatements of one cable-reset procedure). Show the
    # model a bounded, most-informative sample; the tail is still accounted for
    # below, so `occurrences` and `source_ids` stay complete and nothing is
    # dropped from the revocation path.
    from .dedup import MAX_MERGE_CLUSTER

    shown = list(indices[:MAX_MERGE_CLUSTER])
    overflow = list(range(len(shown), len(indices)))

    prompt = _render(units, shown)
    try:
        payload = get_llm().json_call(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            validator=lambda p: _validate(p, len(shown)),
            max_tokens=2400,
        )
    except (LLMPermanentError, Exception) as exc:  # noqa: BLE001
        # Falling back to the single most-informative input is safe: it is a
        # real unit that a human said, just without the consolidation. Losing
        # the whole cluster because one merge failed is not.
        best = max(
            indices,
            key=lambda i: (
                units[i].evidence == "multi_call_case",
                {"high": 2, "medium": 1, "low": 0}.get(units[i].confidence, 1),
                len(units[i].answer or ""),
            ),
        )
        u = units[best]
        return MergeOutcome(
            units=[
                MergedUnit(
                    title=u.title, question=u.question, answer=u.answer,
                    conditions=u.conditions, kind=u.kind,
                    hypothetical_questions=list(u.hypothetical_questions),
                    confidence=u.confidence,
                    source_indices=list(range(len(indices))),
                )
            ],
            action="failed",
            error=str(exc)[:300],
        )

    # Members beyond the shown sample attach to the first emitted unit, so
    # every input index still lands in exactly one canonical unit.
    if overflow and payload["units"]:
        first = payload["units"][0]
        first["source_indices"] = list(first.get("source_indices") or []) + overflow

    out = [
        MergedUnit(
            title=str(u.get("title", "")).strip(),
            question=str(u.get("question", "")).strip(),
            answer=str(u.get("answer", "")).strip(),
            conditions=str(u.get("conditions") or "").strip(),
            kind=str(u.get("kind") or units[indices[0]].kind).strip().lower(),
            hypothetical_questions=[
                str(q) for q in (u.get("hypothetical_questions") or []) if q
            ],
            confidence=str(u.get("confidence") or "medium").strip().lower(),
            source_indices=[int(i) for i in (u.get("source_indices") or [])],
        )
        for u in payload["units"]
    ]
    return MergeOutcome(
        units=out,
        action="split" if len(out) > 1 else "merged",
        split_reason=str(payload.get("split_reason") or ""),
        raw=payload,
    )
