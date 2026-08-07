"""Composes the five guardrails into the two hooks the agent loop calls.

The individual modules are deliberately standalone and independently testable;
this is the only place that knows the order they run in, and the order is part
of the design:

**Input, before any retrieval.** Emissions first -- the index legitimately
contains 432 units mentioning cat-delete, so a gate applied *after* retrieval is
a gate applied to a context window that already holds the thing it was meant to
exclude. Safety second, because a disabled vehicle outranks answering the
question that was actually asked.

**Output, after generation.** Grounding first (a price or availability claim
with no matching tool result in the same turn does not ship), then PII egress at
absolute zero, then escalation -- which runs on clean answers too, because 96%
of handoff need is capability rather than emotion.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import emissions, escalation, grounding, pii, safety

# Cheap language detection for the refusal path only. The customer's language
# is usually known from the channel, but a refusal is exactly the moment a
# wrong-language reply is most damaging, and French is 8.5% of contacts.
_FR_HINTS = re.compile(
    r"\b(je|j'ai|mon|ma|mes|le|la|les|est-ce|combien|puis-je|pouvez|vous|"
    r"catalyseur|supprimer|voiture|besoin|bonjour|merci|prix)\b",
    re.IGNORECASE,
)


def detect_language(text: str, fallback: str = "en") -> str:
    if not text:
        return fallback
    hits = len(_FR_HINTS.findall(text))
    return "fr" if hits >= 2 else fallback


@dataclass
class Verdict:
    """Uniform result for both hooks, so the loop has one shape to handle."""

    blocked: bool = False
    reason: str = ""
    tier: str = "soft"  # "hard" | "soft"
    response: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class Guardrails:
    def __init__(self) -> None:
        self._turns: dict[str, list] = {}
        self._lock = threading.Lock()

    # -- turn bookkeeping ----------------------------------------------------
    def begin_turn(self, session=None) -> str:
        turn_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._turns[turn_id] = []
            # A long-lived process must not accumulate one entry per turn
            # forever; evidence is only meaningful within its own turn.
            if len(self._turns) > 512:
                for stale in list(self._turns)[:256]:
                    self._turns.pop(stale, None)
        if session is not None:
            setattr(session, "_turn_id", turn_id)
        return turn_id

    def record_tool_result(self, session, tool_name: str, call_id: str, payload: Any) -> str:
        """Register a tool result as evidence; returns its provenance token.

        Grounding matches claims against these. This is the mechanism that
        makes an unsourced price unspeakable rather than merely discouraged.
        """
        turn_id = getattr(session, "_turn_id", None) or self.begin_turn(session)
        ev = grounding.record_tool_result(turn_id, tool_name, call_id, payload)
        with self._lock:
            self._turns.setdefault(turn_id, []).append(ev)
        return ev.provenance

    # -- input ---------------------------------------------------------------
    def check_input(self, message: str, *, session=None) -> Verdict:
        language = detect_language(message, getattr(session, "language", "en") or "en")
        if session is not None:
            session.language = language
        self.begin_turn(session)

        verdict = emissions.screen_query(message, language=language)
        if verdict.blocked:
            # Nothing is retrieved at all -- that is the point of pre-retrieval.
            return Verdict(
                blocked=True,
                reason="emissions_request",
                tier="hard",
                response=verdict.refusal,
                detail={"categories": list(verdict.categories)},
            )

        sv = safety.scan(message, language=language)
        if sv.escalate or str(sv.action) not in ("allow", "none", "ok"):
            return Verdict(
                blocked=True,
                reason="safety_stop",
                tier="hard",
                response=sv.message,
                detail={"triggers": [str(t) for t in sv.triggers][:6]},
            )
        return Verdict(blocked=False)

    # -- output --------------------------------------------------------------
    def check_output(
        self, answer: str, *, tool_results: Sequence[dict] | None = None, session=None
    ) -> Verdict:
        language = getattr(session, "language", "en") or "en"
        turn_id = getattr(session, "_turn_id", None) or "unknown"
        with self._lock:
            evidence = list(self._turns.get(turn_id, []))

        gv = grounding.validate(answer, evidence, turn_id=turn_id, language=language)
        if not gv.ok:
            # `action` is descriptive ("block_and_handoff"); `ok` is the
            # decision. Branching on the string would silently stop blocking
            # the day a new action name is introduced.
            return Verdict(
                blocked=True,
                reason=gv.handoff_reason or "ungrounded_claim",
                tier="hard",
                response=gv.answer,
                detail={"violations": [str(v.reason) for v in gv.violations][:6]},
            )
        answer = gv.answer or answer

        report = pii.scan_egress(answer, language=language)
        if not report.clean:
            return Verdict(
                blocked=True,
                reason="pii_egress",
                tier="hard",
                response=(
                    "I need to pass this to a person to answer it safely -- "
                    "let me connect you."
                ),
                detail={"entities": [str(e) for e in report.entities][:6]},
            )

        return Verdict(blocked=False, response=answer)


_instance: Guardrails | None = None
_lock = threading.Lock()


def get_guardrails() -> Guardrails:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Guardrails()
    return _instance
