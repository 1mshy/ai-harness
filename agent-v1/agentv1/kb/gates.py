"""Publication gates and derived risk labels, applied once at Phase 4a.

Order matters and is not arbitrary. ``training_safe`` runs first and is
non-bypassable -- matching the discipline the DGX pipeline already enforces,
where even ``--no-filter`` deliberately does not bypass it.

The gate that is easy to get wrong is #3. ``review.emissions_handling ==
"complied_improperly"`` marks calls where the agent helped with an emissions
defeat. Those documents are ``training_safe: true`` *and* ``useful_content:
true``, so nothing else stops them, and they would embed alongside the 201
correct refusals. ``training_safe`` is a PII gate; it screens nothing about
behavioural correctness. Two different questions, two different gates.

Note the array trap throughout. ``review.incorrect_statements`` is a *list*::

    {"review.incorrect_statements": True}              -> 0 documents
    {"review.incorrect_statements.0": {"$exists": 1}}  -> 535 documents

The same applies to ``threat_flags``, ``agent_knowledge_gaps`` and
``pii_spoken_in_call``. Any dashboard querying these as booleans is silently
reporting zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- Derived-label lexicons --------------------------------------------------
# emissions_risk is deliberately broader than review.emissions_tampering_request.
# That flag fires on the *request* and is true on 268 calls, but the exposure
# lives in the knowledge units: 432 units mention cat-delete/catless/de-cat,
# only 11 (2.5%) carry any refusal language, and 399 (92%) sit on calls the
# review pass never flagged -- they are written as neutral product facts. A
# gate built on the review flag misses 92% of what it is meant to catch.
_EMISSIONS_RE = re.compile(
    r"\b("
    r"cat(?:alytic)?[\s\-]?(?:converter[\s\-]?)?delete|catless|de[\s\-]?cat|test[\s\-]?pipe|"
    r"dpf[\s\-]?delete|egr[\s\-]?delete|gpf[\s\-]?delete|scr[\s\-]?delete|adblue[\s\-]?delete|"
    r"def[\s\-]?delete|o2[\s\-]?(?:spacer|simulator|defoul)|oxygen sensor spacer|"
    r"readiness[\s\-]?(?:defeat|monitor.{0,12}(?:off|disable))|"
    r"emissions?[\s\-]?(?:delete|defeat|removal|tamper)|off[\s\-]?road[\s\-]?only|"
    r"decatalyseur|suppression[\s\-]?(?:du[\s\-]?)?catalyseur|sans[\s\-]?catalyseur"
    r")\b",
    re.IGNORECASE,
)
_EMISSIONS_CODE_RE = re.compile(r"\bP04(?:20|21|22|30|31)\b", re.IGNORECASE)
_SMOG_RE = re.compile(r"\b(smog|emissions? (?:test|inspection)|readiness monitor)\b", re.IGNORECASE)

# safety_gated marks units whose advice, if wrong, immobilises or endangers a
# vehicle. The dominant measured pattern is a car disabled by a failed flash:
# engine damage 301, drivability/limp 205, clutch/transmission 149,
# fuel/fire/EGT 123, turbo overpressure 108 across 1,534 flagged calls. The
# safety surface and the #1 technical category are the same population.
_SAFETY_RE = re.compile(
    r"\b("
    r"won'?t start|will not start|no start|limp mode|limp home|bricked|"
    r"engine damage|rod knock|piston|melted|glowing|smoke|smoking|fire|"
    r"overboost|over[\s\-]?boost|overheat|egt|detonation|knock|misfire|"
    r"towed|tow truck|stranded|clutch slip|transmission fail"
    r")\b",
    re.IGNORECASE,
)

# dealer_pricing marks cost or margin visible to a customer-facing surface.
# 215 pricing units contain dealer cost or margin, 17 of them from
# end_customer calls.
_DEALER_COST_RE = re.compile(
    r"\b(dealer (?:cost|price|net)|your cost|our cost|wholesale|margin|markup|msrp vs|"
    r"distributor price|net price|cost to you)\b",
    re.IGNORECASE,
)

_PRICE_RE = re.compile(r"(?:\$|\bUSD\b|\bCAD\b|\bEUR\b)\s?\d|(?<!\w)\d{2,6}\s?(?:dollars|\$)")

_STAGE_CLAIM_RE = re.compile(
    r"\b(stage\s*[0-4]\s*\+?|released|not (?:yet )?(?:out|available|released)|coming soon)\b",
    re.IGNORECASE,
)


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    labels: dict[str, Any] = field(default_factory=dict)


def unit_blob(unit: dict) -> str:
    """Concatenated text a gate should look at."""
    return " ".join(
        str(unit.get(f) or "")
        for f in ("title", "question", "answer", "conditions", "confidence_reason")
    )


def document_gate(doc: dict, *, is_case: bool = False) -> GateResult:
    """Document-level gates. Runs before any unit is extracted."""
    # 1. training_safe -- non-bypassable, no override flag exists on purpose.
    if doc.get("training_safe") is not True:
        return GateResult(False, "training_safe_false")

    # 2. Unscreened -- fail closed. A case is training_safe only as an AND over
    #    its members, so honour the member-level counters too rather than
    #    trusting the rolled-up boolean alone.
    if is_case:
        if (doc.get("unscreened_members") or 0) > 0:
            return GateResult(False, "unscreened_members")
        if (doc.get("withheld_members") or 0) > 0:
            return GateResult(False, "withheld_members")
        if doc.get("status") != "done":
            return GateResult(False, f"case_status_{doc.get('status')}")
    else:
        if doc.get("useful_content") is not True:
            return GateResult(False, "not_useful_content")
        if doc.get("status") != "done":
            return GateResult(False, f"status_{doc.get('status')}")

    review = doc.get("review") or {}

    # 3. Behavioural correctness -- see module docstring. These documents are
    #    training_safe AND useful_content and nothing else would stop them.
    if review.get("emissions_handling") == "complied_improperly":
        return GateResult(False, "emissions_complied_improperly")

    # 4. The agent said something factually wrong on this call. Array, not bool.
    if isinstance(review.get("incorrect_statements"), list) and review["incorrect_statements"]:
        return GateResult(False, "incorrect_statements")

    return GateResult(True, "", {"review_present": bool(review)})


def derive_labels(unit: dict, doc: dict, *, is_case: bool = False) -> dict[str, Any]:
    """Risk labels carried into the payload and used as retrieval filters."""
    blob = unit_blob(unit)
    review = doc.get("review") or {}

    emissions = bool(
        _EMISSIONS_RE.search(blob)
        or _EMISSIONS_CODE_RE.search(blob)
        or _SMOG_RE.search(blob)
        or review.get("emissions_tampering_request") is True
    )
    safety = bool(_SAFETY_RE.search(blob) or review.get("safety_issue") is True)
    contains_price = bool(_PRICE_RE.search(blob))
    dealer = bool(_DEALER_COST_RE.search(blob)) or (
        contains_price and doc.get("caller_type") in ("dealer_installer", "distributor")
    )
    kind = (unit.get("kind") or "").strip().lower()

    return {
        "emissions_risk": emissions,
        "safety_gated": safety,
        "dealer_pricing": dealer,
        "contains_price": contains_price,
        "stage_claim": bool(_STAGE_CLAIM_RE.search(blob)),
        # internal_process is a *kind*, not a visibility flag, so the mapping is
        # made explicit here rather than inferred at query time by whoever
        # happens to remember. Open question in AGENT_PLAN.md §16.11: these
        # 2,564 units are correct but not customer-facing.
        "internal_only": kind == "internal_process",
        "time_sensitive": bool(unit.get("time_sensitive")),
        "agent_uncertain": bool(unit.get("agent_uncertain")),
    }


def unit_gate(unit: dict, labels: dict) -> GateResult:
    """Unit-level gates that run after the document passed."""
    if not (unit.get("answer") or "").strip():
        return GateResult(False, "empty_answer")
    if not (unit.get("title") or "").strip():
        return GateResult(False, "empty_title")
    # A unit that is itself a piece of emissions-defeat instruction is dropped,
    # not merely labelled. Labelling would leave it retrievable to anyone who
    # passes emissions_risk=True, and no legitimate caller needs to.
    text = unit_blob(unit)
    if labels.get("emissions_risk") and _EMISSIONS_RE.search(text):
        if not re.search(
            r"\b(cannot|can't|will not|won'?t|unable|not (?:able|permitted|legal)|"
            r"refus|decline|illegal|against (?:the )?law|not street legal|"
            r"we do not (?:offer|support|provide))\b",
            text,
            re.IGNORECASE,
        ):
            return GateResult(False, "emissions_instruction_without_refusal")
    return GateResult(True)
