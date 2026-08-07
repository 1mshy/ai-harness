"""Guardrails. AGENT_PLAN.md 9.

Five gates, and the order they run in is part of the design:

    1. emissions.screen_query      PRE-retrieval.  Blocks -> canned refusal,
                                   nothing is embedded or searched.
    2. safety.scan                 PRE-retrieval.  Unconditional stop on a
                                   stated physical-state failure.
    3. safety.check_flash_preconditions
                                   Before any flashing instruction.
    4. grounding.validate          POST-generation. Every price and
                                   availability claim needs a tool result
                                   from this turn.
    5. pii.scan_egress             On the way out. Payload, SSE frame, final
                                   answer -- one call for all three.

    escalation.evaluate            After any of the above, and after an
                                   ordinary turn too.

1 and 2 run before retrieval because a refusal that already searched is a
refusal with the thing it refused sitting in the context window. 4 runs after
generation because it is the only place a claim exists to check. 5 runs last
because it is the last thing that can stop a leak.

Every module here is deterministic. There is not an LLM call in the package,
which is what makes the controls survive a model swap, a provider fallback and
a prompt injection.

Each module has a runnable self-check that measures itself against live Mongo::

    .venv/bin/python -m agentv1.guardrails.emissions
    .venv/bin/python -m agentv1.guardrails.safety
    .venv/bin/python -m agentv1.guardrails.grounding
    .venv/bin/python -m agentv1.guardrails.pii
    .venv/bin/python -m agentv1.guardrails.escalation

Measured 2026-08-06:

    emissions   229/268 (85.4%) of the historically flagged calls refused,
                against a 74.3% human baseline; 5.4% of unflagged calls
                blocked, which is an upper bound on false positives.
    safety      716/1,534 (46.7%) of review.safety_issue calls stopped on
                stated physical state; 2.4% counter-rate.
    grounding   a claim not present in a same-turn tool result is blocked;
                88 currently-unreleased stage rows are live in
                tuning_platforms.
    pii         100% of 3CX recording filenames, 97% of caller numbers,
                0% of person names -- the last figure is the argument for
                Presidio in 9.8.
    escalation  sentiment alone reaches 10.0% of handoff need; there is no
                routing substrate and every decision says so.
"""

from .compose import Guardrails, Verdict, detect_language, get_guardrails
from .emissions import (
    EmissionsMatch,
    EmissionsVerdict,
    LexiconError,
    classify as classify_emissions,
    load_lexicon,
    refusal_text,
    screen_query,
    screen_unit,
)
from .escalation import (
    EscalationDecision,
    EscalationSignals,
    EscalationTrigger,
    RoutingSubstrate,
    customer_asked_for_human,
    ensure_escalation_indexes,
    evaluate as evaluate_escalation,
    open_escalations,
    record_escalation,
    routing_substrate,
)
from .grounding import (
    Claim,
    GroundingVerdict,
    ToolEvidence,
    Violation,
    extract_claims,
    mint_provenance,
    record_tool_result,
    validate as validate_grounding,
    verify as verify_provenance,
)
from .pii import (
    PIIBackend,
    PIIEgressError,
    PIIEntity,
    PIIReport,
    PresidioBackend,
    RegexBackend,
    assert_clean,
    get_backend,
    redact,
    scan_egress,
    scan_text,
)
from .safety import (
    FlashContext,
    Precondition,
    SafetyTrigger,
    SafetyVerdict,
    check_flash_preconditions,
    guard_flash_instruction,
    scan as scan_safety,
)

__all__ = [
    # composition -- the two hooks the agent loop calls
    "Guardrails",
    "Verdict",
    "get_guardrails",
    "detect_language",
    # emissions
    "EmissionsMatch",
    "EmissionsVerdict",
    "LexiconError",
    "classify_emissions",
    "load_lexicon",
    "refusal_text",
    "screen_query",
    "screen_unit",
    # safety
    "FlashContext",
    "Precondition",
    "SafetyTrigger",
    "SafetyVerdict",
    "check_flash_preconditions",
    "guard_flash_instruction",
    "scan_safety",
    # grounding
    "Claim",
    "GroundingVerdict",
    "ToolEvidence",
    "Violation",
    "extract_claims",
    "mint_provenance",
    "record_tool_result",
    "validate_grounding",
    "verify_provenance",
    # pii
    "PIIBackend",
    "PIIEgressError",
    "PIIEntity",
    "PIIReport",
    "PresidioBackend",
    "RegexBackend",
    "assert_clean",
    "get_backend",
    "redact",
    "scan_egress",
    "scan_text",
    # escalation
    "EscalationDecision",
    "EscalationSignals",
    "EscalationTrigger",
    "RoutingSubstrate",
    "customer_asked_for_human",
    "ensure_escalation_indexes",
    "evaluate_escalation",
    "open_escalations",
    "record_escalation",
    "routing_substrate",
]
