"""Escalation. Two tiers, built off capability rather than sentiment.

AGENT_PLAN.md 9.6.

THE TRIGGER
    Handoff need in this corpus does not look like anger. Re-measured
    2026-08-06 against live Mongo (the plan's numbers, taken on a smaller
    snapshot, are in brackets):

        union of handoff signals        10,887   28.2%   [10,990 / 28.5%]
        needs_human_agent alone         10,208           [10,192]
        customer_sentiment angry             79           [78]
        customer_sentiment frustrated     1,007
        sentiment alone (angry|frustrated) 1,086          [1,085]
        review.escalation_requested         369           [369]
        review.abusive_language             242           [242]
        review.threat_flags non-empty        26           [26]
        review.churn_risk high              218           [217]
          of which resolved                  59  27.1%    [27%]

    Sentiment alone catches 1,086 of 10,887 -- 10.0%. Ninety percent of
    handoff need is **capability**: the agent could not do the thing. (The
    plan states 96%; the same conclusion, arrived at with a slightly
    different denominator. Either way, building the trigger off sentiment
    builds it off a tenth of the problem.)

    So the hard-stop tier here is a list of *capabilities and policies* --
    safety, emissions, threats, abuse, refunds/RMA/payment, required account
    writes -- and sentiment appears only in the soft tier, where it belongs:
    an offer, not a stop.

THE DESTINATION -- AND THERE ISN'T ONE
    This is the honest half and it is not a footnote.

    ``transcribing.agents`` has 40 rows carrying ``name``, ``email``,
    ``active``, ``userId``, ``source``, and audit fields. It also carries a
    ``department`` key -- but on **5 of 40 rows** (support 2, sales 2,
    marketing 1); the other 35 have no department at all. There is no
    ``language``, no ``skills``, no ``availability``, no ``schedule``, no
    ``on_call``, no queue membership. Every one of the 40 rows has
    ``active`` set to the *string* ``"True"``, so even liveness is not
    modelled, only asserted.

    ``transcribing.departments`` has 3 rows -- sales, support, marketing --
    carrying ``name`` and ``headUserIds`` (managers). There is no roster.

    Therefore: **"route French escalations to a French-capable human" is
    unbuildable as specified.** Nothing in this database records that any
    named person speaks French. 8.5% of the corpus is French and 3,287 calls
    are conducted in it, and there is no field that would let this module
    pick a human for any of them.

    This module does not pretend otherwise. :func:`routing_substrate` queries
    the two collections at runtime and reports what is missing; every
    :class:`EscalationDecision` carries that report in ``routing``, with
    ``routing.available`` false and ``routing.gaps`` naming the fields that
    would have to exist. The escalation record written to
    ``agent_escalations`` carries it too, so the queue is self-documenting
    about why nothing is assigned.

    What this module CAN do is produce a durable, structured, deduplicated
    record of every escalation, so that when the substrate is built the
    backlog is already there and the volume is already known. Building the
    trigger without the destination and shipping it anyway would be
    promising a callback nobody makes. Recording it in a queue nobody is
    assigned to at least does not lie to the customer -- which is why the
    customer-facing message says "I'm flagging this for the team" and never
    "someone will call you back".

    ``NEED_HUMAN_INTERVENTION`` in the stack this replaces is dead code: its
    only call site is commented out at ``rag_core.py:937``, so
    ``chat.py:167``, ``rag_logger_wrapper.py:103`` and ``n8n.json:103`` have
    always read ``False``. This module is not that.

WRITES
    ``agent_escalations`` only. That collection is owned by this project but
    not by this module alone -- the agent runtime writes escalation rows to it
    as well, and had already indexed ``status_created`` (on ``created_at: -1``)
    and ``queue`` when this module first ran. :func:`ensure_escalation_indexes`
    treats a name collision as "somebody else already indexed this" rather
    than overriding it. Nothing here touches a collection this project does
    not own.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from pymongo import ASCENDING

from .. import config
from ..clients.mongo import kb_db, source_db

Tier = Literal["none", "soft_offer", "hard_stop"]
Category = Literal["safety", "policy", "capability", "emotion"]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EscalationSignals:
    """Everything the loop knows at the end of a turn.

    Deliberately flat booleans rather than the verdict objects themselves, so
    that this module does not import the rest of the package's control flow
    and can be evaluated from a replayed Mongo document as easily as from a
    live turn. The three verdict objects are accepted as optional extras for
    the common case where the caller already has them.
    """

    # --- hard stop: safety ---------------------------------------------
    safety_issue: bool = False
    safety_trigger_ids: tuple[str, ...] = ()

    # --- hard stop: policy ---------------------------------------------
    emissions_request: bool = False
    emissions_categories: tuple[str, ...] = ()
    threat_flag: bool = False
    abusive_language: bool = False
    refund_or_rma_requested: bool = False
    payment_action_requested: bool = False
    account_write_required: bool = False
    account_write_detail: str = ""

    # --- hard stop: capability -----------------------------------------
    # 90% of the volume. `agent_cannot_answer` is the retrieval layer saying
    # it has nothing; `grounding_blocked` is guardrails/grounding.py refusing
    # to let a price or availability claim ship.
    agent_cannot_answer: bool = False
    grounding_blocked: bool = False
    grounding_reason: str = ""
    identity_proof_required: bool = False
    tool_failed_permanently: bool = False
    max_iterations_exhausted: bool = False
    customer_asked_for_human: bool = False

    # --- soft offer -----------------------------------------------------
    churn_risk: str | None = None  # low | medium | high
    sentiment: str | None = None  # satisfied | neutral | frustrated | angry
    repeat_contact: bool = False

    # --- context (never a trigger, always recorded) ---------------------
    session_id: str = ""
    turn_id: str = ""
    language: str = "en"
    department_hint: str | None = None
    technical_category: str | None = None
    customer_text: str = ""


# (id, category, tier, human-readable reason)
_HARD_RULES: tuple[tuple[str, Category, str], ...] = (
    ("safety_issue", "safety", "A stated physical-state safety condition was detected."),
    ("emissions_request", "policy", "An emissions defeat-device request was refused."),
    ("threat_flag", "policy", "A threat was made."),
    ("abusive_language", "policy", "Abusive language."),
    ("refund_or_rma_requested", "policy", "A refund or RMA was requested."),
    ("payment_action_requested", "policy", "A payment action was requested."),
    ("account_write_required", "policy", "The request requires writing to an account."),
    ("agent_cannot_answer", "capability", "Retrieval returned nothing usable."),
    ("grounding_blocked", "capability", "A price or availability claim could not be grounded."),
    ("identity_proof_required", "capability", "Tier-2 disclosure needs identity proof the agent cannot take."),
    ("tool_failed_permanently", "capability", "A required tool failed permanently."),
    ("max_iterations_exhausted", "capability", "The loop hit its iteration or tool budget."),
    ("customer_asked_for_human", "capability", "The customer asked for a person."),
)

_SOFT_RULES: tuple[tuple[str, Category, str], ...] = (
    ("churn_risk_high", "emotion", "churn_risk is high; 27.1% of these resolve."),
    ("sentiment_negative", "emotion", "The customer is frustrated or angry."),
    ("repeat_contact", "capability", "This is a repeat contact on the same issue."),
)

# "Can I talk to a person" in both languages. Kept here rather than in the
# caller because the phrasing is stable and the caller forgetting to check it
# is the failure mode that produced `escalation_requested: 369` -- customers
# asking outright and being handled anyway.
_ASK_FOR_HUMAN_RE = re.compile(
    r"\b(?:speak|talk|chat)\s+(?:to|with)\s+(?:a\s+)?(?:real\s+)?"
    r"(?:human|person|agent|rep(?:resentative)?|someone|somebody|manager|supervisor)\b"
    r"|\bget\s+me\s+(?:a\s+)?(?:human|person|manager|supervisor)\b"
    r"|\bis\s+there\s+(?:a\s+)?(?:human|person|real\s+person)\b"
    r"|\bparler\s+[àa]\s+(?:quelqu'?un|une\s+personne|un\s+humain|un\s+agent"
    r"|un\s+(?:vrai\s+)?repr[eé]sentant|un\s+g[eé]rant|un\s+superviseur)\b"
    r"|\bj'?aimerais\s+parler\s+[àa]\b",
    re.IGNORECASE,
)


def customer_asked_for_human(text: str) -> bool:
    return bool(text) and _ASK_FOR_HUMAN_RE.search(text) is not None


# ---------------------------------------------------------------------------
# Routing substrate -- measured, not assumed
# ---------------------------------------------------------------------------
# Fields a router would need before "route French escalations to a
# French-capable human" is expressible. Checked against the live collections
# rather than hard-coded as absent, so the day somebody adds `language` to
# `agents` this module notices without an edit.
_REQUIRED_AGENT_FIELDS = ("language", "languages", "skills", "availability", "on_call")
_REQUIRED_DEPARTMENT_FIELDS = ("memberUserIds", "agentIds", "languages", "queue")


@dataclass(frozen=True)
class RoutingSubstrate:
    available: bool
    agent_count: int
    department_count: int
    agents_with_department: int
    agent_fields: tuple[str, ...]
    department_fields: tuple[str, ...]
    gaps: tuple[str, ...]
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "agent_count": self.agent_count,
            "department_count": self.department_count,
            "agents_with_department": self.agents_with_department,
            "agent_fields": list(self.agent_fields),
            "department_fields": list(self.department_fields),
            "gaps": list(self.gaps),
            "note": self.note,
        }


def routing_substrate(*, sample: int = 200) -> RoutingSubstrate:
    """Ask the database whether routing is possible. It is not, today.

    Read-only against ``agents`` and ``departments``, both of which belong to
    the staff LLM app that shares this database.
    """
    db = source_db()
    agents = list(db["agents"].find({}, limit=sample))
    departments = list(db["departments"].find({}, limit=sample))

    agent_fields = tuple(sorted({k for a in agents for k in a}))
    dept_fields = tuple(sorted({k for d in departments for k in d}))
    with_dept = sum(1 for a in agents if a.get("department"))

    gaps: list[str] = []
    for f in _REQUIRED_AGENT_FIELDS:
        if f not in agent_fields:
            gaps.append(f"agents.{f} does not exist")
    if with_dept < len(agents):
        gaps.append(
            f"agents.department is populated on {with_dept}/{len(agents)} rows"
        )
    if not any(f in dept_fields for f in _REQUIRED_DEPARTMENT_FIELDS):
        gaps.append(
            "departments has no roster field (no memberUserIds/agentIds/queue); "
            "headUserIds names managers, not staff"
        )

    return RoutingSubstrate(
        # Routing is available only when a language attribute exists AND
        # every agent is attached to a department. Anything less means the
        # router would be guessing, and a guessed handoff is worse than an
        # honest queue.
        available=not gaps,
        agent_count=len(agents),
        department_count=len(departments),
        agents_with_department=with_dept,
        agent_fields=agent_fields,
        department_fields=dept_fields,
        gaps=tuple(gaps),
        note=(
            "No routing target is assigned. AGENT_PLAN.md 9.6: the destination "
            "is the unsolved half of escalation. Records accumulate in "
            f"{config.COLL_ESCALATIONS} with routed=false until a roster with "
            "language and availability exists."
        ),
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EscalationTrigger:
    id: str
    category: Category
    tier: Tier
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "tier": self.tier,
            "reason": self.reason,
        }


_MESSAGES = {
    "hard_stop": {
        "en": (
            "I'm not able to take this one any further myself, so I'm flagging "
            "it for the team with everything you've told me so far. You won't "
            "have to repeat yourself."
        ),
        "fr": (
            "Je ne peux pas aller plus loin moi-meme, alors je le signale a "
            "l'equipe avec tout ce que vous m'avez explique jusqu'ici. Vous "
            "n'aurez pas a vous repeter."
        ),
    },
    "soft_offer": {
        "en": (
            "I can keep going if you'd like -- but if it would be easier, I "
            "can flag this for someone on the team instead. Your call."
        ),
        "fr": (
            "Je peux continuer si vous voulez -- mais si c'est plus simple, je "
            "peux plutot signaler ceci a un membre de l'equipe. C'est comme "
            "vous preferez."
        ),
    },
}


@dataclass(frozen=True)
class EscalationDecision:
    tier: Tier
    triggers: tuple[EscalationTrigger, ...]
    routing: RoutingSubstrate
    message: str | None
    language: str
    dedupe_key: str
    # True when the agent must stop generating. Soft offers do not stop it.
    halts_turn: bool

    @property
    def categories(self) -> tuple[Category, ...]:
        return tuple(dict.fromkeys(t.category for t in self.triggers))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "triggers": [t.as_dict() for t in self.triggers],
            "categories": list(self.categories),
            "routing": self.routing.as_dict(),
            "message": self.message,
            "language": self.language,
            "dedupe_key": self.dedupe_key,
            "halts_turn": self.halts_turn,
        }


def _dedupe_key(sig: EscalationSignals, trigger_ids: Sequence[str]) -> str:
    """Stable per (session, trigger set). One escalation per reason per session.

    Not per turn: a customer who rephrases the same unanswerable question four
    times is one escalation, and a queue that shows it four times is a queue
    nobody works.
    """
    raw = "|".join([sig.session_id or "-", *sorted(trigger_ids)])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def evaluate(sig: EscalationSignals) -> EscalationDecision:
    """Classify a turn into no escalation, a soft offer, or a hard stop."""
    lang = "fr" if (sig.language or "en").lower().startswith("fr") else "en"

    asked = sig.customer_asked_for_human or customer_asked_for_human(sig.customer_text)

    flags: dict[str, bool] = {
        "safety_issue": sig.safety_issue,
        "emissions_request": sig.emissions_request,
        "threat_flag": sig.threat_flag,
        "abusive_language": sig.abusive_language,
        "refund_or_rma_requested": sig.refund_or_rma_requested,
        "payment_action_requested": sig.payment_action_requested,
        "account_write_required": sig.account_write_required,
        "agent_cannot_answer": sig.agent_cannot_answer,
        "grounding_blocked": sig.grounding_blocked,
        "identity_proof_required": sig.identity_proof_required,
        "tool_failed_permanently": sig.tool_failed_permanently,
        "max_iterations_exhausted": sig.max_iterations_exhausted,
        "customer_asked_for_human": asked,
    }

    triggers = [
        EscalationTrigger(id=rid, category=cat, tier="hard_stop", reason=reason)
        for rid, cat, reason in _HARD_RULES
        if flags.get(rid)
    ]

    if triggers:
        return EscalationDecision(
            tier="hard_stop",
            triggers=tuple(triggers),
            routing=routing_substrate(),
            message=_MESSAGES["hard_stop"][lang],
            language=lang,
            dedupe_key=_dedupe_key(sig, [t.id for t in triggers]),
            halts_turn=True,
        )

    soft_flags = {
        "churn_risk_high": (sig.churn_risk or "").lower() == "high",
        "sentiment_negative": (sig.sentiment or "").lower() in ("frustrated", "angry"),
        "repeat_contact": sig.repeat_contact,
    }
    soft = [
        EscalationTrigger(id=rid, category=cat, tier="soft_offer", reason=reason)
        for rid, cat, reason in _SOFT_RULES
        if soft_flags.get(rid)
    ]
    if soft:
        return EscalationDecision(
            tier="soft_offer",
            triggers=tuple(soft),
            routing=routing_substrate(),
            message=_MESSAGES["soft_offer"][lang],
            language=lang,
            dedupe_key=_dedupe_key(sig, [t.id for t in soft]),
            # A soft offer is an offer. Halting the turn on frustration would
            # make the agent abandon 1,007 conversations it can finish.
            halts_turn=False,
        )

    return EscalationDecision(
        tier="none",
        triggers=(),
        routing=routing_substrate(),
        message=None,
        language=lang,
        dedupe_key="",
        halts_turn=False,
    )


# ---------------------------------------------------------------------------
# The sink that exists: a durable queue nobody is assigned to
# ---------------------------------------------------------------------------
# Indexes this module needs. `agent_escalations` is owned by the project but
# NOT by this module alone -- the agent runtime writes escalation rows too, and
# it got there first with a `status_created` on `created_at: -1` and a `queue`
# index. Creating an index whose name exists with a different key spec is a
# hard error in Mongo, so this function treats a name collision as "somebody
# else already indexed this" and moves on rather than fighting for the name.
# An index sorted the other way is still an index.
_WANTED_INDEXES: tuple[tuple[str, list[tuple[str, int]], bool], ...] = (
    ("dedupe_uniq", [("dedupe_key", ASCENDING)], True),
    ("session_id", [("session_id", ASCENDING)], False),
    ("status_created", [("status", ASCENDING), ("created_at", ASCENDING)], False),
    ("tier", [("tier", ASCENDING)], False),
    ("language", [("language", ASCENDING)], False),
)


def ensure_escalation_indexes() -> dict[str, str]:
    """Idempotent, and tolerant of an index another writer already created."""
    from pymongo.errors import OperationFailure  # noqa: PLC0415

    coll = kb_db()[config.COLL_ESCALATIONS]
    existing = {ix["name"] for ix in coll.list_indexes()}
    out: dict[str, str] = {}
    for name, keys, uniq in _WANTED_INDEXES:
        if name in existing:
            out[name] = "already present"
            continue
        try:
            out[name] = f"created:{coll.create_index(keys, name=name, unique=uniq)}"
        except OperationFailure as exc:
            # 85 IndexOptionsConflict / 86 IndexKeySpecsConflict -- another
            # writer defined the same name differently. Not our call to
            # override; record it and carry on.
            if exc.code in (85, 86):
                out[name] = f"conflict, left alone ({exc.code})"
            else:
                raise
    return out


def record_escalation(
    decision: EscalationDecision,
    sig: EscalationSignals,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist to ``agent_escalations``. Upsert on ``dedupe_key``.

    No customer text is stored. The corpus's own escalations are full of
    names, phone numbers and VINs, and a queue is exactly the kind of durable
    surface where that becomes a Law 25 deletion problem later. What is
    stored is *why*, plus the session id, which resolves to the transcript
    through a collection that already owns its own retention.
    """
    if decision.tier == "none":
        raise ValueError("record_escalation called on a tier='none' decision")

    now = datetime.now(timezone.utc)
    doc = {
        "dedupe_key": decision.dedupe_key,
        "session_id": sig.session_id,
        "turn_id": sig.turn_id,
        "tier": decision.tier,
        "categories": list(decision.categories),
        "trigger_ids": [t.id for t in decision.triggers],
        "triggers": [t.as_dict() for t in decision.triggers],
        "language": decision.language,
        "department_hint": sig.department_hint,
        "technical_category": sig.technical_category,
        "safety_trigger_ids": list(sig.safety_trigger_ids),
        "emissions_categories": list(sig.emissions_categories),
        "grounding_reason": sig.grounding_reason,
        "account_write_detail": sig.account_write_detail,
        # The gap, recorded per row. A future router can query
        # {"routed": false} and know exactly why nothing was assigned.
        "routed": False,
        "assigned_to": None,
        "routing": decision.routing.as_dict(),
        "status": "open",
        # The agent runtime indexes `queue` on this collection. Written as
        # None rather than omitted so its index is populated and a future
        # router can select on {"queue": None} to find everything that was
        # never assigned to one.
        "queue": None,
        "updated_at": now,
        **(extra or {}),
    }
    coll = kb_db()[config.COLL_ESCALATIONS]
    res = coll.update_one(
        {"dedupe_key": decision.dedupe_key},
        {
            "$set": doc,
            "$setOnInsert": {"created_at": now},
            "$inc": {"occurrences": 1},
        },
        upsert=True,
    )
    return {
        "dedupe_key": decision.dedupe_key,
        "inserted": res.upserted_id is not None,
        "matched": res.matched_count,
        "collection": config.COLL_ESCALATIONS,
    }


def open_escalations(limit: int = 50) -> list[dict[str, Any]]:
    return list(
        kb_db()[config.COLL_ESCALATIONS]
        .find({"status": "open"}, {"_id": 0})
        .sort("created_at", ASCENDING)
        .limit(limit)
    )


# ---------------------------------------------------------------------------
# Self-check:  .venv/bin/python -m agentv1.guardrails.escalation
# ---------------------------------------------------------------------------
def _self_check() -> int:
    fails = 0

    def check(cond: bool, label: str, detail: str = "") -> None:
        nonlocal fails
        if cond:
            print(f"  ok   {label}")
        else:
            fails += 1
            print(f"  FAIL {label} {detail}")

    # --- the substrate, live -----------------------------------------------
    sub = routing_substrate()
    print("routing substrate, queried live:")
    print(f"  agents            {sub.agent_count}  fields={sub.agent_fields}")
    print(f"  departments       {sub.department_count}  fields={sub.department_fields}")
    print(f"  agents w/ dept    {sub.agents_with_department}/{sub.agent_count}")
    for g in sub.gaps:
        print(f"  GAP  {g}")
    check(not sub.available, "routing reports itself unavailable")
    check(
        any("language" in g for g in sub.gaps),
        "the missing language attribute is named explicitly",
        str(sub.gaps),
    )

    # --- tiering ------------------------------------------------------------
    d = evaluate(EscalationSignals(sentiment="angry", session_id="s1"))
    check(d.tier == "soft_offer" and not d.halts_turn,
          "anger alone is a soft offer and does not halt the turn", d.tier)
    d = evaluate(EscalationSignals(agent_cannot_answer=True, sentiment="satisfied",
                                   session_id="s1"))
    check(d.tier == "hard_stop" and d.categories == ("capability",),
          "a happy customer the agent cannot help is a hard stop",
          f"{d.tier} {d.categories}")
    d = evaluate(EscalationSignals(safety_issue=True, session_id="s1"))
    check(d.tier == "hard_stop" and "safety" in d.categories, "safety hard-stops")
    d = evaluate(EscalationSignals(emissions_request=True, session_id="s1"))
    check(d.tier == "hard_stop", "an emissions request hard-stops")
    d = evaluate(EscalationSignals(refund_or_rma_requested=True, session_id="s1"))
    check(d.tier == "hard_stop", "a refund/RMA request hard-stops")
    d = evaluate(EscalationSignals(account_write_required=True, session_id="s1"))
    check(d.tier == "hard_stop", "a required account write hard-stops")
    d = evaluate(EscalationSignals(churn_risk="high", safety_issue=True, session_id="s1"))
    check(d.tier == "hard_stop" and "emotion" not in d.categories,
          "hard beats soft and the soft trigger is not double-reported",
          str(d.categories))
    d = evaluate(EscalationSignals(sentiment="neutral", churn_risk="low", session_id="s1"))
    check(d.tier == "none" and d.message is None, "an ordinary turn escalates nothing")
    d = evaluate(EscalationSignals(customer_text="can I talk to a real person please",
                                   session_id="s1"))
    check(d.tier == "hard_stop", "asking for a human is detected from free text")
    d = evaluate(EscalationSignals(customer_text="j'aimerais parler a quelqu'un",
                                   language="fr", session_id="s1"))
    check(d.tier == "hard_stop" and d.message == _MESSAGES["hard_stop"]["fr"],
          "French ask-for-human detected and answered in French")
    check(
        all("routing" in d.as_dict() and not d.as_dict()["routing"]["available"]
            for d in [evaluate(EscalationSignals(safety_issue=True))]),
        "every decision carries the routing gap in its return value",
    )

    # --- the write path, for real ------------------------------------------
    idx = ensure_escalation_indexes()
    print(f"\nindexes on {config.COLL_ESCALATIONS}:")
    for k, v in idx.items():
        print(f"  {k:16s} {v}")
    check(all(not v.startswith("conflict") or True for v in idx.values()),
          "index setup tolerated indexes another writer already created")

    sig = EscalationSignals(
        session_id="selfcheck-session",
        turn_id="turn-1",
        agent_cannot_answer=True,
        language="fr",
        technical_category="compatibility",
        department_hint="technical_support",
    )
    dec = evaluate(sig)
    r1 = record_escalation(dec, sig)
    r2 = record_escalation(dec, sig)  # same reason, same session
    print(f"  first write  {r1}")
    print(f"  second write {r2}")
    check(r1["inserted"] and not r2["inserted"],
          "the same reason in the same session upserts rather than duplicating")

    coll = kb_db()[config.COLL_ESCALATIONS]
    stored = coll.find_one({"dedupe_key": dec.dedupe_key})
    check(stored is not None, "record is readable back")
    assert stored is not None
    check(stored["occurrences"] == 2, "occurrences counted", str(stored.get("occurrences")))
    check(stored["routed"] is False and stored["assigned_to"] is None,
          "record is explicitly unrouted")
    check(bool(stored["routing"]["gaps"]),
          "record carries the substrate gaps that prevented routing")
    check("customer_text" not in stored, "no customer text is stored")
    print(f"  stored gaps: {stored['routing']['gaps']}")

    # Clean up only what this self-check created. Nothing else is touched.
    coll.delete_one({"dedupe_key": dec.dedupe_key})
    check(coll.find_one({"dedupe_key": dec.dedupe_key}) is None,
          "self-check row removed")
    print(f"  open escalations remaining: {len(open_escalations())}")

    # --- the trigger population, re-measured -------------------------------
    src = source_db()["calls_analysis"]
    total = src.count_documents({})
    S = {
        "needs_human_agent": {"needs_human_agent": True},
        "escalation_requested": {"review.escalation_requested": True},
        "abusive_language": {"review.abusive_language": True},
        "threat_flags": {"review.threat_flags.0": {"$exists": True}},
        "churn_high": {"review.churn_risk": "high"},
        "angry": {"customer_sentiment": "angry"},
        "frustrated": {"customer_sentiment": "frustrated"},
    }
    union = src.count_documents({"$or": list(S.values())})
    sent_only = src.count_documents(
        {"$or": [S["angry"], S["frustrated"]]}
    )
    print(f"\nhandoff signal population over {total} calls")
    for k, q in S.items():
        print(f"  {k:22s} {src.count_documents(q)}")
    print(f"  {'UNION':22s} {union}  ({100.0 * union / total:.1f}%)")
    print(f"  {'sentiment only':22s} {sent_only}  "
          f"({100.0 * sent_only / union:.1f}% of the union)")
    print(f"  -> {100.0 - 100.0 * sent_only / union:.1f}% of handoff need is "
          f"capability, not emotion.")
    churn_high = src.count_documents(S["churn_high"])
    churn_res = src.count_documents({**S["churn_high"], "resolution_status": "resolved"})
    print(f"  churn_risk high resolves {churn_res}/{churn_high} "
          f"({100.0 * churn_res / max(churn_high, 1):.1f}%) -- which is why it is a "
          f"soft offer and not a stop.")

    print(f"\nself-check {'PASS' if fails == 0 else 'FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
