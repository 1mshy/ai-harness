"""Control tools: the four things the agent is allowed to *do* rather than say.

Everything here writes, and everything here writes only to collections this
project owns -- ``agent_escalations``, ``agent_leads``, ``agent_knowledge_gaps``
and ``agent_approvals``. The source collections (``calls_*``, ``tuning_*``) are
owned by the DGX pipeline and by an internal staff app; nothing in this file
can reach them, because it goes through ``kb_db()`` and never ``source_db()``.

Design notes worth keeping:

*Escalation is built before the trigger.* AGENT_PLAN.md §9.6 is blunt about
this: an escalation that writes to a collection nobody reads is a silent drop
with extra steps. The record carries the routing queue and an explicit
``handoff_context`` block so a human picks up mid-conversation instead of
starting over.

*Escalation is idempotent within a session.* A model that decides to escalate
tends to decide it again on the next iteration of the same loop. The write is
an upsert on ``(session_id, reason, summary digest)``, so a three-iteration
panic produces one ticket with ``repeat_count: 3`` rather than three tickets.

*A lead requires explicit consent.* CASL exposure is real and the corpus
already shows the shape of the problem -- the honest upsell population is
~16,500 accounts, not the 72,262 that a naive "owns less than released" query
returns. ``consent_to_contact`` is a required argument, not a default.

*Knowledge gaps are the eval loop's input.* 5,261 calls carry a non-empty
``agent_unanswered_questions``; this is the live continuation of that signal
and it is what tells you which documentation gap to close next.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING

from .. import config
from ..clients.mongo import kb_db
from .base import Tool, ToolInputError, obj_schema

# Owned by this project. Named here rather than in config because config is
# shared with the KB build and these are runtime artefacts of the agent loop.
COLL_APPROVALS = "agent_approvals"

ESCALATION_REASONS = [
    "emissions_request",
    "safety_issue",
    "pricing_authority",
    "account_action",
    "refund_or_rma",
    "unresolved_after_attempts",
    "customer_requested",
    "abuse_or_threat",
    "out_of_scope",
    "low_confidence",
]

# Where the ticket actually goes. An escalation without a queue is a write to
# a collection nobody polls.
_QUEUE_FOR_REASON = {
    "emissions_request": "compliance",
    "safety_issue": "compliance",
    "abuse_or_threat": "compliance",
    "pricing_authority": "sales_lead",
    "refund_or_rma": "returns",
    "account_action": "accounts",
    "customer_requested": "support_l1",
    "unresolved_after_attempts": "support_l2",
    "out_of_scope": "support_l1",
    "low_confidence": "support_l1",
}

_indexes_ready = False


def _ensure_indexes() -> None:
    """Idempotent, lazy. Called on the first write, not at import, so importing
    the tool layer does not require a reachable Mongo."""
    global _indexes_ready
    if _indexes_ready:
        return
    db = kb_db()
    db[config.COLL_ESCALATIONS].create_index(
        [("session_id", ASCENDING), ("dedupe_key", ASCENDING)],
        unique=True,
        name="session_dedupe",
    )
    db[config.COLL_ESCALATIONS].create_index([("status", ASCENDING), ("created_at", DESCENDING)], name="status_created")
    db[config.COLL_ESCALATIONS].create_index([("queue", ASCENDING)], name="queue")
    db[config.COLL_LEADS].create_index([("session_id", ASCENDING)], name="session")
    db[config.COLL_LEADS].create_index([("created_at", DESCENDING)], name="created")
    db[config.COLL_KNOWLEDGE_GAPS].create_index(
        [("question_key", ASCENDING)], unique=True, name="question_key_uniq"
    )
    db[config.COLL_KNOWLEDGE_GAPS].create_index([("count", DESCENDING)], name="count")
    db[COLL_APPROVALS].create_index([("approval_id", ASCENDING)], unique=True, name="approval_id_uniq")
    db[COLL_APPROVALS].create_index(
        [("session_id", ASCENDING), ("state", ASCENDING)], name="session_state"
    )
    _indexes_ready = True


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _digest(*parts: Any) -> str:
    return hashlib.sha256("\x00".join(str(p or "") for p in parts).encode()).hexdigest()[:16]


def _ticket(prefix: str) -> str:
    return f"{prefix}-{_now():%Y%m%d}-{secrets.token_hex(4)}"


# --- escalate_to_human -------------------------------------------------------


def escalate_to_human(
    reason: str,
    summary: str,
    *,
    session_id: str,
    customer_id: str | None = None,
    persona: str | None = None,
    urgency: str = "normal",
    customer_question: str | None = None,
) -> dict:
    if reason not in ESCALATION_REASONS:
        raise ToolInputError(f"reason must be one of {ESCALATION_REASONS}")
    if len((summary or "").strip()) < 20:
        # A one-word summary makes the human restart the conversation, which
        # is the outcome escalation exists to prevent.
        raise ToolInputError(
            "summary must be a real handoff note (>=20 chars): what the customer "
            "wants, what has been established, and what is blocked."
        )
    _ensure_indexes()
    queue = _QUEUE_FOR_REASON.get(reason, "support_l1")
    dedupe = _digest(reason, summary.strip().lower())
    now = _now()
    doc = {
        "session_id": session_id,
        "dedupe_key": dedupe,
        "reason": reason,
        "queue": queue,
        "urgency": urgency if urgency in ("low", "normal", "high") else "normal",
        "customer_id": customer_id,
        "persona": persona,
        "handoff_context": {
            "summary": summary.strip(),
            "customer_question": (customer_question or "").strip() or None,
        },
        "status": "open",
        "updated_at": now,
    }
    res = kb_db()[config.COLL_ESCALATIONS].find_one_and_update(
        {"session_id": session_id, "dedupe_key": dedupe},
        {
            "$set": doc,
            "$setOnInsert": {"created_at": now, "ticket_id": _ticket("ESC")},
            "$inc": {"repeat_count": 1},
        },
        upsert=True,
        return_document=True,
    )
    return {
        "escalated": True,
        "ticket_id": res["ticket_id"],
        "queue": queue,
        "urgency": doc["urgency"],
        "repeat_count": res.get("repeat_count", 1),
        "tell_customer": (
            "Tell the customer plainly that you are handing this to a person, what "
            "you have already established for them, and that they do not need to "
            "repeat it. Do not promise a callback time."
        ),
    }


# --- record_lead -------------------------------------------------------------


def record_lead(
    interest: str,
    *,
    session_id: str,
    customer_id: str | None = None,
    persona: str | None = None,
    consent_to_contact: bool = False,
    contact_name: str | None = None,
    contact_email: str | None = None,
    contact_phone: str | None = None,
    platform_id: int | None = None,
    vehicle: str | None = None,
    current_stage: str | None = None,
    interested_stage: str | None = None,
    products: list[str] | None = None,
    notes: str | None = None,
) -> dict:
    if len((interest or "").strip()) < 5:
        raise ToolInputError("interest must describe what the customer wants")
    if not consent_to_contact:
        # CASL. A lead recorded without consent is a compliance liability that
        # looks like a sales win on a dashboard, which is the worst possible
        # combination.
        raise ToolInputError(
            "consent_to_contact is false. Ask the customer whether they want to be "
            "contacted about this before recording a lead; do not record one without "
            "an explicit yes."
        )
    if not (contact_email or contact_phone or customer_id):
        raise ToolInputError(
            "a lead needs a way to reach the customer: email, phone, or an "
            "authenticated session"
        )
    _ensure_indexes()
    now = _now()
    doc = {
        "lead_id": _ticket("LEAD"),
        "session_id": session_id,
        "customer_id": customer_id,
        "persona": persona,
        "interest": interest.strip(),
        "consent_to_contact": True,
        "consent_recorded_at": now,
        "contact": {
            "name": (contact_name or "").strip() or None,
            "email": (contact_email or "").strip() or None,
            "phone": (contact_phone or "").strip() or None,
        },
        "vehicle": {
            "platform_id": platform_id,
            "described_as": (vehicle or "").strip() or None,
            "current_stage": current_stage,
            "interested_stage": interested_stage,
        },
        "products": list(products or []),
        "notes": (notes or "").strip() or None,
        "status": "new",
        "created_at": now,
        "updated_at": now,
    }
    kb_db()[config.COLL_LEADS].insert_one(doc)
    return {
        "recorded": True,
        "lead_id": doc["lead_id"],
        "note": "Confirm to the customer that someone will follow up. Do not quote a price.",
    }


# --- log_knowledge_gap -------------------------------------------------------


def log_knowledge_gap(
    question: str,
    *,
    session_id: str,
    persona: str | None = None,
    tools_tried: list[str] | None = None,
    context: str | None = None,
) -> dict:
    if len((question or "").strip()) < 8:
        raise ToolInputError("question must be the customer's actual question")
    _ensure_indexes()
    # Keyed on a normalised form so the same gap asked fifty ways still counts
    # as one row with count=50. A gap list that is really a duplicate list
    # cannot be prioritised.
    key = _digest(" ".join(question.lower().split()))
    now = _now()
    res = kb_db()[config.COLL_KNOWLEDGE_GAPS].find_one_and_update(
        {"question_key": key},
        {
            "$set": {"last_seen_at": now, "persona": persona},
            "$setOnInsert": {
                "question_key": key,
                "question": question.strip(),
                "first_seen_at": now,
                "status": "open",
            },
            "$inc": {"count": 1},
            "$addToSet": {
                "sessions": session_id,
                "tools_tried": {"$each": list(tools_tried or [])},
            },
            "$push": {
                "examples": {
                    "$each": [{"question": question.strip(), "context": context, "at": now}],
                    "$slice": -10,
                }
            },
        },
        upsert=True,
        return_document=True,
    )
    return {
        "logged": True,
        "times_asked": res.get("count", 1),
        "note": (
            "Logged. Now tell the customer you do not have a confident answer and "
            "offer a human -- logging a gap is not answering the question."
        ),
    }


# --- request_approval --------------------------------------------------------


def request_approval(
    action: str,
    justification: str,
    *,
    session_id: str,
    customer_id: str | None = None,
    persona: str | None = None,
    arguments: dict | None = None,
) -> dict:
    """Open a pending approval. The state machine lives in ``executor.py``.

    Imported inside the function on purpose: ``executor`` imports the registry,
    which imports this module, so a top-level import would be a cycle. The
    store is in the executor because the executor is what enforces the gate.
    """
    from .executor import ApprovalStore

    if len((action or "").strip()) < 3:
        raise ToolInputError("action must name what is to be approved")
    if len((justification or "").strip()) < 15:
        raise ToolInputError(
            "justification must state why this is needed and what the customer asked for"
        )
    _ensure_indexes()
    record = ApprovalStore().create(
        session_id=session_id,
        action=action.strip(),
        arguments=dict(arguments or {}),
        justification=justification.strip(),
        customer_id=customer_id,
        persona=persona,
    )
    return {
        "approval_id": record["approval_id"],
        "state": record["state"],
        "action": record["action"],
        "note": (
            "Pending human approval. Tell the customer you have requested it and stop; "
            "do NOT perform the action or describe it as done."
        ),
    }


TOOLS = [
    Tool(
        name="escalate_to_human",
        description=(
            "Hand the conversation to a human, with a handoff note so the customer does "
            "not repeat themselves. Escalate for: any emissions-defeat or tampering "
            "request, any safety concern, refunds/RMA, account actions, anything you "
            "have tried twice without resolving, and any time the customer asks for a "
            "person. Escalating is always better than guessing."
        ),
        parameters=obj_schema(
            {
                "reason": {"type": "string", "enum": ESCALATION_REASONS},
                "summary": {
                    "type": "string",
                    "description": (
                        "Handoff note: what the customer wants, what you established "
                        "(vehicle, platform, stage, error string), and what is blocked."
                    ),
                },
                "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
                "customer_question": {
                    "type": "string",
                    "description": "The customer's question in their own words",
                },
            },
            ["reason", "summary"],
        ),
        handler=escalate_to_human,
        dependency="mongo",
        writes=True,
        injects=("session_id", "customer_id", "persona"),
    ),
    Tool(
        name="record_lead",
        description=(
            "Record a sales lead after the customer has explicitly agreed to be "
            "contacted. Requires consent_to_contact=true -- ask first, and do not call "
            "this if they say no or do not answer."
        ),
        parameters=obj_schema(
            {
                "interest": {
                    "type": "string",
                    "description": "What they want, e.g. 'Stage 2 + downpipe for a 2019 GTI'",
                },
                "consent_to_contact": {
                    "type": "boolean",
                    "description": "True only if the customer explicitly agreed to be contacted",
                },
                "contact_name": {"type": "string"},
                "contact_email": {"type": "string"},
                "contact_phone": {"type": "string"},
                "platform_id": {"type": "integer", "description": "From resolve_vehicle"},
                "vehicle": {"type": "string", "description": "Vehicle as described"},
                "current_stage": {"type": "string"},
                "interested_stage": {"type": "string"},
                "products": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            ["interest", "consent_to_contact"],
        ),
        handler=record_lead,
        dependency="mongo",
        writes=True,
        injects=("session_id", "customer_id", "persona"),
    ),
    Tool(
        name="log_knowledge_gap",
        description=(
            "Record a question you could not answer from the knowledge base or the "
            "tools. Call this BEFORE telling the customer you do not know, every time. "
            "It is the only signal that says which documentation gap to close next."
        ),
        parameters=obj_schema(
            {
                "question": {"type": "string", "description": "The customer's question verbatim"},
                "tools_tried": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which tools you already tried",
                },
                "context": {"type": "string", "description": "Vehicle, platform, error string"},
            },
            ["question"],
        ),
        handler=log_knowledge_gap,
        dependency="mongo",
        writes=True,
        injects=("session_id", "persona"),
    ),
    Tool(
        name="request_approval",
        description=(
            "Ask a human to approve an action you are not allowed to take unilaterally "
            "-- a goodwill credit, a policy exception, an out-of-window return. Creates "
            "a pending request and returns immediately. The action does NOT happen "
            "until a human approves it; never tell the customer it is done."
        ),
        parameters=obj_schema(
            {
                "action": {"type": "string", "description": "What you want approved"},
                "justification": {
                    "type": "string",
                    "description": "Why, and what the customer asked for",
                },
                "arguments": {
                    "type": "object",
                    "description": "Structured detail: amounts, order ids, stages",
                },
            },
            ["action", "justification"],
        ),
        handler=request_approval,
        dependency="mongo",
        writes=True,
        requires_approval=False,  # this tool *creates* approvals; it is not gated by one
        injects=("session_id", "customer_id", "persona"),
    ),
]


def self_check() -> None:
    import json

    sid = f"selfcheck-{secrets.token_hex(4)}"
    db = kb_db()

    r1 = escalate_to_human(
        reason="emissions_request",
        summary="Customer asked for a tune with the EGR and DPF deleted on a 2016 Golf TDI. Refused; routing to compliance.",
        session_id=sid,
        persona="support",
    )
    print("escalate_to_human:", json.dumps(r1, indent=1))
    r2 = escalate_to_human(
        reason="emissions_request",
        summary="Customer asked for a tune with the EGR and DPF deleted on a 2016 Golf TDI. Refused; routing to compliance.",
        session_id=sid,
        persona="support",
    )
    assert r1["ticket_id"] == r2["ticket_id"], "idempotency broken"
    assert r2["repeat_count"] == 2
    print("second identical escalation -> same ticket, repeat_count:", r2["repeat_count"])

    try:
        escalate_to_human(reason="nope", summary="x" * 30, session_id=sid)
        raise AssertionError("bad reason accepted")
    except ToolInputError as exc:
        print("ToolInputError:", exc)
    try:
        escalate_to_human(reason="safety_issue", summary="help", session_id=sid)
        raise AssertionError("thin summary accepted")
    except ToolInputError as exc:
        print("ToolInputError:", exc)

    try:
        record_lead("Stage 2 for a 2019 GTI", session_id=sid, contact_email="a@b.com")
        raise AssertionError("lead without consent accepted")
    except ToolInputError as exc:
        print("ToolInputError:", exc)
    lead = record_lead(
        "Stage 2 plus downpipe for a 2019 GTI",
        session_id=sid,
        consent_to_contact=True,
        contact_email="selfcheck@example.invalid",
        platform_id=80,
        persona="sales",
    )
    print("record_lead:", json.dumps(lead, indent=1))

    g1 = log_knowledge_gap(
        "Does the UniFLEX kit work with a 2026 Golf R?", session_id=sid, persona="sales"
    )
    g2 = log_knowledge_gap(
        "does the uniflex kit work with a 2026 golf r?", session_id=sid, persona="sales"
    )
    print("log_knowledge_gap:", g1["times_asked"], "->", g2["times_asked"])
    assert g2["times_asked"] == g1["times_asked"] + 1, "gap dedupe broken"

    ap = request_approval(
        action="one-time goodwill licence transfer waiver",
        justification="Customer was charged twice for the same transfer in March; order 78098 shows the duplicate.",
        session_id=sid,
        persona="support",
    )
    print("request_approval:", json.dumps(ap, indent=1))
    assert ap["state"] == "pending"

    print("collections written:", sorted(
        {config.COLL_ESCALATIONS, config.COLL_LEADS, config.COLL_KNOWLEDGE_GAPS, COLL_APPROVALS}
    ))
    # Clean up only what this self-check created. Nothing else is touched.
    db[config.COLL_ESCALATIONS].delete_many({"session_id": sid})
    db[config.COLL_LEADS].delete_many({"session_id": sid})
    db[config.COLL_KNOWLEDGE_GAPS].delete_many({"sessions": sid})
    db[COLL_APPROVALS].delete_many({"session_id": sid})
    print("control.py self-check OK")


if __name__ == "__main__":
    self_check()
