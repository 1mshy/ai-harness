"""Scenario: tool_calling_crm — multi-step CRM tool calling + grounded answer.

A customer email lands in the shared inbox asking the agent to check on an
order. The agent must (1) recognize it needs live data, (2) call the CRM tools
with the IDs extracted from the email, and (3) report the order's status
grounded on what the tools returned — not invented.

This exercises the OpenAI function-calling path end to end: argument extraction
from natural language, a deterministic tool_router, and a final answer that
cites the canned data. The grader is diagnostic: an informational check flags
whether the model emitted *any* tool call, so a model that simply lacks tool
support reads differently from one that called the wrong tool with wrong args.
"""

from __future__ import annotations

import re

from sales_eval.harness import (
    GradeResult,
    RunOutcome,
    Scenario,
    contains_any,
    norm,
    register,
)

# --------------------------------------------------------------------------- #
# Fixtures — canned CRM data (deterministic, the grader's source of truth)
# --------------------------------------------------------------------------- #
KNOWN_EMAIL = "dana@northwind.co"
KNOWN_ORDER_ID = "SO-4821"

_CUSTOMERS = {
    KNOWN_EMAIL: {
        "account_id": "ACCT-3391",
        "name": "Dana Whitfield",
        "company": "Northwind Logistics",
        "plan": "Growth (annual)",
    },
}

_ORDERS = {
    KNOWN_ORDER_ID: {
        "order_id": KNOWN_ORDER_ID,
        "status": "In transit",
        "carrier": "FedEx Ground",
        "eta_date": "2026-06-25",
        "tracking": "FX772104553801",
    },
}

# Pre-extract the exact canned values the grounded answer should cite.
_CANNED_ORDER = _ORDERS[KNOWN_ORDER_ID]
_GROUNDING_TOKENS = [
    _CANNED_ORDER["eta_date"],          # "2026-06-25"
    _CANNED_ORDER["status"],            # "In transit"
    _CANNED_ORDER["tracking"],          # tracking number
    _CANNED_ORDER["carrier"],           # "FedEx Ground"
]


def _normalize_order_id(raw: object) -> str:
    """Canonicalize an order id: uppercase, strip spaces, normalize SO-#### form.

    Tolerates "SO-4821", "so-4821", "SO4821", "so 4821", " SO-4821 " -> "SO4821".
    """
    s = re.sub(r"[\s\-_]", "", str(raw or "")).upper()
    return s


_CANON_KNOWN_ORDER = _normalize_order_id(KNOWN_ORDER_ID)  # "SO4821"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": (
                "Look up a CRM customer account by their email address. "
                "Returns the account_id, name, company, and plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The customer's email address, e.g. dana@northwind.co",
                    }
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Get the shipping status of an order by its order id. Returns "
                "status, carrier, eta_date (ISO yyyy-mm-dd), and tracking number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, e.g. SO-4821",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
]


def tool_router(name: str, args: dict):
    """Deterministic CRM. Known email/order -> canned data; anything else -> error."""
    if name == "lookup_customer":
        email = norm(args.get("email"))
        rec = _CUSTOMERS.get(email)
        if rec is None:
            return {"error": f"no customer found for email {args.get('email')!r}"}
        return rec
    if name == "get_order_status":
        if _normalize_order_id(args.get("order_id")) == _CANON_KNOWN_ORDER:
            return _CANNED_ORDER
        return {"error": f"no order found for id {args.get('order_id')!r}"}
    return {"error": f"unknown tool {name!r}"}


# --------------------------------------------------------------------------- #
# Grader
# --------------------------------------------------------------------------- #
def grade(out: RunOutcome) -> GradeResult:
    g = GradeResult()

    names = out.tool_names()
    order_call = out.first_call("get_order_status")

    # INFORMATIONAL: did the model use the tools API at all? Distinguishes a
    # no-tool-support model from one that called the wrong tool with wrong args.
    g.add(
        "model emitted >=1 tool call",
        len(out.tool_calls) > 0,
        detail=f"tool calls observed: {names or 'none'}",
        required=False,
    )

    # REQUIRED 1: get_order_status was actually called.
    g.add(
        "called get_order_status",
        order_call is not None,
        detail=f"tools called: {names or 'none'}",
    )

    # REQUIRED 2: it was called with the correct, normalized order id.
    got_id = order_call.arguments.get("order_id") if order_call else None
    id_ok = bool(order_call) and _normalize_order_id(got_id) == _CANON_KNOWN_ORDER
    g.add(
        "get_order_status called with order_id SO-4821",
        id_ok,
        detail=f"order_id arg: {got_id!r} (normalized {_normalize_order_id(got_id)!r}, "
               f"expected {_CANON_KNOWN_ORDER!r})",
    )

    # REQUIRED 3: final answer is grounded on the router's returned data — at
    # least one exact canned value (eta_date / status / tracking / carrier)
    # appears in the final text. NOT a guessed date.
    grounded = contains_any(out.final_text, _GROUNDING_TOKENS)
    g.add(
        "final answer grounded on canned order data",
        grounded,
        detail=f"expected one of {_GROUNDING_TOKENS!r} in final_text: "
               f"{out.final_text[:200]!r}",
    )

    return g


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
register(
    Scenario(
        name="tool_calling_crm",
        category="tool_use",
        description="CRM tool calling: extract email+order id from a customer "
                    "email, look up the order, report grounded status.",
        system=(
            "You are a B2B SaaS customer-success agent for Northwind's vendor "
            "portal. You have access to CRM tools. When a customer asks about an "
            "order, ALWAYS use the tools to fetch live data before answering — "
            "never guess shipping details. After calling the tools, reply to the "
            "customer with the order's current status and ETA, citing only the "
            "data the tools returned."
        ),
        user_messages=[
            {
                "role": "user",
                "content": (
                    "Subject: where's my shipment?\n\n"
                    "Hi, this is Dana (dana@northwind.co) — can you check on order "
                    "SO-4821? It was supposed to arrive yesterday and I haven't "
                    "seen anything. Thanks!"
                ),
            }
        ],
        grade=grade,
        tools=TOOLS,
        tool_router=tool_router,
        max_tokens=512,
        temperature=0.0,
        sample_good={
            "tool_calls": [
                {"name": "lookup_customer", "arguments": {"email": "dana@northwind.co"}},
                {"name": "get_order_status", "arguments": {"order_id": "SO-4821"}},
            ],
            "final_text": (
                "Hi Dana — I checked on order SO-4821. It's currently In transit "
                "via FedEx Ground (tracking FX772104553801) with an updated ETA of "
                "2026-06-25. Sorry it slipped past the original date; it's moving "
                "and should reach you in the next couple of days."
            ),
        },
        sample_bad={
            # Plausible but WRONG: no tool call, invents a delivery date that is
            # not in the canned data and asserts it was already delivered.
            "final_text": (
                "Hi Dana — good news, order SO-4821 was delivered on 2026-06-22. "
                "It should be waiting at your dock now. Let me know if you can't "
                "locate it!"
            ),
        },
    )
)
