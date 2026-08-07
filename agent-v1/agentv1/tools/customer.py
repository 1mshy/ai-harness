"""Tier-2, customer-scoped reads over ``tuning_customers``. Read-only.

**No function here takes a customer identifier from the model.** The public
JSON schema of every tool below is ``{}`` -- literally zero properties. The
handler receives ``customer_id`` as a keyword the executor binds from
server-side session state, and ``Tool.__post_init__`` refuses at import if that
name ever appears in the schema. "Look up someone else's order because the user
asked nicely" is therefore not a malformed request that gets rejected; it is a
sentence with nowhere to put the other person's id.

That is a different guarantee from a prompt saying "only access the current
customer's data". A prompt is a request. This is a type error.

The customer key is ``tuning_customers.customer_id`` (a string) with ``_id`` as
the integer mirror; both are accepted because sessions established from
different upstreams carry different ones.

**Identity is not a phone match.** AGENT_PLAN.md §9.3: a phone number matching
a record is evidence, not authentication. Establishing the session is the
auth layer's job. These tools trust the session and nothing else, which is why
none of them takes a phone number either.

**PII discipline** — the caller is entitled to their own data, but the answer
text goes into an SSE frame and a transcript. VINs and licence plates are
therefore masked to a confirmable suffix: enough for "yes, that's my car",
not enough to be a leaked identifier if the transcript escapes.
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..clients.mongo import source_db
from .base import TIER_CUSTOMER, Tool, ToolInputError, obj_schema

MAX_VEHICLES = 10
MAX_ORDERS = 10
MAX_FLASHES = 25


def _customer(customer_id: str) -> dict:
    """Resolve the session's customer. Never called with model-supplied input."""
    if not customer_id:
        # Defence in depth. The executor already refuses Tier 2 without a
        # session, so reaching here means a caller bypassed dispatch.
        raise ToolInputError("no authenticated customer on this session")
    coll = source_db()[config.COLL_CUSTOMERS]
    doc = coll.find_one({"customer_id": str(customer_id)})
    if doc is None and str(customer_id).isdigit():
        doc = coll.find_one({"_id": int(customer_id)})
    if doc is None:
        raise ToolInputError(f"no customer record for the current session")
    if doc.get("is_deleted"):
        raise ToolInputError(
            "this account is marked deleted; account data cannot be served"
        )
    return doc


def _mask(value: Any, keep: int = 4) -> str | None:
    """Last ``keep`` characters only. Enough to confirm, not enough to reuse."""
    if not value:
        return None
    text = str(value).strip()
    if len(text) <= keep:
        return "*" * len(text)
    return "*" * (len(text) - keep) + text[-keep:]


def _vehicle_view(v: dict) -> dict:
    owned = v.get("owned_stage") or {}
    flashed = v.get("flashed_stage") or {}
    return {
        "display_name": v.get("display_name"),
        "year": v.get("year"),
        "make": v.get("make"),
        "model": v.get("model"),
        "vin_last4": _mask(v.get("vin")),
        "transmission": v.get("transmission"),
        "is_dsg": bool(v.get("is_dsg")),
        # The join key into check_stage_availability. This is the whole reason
        # a Tier-2 session is worth having on a compatibility question.
        "platform_id": v.get("platform_id"),
        "platform_name": v.get("platform_name"),
        "owned_stage": owned.get("label"),
        "owned_stage_at": owned.get("at"),
        "flashed_stage": flashed.get("label"),
        "max_released_stage": v.get("max_released_stage"),
        "released_stage_labels": v.get("released_stage_labels") or [],
        "unreleased_stage_labels": v.get("unreleased_stage_labels") or [],
        "flash_count": v.get("flash_count"),
        "last_flash_at": v.get("last_flash_at"),
    }


def get_my_vehicles(*, customer_id: str) -> dict:
    doc = _customer(customer_id)
    vehicles = [
        _vehicle_view(v)
        for v in (doc.get("vehicles") or [])
        if not v.get("removed_at")
    ][:MAX_VEHICLES]
    unmatched = [v for v in vehicles if v["platform_id"] is None]
    return {
        "vehicle_count": doc.get("vehicle_count") or len(vehicles),
        "returned": len(vehicles),
        "vehicles": vehicles,
        "active_platform_ids": doc.get("active_platform_ids") or [],
        "note": (
            # 93.4% of vehicle rows join to a platform. The misses are null on
            # the vehicle side, so the right response is "confirm the car",
            # not "the platform table is broken".
            f"{len(unmatched)} vehicle(s) have no platform match; resolve_vehicle "
            f"can identify them from make/model/year."
            if unmatched
            else None
        ),
    }


def _order_view(o: dict) -> dict:
    return {
        "order_id": o.get("po_id"),
        "order_type": o.get("order_type"),
        "status": o.get("status_label"),
        "is_open": bool(o.get("is_open")),
        "created_at": o.get("created_at"),
        "closed_at": o.get("closed_at"),
        "carrier": o.get("carrier"),
        "has_tracking": bool(o.get("has_tracking")),
        # grand_total is what the customer already paid. It is history, not a
        # quote, and must never be reused as the price of a new purchase.
        "grand_total": o.get("grand_total"),
        "currency": o.get("currency"),
        "item_count": o.get("item_count"),
        "items": [
            {
                "part_number": i.get("part_number"),
                "name": i.get("name"),
                "quantity": i.get("quantity"),
                "category": i.get("category"),
            }
            for i in (o.get("items") or [])[:12]
        ],
        "categories": o.get("categories") or [],
    }


def get_my_orders(*, customer_id: str) -> dict:
    doc = _customer(customer_id)
    orders = sorted(
        doc.get("orders") or [],
        key=lambda o: str(o.get("created_at") or ""),
        reverse=True,
    )
    views = [_order_view(o) for o in orders[:MAX_ORDERS]]
    return {
        "order_count": doc.get("order_count") or len(orders),
        "open_order_count": doc.get("open_order_count") or 0,
        "last_order_at": doc.get("last_order_at"),
        "returned": len(views),
        "orders": views,
        "ordered_categories": doc.get("ordered_categories") or [],
        "price_note": (
            "Totals shown are what was historically paid. They are not current "
            "pricing -- use get_fee_schedule for anything quotable."
        ),
    }


def get_my_tune_history(*, customer_id: str) -> dict:
    """Flash history across the account's vehicles, newest first.

    Carries the upgrade signal without computing an upsell: 78.8% of accounts
    own a stage below what is released for their platform, but that number is
    inflated by decade-old flashes (``owned_stage.at`` reaches back to 2007 and
    only 20.4% of vehicles were flashed since 2024). So the tool reports the
    gap and the date it was last touched, and lets the persona decide whether
    a fourteen-year-old Stage 1 is a lead or a museum piece.
    """
    doc = _customer(customer_id)
    events: list[dict] = []
    vehicles: list[dict] = []
    for v in doc.get("vehicles") or []:
        owned = (v.get("owned_stage") or {}).get("label")
        top = v.get("max_released_stage")
        vehicles.append(
            {
                "display_name": v.get("display_name"),
                "platform_id": v.get("platform_id"),
                "platform_name": v.get("platform_name"),
                "owned_stage": owned,
                "max_released_stage": top,
                "has_headroom": bool(owned and top and owned != top),
                "last_flash_at": v.get("last_flash_at"),
                "flash_count": v.get("flash_count"),
            }
        )
        for f in v.get("flash_history") or []:
            events.append(
                {
                    "at": f.get("at"),
                    "vehicle": v.get("display_name"),
                    "platform_id": f.get("platform_id"),
                    "stage": f.get("stage_label"),
                    "variant": f.get("stage_variant") or None,
                    "type": f.get("trans_type"),
                    "flashed": bool(f.get("is_flashed")),
                    "cancelled": bool(f.get("is_cancelled")),
                    "is_dsg": bool(f.get("is_dsg")),
                    # file_name is an internal calibration filename. Excluded:
                    # it is not actionable for the customer and it is a direct
                    # pointer into the calibration store.
                }
            )
    events.sort(key=lambda e: str(e.get("at") or ""), reverse=True)
    return {
        "vehicles": vehicles[:MAX_VEHICLES],
        "flash_event_count": len(events),
        "returned": min(len(events), MAX_FLASHES),
        "flash_history": events[:MAX_FLASHES],
    }


def get_my_open_case(*, customer_id: str) -> dict:
    """The caller's most recent unresolved support case, if there is one.

    Joined on ``phone_keys`` because that is the only key linking the CRM
    account to the call corpus. The phone number is read from the *account
    record*, never from anything the model said, and it is not returned.

    This is the fix for the measured repeat-contact driver: 267 returning
    callers were re-quoted because the second agent did not have the first
    conversation.
    """
    doc = _customer(customer_id)
    keys = [k for k in (doc.get("phone_keys") or []) if k]
    if not keys:
        return {
            "has_open_case": False,
            "reason": "this account has no phone key, so it cannot be linked to a case",
        }

    cases = list(
        source_db()[config.COLL_CASES]
        .find(
            {
                "phone_key": {"$in": keys},
                "training_safe": True,
                "case_resolution_status": {"$ne": "resolved"},
            },
            {
                "case_id": 1,
                "case_label": 1,
                "issue": 1,
                "issue_category": 1,
                "root_cause": 1,
                "case_resolution_status": 1,
                "open_questions": 1,
                "last_call_ts": 1,
                "first_call_ts": 1,
                "attempts": 1,
                "what_finally_worked": 1,
                "case_metrics.contacts": 1,
            },
        )
        .sort("last_call_ts", -1)
        .limit(1)
    )
    if not cases:
        return {"has_open_case": False}

    case = cases[0]
    attempts = [a for a in case.get("attempts") or [] if isinstance(a, dict)]
    return {
        "has_open_case": True,
        # Hand this to get_case for the full chronology and every failed
        # attempt -- do not duplicate that projection here.
        "case_id": case.get("case_id"),
        "case_label": case.get("case_label"),
        "issue": case.get("issue"),
        "issue_category": case.get("issue_category"),
        "root_cause": case.get("root_cause"),
        "status": case.get("case_resolution_status"),
        "open_questions": case.get("open_questions") or [],
        "first_contact": case.get("first_call_ts"),
        "last_contact": case.get("last_call_ts"),
        "contacts": (case.get("case_metrics") or {}).get("contacts"),
        "already_failed": [
            {"attempt": a.get("attempt"), "made_by": a.get("made_by")}
            for a in attempts
            if str(a.get("result")).lower() == "failed"
        ],
        "next_step": "Call get_case with this case_id before suggesting anything.",
    }


_NO_ARGS = obj_schema({})

TOOLS = [
    Tool(
        name="get_my_vehicles",
        description=(
            "The signed-in customer's vehicles on file, with the platform id and the "
            "stage they currently own. Use this instead of asking them to describe "
            "their car. Takes no arguments -- it always and only returns the current "
            "session's data."
        ),
        parameters=_NO_ARGS,
        handler=get_my_vehicles,
        tier=TIER_CUSTOMER,
        dependency="mongo",
        injects=("customer_id",),
    ),
    Tool(
        name="get_my_orders",
        description=(
            "The signed-in customer's order history, newest first, including open "
            "orders, carrier and line items. Totals are historical and are not a quote. "
            "Takes no arguments."
        ),
        parameters=_NO_ARGS,
        handler=get_my_orders,
        tier=TIER_CUSTOMER,
        dependency="mongo",
        injects=("customer_id",),
    ),
    Tool(
        name="get_my_tune_history",
        description=(
            "Every flash on the signed-in customer's vehicles, newest first, plus "
            "whether each vehicle owns a stage below what is released for its platform. "
            "Use before discussing an upgrade. Takes no arguments."
        ),
        parameters=_NO_ARGS,
        handler=get_my_tune_history,
        tier=TIER_CUSTOMER,
        dependency="mongo",
        injects=("customer_id",),
    ),
    Tool(
        name="get_my_open_case",
        description=(
            "The signed-in customer's most recent unresolved support case, if any, "
            "including what has already been tried and failed. Call this at the start "
            "of a support conversation so the customer does not have to repeat "
            "themselves. Takes no arguments."
        ),
        parameters=_NO_ARGS,
        handler=get_my_open_case,
        tier=TIER_CUSTOMER,
        dependency="mongo",
        injects=("customer_id",),
    ),
]


def self_check() -> None:
    import json

    db = source_db()
    # A live account with vehicles, orders and flash history, chosen by shape
    # rather than hardcoded so the check does not rot.
    doc = db[config.COLL_CUSTOMERS].find_one(
        {
            "order_count": {"$gt": 0},
            "vehicle_count": {"$gt": 0},
            "is_deleted": {"$ne": True},
            "vehicles.platform_id": {"$ne": None},
        },
        {"customer_id": 1},
    )
    cid = doc["customer_id"]
    print(f"--- using live customer_id (not printed in full): ***{str(cid)[-3:]} ---")

    v = get_my_vehicles(customer_id=cid)
    print("get_my_vehicles:", json.dumps(v, indent=1, default=str)[:1200])
    assert v["returned"] >= 1
    blob = json.dumps(v, default=str)
    assert "vin" not in blob or "vin_last4" in blob
    for veh in v["vehicles"]:
        assert veh.get("vin_last4") is None or veh["vin_last4"].startswith("*")

    o = get_my_orders(customer_id=cid)
    print("get_my_orders:", json.dumps(o, indent=1, default=str)[:900])
    assert o["order_count"] >= 1

    h = get_my_tune_history(customer_id=cid)
    print("get_my_tune_history:", json.dumps(h, indent=1, default=str)[:900])

    # An account that actually has an unresolved case, so both branches run.
    case = db[config.COLL_CASES].find_one(
        {"training_safe": True, "case_resolution_status": {"$ne": "resolved"}},
        {"phone_key": 1},
    )
    linked = db[config.COLL_CUSTOMERS].find_one(
        {"phone_keys": case["phone_key"]}, {"customer_id": 1}
    )
    if linked:
        c = get_my_open_case(customer_id=linked["customer_id"])
        print("get_my_open_case:", json.dumps(c, indent=1, default=str)[:1400])
        assert c["has_open_case"] is True
        assert "phone" not in json.dumps(c, default=str).lower()
    else:
        print("get_my_open_case: no customer linked to an unresolved case; "
              "exercising the negative branch instead")
        print(json.dumps(get_my_open_case(customer_id=cid), indent=1, default=str))

    try:
        get_my_vehicles(customer_id="")
        raise AssertionError("empty customer_id accepted")
    except ToolInputError as exc:
        print("ToolInputError:", exc)

    for tool in TOOLS:
        assert tool.parameters["properties"] == {}, tool.name
        assert tool.injects == ("customer_id",)
    print("customer.py self-check OK")


if __name__ == "__main__":
    self_check()
