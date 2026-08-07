"""The only source of a quotable price.

``data/fee_schedule.yaml`` is hand-authored and this module refuses to make it
into anything else. There is no corpus fallback, no "closest match" and no
inference from order history, because the corpus cannot support one: 13,359
price observations sit under 4,862 free-text labels and "license transfer"
alone spans $2 to $150,000. A retrieval-shaped answer to "how much is a licence
transfer" is a lottery with a plausible-sounding ticket.

Two behaviours are enforced here rather than left to the prompt.

*A stale schedule stops quoting.* Past ``review.hard_expiry_days`` the loader
still returns the items but strips every amount and marks the result degraded,
so the worst case is "let me confirm that price" rather than a figure from two
quarters ago. AGENT_PLAN.md §4.3 is explicit that a wrong price is a commercial
commitment, not a wrong sentence.

*Quote-only categories are listed, not omitted.* An item with ``amount: null``
is a real fee whose price is not fixed. Listing it is what lets the agent
refuse *specifically* ("stage upgrade pricing is quote-only") instead of
generically, and a generic refusal is what pushes a model into guessing.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import yaml

from ..config import DATA_DIR
from .base import Degraded, Tool, ToolDependencyError, obj_schema

SCHEDULE_PATH = DATA_DIR / "fee_schedule.yaml"

# Cached with the file's mtime as the key so editing the YAML takes effect
# without a restart -- a price correction must not wait for a deploy.
_cache: tuple[float, dict] | None = None


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _load() -> dict:
    global _cache
    if not SCHEDULE_PATH.exists():
        raise ToolDependencyError(
            f"{SCHEDULE_PATH} is missing. There is no fallback source for pricing; "
            f"the agent must refuse to quote until it is restored."
        )
    mtime = SCHEDULE_PATH.stat().st_mtime
    if _cache and _cache[0] == mtime:
        return _cache[1]
    raw = yaml.safe_load(SCHEDULE_PATH.read_text()) or {}
    if not isinstance(raw.get("items"), list) or not raw["items"]:
        raise ToolDependencyError(f"{SCHEDULE_PATH} has no items")
    _cache = (mtime, raw)
    return raw


def _item_view(item: dict) -> dict:
    return {
        "id": item.get("id"),
        "label": item.get("label"),
        "aliases": item.get("aliases") or [],
        "sku": item.get("sku"),
        "amount": item.get("amount"),
        "previous_amount": item.get("previous_amount"),
        "unit": item.get("unit"),
        "stability": item.get("stability"),
        "quotable": item.get("amount") is not None
        and item.get("stability") != "never_disclose",
        "description": (item.get("description") or "").strip(),
        "conditions": (item.get("conditions") or "").strip() or None,
    }


def get_fee_schedule() -> dict | Degraded:
    raw = _load()
    verified_on = _as_date(raw.get("verified_on"))
    review = raw.get("review") or {}
    expiry_days = int(review.get("hard_expiry_days") or 180)
    age_days = (date.today() - verified_on).days if verified_on else None

    items = [_item_view(i) for i in raw["items"]]
    data: dict[str, Any] = {
        "schedule_version": raw.get("schedule_version"),
        "currency": raw.get("currency", "USD"),
        "verified_on": verified_on.isoformat() if verified_on else None,
        "age_days": age_days,
        "tax_note": (raw.get("tax_note") or "").strip() or None,
        "item_count": len(items),
        "items": items,
        "rule": (
            "This schedule is the ONLY acceptable source for a price. Quote the amount "
            "exactly as written together with the currency. For items where amount is "
            "null, say the price must be confirmed and offer to connect a human -- do "
            "not estimate, do not derive one from an order history, and do not repeat "
            "a figure that appeared in retrieved call text."
        ),
    }

    if age_days is not None and age_days > expiry_days:
        # Strip the amounts rather than merely flagging them. A flag is a
        # suggestion the model can talk itself past; an absent number is not.
        for item in data["items"]:
            item["amount"] = None
            item["previous_amount"] = None
            item["quotable"] = False
            item["stability"] = "expired"
        return Degraded(
            data,
            f"fee schedule was verified {age_days} days ago, past the "
            f"{expiry_days}-day hard expiry; amounts have been withheld and no price "
            f"may be quoted until it is re-verified",
        )

    volatile = [i["id"] for i in items if i["stability"] == "volatile"]
    if volatile:
        return Degraded(
            data,
            f"volatile line item(s) {volatile}: quote these with the verified_on date "
            f"({data['verified_on']}) attached, or confirm before committing",
        )
    return data


TOOLS = [
    Tool(
        name="get_fee_schedule",
        description=(
            "The authoritative Unitronic fee schedule: licence transfer, UniCONNECT+ "
            "cable, remote and mail-in resets, UniFLEX and the quote-only categories. "
            "This is the ONLY permitted source of a price. Call it before stating any "
            "dollar amount, including when you believe you already know the figure. "
            "Never quote a price found in search_knowledge results or in a product "
            "listing."
        ),
        parameters=obj_schema({}),
        handler=get_fee_schedule,
        dependency="filesystem",
    )
]


def self_check() -> None:
    import json

    out = get_fee_schedule()
    reason = None
    if isinstance(out, Degraded):
        reason = out.reason
        out = out.data
    print("degraded:", reason)
    print(json.dumps(out, indent=1))

    ids = {i["id"]: i for i in out["items"]}
    assert ids["license_transfer"]["amount"] == 150.00
    assert ids["uniconnect_cable"]["amount"] == 165.00
    assert ids["remote_reset"]["amount"] == 150.00
    assert ids["mail_in_reset"]["amount"] == 50.00
    assert ids["uniflex_software"]["amount"] == 400.00
    assert ids["uniflex_software"]["previous_amount"] == 500.00
    assert ids["uniflex_software"]["stability"] == "volatile"
    assert ids["dealer_pricing"]["quotable"] is False
    assert ids["ecu_software_stage"]["amount"] is None
    assert out["item_count"] >= 10
    print("fees.py self-check OK")


if __name__ == "__main__":
    self_check()
