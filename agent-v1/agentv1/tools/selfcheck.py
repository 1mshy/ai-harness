"""Runnable end-to-end check of the whole tool layer against live systems.

``python -m agentv1.tools.selfcheck`` runs every module's own self-check and
then the cross-module assertions that no single module can make: that the
registry matches the contract, that a Tier-2 call from an anonymous session is
refused, that the competitor briefs are consistent with their index, and that
a full sales-shaped conversation resolves a vehicle, checks a stage and quotes
a fee -- each with a verifiable provenance token.

Every module's self-check runs against the real Mongo, the real Qdrant and the
real files. There are no fixtures here on purpose: the failure modes this layer
exists to prevent (a stale platform table, a missing collection, a payload
field that changed shape) are invisible against a mock.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

import yaml

from .. import config
from ..clients.mongo import kb_db
from .base import TIER_ANONYMOUS, TIER_CUSTOMER, verify_provenance
from .executor import Budget, SessionContext, ToolExecutor
from .registry import EXPECTED_TOOLS, REGISTRY

BRIEF_DIR = config.DATA_DIR / "competitor_briefs"


def check_briefs() -> None:
    index = yaml.safe_load((BRIEF_DIR / "_index.yaml").read_text())
    brands = index["brands"]
    print(f"competitor brands indexed: {len(brands)}")

    covered = 0
    for brand in brands:
        rel = brand["relationship"]
        assert rel in ("software_rival", "complementary_hardware"), brand
        if brand.get("brief"):
            path = BRIEF_DIR / brand["brief"]
            assert path.exists(), f"missing brief {path}"
            text = path.read_text()
            assert len(text) > 800, f"{path} is too thin to be useful"
            covered += brand.get("mentions", 0)
        print(f"  {brand['name']:24s} {rel:22s} mentions={brand.get('mentions', '-')}")

    # The six that must never be argued with.
    complementary = {
        b["id"] for b in brands if b["relationship"] == "complementary_hardware"
    }
    required = {"cts_turbo", "integrated_engineering", "ecs_tuning", "awe_tuning",
                "eventuri", "csf"}
    assert required <= complementary, (
        f"these must be tagged complementary_hardware: {sorted(required - complementary)}"
    )
    assert covered == index["covered_by_briefs"] == 2785, covered
    share = covered / index["total_mentions_measured"]
    print(f"  briefs cover {covered}/{index['total_mentions_measured']} = {share:.1%}")

    # No brief may assert a competitor price or a power figure.
    for path in sorted(BRIEF_DIR.glob("*.md")):
        text = path.read_text()
        for banned in ("$", " hp", "HP ", "lb-ft", "whp"):
            assert banned not in text, f"{path.name} contains {banned!r}"
    print("  no competitor prices or power figures asserted in any brief")


def check_conversation() -> None:
    """A sales-shaped turn, end to end, through the executor."""
    ex = ToolExecutor(budget=Budget(per_turn=12, per_tool_per_turn=4, per_session=200))
    sid = f"selfcheck-e2e-{secrets.token_hex(4)}"
    ctx = SessionContext(session_id=sid, persona="sales")

    print("\nschemas offered to an anonymous sales session:")
    names = [s["function"]["name"] for s in ex.tool_schemas(ctx)]
    print(" ", names)
    assert not any(n.startswith("get_my_") for n in names)

    steps = [
        ("resolve_vehicle", {"make": "Audi", "model": "S3", "year": 2017}),
        ("check_stage_availability", {"platform_id": 80, "stage": "Stage 2+"}),
        ("get_fee_schedule", {}),
        ("lookup_error_string", {"text": "the request is not supported"}),
        ("search_knowledge", {"query": "does Stage 2 need a downpipe"}),
    ]
    tokens = []
    for name, args in steps:
        res = ex.dispatch(name, args, ctx)
        assert res.ok, f"{name} failed: {res.error}"
        assert verify_provenance(res.provenance, name, res.data)
        tokens.append(res.provenance)
        summary = {
            "resolve_vehicle": lambda d: {
                "candidates": d["candidate_count"],
                "agree_on_stages": d["agree_on_stages"],
                "top_platform": d["candidates"][0]["platform_id"],
                "clarify": d["clarify_on"],
            },
            "check_stage_availability": lambda d: {
                "available": d["available"],
                "files_available": d.get("files_available"),
                "synced_at": d["synced_at"],
            },
            "get_fee_schedule": lambda d: {
                "items": d["item_count"],
                "license_transfer": next(
                    i["amount"] for i in d["items"] if i["id"] == "license_transfer"
                ),
            },
            "lookup_error_string": lambda d: {
                "matched": d["matched"],
                "occurrences": d["occurrences"],
                "cases": d["related_case_ids"][:2],
            },
            "search_knowledge": lambda d: {
                "source": d["source"],
                "results": d["result_count"],
                "missing_collections": d.get("retrieval", {}).get("collections_missing"),
            },
        }[name](res.data)
        # Every tool message must survive json.dumps with no default= hook;
        # anything that does not is a dataclass that leaked out of a handler.
        json.dumps(res.to_dict())
        print(
            f"  {name:26s} ok degraded={str(res.degraded):5s} "
            f"{res.latency_ms:7.1f}ms {json.dumps(summary, default=str)}"
        )

    assert len(set(tokens)) == len(tokens), "provenance tokens collided"

    print("\nTier-2 refusal from this same anonymous session:")
    for name in ("get_my_vehicles", "get_my_orders", "get_my_tune_history",
                 "get_my_open_case"):
        res = ex.dispatch(name, {"customer_id": "1178"}, ctx)
        assert res.ok is False and res.error_kind == "policy", name
        assert res.provenance == "", "a refused call must carry no provenance"
        print(f"  {name:22s} refused: {res.error_kind}")

    for coll in (config.COLL_AGENT_EVENTS, config.COLL_AGENT_SESSIONS):
        kb_db()[coll].delete_many({"session_id": sid})


def main() -> int:
    from . import base, control, customer, fees, knowledge, products, registry, vehicle
    from . import executor as executor_mod

    modules = [
        ("base", base),
        ("registry", registry),
        ("vehicle", vehicle),
        ("knowledge", knowledge),
        ("products", products),
        ("fees", fees),
        ("customer", customer),
        ("control", control),
        ("executor", executor_mod),
    ]
    failures = []
    for name, module in modules:
        print(f"\n{'=' * 70}\n== {name}.self_check()\n{'=' * 70}")
        try:
            module.self_check()
        except Exception as exc:  # noqa: BLE001 - report all, fail once
            failures.append((name, exc))
            print(f"!! {name} FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 70}\n== cross-module\n{'=' * 70}")
    try:
        assert set(REGISTRY.names()) == EXPECTED_TOOLS
        print(f"registry holds {len(REGISTRY)} tools matching the contract")
        anon = len(REGISTRY.get_tool_schemas("support", TIER_ANONYMOUS))
        auth = len(REGISTRY.get_tool_schemas("support", TIER_CUSTOMER))
        print(f"support persona: {anon} tools anonymous, {auth} authenticated")
        check_briefs()
        check_conversation()
    except Exception as exc:  # noqa: BLE001
        failures.append(("cross-module", exc))
        print(f"!! cross-module FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 70}")
    if failures:
        for name, exc in failures:
            print(f"FAILED {name}: {type(exc).__name__}: {exc}")
        return 1
    print("ALL TOOL-LAYER SELF-CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
