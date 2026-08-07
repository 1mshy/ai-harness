"""The tool registry: 15 tools, one place that knows all of them.

``get_tool_schemas(persona, tier_allowed)`` is what the agent loop hands to the
endpoint. It filters twice, and the second filter is the one that matters: a
tool above the session's tier is **not in the schema list at all**. An
unauthenticated session is not told that ``get_my_orders`` exists, so it is not
in a position to try. The executor refuses it a second time at dispatch, which
is the belt to this file's braces -- schema filtering alone would be defeated
by a model that has seen the tool in an earlier turn of the same conversation.

Persona filtering is a relevance filter, not a security one, and is treated as
such: ``record_lead`` is hidden from the support persona because a support
agent that starts capturing leads mid-troubleshoot is annoying, not dangerous.

Registration is import-time and eager. Importing this module imports every tool
module, which makes "is the tool layer importable" a single check rather than
fifteen, and makes the ``Tool.__post_init__`` assertions (no injected argument
in a public schema; Tier 2 must inject ``customer_id``) run at startup rather
than on the request that would have leaked something.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .base import (
    ALL_PERSONAS,
    PERSONA_SALES,
    PERSONA_SUPPORT,
    TIER_ANONYMOUS,
    Tool,
)
from . import control, customer, fees, knowledge, products, vehicle


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def visible(self, persona: str, tier_allowed: int) -> list[Tool]:
        return [
            t
            for t in self._tools.values()
            if t.tier <= tier_allowed and (persona in t.personas)
        ]

    def get_tool_schemas(
        self, persona: str = PERSONA_SUPPORT, tier_allowed: int = TIER_ANONYMOUS
    ) -> list[dict]:
        """OpenAI-format schemas for exactly the tools this session may call."""
        return [
            t.to_openai_schema()
            for t in sorted(self.visible(persona, tier_allowed), key=lambda t: t.name)
        ]


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for module in (knowledge, vehicle, products, fees, customer, control):
        reg.register_all(module.TOOLS)

    # Persona narrowing. Everything else is shared, which is the point of
    # "one platform, two personas" -- a support conversation that turns into a
    # sale must not need a different tool layer.
    _restrict(reg, "record_lead", {PERSONA_SALES})
    return reg


def _restrict(reg: ToolRegistry, name: str, personas: set[str]) -> None:
    tool = reg.get(name)
    if tool is None:
        raise KeyError(name)
    # Tool is frozen; rebuild rather than mutate so the invariants in
    # __post_init__ are re-checked.
    reg._tools[name] = Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        handler=tool.handler,
        tier=tool.tier,
        dependency=tool.dependency,
        personas=frozenset(personas),
        injects=tool.injects,
        writes=tool.writes,
        requires_approval=tool.requires_approval,
    )


REGISTRY = build_registry()

# The contract with the rest of the platform. A tool appearing or vanishing
# without this set changing is a merge accident, and the agent loop would
# happily run with a silently smaller surface.
EXPECTED_TOOLS = {
    # Tier 0
    "search_knowledge",
    "get_case",
    "lookup_error_string",
    "resolve_vehicle",
    "check_stage_availability",
    "search_products",
    "lookup_product_by_sku",
    "get_fee_schedule",
    # Tier 2 -- customer-scoped, no identifier in any signature
    "get_my_vehicles",
    "get_my_orders",
    "get_my_tune_history",
    "get_my_open_case",
    # Control
    "escalate_to_human",
    "record_lead",
    "log_knowledge_gap",
    "request_approval",
}


def get_tool_schemas(
    persona: str = PERSONA_SUPPORT, tier_allowed: int = TIER_ANONYMOUS
) -> list[dict]:
    return REGISTRY.get_tool_schemas(persona, tier_allowed)


def get_tool(name: str) -> Tool | None:
    return REGISTRY.get(name)


def self_check() -> None:
    import json

    print(f"registered tools ({len(REGISTRY)}):")
    for tool in sorted(REGISTRY, key=lambda t: (t.tier, t.name)):
        print(
            f"  tier {tool.tier}  {tool.name:26s} dep={tool.dependency:10s} "
            f"writes={str(tool.writes):5s} personas={','.join(sorted(tool.personas))} "
            f"injects={tool.injects or '()'}"
        )
    # AGENT_PLAN.md §7 says "15 tools" and then enumerates 16: its Tier-0 table
    # puts search_products and lookup_product_by_sku on one row. They are two
    # callables with two schemas, so the registry holds 16. Asserting the name
    # set rather than the count keeps the check honest about that.
    assert set(REGISTRY.names()) == EXPECTED_TOOLS, (
        f"registry drift: missing {sorted(EXPECTED_TOOLS - set(REGISTRY.names()))}, "
        f"unexpected {sorted(set(REGISTRY.names()) - EXPECTED_TOOLS)}"
    )

    anon = get_tool_schemas(PERSONA_SUPPORT, TIER_ANONYMOUS)
    auth = get_tool_schemas(PERSONA_SUPPORT, 2)
    anon_names = {s["function"]["name"] for s in anon}
    auth_names = {s["function"]["name"] for s in auth}
    print("\nanonymous support sees:", sorted(anon_names))
    print("authenticated support gains:", sorted(auth_names - anon_names))
    assert not any(n.startswith("get_my_") for n in anon_names), "Tier 2 leaked into Tier 0 schemas"
    assert {"get_my_vehicles", "get_my_orders", "get_my_tune_history", "get_my_open_case"} <= auth_names

    sales = {s["function"]["name"] for s in get_tool_schemas(PERSONA_SALES, TIER_ANONYMOUS)}
    print("sales-only:", sorted(sales - anon_names))
    assert "record_lead" in sales and "record_lead" not in anon_names

    # No public schema may carry a customer identifier anywhere.
    for tool in REGISTRY:
        props = set(tool.parameters.get("properties", {}))
        assert not (props & {"customer_id", "customer", "user_id", "account_id", "phone"}), tool.name

    print("\nexample schema fed to the endpoint:")
    print(json.dumps(REGISTRY.get("check_stage_availability").to_openai_schema(), indent=1))
    print("registry.py self-check OK")


if __name__ == "__main__":
    self_check()
