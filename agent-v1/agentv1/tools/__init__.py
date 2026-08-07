"""Native in-process tool layer.

Everything the agent loop needs is re-exported here, but note what is *not*:
the individual handler functions. Callers go through ``ToolExecutor.dispatch``,
because dispatch is where tier enforcement, budgets, breakers, the approval
gate and the provenance token live. Importing ``get_my_orders`` and calling it
directly skips all five, which is why the tool modules are not re-exported.

MCP is deliberately absent. A tool call inside a loop making 2-6 per turn
should cost microseconds; an HTTP round trip per call spends the latency budget
on transport. MCP is retained upstream only where a second client needs the
same surface.
"""

from .base import (
    ALL_PERSONAS,
    PERSONA_SALES,
    PERSONA_SUPPORT,
    TIER_ANONYMOUS,
    TIER_CUSTOMER,
    Degraded,
    Tool,
    ToolDependencyError,
    ToolInputError,
    ToolPolicyError,
    ToolResult,
    find_provenance_tokens,
    make_provenance,
    verify_provenance,
)
from .executor import (
    ApprovalStore,
    Budget,
    CircuitBreaker,
    SessionContext,
    ToolExecutor,
    get_executor,
    is_transient,
)
from .registry import EXPECTED_TOOLS, REGISTRY, ToolRegistry, get_tool, get_tool_schemas

__all__ = [
    "ALL_PERSONAS",
    "PERSONA_SALES",
    "PERSONA_SUPPORT",
    "TIER_ANONYMOUS",
    "TIER_CUSTOMER",
    "Degraded",
    "Tool",
    "ToolDependencyError",
    "ToolInputError",
    "ToolPolicyError",
    "ToolResult",
    "find_provenance_tokens",
    "make_provenance",
    "verify_provenance",
    "ApprovalStore",
    "Budget",
    "CircuitBreaker",
    "SessionContext",
    "ToolExecutor",
    "get_executor",
    "is_transient",
    "EXPECTED_TOOLS",
    "REGISTRY",
    "ToolRegistry",
    "get_tool",
    "get_tool_schemas",
]
