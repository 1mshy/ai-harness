"""Tool definition, result envelope and the provenance token.

Three things are decided here rather than in each tool module.

*The JSON schema is the security boundary, not the docstring.* A Tier-2 tool's
schema has no ``customer_id`` property, so "look up someone else's order" is
not a request the model can even encode. ``Tool.injects`` names the arguments
the executor supplies from server-side session state, and registration asserts
that no injected name also appears in the schema -- otherwise a model-supplied
value would silently shadow the session one.

*Every result carries a provenance token.* AGENT_PLAN.md §9.5 requires that a
price or a stage-availability claim be traceable to a tool result in the same
turn. The token is a digest of ``(tool name, result payload)``, so
``guardrails/grounding.py`` can recompute it from the tool message it already
has in the transcript and does not need to share mutable state with the
executor. That matters because the two run in different processes under
multiple workers.

*Argument validation is code, not trust.* The endpoint's tool parser emits
well-formed JSON but not necessarily *correct* JSON -- extra properties and
string-typed integers both occur. Unknown properties are dropped rather than
passed through, because a dropped argument is a narrower query and a passed-
through one is an unvalidated field in a Mongo filter.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Iterable

# Tier 0 needs no identity at all. Tier 2 is an authenticated, customer-scoped
# session. There is deliberately no Tier 1: a half-identified caller is the
# case AGENT_PLAN.md §9.3 warns about ("a phone match is evidence, not
# identity"), and giving it its own tier invites tools to be written against
# it.
TIER_ANONYMOUS = 0
TIER_CUSTOMER = 2

PERSONA_SALES = "sales"
PERSONA_SUPPORT = "support"
ALL_PERSONAS = frozenset({PERSONA_SALES, PERSONA_SUPPORT})


class ToolInputError(ValueError):
    """The arguments are wrong. Retrying the same call will not help."""


class ToolDependencyError(RuntimeError):
    """A backing system failed. May be transient; the breaker decides."""


class ToolPolicyError(RuntimeError):
    """The call was well-formed but must not be answered.

    Distinct from a dependency failure because it must never trip a circuit
    breaker: a freshness refusal on a stale platform table is the system
    working, and opening a breaker on it would take down every other tool
    sharing the dependency.
    """


@dataclass
class Degraded:
    """Handler return wrapper: usable data that must be labelled as such.

    A stale platform snapshot still answers "does Stage 3 exist for platform
    80"; it just cannot claim to be current. Returning a bare dict here would
    lose that distinction by the time the model reads it.
    """

    data: Any
    reason: str


# --- Provenance --------------------------------------------------------------

_PROVENANCE_RE = re.compile(r"\btool:([a-z_][a-z0-9_]*):([0-9a-f]{16})\b")


_PRIMITIVES = (str, int, float, bool, type(None))


def jsonable(value: Any) -> Any:
    """Coerce a handler's return value into something ``json.dumps`` accepts.

    Applied once, centrally, rather than per tool. Mongo hands back
    ``datetime`` and ``ObjectId``; Qdrant hands back numpy scalars. Any of them
    reaching ``json.dumps`` without a ``default=`` hook is a 500 on the tool
    message, and adding ``default=str`` at the serialisation site instead would
    mean the provenance digest is computed over one representation and the
    wire format carries another.
    """
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    # numpy scalars and anything else that quacks like a number.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _canonical(data: Any) -> str:
    """Stable serialisation so the digest survives dict ordering and floats."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def make_provenance(tool: str, data: Any) -> str:
    """``tool:<name>:<16 hex>`` -- deterministic over the result payload.

    Deterministic rather than random precisely so the grounding check is
    stateless: given the tool message in the transcript it can recompute the
    token and reject one that was invented by the model.
    """
    digest = hashlib.sha256(f"{tool}\x00{_canonical(data)}".encode()).hexdigest()[:16]
    return f"tool:{tool}:{digest}"


def verify_provenance(token: str, tool: str, data: Any) -> bool:
    return token == make_provenance(tool, data)


def find_provenance_tokens(text: str) -> list[tuple[str, str]]:
    """Every ``(tool, digest)`` pair mentioned in a blob of text."""
    return _PROVENANCE_RE.findall(text or "")


# --- Result envelope ---------------------------------------------------------


@dataclass
class ToolResult:
    """What the executor puts on the wire as the ``tool`` role message.

    ``degraded`` is separate from ``ok`` on purpose. A stale platform table
    still answers, but the answer must be labelled; collapsing that into a
    boolean is how a six-day-old snapshot gets quoted as current stock.
    """

    tool: str
    ok: bool
    data: Any = None
    provenance: str = ""
    error: str | None = None
    error_kind: str | None = None  # "input" | "dependency" | "policy" | "budget"
    degraded: bool = False
    degraded_reason: str | None = None
    tier: int = TIER_ANONYMOUS
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)

    @classmethod
    def success(
        cls,
        tool: str,
        data: Any,
        *,
        tier: int = TIER_ANONYMOUS,
        degraded: bool = False,
        degraded_reason: str | None = None,
    ) -> "ToolResult":
        # Coerce first, then digest, so the token is computed over exactly the
        # bytes that go on the wire and grounding can recompute it from the
        # transcript alone.
        data = jsonable(data)
        return cls(
            tool=tool,
            ok=True,
            data=data,
            provenance=make_provenance(tool, data),
            tier=tier,
            degraded=degraded,
            degraded_reason=degraded_reason,
        )

    @classmethod
    def failure(
        cls, tool: str, error: str, *, kind: str = "dependency", tier: int = TIER_ANONYMOUS
    ) -> "ToolResult":
        # A failure carries no provenance token. That is the whole point: the
        # model cannot cite a call that did not return data.
        return cls(tool=tool, ok=False, error=error, error_kind=kind, tier=tier)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"tool": self.tool, "ok": self.ok}
        if self.ok:
            out["data"] = self.data
            out["provenance"] = self.provenance
        else:
            out["error"] = self.error
            out["error_kind"] = self.error_kind
        if self.degraded:
            out["degraded"] = True
            out["degraded_reason"] = self.degraded_reason
        out["latency_ms"] = round(self.latency_ms, 1)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


# --- Tool --------------------------------------------------------------------

_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema, object type, fed straight to the endpoint
    handler: Callable[..., Any]
    tier: int = TIER_ANONYMOUS
    # Which backing system this call touches. The circuit breaker is keyed on
    # this, so a Qdrant outage does not open the breaker on Mongo tools.
    dependency: str = "none"
    personas: frozenset = ALL_PERSONAS
    # Server-side values the executor binds. Never in `parameters`.
    injects: tuple[str, ...] = ()
    # Writes to a collection this project owns. Used by the executor to route
    # through the approval state machine and to refuse in dry-run mode.
    writes: bool = False
    requires_approval: bool = False

    def __post_init__(self) -> None:
        props = set((self.parameters or {}).get("properties", {}))
        clash = props.intersection(self.injects)
        if clash:
            # Fail at import, not at request time. A model-supplied
            # `customer_id` that shadows the session one is the exact hole
            # Tier 2 exists to close.
            raise ValueError(
                f"{self.name}: injected argument(s) {sorted(clash)} also appear in the "
                f"public schema. Injected values must be inexpressible by the model."
            )
        if self.tier == TIER_CUSTOMER and "customer_id" not in self.injects:
            raise ValueError(
                f"{self.name}: Tier-2 tools must inject customer_id from session state."
            )

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "required": []},
            },
        }

    def validate_arguments(self, args: dict | None) -> dict:
        """Drop unknown keys, check required, coerce the two types the model
        gets wrong (int-as-string, scalar-as-single-element-array)."""
        args = dict(args or {})
        schema = self.parameters or {}
        props: dict[str, dict] = schema.get("properties", {}) or {}
        required: Iterable[str] = schema.get("required", []) or []

        clean: dict[str, Any] = {}
        for key, spec in props.items():
            if key not in args or args[key] is None:
                continue
            value = args[key]
            want = spec.get("type")
            if want == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
                value = int(value.strip())
            elif want == "number" and isinstance(value, str):
                try:
                    value = float(value.strip())
                except ValueError:
                    pass
            elif want == "array" and not isinstance(value, list):
                value = [value]
            elif want == "string" and isinstance(value, (int, float)):
                value = str(value)
            py = _JSON_TYPES.get(want)
            if py and not isinstance(value, py):
                raise ToolInputError(
                    f"{self.name}.{key}: expected {want}, got {type(value).__name__}"
                )
            enum = spec.get("enum")
            if enum and value not in enum:
                raise ToolInputError(
                    f"{self.name}.{key}: {value!r} not one of {enum}"
                )
            clean[key] = value

        missing = [k for k in required if k not in clean]
        if missing:
            raise ToolInputError(f"{self.name}: missing required argument(s) {missing}")
        return clean


def obj_schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def self_check() -> None:
    t = Tool(
        name="demo",
        description="demo",
        parameters=obj_schema({"n": {"type": "integer"}}, ["n"]),
        handler=lambda n: n,
    )
    assert t.to_openai_schema()["function"]["name"] == "demo"
    assert t.validate_arguments({"n": "7", "junk": 1}) == {"n": 7}
    try:
        t.validate_arguments({})
        raise AssertionError("missing required not caught")
    except ToolInputError:
        pass

    tok = make_provenance("demo", {"a": 1, "b": [2, 3]})
    assert verify_provenance(tok, "demo", {"b": [2, 3], "a": 1}), "digest not order-stable"
    assert not verify_provenance(tok, "demo", {"a": 2})
    assert find_provenance_tokens(f"see {tok} here") == [("demo", tok.split(":")[2])]

    try:
        Tool(
            name="bad",
            description="",
            parameters=obj_schema({"customer_id": {"type": "string"}}),
            handler=lambda **_: None,
            tier=TIER_CUSTOMER,
            injects=("customer_id",),
        )
        raise AssertionError("injected/public clash not caught")
    except ValueError:
        pass

    r = ToolResult.success("demo", {"x": 1})
    assert verify_provenance(r.provenance, "demo", {"x": 1})
    assert ToolResult.failure("demo", "boom").to_dict().get("provenance") is None

    # A datetime from Mongo must survive to the wire, and the token must be
    # derivable from the wire form without knowing it was ever a datetime.
    dirty = ToolResult.success("demo", {"when": datetime(2026, 8, 5, 12, 0), "n": Decimal("1.5")})
    assert json.dumps(dirty.to_dict())  # no default= hook
    assert dirty.data == {"when": "2026-08-05T12:00:00", "n": 1.5}
    assert verify_provenance(dirty.provenance, "demo", dirty.data)
    print("base.py self-check OK")


if __name__ == "__main__":
    self_check()
