"""The enforcement layer. Four mechanisms, none of them prompts.

AGENT_PLAN.md §7 is explicit that instructing a model not to do something is
not a control. What is implemented here instead:

1. **Read-only at the credential.** Reads go through ``mongo.source_db()``,
   which is a different connection from ``kb_db()``. Only the four control
   tools touch the write side, and they can only reach ``agent_*``. In
   production the read connection carries a Mongo user with ``read`` on
   ``transcribing``; an agent cannot write what the connection string cannot
   write, and no amount of prompt injection changes a connection string.

2. **Scoping by omission.** Tier-2 tools have no customer identifier in their
   public schema. ``_bind_injections`` supplies ``customer_id`` from
   ``SessionContext``, and a Tier-2 call from an unauthenticated session is
   refused *before dispatch* -- before argument validation, before the breaker,
   before the handler is looked at. The refusal is not "you may not"; there is
   no argument in which to put someone else's id.

3. **Budgets.** A hard per-turn cap (``config.MAX_TOOL_CALLS``), a per-tool
   per-turn cap that stops the loop calling ``search_knowledge`` six times with
   reworded queries, and a per-session budget counted in Mongo so it survives
   multiple workers. The per-turn cap is in-process because a turn is served by
   one worker; the session budget is not, because a session is not.

4. **Circuit breakers, one per dependency, with a transient/permanent split.**
   Only transient failures trip a breaker. A ``ToolInputError`` is the model's
   fault and a ``ToolPolicyError`` (a stale-data refusal) is the system working
   as designed -- counting either toward a trip would let a confused model take
   Mongo "down" for every other session by calling ``get_case('banana')`` five
   times.

The approval state machine (pending -> approve/deny -> execute) is persisted to
``agent_approvals``. It is deliberately not a module global: the implementation
this replaces used one, which meant approval state was per-worker and a request
approved on worker 1 was still pending on worker 2. Transitions are
``find_one_and_update`` guarded on the current state, so a double-approve is a
no-op rather than a race. An approval is also bound to a digest of the
arguments it was granted for, so it cannot be spent on a different call to the
same tool -- see ``arguments_digest``.

Every successful result carries a provenance token from ``base.make_provenance``
so ``guardrails/grounding.py`` can verify that a price or an availability claim
had a tool result behind it in the same turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import config
from ..clients.mongo import kb_db
from .base import (
    TIER_ANONYMOUS,
    TIER_CUSTOMER,
    Degraded,
    Tool,
    ToolInputError,
    ToolPolicyError,
    ToolResult,
    jsonable,
    make_provenance,
)
from .control import COLL_APPROVALS, _ensure_indexes
from .registry import REGISTRY, ToolRegistry

log = logging.getLogger(__name__)

# --- Session -----------------------------------------------------------------


@dataclass
class SessionContext:
    """Server-side session state. Nothing here is ever taken from a request body.

    ``authenticated`` and ``customer_id`` are set by the auth layer. A phone
    number that happens to match a record does not set them -- AGENT_PLAN.md
    §9.3: a phone match is evidence, not identity.
    """

    session_id: str
    persona: str = "support"
    customer_id: str | None = None
    authenticated: bool = False
    turn_id: str = field(default_factory=lambda: secrets.token_hex(6))

    @property
    def tier_allowed(self) -> int:
        return (
            TIER_CUSTOMER
            if (self.authenticated and self.customer_id)
            else TIER_ANONYMOUS
        )

    def new_turn(self) -> "SessionContext":
        return SessionContext(
            session_id=self.session_id,
            persona=self.persona,
            customer_id=self.customer_id,
            authenticated=self.authenticated,
            turn_id=secrets.token_hex(6),
        )


# --- Budgets -----------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    per_turn: int = config.MAX_TOOL_CALLS
    per_tool_per_turn: int = 3
    per_session: int = 40


class BudgetLedger:
    """Per-turn counts in memory, per-session counts in Mongo.

    Split on purpose. A turn never spans workers, so an in-process counter is
    both correct and free. A session does span workers, so its counter lives in
    ``agent_sessions`` where a ``$inc`` is atomic. Putting both in memory is the
    bug that makes a budget evaporate the moment you scale past one process.
    """

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self._lock = threading.Lock()
        self._turns: dict[str, dict[str, int]] = {}

    def check_and_consume(self, ctx: SessionContext, tool_name: str) -> str | None:
        """Returns a refusal string, or None if the call is within budget."""
        key = f"{ctx.session_id}:{ctx.turn_id}"
        with self._lock:
            counts = self._turns.setdefault(key, {})
            total = sum(counts.values())
            if total >= self.budget.per_turn:
                return (
                    f"per-turn tool budget exhausted ({self.budget.per_turn} calls). "
                    f"Answer with what you have, or escalate."
                )
            if counts.get(tool_name, 0) >= self.budget.per_tool_per_turn:
                return (
                    f"{tool_name} has already been called "
                    f"{self.budget.per_tool_per_turn} times this turn. Rewording the "
                    f"query will not change the answer -- use a different tool or say "
                    f"you do not know."
                )
            counts[tool_name] = counts.get(tool_name, 0) + 1

        try:
            doc = kb_db()[config.COLL_AGENT_SESSIONS].find_one_and_update(
                {"session_id": ctx.session_id},
                {
                    "$inc": {"tool_calls": 1},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                    "$setOnInsert": {"session_id": ctx.session_id, "persona": ctx.persona},
                },
                upsert=True,
                return_document=True,
            )
        except Exception as exc:  # noqa: BLE001 - budget must not take the agent down
            # Fail open on the durable half only. The in-process per-turn cap
            # has already been applied above, so the blast radius of a Mongo
            # hiccup here is "a long session gets a few extra calls", not
            # "the agent stops working".
            log.warning("session budget counter unavailable: %s", exc)
            return None
        if (doc.get("tool_calls") or 0) > self.budget.per_session:
            return (
                f"per-session tool budget exhausted ({self.budget.per_session} calls). "
                f"Escalate to a human."
            )
        return None

    def reset_turn(self, ctx: SessionContext) -> None:
        with self._lock:
            self._turns.pop(f"{ctx.session_id}:{ctx.turn_id}", None)

    def turn_usage(self, ctx: SessionContext) -> dict[str, int]:
        with self._lock:
            return dict(self._turns.get(f"{ctx.session_id}:{ctx.turn_id}", {}))


# --- Circuit breaker ---------------------------------------------------------

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


def is_transient(exc: BaseException) -> bool:
    """Transient-vs-permanent, adopted from the DGX pipeline's tested split.

    The default is *permanent*. An unrecognised exception is far more likely to
    be a bug in a handler than a network blip, and treating bugs as transient
    produces a breaker that flaps instead of one that protects.
    """
    from ..clients.llm import LLMTransientError

    if isinstance(exc, (ToolInputError, ToolPolicyError)):
        return False
    if isinstance(exc, LLMTransientError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    name = type(exc).__name__
    transient_names = {
        # pymongo
        "AutoReconnect",
        "NetworkTimeout",
        "ServerSelectionTimeoutError",
        "ConnectionFailure",
        "ExecutionTimeout",
        "WriteConcernError",
        "NotPrimaryError",
        # qdrant / httpx
        "ResponseHandlingException",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
    }
    if name in transient_names:
        return True
    if name == "UnexpectedResponse":
        status = getattr(exc, "status_code", None)
        return status is None or status >= 500 or status == 429
    if isinstance(exc, OSError):
        return True
    return False


class CircuitBreaker:
    """One per dependency. Trips on transient failures only."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        half_open_successes: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.half_open_successes = half_open_successes
        self._lock = threading.Lock()
        self.state = CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self.permanent_errors = 0
        self.trips = 0

    def allow(self) -> tuple[bool, str | None]:
        with self._lock:
            if self.state == CLOSED:
                return True, None
            if self.state == OPEN:
                if time.time() - self._opened_at >= self.recovery_seconds:
                    self.state = HALF_OPEN
                    self._successes = 0
                    return True, None
                wait = self.recovery_seconds - (time.time() - self._opened_at)
                return False, (
                    f"{self.name} is unavailable (circuit open, retry in {wait:.0f}s)"
                )
            return True, None  # HALF_OPEN: let a probe through

    def record_success(self) -> None:
        with self._lock:
            if self.state == HALF_OPEN:
                self._successes += 1
                if self._successes >= self.half_open_successes:
                    self.state = CLOSED
                    self._failures = 0
            else:
                self._failures = 0

    def record_failure(self, exc: BaseException) -> None:
        with self._lock:
            if not is_transient(exc):
                # Counted for observability, never toward a trip. A model
                # passing a bad case id five times must not open Mongo's
                # breaker for every other session on the box.
                self.permanent_errors += 1
                return
            self._failures += 1
            if self.state == HALF_OPEN or self._failures >= self.failure_threshold:
                self.state = OPEN
                self._opened_at = time.time()
                self.trips += 1

    def snapshot(self) -> dict:
        return {
            "dependency": self.name,
            "state": self.state,
            "consecutive_transient_failures": self._failures,
            "permanent_errors": self.permanent_errors,
            "trips": self.trips,
        }


# --- Approval state machine --------------------------------------------------

PENDING, APPROVED, DENIED, EXECUTED, EXPIRED = (
    "pending",
    "approved",
    "denied",
    "executed",
    "expired",
)


def arguments_digest(arguments: dict | None) -> str:
    """Stable digest of a call's arguments, used to bind an approval to them.

    An approval is granted for a specific action *with specific arguments*. A
    human who approved a $150 goodwill credit has not approved a $99,999 one,
    so ``dispatch`` only spends an approval whose digest matches the call in
    front of it. Keyed on ``(session, tool)`` alone the gate is bypassable: the
    model requests a modest action, a human approves it, and the next call
    reaches the handler with different arguments and an approval to spend.
    """
    canon = json.dumps(
        jsonable(dict(arguments or {})), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


class ApprovalStore:
    """pending -> approve/deny -> execute, persisted to ``agent_approvals``.

    Every transition is a guarded ``find_one_and_update``: the filter names the
    state being left, so two humans clicking approve produce one approval and
    one "already handled", not two executions.
    """

    TTL_HOURS = 48.0

    def __init__(self, collection: str = COLL_APPROVALS) -> None:
        self._coll_name = collection

    @property
    def _coll(self):
        return kb_db()[self._coll_name]

    def create(
        self,
        *,
        session_id: str,
        action: str,
        arguments: dict,
        justification: str,
        customer_id: str | None = None,
        persona: str | None = None,
        tool_name: str | None = None,
    ) -> dict:
        _ensure_indexes()
        now = datetime.now(timezone.utc)
        doc = {
            "approval_id": f"APR-{now:%Y%m%d}-{secrets.token_hex(4)}",
            "session_id": session_id,
            "customer_id": customer_id,
            "persona": persona,
            "tool_name": tool_name,
            "action": action,
            "arguments": arguments,
            "arguments_digest": arguments_digest(arguments),
            "justification": justification,
            "state": PENDING,
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(hours=self.TTL_HOURS),
            "decided_by": None,
            "decided_at": None,
            "decision_note": None,
            "result_summary": None,
        }
        self._coll.insert_one(doc)
        doc.pop("_id", None)
        return doc

    def get(self, approval_id: str) -> dict | None:
        doc = self._coll.find_one({"approval_id": approval_id}, {"_id": 0})
        if doc and doc["state"] == PENDING and self._is_expired(doc):
            return self._expire(approval_id)
        return doc

    def _is_expired(self, doc: dict) -> bool:
        exp = doc.get("expires_at")
        if not isinstance(exp, datetime):
            return False
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp

    def _expire(self, approval_id: str) -> dict | None:
        return self._coll.find_one_and_update(
            {"approval_id": approval_id, "state": PENDING},
            {"$set": {"state": EXPIRED, "updated_at": datetime.now(timezone.utc)}},
            projection={"_id": 0},
            return_document=True,
        )

    def _decide(self, approval_id: str, new_state: str, approver: str, note: str | None) -> dict:
        now = datetime.now(timezone.utc)
        doc = self._coll.find_one_and_update(
            {"approval_id": approval_id, "state": PENDING},
            {
                "$set": {
                    "state": new_state,
                    "decided_by": approver,
                    "decided_at": now,
                    "decision_note": note,
                    "updated_at": now,
                }
            },
            projection={"_id": 0},
            return_document=True,
        )
        if doc is None:
            current = self.get(approval_id)
            if current is None:
                raise ToolInputError(f"no approval {approval_id}")
            raise ToolPolicyError(
                f"approval {approval_id} is already {current['state']}; "
                f"it cannot be moved to {new_state}"
            )
        return doc

    def approve(self, approval_id: str, approver: str, note: str | None = None) -> dict:
        return self._decide(approval_id, APPROVED, approver, note)

    def deny(self, approval_id: str, approver: str, reason: str) -> dict:
        return self._decide(approval_id, DENIED, approver, reason)

    def mark_executed(self, approval_id: str, result_summary: str) -> dict:
        now = datetime.now(timezone.utc)
        doc = self._coll.find_one_and_update(
            {"approval_id": approval_id, "state": APPROVED},
            {
                "$set": {
                    "state": EXECUTED,
                    "executed_at": now,
                    "updated_at": now,
                    "result_summary": result_summary[:500],
                }
            },
            projection={"_id": 0},
            return_document=True,
        )
        if doc is None:
            raise ToolPolicyError(
                f"approval {approval_id} is not in state {APPROVED}; refusing to execute"
            )
        return doc

    def pending_for_session(self, session_id: str) -> list[dict]:
        return list(
            self._coll.find({"session_id": session_id, "state": PENDING}, {"_id": 0})
        )

    def find_approved(
        self, session_id: str, tool_name: str, arguments_digest: str | None = None
    ) -> dict | None:
        """An approval usable for this exact call.

        ``arguments_digest`` is not optional in practice -- ``dispatch`` always
        passes it. The default exists only for callers that legitimately do not
        have arguments in hand (the human console listing what is outstanding).
        """
        query: dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
            "state": APPROVED,
        }
        if arguments_digest is not None:
            query["arguments_digest"] = arguments_digest
        return self._coll.find_one(query, {"_id": 0}, sort=[("created_at", 1)])


# --- Executor ----------------------------------------------------------------


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        budget: Budget | None = None,
        approvals: ApprovalStore | None = None,
        audit: bool = True,
    ) -> None:
        self.registry = registry or REGISTRY
        self.budgets = BudgetLedger(budget)
        self.approvals = approvals or ApprovalStore()
        self.audit = audit
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breaker_lock = threading.Lock()

    # -- breakers
    def breaker(self, dependency: str) -> CircuitBreaker:
        with self._breaker_lock:
            if dependency not in self._breakers:
                self._breakers[dependency] = CircuitBreaker(dependency)
            return self._breakers[dependency]

    def breaker_states(self) -> list[dict]:
        with self._breaker_lock:
            return [b.snapshot() for b in self._breakers.values()]

    # -- schemas
    def tool_schemas(self, ctx: SessionContext) -> list[dict]:
        return self.registry.get_tool_schemas(ctx.persona, ctx.tier_allowed)

    # -- injection
    def _bind_injections(self, tool: Tool, ctx: SessionContext) -> dict:
        available = {
            "session_id": ctx.session_id,
            "customer_id": ctx.customer_id,
            "persona": ctx.persona,
            "turn_id": ctx.turn_id,
        }
        missing = [name for name in tool.injects if name not in available]
        if missing:
            raise ToolPolicyError(
                f"{tool.name} requires session values {missing} that this executor "
                f"cannot supply"
            )
        return {name: available[name] for name in tool.injects}

    # -- dispatch
    def dispatch(
        self, name: str, arguments: dict | None, ctx: SessionContext
    ) -> ToolResult:
        started = time.perf_counter()

        tool = self.registry.get(name)
        if tool is None:
            return self._finish(
                ToolResult.failure(
                    name,
                    f"no tool named {name!r}. Available: "
                    f"{[s['function']['name'] for s in self.tool_schemas(ctx)]}",
                    kind="input",
                ),
                started,
                ctx,
                arguments,
            )

        # (2) Scoping by omission -- checked before anything else touches the
        # arguments, so an unauthenticated Tier-2 attempt never reaches code
        # that could be tricked into looking something up.
        if tool.tier > ctx.tier_allowed:
            return self._finish(
                ToolResult.failure(
                    name,
                    f"{name} requires an authenticated customer session (tier "
                    f"{tool.tier}); this session is tier {ctx.tier_allowed}. Ask the "
                    f"customer to sign in. Do not ask them for an account number -- "
                    f"there is nowhere to put one.",
                    kind="policy",
                    tier=tool.tier,
                ),
                started,
                ctx,
                arguments,
            )

        if ctx.persona not in tool.personas:
            return self._finish(
                ToolResult.failure(
                    name,
                    f"{name} is not available to the {ctx.persona} persona",
                    kind="policy",
                ),
                started,
                ctx,
                arguments,
            )

        # (3) Budgets.
        refusal = self.budgets.check_and_consume(ctx, name)
        if refusal:
            return self._finish(
                ToolResult.failure(name, refusal, kind="budget"), started, ctx, arguments
            )

        # Human-in-the-loop gate, for tools declared as needing one. The
        # approval is looked up by (session, tool, arguments digest): an
        # approval granted for one set of arguments cannot be spent on another.
        arg_digest = arguments_digest(arguments)
        if tool.requires_approval:
            approved = self.approvals.find_approved(ctx.session_id, name, arg_digest)
            if approved is None:
                record = self.approvals.create(
                    session_id=ctx.session_id,
                    action=name,
                    arguments=dict(arguments or {}),
                    justification=f"model requested {name}",
                    customer_id=ctx.customer_id,
                    persona=ctx.persona,
                    tool_name=name,
                )
                return self._finish(
                    ToolResult.failure(
                        name,
                        f"{name} needs human approval. Request {record['approval_id']} "
                        f"is pending. Tell the customer it has been requested; do not "
                        f"describe the action as done.",
                        kind="policy",
                    ),
                    started,
                    ctx,
                    arguments,
                )

        # (4) Breakers.
        breaker = self.breaker(tool.dependency)
        ok, why = breaker.allow()
        if not ok:
            return self._finish(
                ToolResult.failure(name, why or "dependency unavailable", kind="dependency"),
                started,
                ctx,
                arguments,
            )

        try:
            clean = tool.validate_arguments(arguments)
            clean.update(self._bind_injections(tool, ctx))
            raw = tool.handler(**clean)
        except ToolInputError as exc:
            breaker.record_failure(exc)
            return self._finish(
                ToolResult.failure(name, str(exc), kind="input", tier=tool.tier),
                started,
                ctx,
                arguments,
            )
        except ToolPolicyError as exc:
            breaker.record_failure(exc)
            return self._finish(
                ToolResult.failure(name, str(exc), kind="policy", tier=tool.tier),
                started,
                ctx,
                arguments,
            )
        except Exception as exc:  # noqa: BLE001 - classified, then re-shaped
            breaker.record_failure(exc)
            kind = "dependency" if is_transient(exc) else "input"
            log.exception("tool %s failed", name)
            return self._finish(
                ToolResult.failure(
                    name, f"{type(exc).__name__}: {exc}", kind=kind, tier=tool.tier
                ),
                started,
                ctx,
                arguments,
            )

        breaker.record_success()

        if isinstance(raw, Degraded):
            result = ToolResult.success(
                name, raw.data, tier=tool.tier, degraded=True, degraded_reason=raw.reason
            )
        else:
            result = ToolResult.success(name, raw, tier=tool.tier)

        if tool.requires_approval:
            approved = self.approvals.find_approved(ctx.session_id, name, arg_digest)
            if approved:
                try:
                    self.approvals.mark_executed(
                        approved["approval_id"], f"{name} executed"
                    )
                except ToolPolicyError as exc:
                    # Another worker consumed the same approval between the
                    # gate check and here. The handler has already run, so the
                    # honest thing is to record it and return the result rather
                    # than raise out of dispatch with a 500.
                    log.warning("approval %s already spent: %s", approved["approval_id"], exc)

        return self._finish(result, started, ctx, arguments)

    def _finish(
        self,
        result: ToolResult,
        started: float,
        ctx: SessionContext,
        arguments: dict | None,
    ) -> ToolResult:
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        if self.audit:
            self._audit(result, ctx, arguments)
        return result

    def _audit(self, result: ToolResult, ctx: SessionContext, arguments: dict | None) -> None:
        """Append-only trail in ``agent_events``.

        Argument *keys* are recorded, not values: a lead's email address and a
        customer's error text are both arguments, and an audit log that
        accumulates them is a second PII store nobody remembers to purge.
        """
        try:
            kb_db()[config.COLL_AGENT_EVENTS].insert_one(
                {
                    "session_id": ctx.session_id,
                    "turn_id": ctx.turn_id,
                    "ts": datetime.now(timezone.utc),
                    "kind": "tool_call",
                    "tool": result.tool,
                    "persona": ctx.persona,
                    "tier": result.tier,
                    "authenticated": ctx.authenticated,
                    "ok": result.ok,
                    "error_kind": result.error_kind,
                    "degraded": result.degraded,
                    "latency_ms": round(result.latency_ms, 1),
                    "argument_keys": sorted(arguments or {}),
                    "provenance": result.provenance or None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("audit write failed: %s", exc)

    # -- approval driving, for the human console
    def approve_and_execute(
        self, approval_id: str, approver: str, ctx: SessionContext, note: str | None = None
    ) -> ToolResult:
        """pending -> approved -> execute, in that order and only in that order.

        The ``approved -> executed`` transition is left to ``dispatch``, which
        is the single place that knows the handler actually ran. Marking it
        here as well would be a second transition attempt on a record already
        in ``executed`` -- which the guarded update correctly rejects, but
        loudly and for the wrong reason.
        """
        self.approvals.approve(approval_id, approver, note)
        record = self.approvals.get(approval_id) or {}
        tool_name = record.get("tool_name") or record.get("action")
        return self.dispatch(tool_name, record.get("arguments") or {}, ctx)

    # -- grounding support
    def verify_result_token(self, token: str, tool: str, data: Any) -> bool:
        """Re-derivable by ``guardrails/grounding.py`` without shared state."""
        return token == make_provenance(tool, data)


_default: ToolExecutor | None = None
_default_lock = threading.Lock()


def get_executor() -> ToolExecutor:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = ToolExecutor()
    return _default


def self_check() -> None:
    import json

    from .base import verify_provenance

    ex = ToolExecutor()
    sid = f"selfcheck-{secrets.token_hex(4)}"

    anon = SessionContext(session_id=sid, persona="support")
    print("--- (2) Tier-2 call from an unauthenticated session ---")
    r = ex.dispatch("get_my_orders", {}, anon)
    print(json.dumps(r.to_dict(), indent=1))
    assert r.ok is False and r.error_kind == "policy"
    assert "get_my_orders" not in {
        s["function"]["name"] for s in ex.tool_schemas(anon)
    }, "Tier-2 tool visible to an anonymous session"

    print("--- and the same call cannot smuggle an id ---")
    r2 = ex.dispatch("get_my_orders", {"customer_id": "1178"}, anon)
    print(json.dumps(r2.to_dict(), indent=1))
    assert r2.ok is False

    print("--- Tier-0 call succeeds and carries provenance ---")
    r3 = ex.dispatch("check_stage_availability", {"platform_id": 80, "stage": "Stage 2"}, anon)
    print("ok:", r3.ok, "degraded:", r3.degraded)
    print("degraded_reason:", r3.degraded_reason)
    print("provenance:", r3.provenance)
    assert r3.ok and verify_provenance(r3.provenance, "check_stage_availability", r3.data)
    assert r3.data["available"] is True

    print("--- authenticated session reaches Tier 2 ---")
    from ..clients.mongo import source_db

    cust = source_db()[config.COLL_CUSTOMERS].find_one(
        {"order_count": {"$gt": 0}, "is_deleted": {"$ne": True}}, {"customer_id": 1}
    )
    auth = SessionContext(
        session_id=sid + "-auth",
        persona="support",
        customer_id=cust["customer_id"],
        authenticated=True,
    )
    r4 = ex.dispatch("get_my_orders", {}, auth)
    print("ok:", r4.ok, "orders:", r4.data["order_count"] if r4.ok else r4.error)
    assert r4.ok

    print("--- argument validation: string int is coerced, junk is dropped ---")
    r5 = ex.dispatch(
        "check_stage_availability", {"platform_id": "80", "nonsense": True}, anon
    )
    assert r5.ok, r5.error
    r6 = ex.dispatch("check_stage_availability", {}, anon)
    print("missing required ->", r6.error_kind, "|", r6.error)
    assert r6.error_kind == "input"

    print("--- (3) budgets ---")
    tight = ToolExecutor(budget=Budget(per_turn=3, per_tool_per_turn=2, per_session=100))
    bctx = SessionContext(session_id=sid + "-budget")
    outcomes = [
        tight.dispatch("get_fee_schedule", {}, bctx).to_dict() for _ in range(4)
    ]
    for i, o in enumerate(outcomes, 1):
        print(f"  call {i}: ok={o['ok']} {o.get('error_kind') or ''} {o.get('error','')[:70]}")
    assert outcomes[0]["ok"] and outcomes[1]["ok"]
    assert outcomes[2]["ok"] is False and outcomes[2]["error_kind"] == "budget"

    print("--- (4) breaker: permanent errors do NOT trip it ---")
    br_ctx = SessionContext(session_id=sid + "-br")
    loose = ToolExecutor(budget=Budget(per_turn=50, per_tool_per_turn=50, per_session=500))
    for _ in range(8):
        loose.dispatch("get_case", {"case_id": "definitely-not-a-case"}, br_ctx)
    mongo_breaker = loose.breaker("mongo")
    print(" ", json.dumps(mongo_breaker.snapshot(), indent=1))
    assert mongo_breaker.state == CLOSED, "permanent errors tripped the breaker"
    assert mongo_breaker.permanent_errors >= 8

    print("--- breaker: transient failures DO trip it, then recover ---")
    probe = CircuitBreaker("probe", failure_threshold=3, recovery_seconds=0.05)
    for _ in range(3):
        probe.record_failure(TimeoutError("boom"))
    print("  after 3 transient failures:", probe.snapshot())
    assert probe.state == OPEN
    allowed, why = probe.allow()
    assert allowed is False and why
    print("  refusal:", why)
    time.sleep(0.06)
    allowed, _ = probe.allow()
    assert allowed and probe.state == HALF_OPEN
    probe.record_success()
    probe.record_success()
    assert probe.state == CLOSED
    print("  recovered:", probe.snapshot())

    print("--- approval state machine (persisted, guarded transitions) ---")
    store = ApprovalStore()
    rec = store.create(
        session_id=sid,
        action="goodwill_credit",
        arguments={"amount": 150, "order_id": 78098},
        justification="duplicate licence transfer charge",
        tool_name="goodwill_credit",
    )
    print("  created:", rec["approval_id"], rec["state"])
    assert rec["state"] == PENDING
    approved = store.approve(rec["approval_id"], "jordan@unitronic")
    print("  approved by:", approved["decided_by"], "->", approved["state"])
    try:
        store.approve(rec["approval_id"], "someone-else")
        raise AssertionError("double approve permitted")
    except ToolPolicyError as exc:
        print("  second approve refused:", exc)
    executed = store.mark_executed(rec["approval_id"], "credit issued")
    assert executed["state"] == EXECUTED
    try:
        store.mark_executed(rec["approval_id"], "again")
        raise AssertionError("double execute permitted")
    except ToolPolicyError as exc:
        print("  second execute refused:", exc)

    denied = store.create(
        session_id=sid, action="refund", arguments={}, justification="outside window",
        tool_name="refund",
    )
    store.deny(denied["approval_id"], "jordan@unitronic", "outside the 15-day window")
    assert store.get(denied["approval_id"])["state"] == DENIED
    print("  deny path OK")

    print("--- approval-gated dispatch (pending -> approve -> execute) ---")
    from .base import obj_schema

    executions: list[dict] = []

    def _issue_credit(*, amount: int, session_id: str) -> dict:
        executions.append({"amount": amount, "session_id": session_id})
        return {"issued": True, "amount": amount}

    gated_registry = ToolRegistry()
    gated_registry.register(
        Tool(
            name="issue_goodwill_credit",
            description="gated demo tool",
            parameters=obj_schema({"amount": {"type": "integer"}}, ["amount"]),
            handler=_issue_credit,
            dependency="mongo",
            writes=True,
            requires_approval=True,
            injects=("session_id",),
        )
    )
    gex = ToolExecutor(gated_registry, budget=Budget(per_turn=20, per_tool_per_turn=20))
    gctx = SessionContext(session_id=sid + "-gate", persona="support")

    first = gex.dispatch("issue_goodwill_credit", {"amount": 150}, gctx)
    print("  first call:", first.error_kind, "|", first.error)
    assert first.ok is False and first.error_kind == "policy"
    assert not executions, "gated tool ran without approval"

    pending = gex.approvals.pending_for_session(gctx.session_id)
    assert len(pending) == 1
    aid = pending[0]["approval_id"]
    ran = gex.approve_and_execute(aid, "jordan@unitronic", gctx)
    print("  after approval:", json.dumps(ran.to_dict(), indent=1))
    assert ran.ok and executions == [{"amount": 150, "session_id": gctx.session_id}]
    assert gex.approvals.get(aid)["state"] == EXECUTED

    print("--- an approval cannot be spent on different arguments ---")
    sctx = SessionContext(session_id=sid + "-swap", persona="support")
    executions.clear()
    first = gex.dispatch("issue_goodwill_credit", {"amount": 150}, sctx)
    assert first.ok is False
    swap_id = gex.approvals.pending_for_session(sctx.session_id)[0]["approval_id"]
    gex.approvals.approve(swap_id, "jordan@unitronic")
    # The human approved 150. The model now asks for 99999 against it.
    attack = gex.dispatch("issue_goodwill_credit", {"amount": 99999}, sctx)
    print("  swapped-argument call:", attack.error_kind, "|", attack.error)
    assert attack.ok is False and attack.error_kind == "policy", "approval spent on swapped arguments"
    assert not executions, f"gated handler ran with unapproved arguments: {executions}"
    assert gex.approvals.get(swap_id)["state"] == APPROVED, "approval consumed by the swap"
    # The originally approved arguments still execute exactly once.
    honest = gex.dispatch("issue_goodwill_credit", {"amount": 150}, sctx)
    assert honest.ok and executions == [{"amount": 150, "session_id": sctx.session_id}]
    assert gex.approvals.get(swap_id)["state"] == EXECUTED
    print("  original arguments still execute once; approval now:", EXECUTED)
    for s in (gctx.session_id, sctx.session_id):
        kb_db()[COLL_APPROVALS].delete_many({"session_id": s})
        for coll in (config.COLL_AGENT_EVENTS, config.COLL_AGENT_SESSIONS):
            kb_db()[coll].delete_many({"session_id": s})
    kb_db()[COLL_APPROVALS].delete_many({"session_id": gctx.session_id})
    for coll in (config.COLL_AGENT_EVENTS, config.COLL_AGENT_SESSIONS):
        kb_db()[coll].delete_many({"session_id": gctx.session_id})

    print("--- audit trail ---")
    events = kb_db()[config.COLL_AGENT_EVENTS].count_documents({"session_id": sid})
    print("  agent_events rows for this session:", events)
    assert events > 0
    sample = kb_db()[config.COLL_AGENT_EVENTS].find_one({"session_id": sid}, {"_id": 0})
    print("  sample:", json.dumps(sample, indent=1, default=str))

    # Clean up only what this self-check created.
    for coll in (config.COLL_AGENT_EVENTS, config.COLL_AGENT_SESSIONS):
        kb_db()[coll].delete_many({"session_id": {"$regex": f"^{sid}"}})
    kb_db()[COLL_APPROVALS].delete_many({"session_id": sid})
    print("executor.py self-check OK")


if __name__ == "__main__":
    self_check()
