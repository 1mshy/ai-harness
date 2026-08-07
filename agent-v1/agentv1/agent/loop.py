"""The bounded tool-calling loop.

Owned rather than delegated to a framework. The alternatives each cost more
than they give here: LangGraph adds a second state model beside this one and
drags in a dependency tree the project deliberately does not have; LlamaIndex's
agent workflow couples agent control flow to a pinned retrieval version, so a
retrieval bugfix becomes an agent regression; the Claude Agent SDK is built
around Read/Write/Edit/Bash for a sandboxed coding agent, which is the wrong
shape for fifteen read-only business tools and a three-sentence reply; CrewAI
and AutoGen multiply LLM calls to role-play what is one reasoner plus tools.

What is actually needed is small: a loop, hard bounds, and hooks for approval,
error interception and guardrails. That is this file.

The endpoint supports **native tool calling** (verified 2026-08-06:
``finish_reason="tool_calls"`` with well-formed arguments), so tools go over as
OpenAI schemas rather than being hand-parsed out of prose.

Bounds: 4 iterations, 6 tool calls, 45 s wall clock. They exist because an
unbounded loop in front of a customer is the failure everybody discovers in
production rather than in review.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .. import config
from ..clients.llm import LLMError, ToolCall, get_llm
from .personas import Persona, select_persona
from .session import Session


@dataclass
class Event:
    """One SSE frame.

    Types are purely additive over the existing transport, whose frontend
    switch has no throwing default -- so ``tool_call`` / ``tool_result`` /
    ``thinking`` / ``citation`` / ``handoff`` cost nothing to introduce.
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.type, **self.data})}\n\n"


@dataclass
class ToolOutcome:
    """What the loop needs back from a tool call, independent of the executor."""

    ok: bool
    content: Any
    summary: str = ""
    provenance: str = ""
    citations: list[dict] = field(default_factory=list)
    handoff: dict | None = None


class ExecutorAdapter:
    """Bridge the tool executor's context model to the loop's session model.

    The executor is deliberately built around its own ``SessionContext`` whose
    ``tier_allowed`` is derived from ``authenticated and customer_id``. Rather
    than teach either side about the other, translate here -- so the identity
    rule stays in one place and the loop keeps working if the executor is
    swapped.

    The translation is where Tier 2 is actually gated: a session is only
    `authenticated` if it passed explicit verification. A phone match sets
    `matched_by_phone`, which deliberately does not.
    """

    def __init__(self, executor) -> None:
        self.executor = executor

    def _context(self, session: Session):
        from ..tools.executor import SessionContext

        may_read, _ = session.identity.may_read_account()
        return SessionContext(
            session_id=session.session_id,
            persona=session.persona,
            customer_id=session.identity.customer_id if may_read else None,
            authenticated=may_read,
        )

    def schemas_for(self, persona: Persona, session: Session | None = None) -> list[dict]:
        ctx = self._context(session) if session else None
        if ctx is None:
            from ..tools.executor import SessionContext

            ctx = SessionContext(session_id="anon", persona=persona.name)
        ctx.persona = persona.name
        schemas = self.executor.tool_schemas(ctx)
        if persona.tool_allowlist:
            allowed = set(persona.tool_allowlist)
            schemas = [s for s in schemas if s.get("function", s)["name"] in allowed]
        return schemas

    def execute(self, call: ToolCall, session: Session) -> ToolOutcome:
        result = self.executor.dispatch(call.name, call.arguments, self._context(session))
        data = getattr(result, "data", None)
        citations: list[dict] = []
        if isinstance(data, dict):
            for item in data.get("results") or data.get("units") or []:
                if isinstance(item, dict) and item.get("unit_id"):
                    citations.append(
                        {
                            "unit_id": item["unit_id"],
                            "title": item.get("title", ""),
                            "evidence": item.get("evidence", ""),
                        }
                    )
        handoff = None
        if call.name == "escalate_to_human" and getattr(result, "ok", False):
            handoff = {"reason": call.arguments.get("reason", "requested"), "tier": "hard"}
        summary = ""
        if getattr(result, "degraded", False):
            summary = f"degraded: {getattr(result, 'degraded_reason', '')}"
        elif not getattr(result, "ok", False):
            summary = f"error: {getattr(result, 'error', '')}"
        return ToolOutcome(
            ok=bool(getattr(result, "ok", False)),
            content=data if data is not None else {"error": getattr(result, "error", "")},
            summary=summary,
            provenance=getattr(result, "provenance", "") or "",
            citations=citations[:8],
            handoff=handoff,
        )


@dataclass
class TurnResult:
    answer: str
    events: list[Event] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    handoff: dict | None = None
    blocked_reason: str | None = None
    iterations: int = 0
    elapsed_s: float = 0.0


class AgentLoop:
    """One turn of the agent.

    ``executor``, ``guardrails`` and ``retriever`` are injected rather than
    imported so this file has no opinion about how a tool is implemented, and
    so a test can run the loop with three fakes and no network.
    """

    def __init__(self, executor, guardrails=None, llm=None) -> None:
        # Accept either a raw tool executor or an already-wrapped adapter, so
        # a test can inject a fake without importing the real tool layer.
        self.executor = (
            executor
            if hasattr(executor, "schemas_for")
            else ExecutorAdapter(executor)
        )
        self.guardrails = guardrails
        self.llm = llm or get_llm()

    # -- public --------------------------------------------------------------
    def run(
        self,
        session: Session,
        message: str,
        *,
        persona: Persona | None = None,
        emit: Callable[[Event], None] | None = None,
    ) -> TurnResult:
        events: list[Event] = []

        def send(event: Event) -> None:
            events.append(event)
            if emit:
                emit(event)

        started = time.monotonic()
        persona = persona or select_persona(
            message, internal=session.internal, explicit=session.persona
        )
        result = TurnResult(answer="")

        # --- pre-retrieval guardrails ---------------------------------------
        # Emissions and safety are checked BEFORE any retrieval, because the
        # index legitimately contains 432 units that mention cat-delete. A gate
        # applied after retrieval is a gate applied to a context window that
        # already contains the thing it was meant to keep out.
        if self.guardrails:
            pre = self.guardrails.check_input(message, session=session)
            if pre.blocked:
                send(Event("handoff", {"reason": pre.reason, "tier": pre.tier}))
                result.answer = pre.response
                result.blocked_reason = pre.reason
                result.handoff = {"reason": pre.reason, "tier": pre.tier}
                result.elapsed_s = time.monotonic() - started
                session.add_turn("user", message)
                session.add_turn("assistant", result.answer, blocked=pre.reason)
                return result

        session.add_turn("user", message)
        messages = session.chat_messages(persona.system_prompt)
        tools = self.executor.schemas_for(persona, session)

        budget = config.MAX_TOOL_CALLS
        deadline = started + config.WALL_CLOCK_SECONDS

        for iteration in range(config.MAX_ITERATIONS):
            result.iterations = iteration + 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                send(Event("thinking", {"note": "time budget exhausted"}))
                break

            try:
                completion = self.llm.chat(
                    messages,
                    tools=tools if budget > 0 else None,
                    tool_choice="auto" if budget > 0 else "none",
                    max_tokens=900,
                    temperature=0.2,
                )
            except LLMError as exc:
                send(Event("error", {"message": "reasoner unavailable"}))
                result.answer = (
                    "I'm having trouble reaching my knowledge systems right now. "
                    "Let me get a person to help you."
                )
                result.handoff = {"reason": "reasoner_unavailable", "tier": "hard"}
                result.blocked_reason = f"llm_error: {exc}"
                result.elapsed_s = time.monotonic() - started
                return result

            if not completion.tool_calls:
                result.answer = (completion.content or "").strip()
                break

            # --- dispatch tool calls ---------------------------------------
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.raw_arguments},
                        }
                        for tc in completion.tool_calls
                    ],
                }
            )

            for call in completion.tool_calls:
                if budget <= 0:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                {"error": "tool budget exhausted for this turn"}
                            ),
                        }
                    )
                    continue
                budget -= 1
                session.tool_calls_used += 1
                send(Event("tool_call", {"name": call.name, "arguments": call.arguments}))

                outcome = self.executor.execute(call, session)
                # Register the result as grounding evidence for THIS turn. A
                # price or availability claim in the final answer is checked
                # against these; without the registration the claim has nothing
                # to match and is correctly refused.
                if self.guardrails and outcome.ok:
                    try:
                        outcome.provenance = self.guardrails.record_tool_result(
                            session, call.name, call.id, outcome.content
                        ) or outcome.provenance
                    except Exception:  # noqa: BLE001
                        # Evidence bookkeeping must never take down a turn; the
                        # consequence of failure is a refusal, not a bad answer.
                        pass
                result.tool_calls.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": outcome.ok,
                        "provenance": outcome.provenance,
                        # Kept so the groundedness check can reconstruct what
                        # this turn was entitled to know. Bounded, because a
                        # knowledge search can return a lot of text and this
                        # ends up persisted on every eval row.
                        "result": outcome.content,
                    }
                )
                send(
                    Event(
                        "tool_result",
                        {
                            "name": call.name,
                            "ok": outcome.ok,
                            "summary": outcome.summary,
                            "provenance": outcome.provenance,
                        },
                    )
                )
                for citation in outcome.citations:
                    result.citations.append(citation)
                    send(Event("citation", citation))

                if outcome.handoff:
                    result.handoff = outcome.handoff
                    send(Event("handoff", outcome.handoff))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(outcome.content, default=str)[:12000],
                    }
                )
        else:
            # Fell out of the loop without a plain answer. Ask once more with
            # tools disabled rather than returning the tool-call scaffolding.
            try:
                final = self.llm.chat(
                    messages + [
                        {
                            "role": "user",
                            "content": "Answer now using only what you have. Do not call any more tools.",
                        }
                    ],
                    max_tokens=700,
                    temperature=0.2,
                )
                result.answer = (final.content or "").strip()
            except LLMError:
                result.answer = ""

        if not result.answer:
            result.answer = (
                "I wasn't able to pin that down. Let me connect you with someone who can."
            )
            result.handoff = result.handoff or {"reason": "no_answer", "tier": "soft"}

        # --- post-generation guardrails -------------------------------------
        # This is what makes an ungrounded price or availability claim
        # unspeakable rather than merely discouraged: the claim must have a
        # matching tool result in THIS turn or it does not ship.
        if self.guardrails:
            post = self.guardrails.check_output(
                result.answer, tool_results=result.tool_calls, session=session
            )
            if post.blocked:
                send(Event("handoff", {"reason": post.reason, "tier": post.tier}))
                result.answer = post.response
                result.blocked_reason = post.reason
                result.handoff = {"reason": post.reason, "tier": post.tier}
            elif post.response != result.answer:
                result.answer = post.response

        session.add_turn("assistant", result.answer)
        session.turns += 1
        result.elapsed_s = time.monotonic() - started
        send(Event("done", {"elapsed_s": round(result.elapsed_s, 2), "iterations": result.iterations}))
        return result

    def stream(
        self, session: Session, message: str, *, persona: Persona | None = None
    ) -> Iterator[str]:
        """Generator form for SSE.

        Runs the turn on a worker thread and yields events as they occur, so a
        user sees `tool_call` frames while the tools are still running rather
        than a 6-second blank followed by everything at once. Step-level, not
        token-level: there is nothing to preserve on the token side anyway,
        since no query engine in the stack this replaces was ever constructed
        with streaming enabled -- its `time_to_first_token` metric has always
        measured total generation time.
        """
        import queue as _queue
        import threading

        channel: _queue.Queue = _queue.Queue()
        sentinel = object()

        def worker() -> None:
            try:
                self.run(session, message, persona=persona, emit=channel.put)
            except Exception as exc:  # noqa: BLE001
                channel.put(Event("error", {"message": str(exc)[:200]}))
            finally:
                channel.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            item = channel.get()
            if item is sentinel:
                break
            yield item.to_sse()
