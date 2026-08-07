"""Agent runtime package, plus the single entry point everything else calls.

``eval_turn`` exists so the eval harness measures the *production* path --
persona selection, tools, guardrails and all -- rather than a convenience
wrapper that quietly skips a gate. The harness deliberately refuses to guess an
entry point; wiring it here is what makes ``--sut agent`` mean what it says.
"""

from __future__ import annotations

import threading
from typing import Any

_loop = None
_lock = threading.Lock()


def get_loop():
    """Build the production loop once: real tools, real guardrails."""
    global _loop
    if _loop is None:
        with _lock:
            if _loop is None:
                from ..guardrails import get_guardrails
                from ..tools.executor import get_executor
                from .loop import AgentLoop

                _loop = AgentLoop(get_executor(), get_guardrails())
    return _loop


def eval_turn(
    question: str,
    *,
    session_id: str | None = None,
    language: str = "en",
    persona: str | None = None,
    internal: bool = False,
    **_ignored: Any,
) -> dict:
    """One production turn, flattened to what a grader needs.

    A fresh session per call: carrying state between unrelated eval examples
    would let one example's slots answer another's question, which inflates
    every number in a way that is invisible in the summary.
    """
    from .personas import select_persona
    from .session import get_store

    store = get_store()
    session = store.create(session_id=session_id, internal=internal)
    session.language = language
    chosen = select_persona(question, internal=internal, explicit=persona)

    result = get_loop().run(session, question, persona=chosen)
    store.save(session)

    # `tool_results` carries the actual payloads, not just tool names. The
    # groundedness gate reconstructs "what was this turn allowed to know" from
    # exactly these fields -- reporting names alone makes every correctly
    # sourced price look unsourced, which reads as a failing gate rather than
    # as missing instrumentation.
    tool_results = []
    for call in result.tool_calls:
        payload = call.get("result", call.get("content"))
        tool_results.append(
            {
                "name": call.get("name"),
                "arguments": call.get("arguments"),
                "ok": call.get("ok"),
                "result": payload,
            }
        )

    return {
        "answer": result.answer,
        "persona": chosen.name,
        "tool_calls": [c.get("name") for c in result.tool_calls],
        "tool_results": tool_results,
        "citations": result.citations,
        "handoff": result.handoff,
        "blocked_reason": result.blocked_reason,
        "blocked_by_grounding": result.blocked_reason is not None
        and "ground" in str(result.blocked_reason).lower(),
        "iterations": result.iterations,
        "retrieved": result.citations,
        "session_id": session.session_id,
    }


# Aliases for the other names callers look for.
run_turn = eval_turn
answer = eval_turn

__all__ = ["eval_turn", "run_turn", "answer", "get_loop"]
