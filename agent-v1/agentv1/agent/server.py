"""FastAPI agent runtime on :5124.

ASGI rather than the existing Flask app, for one specific reason: an agent turn
is multi-second, multi-tool and bound on *waiting*, and the Flask path creates a
fresh event loop per request, which breaks any pooled async client the tool
layer needs. Flask stays up on :5123 untouched and serves as the degradation
target -- if this process is down, the old RAG path still answers.

The SSE transport is reused as-is. The existing frontend switch has no throwing
default, so ``tool_call`` / ``tool_result`` / ``thinking`` / ``citation`` /
``handoff`` are additive and frontend-only.

    uvicorn agentv1.agent.server:app --host 127.0.0.1 --port 5124
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import config
from ..clients import qdrant
from .loop import AgentLoop
from .personas import PERSONAS, select_persona
from .session import get_store

log = logging.getLogger("agentv1.server")

_state: dict[str, Any] = {}


def _build_runtime():
    """Wire the loop to whatever is actually importable.

    The tool and guardrail layers are separate modules; if one is absent the
    service still starts in a reduced mode and says so at /health, rather than
    failing to boot and taking the whole surface down.
    """
    executor = None
    guardrails = None
    errors: list[str] = []
    try:
        from ..tools.executor import get_executor

        executor = get_executor()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tools: {exc}")
    try:
        from ..guardrails import get_guardrails

        guardrails = get_guardrails()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"guardrails: {exc}")
    return executor, guardrails, errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    executor, guardrails, errors = _build_runtime()
    _state["executor"] = executor
    _state["guardrails"] = guardrails
    _state["errors"] = errors
    _state["loop"] = AgentLoop(executor, guardrails) if executor else None

    # Refuse to claim readiness on an unusable index. The assertion is meant to
    # fire: today `unitronic_faq_0_6b` holds 0 points and is routed to.
    routed = [config.ALIAS_KB_UNITS]
    try:
        qdrant.assert_routed_collections_populated(routed)
        _state["index_ready"] = True
        _state["index_error"] = None
    except Exception as exc:  # noqa: BLE001
        _state["index_ready"] = False
        _state["index_error"] = str(exc)
        log.warning("index not ready: %s", exc)
    yield


app = FastAPI(title="agentv1", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5123"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    persona: str | None = None
    internal: bool = False
    actor: str | None = None
    # Deliberately absent: customer_id. Identity is server-side state only.
    # A client that could name a customer id could name somebody else's.


def _session_for(req: ChatRequest):
    store = get_store()
    session = store.get_or_create(
        req.session_id, internal=req.internal, actor=req.actor
    )
    if req.persona and req.persona in PERSONAS:
        session.persona = req.persona
    return session


def _require_loop():
    if not _state.get("loop"):
        raise HTTPException(
            status_code=503,
            detail={"error": "agent runtime unavailable", "causes": _state.get("errors", [])},
        )
    return _state["loop"]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state.get("loop") and _state.get("index_ready") else "degraded",
        "index_ready": _state.get("index_ready", False),
        "index_error": _state.get("index_error"),
        "runtime_errors": _state.get("errors", []),
        "personas": sorted(PERSONAS),
        "settings": config.SETTINGS.as_dict(),
    }


@app.post("/api/agent/chat")
def chat(req: ChatRequest) -> JSONResponse:
    loop = _require_loop()
    session = _session_for(req)
    persona = select_persona(
        req.message, internal=session.internal, explicit=req.persona or session.persona
    )
    result = loop.run(session, req.message, persona=persona)
    get_store().save(session)
    return JSONResponse(
        {
            "session_id": session.session_id,
            "persona": persona.name,
            "answer": result.answer,
            "citations": result.citations,
            "tool_calls": result.tool_calls,
            "handoff": result.handoff,
            "blocked_reason": result.blocked_reason,
            "elapsed_s": round(result.elapsed_s, 2),
            "iterations": result.iterations,
        }
    )


@app.post("/api/agent/stream")
async def stream(req: ChatRequest) -> StreamingResponse:
    loop = _require_loop()
    session = _session_for(req)
    persona = select_persona(
        req.message, internal=session.internal, explicit=req.persona or session.persona
    )

    def generate():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session.session_id, 'persona': persona.name})}\n\n"
        try:
            for frame in loop.stream(session, req.message, persona=persona):
                yield frame
        finally:
            get_store().save(session)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent/verify_identity")
def verify_identity(payload: dict, request: Request) -> JSONResponse:
    """Promote a session to Tier 2.

    A phone match is evidence, never identity: `vehicle_context.matched` is
    true on 61.9% of calls but `name_agrees` holds only 31.2% of those, and
    placeholder keys are real (`1111111111` maps to 374 accounts). So this
    endpoint requires an explicit proof, and more than one candidate keeps the
    session at Tier 0 regardless of what was supplied.
    """
    session = get_store().get(payload.get("session_id", ""))
    if not session:
        raise HTTPException(404, "unknown session")
    executor = _state.get("executor")
    if not executor or not hasattr(executor, "verify_identity"):
        raise HTTPException(503, "identity verification unavailable")
    ok, detail = executor.verify_identity(session, payload)
    get_store().save(session)
    return JSONResponse({"verified": ok, "detail": detail, "tier": 2 if ok else 0})


@app.get("/api/agent/session/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    session = get_store().get(session_id)
    if not session:
        raise HTTPException(404, "unknown session")
    return JSONResponse(
        {
            "session_id": session.session_id,
            "persona": session.persona,
            "turns": session.turns,
            "slots": session.slots,
            "tool_calls_used": session.tool_calls_used,
            # identity is intentionally summarised, never echoed in full
            "identity_verified": session.identity.verified,
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "agentv1.agent.server:app",
        host=config.AGENT_HOST,
        port=config.AGENT_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
