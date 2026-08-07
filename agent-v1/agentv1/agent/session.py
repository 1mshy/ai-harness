"""Server-side session state: identity, durable slots, real conversation memory.

Two things here are direct responses to how the previous stack failed.

**Identity never comes from the client.** ``customer_id`` lives only here and is
injected into Tier-2 tool calls by the executor. A caller cannot name a
customer id in a tool argument because the argument does not exist.

**Memory is not a truncated tail.** The stack this replaces kept
``conversation_memory[session_id][-5:]`` with each message cut to 200
characters and a literal ``'...'`` appended -- so the agent could ask "what
engine code?" and then genuinely lose the answer. That is the concrete
difference between a prompt mode and an agent: durable slots and state.

**A phone match is context, not identity.** ``vehicle_context.matched`` is true
on 61.9% of calls, but among those, ``name_agrees`` holds only 31.2% of the
time, and placeholder numbers are real -- ``1111111111`` maps to 374 accounts.
So caller-ID resolves *context* and never authorises *disclosure*. Tier-2 reads
require an explicit identity proof, and more than one candidate forces Tier 0.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .. import config
from ..clients.mongo import kb_db

MAX_HISTORY_TURNS = 24

# Slots worth remembering across turns. The largest measured theme among
# avoidable repeat contacts is "the agent did not collect required data
# upfront", so the checklist is part of the state rather than a prompt
# suggestion the model may drop.
SLOT_FIELDS = (
    "vin", "ecu_box_code", "tcu_id", "tcu_revision", "cable_serial", "fuel_octane",
    "vehicle_make", "vehicle_model", "vehicle_year", "vehicle_engine",
    "vehicle_chassis", "platform_id", "current_stage", "desired_stage",
    "error_string", "transmission",
)

REQUIRED_SLOTS_BY_INTENT = {
    "flashing_error": ("vin", "ecu_box_code", "cable_serial", "error_string"),
    "compatibility": ("vehicle_make", "vehicle_model", "vehicle_year"),
    "tcu": ("vin", "tcu_id", "tcu_revision"),
    "pricing_quote": ("vehicle_make", "vehicle_model", "vehicle_year", "desired_stage"),
    "performance_issue": ("vin", "fuel_octane", "current_stage"),
}


@dataclass
class Identity:
    """Who we believe the caller is, and how sure we are.

    ``verified`` is the only field that unlocks Tier 2. ``matched_by_phone``
    deliberately does not.
    """

    customer_id: str | None = None
    verified: bool = False
    verification_method: str | None = None
    matched_by_phone: bool = False
    candidate_count: int = 0
    name_agrees: bool | None = None

    def may_read_account(self) -> tuple[bool, str]:
        if not self.customer_id:
            return False, "no_customer_resolved"
        if self.candidate_count > 1:
            # More than one account behind the same key. Serving either is a
            # coin flip with a privacy incident on one side.
            return False, "ambiguous_identity"
        if not self.verified:
            return False, "identity_not_verified"
        return True, ""


@dataclass
class Session:
    session_id: str
    persona: str = "support"
    internal: bool = False
    actor: str | None = None  # staff user id, for the copilot surfaces
    language: str = "en"
    identity: Identity = field(default_factory=Identity)
    slots: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    tool_calls_used: int = 0
    turns: int = 0
    created_at: str = ""
    updated_at: str = ""

    # -- memory --------------------------------------------------------------
    def add_turn(self, role: str, content: str, **extra) -> None:
        """Append a full-fidelity turn. No truncation -- see module docstring."""
        self.history.append(
            {"role": role, "content": content, "ts": _now(), **extra}
        )
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[: len(self.history) - MAX_HISTORY_TURNS]

    def chat_messages(self, system_prompt: str) -> list[dict]:
        msgs = [{"role": "system", "content": system_prompt}]
        if self.slots:
            known = ", ".join(f"{k}={v}" for k, v in self.slots.items() if v)
            if known:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"Already established this conversation: {known}. Do not ask for these again.",
                    }
                )
        for turn in self.history:
            if turn["role"] in ("user", "assistant"):
                msgs.append({"role": turn["role"], "content": turn["content"]})
        return msgs

    # -- slots ---------------------------------------------------------------
    def set_slot(self, name: str, value: Any) -> None:
        if name in SLOT_FIELDS and value not in (None, ""):
            self.slots[name] = value

    def missing_slots(self, intent: str) -> list[str]:
        return [s for s in REQUIRED_SLOTS_BY_INTENT.get(intent, ()) if not self.slots.get(s)]

    def as_document(self) -> dict:
        doc = asdict(self)
        doc["updated_at"] = _now()
        return doc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    """In-process cache with Mongo write-through.

    Mongo-backed rather than a module global because the approval state machine
    and the tool budgets have to survive multiple uvicorn workers. A global
    flag does not, which is exactly why the developer-console version of this
    gated nothing in practice.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(self, **kwargs) -> Session:
        sid = kwargs.pop("session_id", None) or uuid.uuid4().hex
        session = Session(session_id=sid, created_at=_now(), updated_at=_now(), **kwargs)
        with self._lock:
            self._cache[sid] = session
        self._persist(session)
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]
        doc = kb_db()[config.COLL_AGENT_SESSIONS].find_one({"session_id": session_id})
        if not doc:
            return None
        doc.pop("_id", None)
        ident = Identity(**(doc.pop("identity", {}) or {}))
        session = Session(identity=ident, **doc)
        with self._lock:
            self._cache[session_id] = session
        return session

    def get_or_create(self, session_id: str | None, **kwargs) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing
            kwargs["session_id"] = session_id
        return self.create(**kwargs)

    def save(self, session: Session) -> None:
        with self._lock:
            self._cache[session.session_id] = session
        self._persist(session)

    def _persist(self, session: Session) -> None:
        kb_db()[config.COLL_AGENT_SESSIONS].update_one(
            {"session_id": session.session_id},
            {"$set": session.as_document()},
            upsert=True,
        )


_store: SessionStore | None = None
_store_lock = threading.Lock()


def get_store() -> SessionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore()
    return _store
