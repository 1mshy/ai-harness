"""Mongo access, split by intent so that write capability is visible in types.

The source collections (``calls_analysis``, ``calls_cases``, ``tuning_*``) are
owned by the DGX pipeline and by an internal staff LLM app that shares this
database. Nothing here writes to them. The collections this project owns are
reachable only through :func:`kb_db`, which is a different function so that
"did I just write to the analyzer's collection" is answerable by reading the
call site.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

from .. import config


@lru_cache(maxsize=2)
def _client(read_only: bool) -> MongoClient:
    return MongoClient(
        config.MONGO_URL,
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
        appname="agentv1-ro" if read_only else "agentv1-kb",
        tz_aware=False,
    )


def source_db() -> Database:
    """Read side. Treat every collection reachable from here as append-only."""
    return _client(True)[config.MONGO_DB]


def kb_db() -> Database:
    """Write side, for the four collections this project owns."""
    return _client(False)[config.MONGO_DB]


def ensure_kb_indexes() -> dict[str, list[str]]:
    """Create the indexes Phase 4a and the revocation path depend on.

    Idempotent. ``kb_units`` is queried three ways -- by ``unit_id`` for
    upsert, by ``(gen, status)`` for the projection scan, and by
    ``training_safe`` for the revocation sweep -- and the generation swap does
    a bulk delete by ``gen``, which is unusable without an index at this size.
    """
    db = kb_db()
    created: dict[str, list[str]] = {}

    units = db[config.COLL_KB_UNITS]
    created[config.COLL_KB_UNITS] = [
        units.create_index([("unit_id", ASCENDING)], unique=True, name="unit_id_uniq"),
        units.create_index([("gen", ASCENDING), ("status", ASCENDING)], name="gen_status"),
        units.create_index([("training_safe", ASCENDING)], name="training_safe"),
        units.create_index([("source_ids", ASCENDING)], name="source_ids"),
        units.create_index([("kb_version", ASCENDING), ("merge_version", ASCENDING)], name="versions"),
        units.create_index([("cluster_id", ASCENDING)], name="cluster_id"),
    ]

    state = db[config.COLL_KB_STATE]
    created[config.COLL_KB_STATE] = [
        state.create_index([("key", ASCENDING)], unique=True, name="key_uniq"),
    ]

    rev = db[config.COLL_KB_REVOCATIONS]
    created[config.COLL_KB_REVOCATIONS] = [
        rev.create_index([("unit_id", ASCENDING)], name="unit_id"),
        rev.create_index([("applied", ASCENDING)], name="applied"),
        rev.create_index([("source_id", ASCENDING)], name="source_id"),
    ]

    sess = db[config.COLL_AGENT_SESSIONS]
    created[config.COLL_AGENT_SESSIONS] = [
        sess.create_index([("session_id", ASCENDING)], unique=True, name="session_id_uniq"),
        sess.create_index([("updated_at", ASCENDING)], name="updated_at"),
    ]

    ev = db[config.COLL_AGENT_EVENTS]
    created[config.COLL_AGENT_EVENTS] = [
        ev.create_index([("session_id", ASCENDING), ("ts", ASCENDING)], name="session_ts"),
    ]
    return created


def get_state(key: str) -> dict[str, Any] | None:
    return kb_db()[config.COLL_KB_STATE].find_one({"key": key})


def put_state(key: str, payload: dict[str, Any]) -> None:
    """Stamp build state.

    Written *last* in every build, after the new generation is live and the old
    one is gone. A crash between those steps leaves a stale stamp pointing at a
    generation that still exists, which is recoverable; the reverse is not.
    """
    kb_db()[config.COLL_KB_STATE].update_one(
        {"key": key}, {"$set": {"key": key, **payload}}, upsert=True
    )
