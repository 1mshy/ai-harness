"""Pinned configuration. Fails fast rather than silently degrading.

Two invariants from AGENT_PLAN.md §3.3 are enforced here rather than left to
convention:

* ``QWEN_SIZE`` is explicit. The production stack it replaces left it unset, so
  prod silently ran the dev-sized default. There is no default here.
* Collections are never created on a read path. ``qdrant.py`` takes
  ``create=False`` and the startup assertion refuses to boot when a routed
  collection is empty.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(REPO_ROOT / ".env")


def _req(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is required and unset. It is deliberately not defaulted -- "
            f"an unset embedding size or connection string degrades silently."
        )
    return val


# --- Embedding generation ----------------------------------------------------
# The suffix is part of every collection name. Flipping it is a full rebuild,
# never an in-place change: 0.6b is 1024-dim and 8b is 4096-dim, and Qdrant
# will happily accept writes to a collection whose dimension you have changed
# your mind about only by refusing them at query time.
QWEN_SIZE = os.environ.get("QWEN_SIZE", "0_6b")
_DIMS = {"0_6b": 1024, "8b": 4096}
if QWEN_SIZE not in _DIMS:
    raise RuntimeError(f"QWEN_SIZE must be one of {sorted(_DIMS)}, got {QWEN_SIZE!r}")
EMBED_DIM = _DIMS[QWEN_SIZE]

EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# "local" runs sentence-transformers on MPS/CPU; "remote" uses an
# OpenAI-compatible /v1/embeddings endpoint. Measured 2026-08-05: local MPS
# 256 texts/s, DGX vLLM 121 texts/s (contended by the 31B chat model).
# Cross-backend cosine agreement is 1.0000, so they are interchangeable.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "local")
EMBED_REMOTE_URL = os.environ.get("EMBED_REMOTE_URL", "http://10.150.0.30:1235/v1")
EMBED_REMOTE_MODEL = os.environ.get("EMBED_REMOTE_MODEL", "qwen3-embed")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "auto")
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))

# Qwen3 embedding models are instruct-tuned and asymmetric: the query side
# takes a task prefix and the document side does not. Getting this backwards
# costs real recall and is invisible in every smoke test.
QUERY_INSTRUCTION = (
    "Instruct: Given a customer support or sales question about automotive "
    "performance tuning, retrieve the knowledge unit that answers it\nQuery:"
)

# --- Reranker ----------------------------------------------------------------
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") not in ("0", "false", "False")
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "auto")

# --- Retrieval budget --------------------------------------------------------
# The stack this replaces ran similarity_top_k=2 with no reranker, which is
# ~200 words of evidence for an entire answer. Retrieve wide, rerank, cut.
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "30"))
FINAL_TOP_K = int(os.environ.get("FINAL_TOP_K", "8"))
RRF_K = int(os.environ.get("RRF_K", "60"))

# --- Mongo -------------------------------------------------------------------
MONGO_URL = _req("MONGO_URL")
MONGO_DB = os.environ.get("MONGO_DB", "transcribing")

COLL_ANALYSIS = "calls_analysis"
COLL_CASES = "calls_cases"
COLL_PLATFORMS = "tuning_platforms"
COLL_CUSTOMERS = "tuning_customers"
COLL_CRM = "crm_contacts"
COLL_SYNC_STATE = "tuning_sync_state"

# Owned by this project -- Phase 4a writes them, nothing else does.
COLL_KB_UNITS = "kb_units"
COLL_KB_STATE = "kb_units_state"
COLL_KB_REVOCATIONS = "kb_revocations"
COLL_AGENT_SESSIONS = "agent_sessions"
COLL_AGENT_EVENTS = "agent_events"
COLL_KNOWLEDGE_GAPS = "agent_knowledge_gaps"
COLL_LEADS = "agent_leads"
COLL_ESCALATIONS = "agent_escalations"

# --- Qdrant ------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "60"))

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def _sfx(base: str) -> str:
    return f"{base}_{QWEN_SIZE}"


# New collections, hybrid dense+sparse from birth. Built behind an alias so a
# bad build is a seconds-long alias flip rather than a re-embed.
ALIAS_KB_UNITS = "unitronic_kb_units"
ALIAS_CASE_NARRATIVES = "unitronic_case_narratives"
ALIAS_CALL_RESIDUAL = "unitronic_call_residual"
ALIAS_PLATFORM_STAGES = "unitronic_platform_stages"

NEW_ALIASES = [
    ALIAS_KB_UNITS,
    ALIAS_CASE_NARRATIVES,
    ALIAS_CALL_RESIDUAL,
    ALIAS_PLATFORM_STAGES,
]

# Pre-existing collections kept as-is. These are legacy single-vector
# collections; the retrieval layer must not assume named vectors on them.
LEGACY_COLLECTIONS = [
    _sfx("unitronic_comprehensive"),
    _sfx("unitronic_products"),
    _sfx("unitronic_products_tuning"),
    _sfx("unitronic_tuning"),
    _sfx("unitronic_company_info"),
]

# PII: carries `file_name` embedding the caller's phone number plus agent_name
# and caller_area_code, duplicated inside _node_content. Superseded by
# kb_units. Never routed to; ops/p0_remediate.py snapshots and drops it.
QUARANTINED_COLLECTIONS = [_sfx("unitronic_call_transcriptions")]


def collection_for(alias: str, generation: str) -> str:
    """Physical collection name behind an alias for a given build generation."""
    return f"{alias}_{QWEN_SIZE}__{generation}"


# --- Reasoner LLM ------------------------------------------------------------
LLM_URL = os.environ.get("OPENAI_URL", "http://10.150.0.30:1234/v1")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
LLM_MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "8"))

# Verified 2026-08-05 against the live endpoint: it is started with
# --enable-auto-tool-choice --tool-call-parser gemma4 and returns well-formed
# tool_calls with finish_reason="tool_calls". AGENT_PLAN.md §10 assumed we
# would have to hand-roll free-form JSON tool calls; we do not. The separate
# `json_schema` response-format invariant (guided decoding returns empty
# content) is real and is why structured extraction still uses free-form JSON
# plus code-side validation -- see clients/llm.py.
LLM_NATIVE_TOOL_CALLING = os.environ.get("LLM_NATIVE_TOOLS", "1") not in ("0", "false")

# The box does not hold a model still. Measured 2026-08-07: the endpoint moved
# from Gemma-4-31B to Nemotron-3-Super-120B inside ten minutes, and a pinned
# OPENAI_MODEL turns every reasoner call into a 404 -- which surfaces as a
# healthy /health next to an agent that answers "I'm having trouble reaching my
# knowledge systems" on every turn. So the model is *discovered* by default and
# pinned only when someone pins it on purpose.
#
# Resolution order: OPENAI_MODEL -> first id from GET {LLM_URL}/models ->
# LLM_MODEL_FALLBACK. Discovery is lazy (nothing imports config to make a
# network call) and cached, so the cost is one request per process.
LLM_MODEL_FALLBACK = os.environ.get("LLM_MODEL_FALLBACK", "nvidia/Gemma-4-31B-IT-NVFP4")
LLM_MODEL_DISCOVERY_TIMEOUT = float(os.environ.get("LLM_MODEL_DISCOVERY_TIMEOUT", "5"))

_model_lock = threading.Lock()
_model_cache: str | None = None


def discover_llm_model(base_url: str | None = None, timeout: float | None = None) -> str | None:
    """First model id the endpoint serves, or None if it cannot be asked.

    Deliberately returns None rather than raising: an unreachable endpoint at
    import time must not stop the process from booting. The caller falls back
    to the pinned literal and the reasoner fails later, loudly, at the point
    where it actually matters.
    """
    url = (base_url or LLM_URL).rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {LLM_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or LLM_MODEL_DISCOVERY_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - urllib, socket, JSON and key errors alike
        return None
    for entry in payload.get("data") or []:
        model_id = (entry or {}).get("id")
        if model_id:
            return str(model_id)
    return None


def resolve_llm_model(refresh: bool = False) -> str:
    """The model id to send. Cached; pass refresh=True after a 404."""
    global _model_cache
    pinned = os.environ.get("OPENAI_MODEL")
    if pinned:
        return pinned
    with _model_lock:
        if _model_cache is None or refresh:
            _model_cache = discover_llm_model() or LLM_MODEL_FALLBACK
        return _model_cache


_settings_cache: "Settings | None" = None


def __getattr__(name: str):
    """Keep ``config.LLM_MODEL`` and ``config.SETTINGS`` working lazily.

    Both would otherwise resolve the model at import, which would put a network
    call in the path of every ``import agentv1.config`` -- including the test
    suite, which has no business talking to the reasoner.
    """
    global _settings_cache
    if name == "LLM_MODEL":
        return resolve_llm_model()
    if name == "SETTINGS":
        if _settings_cache is None:
            _settings_cache = Settings()
        return _settings_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Agent runtime -----------------------------------------------------------
AGENT_HOST = os.environ.get("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "5124"))
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "4"))
MAX_TOOL_CALLS = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "6"))
WALL_CLOCK_SECONDS = float(os.environ.get("AGENT_WALL_CLOCK", "45"))

# --- Freshness gates ---------------------------------------------------------
# sync_tuning_db.py is unscheduled and Mac-pinned. If the platform table is
# stale the compatibility oracle must say so rather than answer confidently
# from a snapshot. Set generously: the measured staleness on 2026-08-05 was
# 5.8 days, so a 48h gate would have parked the oracle in permanent refusal.
PLATFORM_STALE_WARN_HOURS = float(os.environ.get("PLATFORM_STALE_WARN_HOURS", "48"))
PLATFORM_STALE_REFUSE_HOURS = float(os.environ.get("PLATFORM_STALE_REFUSE_HOURS", "336"))

# --- Build versions ----------------------------------------------------------
# Bumping either forces the corresponding stage to re-run under
# `build_kb --redo-outdated`.
KB_VERSION = 1
MERGE_VERSION = 1
INDEX_VERSION = 1


@dataclass(frozen=True)
class Settings:
    """Snapshot of the pinned values, for logging into a build stamp."""

    qwen_size: str = QWEN_SIZE
    embed_dim: int = EMBED_DIM
    embed_model: str = EMBED_MODEL
    rerank_model: str = RERANK_MODEL
    kb_version: int = KB_VERSION
    merge_version: int = MERGE_VERSION
    index_version: int = INDEX_VERSION
    llm_model: str = field(default_factory=resolve_llm_model)
    aliases: tuple = field(default_factory=lambda: tuple(NEW_ALIASES))

    def as_dict(self) -> dict:
        return {
            "qwen_size": self.qwen_size,
            "embed_dim": self.embed_dim,
            "embed_model": self.embed_model,
            "rerank_model": self.rerank_model,
            "kb_version": self.kb_version,
            "merge_version": self.merge_version,
            "index_version": self.index_version,
            "llm_model": self.llm_model,
        }

