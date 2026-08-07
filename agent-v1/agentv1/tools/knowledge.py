"""Knowledge tools: semantic search, case recall, and the error-string index.

Three tools, three different retrieval philosophies, chosen by what the data
actually is.

``search_knowledge`` is the semantic path and the only one that touches the
vector store. It delegates to ``agentv1.retrieval.pipeline.search`` when that
module is available so the ``training_safe`` floor, supersession filtering and
reranking are enforced in exactly one place. It is written to bind late --
``importlib`` at call time, not ``from ... import`` at module load -- because
the retrieval layer is built by a different phase and this module must stay
importable before it lands.

``get_case`` is a primary-key read of ``calls_cases``. It exists for the
``attempts`` array: 2,516 failed attempts across 6,679 cases with attempts are
the corpus's **only** negative evidence. Every other surface -- knowledge
units, solution steps, case narratives -- is success-biased (solution_steps are
66% populated on resolved calls and 19% on unresolved), so nothing else in the
system can answer "what did people already try that did not work". Used two
ways per AGENT_PLAN.md §8.2: as a pre-check before suggesting a fix, and as an
anti-answer.

``lookup_error_string`` is an exact-match index and deliberately not a search.
``flashing_error`` is a closed vocabulary -- a couple of dozen verbatim strings
cover most of 5,663 calls -- so the right data structure is a dict, not a
1024-dimension nearest-neighbour query. It is the cheapest high-volume win
available and it costs one Mongo scan at process start.

**PII**: none of these return ``members``, ``file_name``, ``phone_key`` or a
customer name. Case member ids are 3CX WAV filenames with the caller's phone
number embedded in them (see ``text.normalize.wav_filename_phone``), so the
projection here is an allowlist, not a blocklist.
"""

from __future__ import annotations

import importlib
import inspect
import re
import threading
import time
from collections import Counter, defaultdict
from typing import Any

from .. import config
from ..clients.mongo import source_db
from .base import Degraded, Tool, ToolDependencyError, ToolInputError, obj_schema

# --- search_knowledge --------------------------------------------------------

_KINDS = [
    "product_info",
    "procedure",
    "troubleshooting",
    "compatibility",
    "policy",
    "pricing",
    "faq",
]


# Payload keys worth showing the model, in preference order. Two shapes have
# to be handled: `kb_units` (title/question/answer/conditions, per CONTRACT.md)
# and the legacy scraped collections (context/original_text/product_name). The
# pipeline routes across both, so the projection cannot assume either.
_UNIT_TEXT_KEYS = ("answer", "context", "original_text", "text", "content")
_UNIT_META_KEYS = (
    "kind",
    "title",
    "question",
    "conditions",
    "confidence",
    "evidence",
    "case_id",
    "language",
    "department",
    "occurrences",
    "product_name",
    "product_url",
    "url",
    "sku",
)
_TEXT_CHARS = 1200


def _pipeline_search(**kwargs):
    """Call the retrieval layer if it exists, adapting to its signature.

    Late-bound and signature-filtered on purpose. This module was written
    before ``retrieval/pipeline.py`` landed, and a hard import would have made
    the whole tool layer unimportable in the interim. The kwarg filtering is
    what let ``limit=`` become ``top_k=`` on the pipeline side without breaking
    this call site, and it is why the filter set can grow there without an edit
    here.
    """
    try:
        mod = importlib.import_module("agentv1.retrieval.pipeline")
    except ImportError:
        return None
    fn = getattr(mod, "search", None)
    if not callable(fn):
        return None
    try:
        params = inspect.signature(fn).parameters
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    return fn(**kwargs)


def _unit_view(unit: Any) -> dict:
    """One retrieved unit, flattened to something a model can read.

    Duck-typed rather than isinstance-checked against ``RetrievedUnit``: the
    tool layer must not import the retrieval layer's private types just to
    project them, or every change to that dataclass becomes a change here.
    """
    payload = dict(getattr(unit, "payload", None) or {})
    text = getattr(unit, "text", None)
    if not isinstance(text, str) or not text.strip():
        text = next(
            (payload[k] for k in _UNIT_TEXT_KEYS if isinstance(payload.get(k), str) and payload[k].strip()),
            "",
        )
    out: dict[str, Any] = {
        "unit_id": getattr(unit, "unit_id", None) or payload.get("unit_id"),
        "score": round(float(getattr(unit, "final_score", 0.0) or 0.0), 4),
        "text": text.strip()[:_TEXT_CHARS],
        "collections": list(getattr(unit, "collections", None) or []),
    }
    for key in _UNIT_META_KEYS:
        val = payload.get(key)
        if val not in (None, "", []):
            out[key] = val
    return out


def _normalize_pipeline_result(raw: Any, limit: int) -> tuple[list[dict], dict]:
    """Accept either a ``SearchResult`` or a plain list of dicts."""
    if isinstance(raw, list):
        return raw[:limit], {}
    units = list(getattr(raw, "units", None) or [])
    meta = {
        "collections_searched": list(getattr(raw, "collections_searched", None) or []),
        "collections_missing": list(getattr(raw, "collections_missing", None) or []),
        "collections_failed": list(getattr(raw, "collections_failed", None) or []),
        "dropped_superseded": getattr(raw, "dropped_superseded", None),
        "timings_ms": getattr(raw, "timings_ms", None),
    }
    return [_unit_view(u) for u in units[:limit]], meta


def _fallback_search(
    query: str,
    *,
    kind: str | None,
    department: str | None,
    language: str | None,
    limit: int,
) -> list[dict]:
    """Direct Qdrant hybrid search against the ``kb_units`` alias.

    DEGRADED PATH, reached only when ``retrieval.pipeline`` cannot be imported
    or has no ``search``. It was written for the window before that module
    landed; it has since landed, so this is now a fuse rather than the normal
    route. It is knowingly weaker than the real thing: no reranker, no
    supersession filter, single collection, and the sparse half is empty
    because the fitted BM25 model is a build artefact of the index phase.

    The one thing it does not skip is the ``training_safe`` floor. That is
    AND-ed into the Qdrant filter here exactly as the pipeline does it, because
    a safety floor that holds only on the happy path is not a floor -- and the
    unhappy path is precisely when this function runs.
    """
    from qdrant_client import models as qm

    from ..clients.embeddings import get_embedder
    from ..clients.qdrant import CollectionMissing, hybrid_search

    must: list[Any] = [
        qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=True)),
        # internal_process units are staff-facing process notes. They are never
        # publishable, which is a property of the unit and not of the caller.
        qm.FieldCondition(key="internal_only", match=qm.MatchValue(value=False)),
    ]
    if kind:
        must.append(qm.FieldCondition(key="kind", match=qm.MatchValue(value=kind)))
    if department:
        must.append(
            qm.FieldCondition(key="department", match=qm.MatchValue(value=department))
        )
    if language:
        must.append(
            qm.FieldCondition(key="language", match=qm.MatchValue(value=language))
        )

    vec = get_embedder().embed_query(query)
    try:
        hits = hybrid_search(
            config.ALIAS_KB_UNITS,
            dense=[float(x) for x in vec],
            sparse_indices=[],
            sparse_values=[],
            limit=limit * 2,  # two points per unit; dedupe on unit_id below
            query_filter=qm.Filter(must=must),
        )
    except CollectionMissing as exc:
        raise ToolDependencyError(
            f"knowledge index not built yet: {exc}"
        ) from exc

    seen: set[str] = set()
    out: list[dict] = []
    for hit in hits:
        uid = hit.payload.get("unit_id")
        if uid in seen:
            continue
        seen.add(uid)
        out.append(
            {
                "unit_id": uid,
                "score": round(hit.score, 4),
                "kind": hit.payload.get("kind"),
                "title": hit.payload.get("title"),
                "question": hit.payload.get("question"),
                "answer": hit.payload.get("answer"),
                "conditions": hit.payload.get("conditions"),
                "confidence": hit.payload.get("confidence"),
                "evidence": hit.payload.get("evidence"),
                "case_id": hit.payload.get("case_id"),
                "language": hit.payload.get("language"),
            }
        )
        if len(out) >= limit:
            break
    return out

# search_knowledge answers *knowledge* questions, so it searches the knowledge
# collections only. The scraped product catalogue is reachable through
# search_products, which is a different question with a different answer shape.
#
# This is not tidiness. Measured on the live cluster, fanning out across all
# nine routable collections and fusing into a single 30-candidate rerank window
# let the catalogue crowd the knowledge units out entirely: "how do I transfer
# a license to a new owner" came back as four product pages scoring about -9,
# while the correct unit ("Software License Transfer for New Vehicle Owners",
# rerank +6.0) never reached the cross-encoder at all. RRF fuses by rank, and a
# catalogue carrying three near-duplicate copies of every product supplies more
# high-rank candidates than the corpus does.
KNOWLEDGE_COLLECTIONS = (
    config.ALIAS_KB_UNITS,
    config.ALIAS_CASE_NARRATIVES,
    config.ALIAS_CALL_RESIDUAL,
)




def search_knowledge(
    query: str,
    kind: str | None = None,
    department: str | None = None,
    language: str | None = None,
    *,
    persona: str | None = None,
) -> dict | Degraded:
    """``persona`` is injected by the executor, not supplied by the model.

    It selects the retrieval route, which is a server-side policy decision. If
    it were an argument the model could ask for the sales route on a support
    call, which is a quiet way to change which collections get searched.
    """
    if not (query or "").strip():
        raise ToolInputError("search_knowledge needs a non-empty query")
    limit = config.FINAL_TOP_K

    raw = _pipeline_search(
        query=query,
        persona=persona,
        kind=kind,
        department=department,
        language=language,
        limit=limit,
        top_k=limit,
        collections=KNOWLEDGE_COLLECTIONS,
    )
    if raw is not None:
        results, meta = _normalize_pipeline_result(raw, limit)
        data = {
            "query": query,
            "filters": {"kind": kind, "department": department, "language": language},
            "source": "retrieval.pipeline",
            "result_count": len(results),
            "results": results,
            "retrieval": meta,
        }
        # A routed collection that is missing is not a warning to be logged and
        # forgotten -- it means part of the corpus was silently not searched,
        # and the answer must be labelled accordingly rather than presented as
        # a complete search.
        missing = meta.get("collections_missing") or []
        failed = meta.get("collections_failed") or []
        if missing or failed:
            return Degraded(
                data,
                f"retrieval ran without {sorted(set(missing) | set(failed))}; "
                f"coverage is incomplete, so absence of a result is not evidence "
                f"that no answer exists",
            )
        return data

    hits = _fallback_search(
        query, kind=kind, department=department, language=language, limit=limit
    )
    return Degraded(
        {
            "query": query,
            "filters": {"kind": kind, "department": department, "language": language},
            "source": "qdrant_direct_fallback",
            "result_count": len(hits),
            "results": hits,
        },
        "retrieval.pipeline is not available; ran a dense-only Qdrant query with the "
        "training_safe floor but without reranking or supersession filtering",
    )


# --- get_case ----------------------------------------------------------------

_ATTEMPT_RESULTS = ("worked", "failed", "partial", "untested", "unknown")


def _case_view(case: dict) -> dict:
    """Allowlist projection. ``members``/``phone_key``/``customer_match`` are
    excluded by construction, not filtered out afterwards."""
    attempts = [a for a in case.get("attempts") or [] if isinstance(a, dict)]
    by_result: dict[str, list[dict]] = defaultdict(list)
    for a in attempts:
        view = {
            "attempt": a.get("attempt"),
            "made_by": a.get("made_by"),
            "result": a.get("result"),
        }
        by_result[str(a.get("result") or "unknown").lower()].append(view)

    metrics = case.get("case_metrics") or {}
    vehicles = []
    for v in (case.get("vehicle_context") or {}).get("vehicles") or []:
        vehicles.append(
            {
                "display_name": v.get("display_name"),
                "year": v.get("year"),
                "platform_id": v.get("platform_id"),
                "platform_name": v.get("platform_name"),
                "owned_stage": v.get("owned_stage"),
                "max_released_stage": v.get("max_released_stage"),
            }
        )

    return {
        "case_id": case.get("case_id"),
        "case_label": case.get("case_label"),
        "issue": case.get("issue"),
        "issue_category": case.get("issue_category"),
        "root_cause": case.get("root_cause"),
        "root_cause_detail": case.get("root_cause_detail"),
        "chronology": [
            {
                "contact": c.get("contact"),
                "who": c.get("who"),
                "what_happened": c.get("what_happened"),
            }
            for c in case.get("chronology") or []
        ],
        # The whole reason this tool exists. Failed attempts are surfaced
        # first and separately so the model cannot skim past them.
        "failed_attempts": by_result.get("failed", []),
        "partial_attempts": by_result.get("partial", []),
        "worked_attempts": by_result.get("worked", []),
        "untested_attempts": by_result.get("untested", []) + by_result.get("unknown", []),
        "attempt_counts": {k: len(by_result.get(k, [])) for k in _ATTEMPT_RESULTS},
        "what_finally_worked": case.get("what_finally_worked"),
        "final_resolution": case.get("final_resolution"),
        "case_resolution_status": case.get("case_resolution_status"),
        "repeat_contact_avoidable": case.get("repeat_contact_avoidable"),
        "repeat_contact_reason": case.get("repeat_contact_reason"),
        "open_questions": case.get("open_questions") or [],
        "customer_effort": case.get("customer_effort"),
        "language": (metrics.get("languages") or [None])[0],
        "contacts": metrics.get("contacts"),
        "span_days": metrics.get("span_days"),
        "link_confidence": case.get("link_confidence"),
        "vehicles": vehicles,
        # `internal_process` units are staff-facing process notes and are never
        # publishable -- that is a property of the unit, not of the caller, and
        # `_fallback_search` filters them out on the vector path for the same
        # reason. Zero case units carry that kind today; the filter is here so
        # the day the analyzer emits one it does not arrive in a customer
        # answer through the one read path that had no gate.
        "knowledge_units": [
            {
                "kind": u.get("kind"),
                "title": u.get("title"),
                "question": u.get("question"),
                "answer": u.get("answer"),
                "conditions": u.get("conditions"),
                "confidence": u.get("confidence"),
            }
            for u in case.get("case_knowledge_units") or []
            if u.get("kind") != "internal_process" and not u.get("internal_only")
        ],
    }


def get_case(case_id: str) -> dict:
    if not (case_id or "").strip():
        raise ToolInputError("get_case needs a case_id")
    case = source_db()[config.COLL_CASES].find_one(
        {"case_id": case_id.strip(), "training_safe": True}
    )
    if not case:
        # Distinguish "no such case" from "case exists but is not publishable".
        # Silently returning empty for a withheld case teaches the model the
        # id was wrong and it will go looking for another one.
        exists = source_db()[config.COLL_CASES].count_documents(
            {"case_id": case_id.strip()}, limit=1
        )
        if exists:
            raise ToolInputError(
                f"case {case_id} exists but is not training_safe and cannot be used."
            )
        raise ToolInputError(f"no case with id {case_id}")
    return _case_view(case)


# --- lookup_error_string -----------------------------------------------------

# Canonicalisation for the closed vocabulary. Every rule here was derived from
# the measured surface forms: "cable is not responding" (149), "cable not
# responding" (85), "cable s not responding" (22) are one error and three
# spellings, and an index that keeps them apart answers 149 of 256 calls.
_STEM = {
    "flashing": "flash",
    "failed": "fail",
    "failure": "fail",
    "fails": "fail",
    "programming": "program",
    "initialize": "init",
    "initialization": "init",
    "initialise": "init",
    "initialisation": "init",
    "tuned": "tune",
    "files": "file",
    "errors": "error",
    "cables": "cable",
    "responding": "respond",
    "responds": "respond",
    "activated": "activate",
    "connecting": "connect",
    "connection": "connect",
    "invalid": "invalid",
    "supported": "support",
    "resolved": "resolve",
    "collect": "collect",
}
# Dropped everywhere: articles, copulas, and the possessive-'s the transcriber
# emits as a bare "s".
_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "s", "to", "of"}
# Dropped only in trailing position: they annotate the error, they are not it.
_TRAILING_NOISE = {"detected", "error", "occurred", "message", "again"}

_NEG_PHRASES = [
    (re.compile(r"\bcould not\b"), "not"),
    (re.compile(r"\bcan not\b"), "not"),
    (re.compile(r"\bcannot\b"), "not"),
    (re.compile(r"\bdoes not\b"), "not"),
    (re.compile(r"\bdo not\b"), "not"),
    (re.compile(r"\bdid not\b"), "not"),
    (re.compile(r"\bisn t\b"), "not"),
    (re.compile(r"\bwon t\b"), "not"),
    (re.compile(r"\bunable to\b"), "not"),
]
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
# Leading label the customer pastes along with the message.
_LEAD = re.compile(
    r"^(error|err|fault|warning|message|msg|code|exception)\s*[:#\-]?\s*", re.IGNORECASE
)


def canonical_error(text: str) -> str:
    """Normalised lookup key. Deterministic, no model involved."""
    if not text:
        return ""
    low = _LEAD.sub("", str(text).strip().lower())
    low = _PUNCT.sub(" ", low)
    for pat, rep in _NEG_PHRASES:
        low = pat.sub(rep, low)
    words = [_STEM.get(w, w) for w in _WS.sub(" ", low).strip().split() if w not in _STOP]
    while words and words[-1] in _TRAILING_NOISE:
        words.pop()
    return " ".join(words)


class _ErrorIndex:
    """In-process exact-match index over the flashing-error vocabulary.

    Rebuilt on a TTL rather than cached forever: ``calls_analysis`` grows
    continuously and a worker that ran for a week would answer from a week-old
    vocabulary. Rebuilt lazily rather than at import so that importing the tool
    layer does not require Mongo.
    """

    TTL_SECONDS = 3600.0
    # A string seen once is a transcription artefact, not a vocabulary entry.
    MIN_OCCURRENCES = 2

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._built_at = 0.0
        self._entries: dict[str, dict] = {}
        self._doc_count = 0

    def _build(self) -> None:
        db = source_db()
        # filename -> case_id, so a matched error string can hand the agent a
        # case id and therefore the failed attempts. Cheap: 6.8k docs.
        file_to_case: dict[str, str] = {}
        for case in db[config.COLL_CASES].find(
            {"training_safe": True}, {"case_id": 1, "members": 1}
        ):
            for member in case.get("members") or []:
                file_to_case[member] = case["case_id"]

        agg: dict[str, dict] = {}
        docs = 0
        cursor = db[config.COLL_ANALYSIS].find(
            {"error_messages_verbatim.0": {"$exists": True}, "training_safe": True},
            {
                "error_messages_verbatim": 1,
                "technical_category": 1,
                "solution_steps": 1,
                "canonical_problem": 1,
                "resolved_success": 1,
                "file_name": 1,
                "language": 1,
            },
        )
        for doc in cursor:
            docs += 1
            case_id = file_to_case.get(doc.get("file_name") or "")
            for raw in doc.get("error_messages_verbatim") or []:
                if not isinstance(raw, str):
                    continue
                key = canonical_error(raw)
                if not key or len(key) < 4:
                    continue
                entry = agg.setdefault(
                    key,
                    {
                        "key": key,
                        "occurrences": 0,
                        "surfaces": Counter(),
                        "categories": Counter(),
                        "problems": Counter(),
                        "steps": Counter(),
                        "case_ids": [],
                        "resolved": 0,
                        "unresolved": 0,
                        "languages": Counter(),
                    },
                )
                entry["occurrences"] += 1
                entry["surfaces"][raw.strip()] += 1
                if doc.get("technical_category"):
                    entry["categories"][doc["technical_category"]] += 1
                if doc.get("canonical_problem"):
                    entry["problems"][doc["canonical_problem"]] += 1
                for step in doc.get("solution_steps") or []:
                    if isinstance(step, str) and step.strip():
                        entry["steps"][step.strip()] += 1
                if doc.get("resolved_success") is True:
                    entry["resolved"] += 1
                elif doc.get("resolved_success") is False:
                    entry["unresolved"] += 1
                if doc.get("language"):
                    entry["languages"][doc["language"]] += 1
                if case_id and case_id not in entry["case_ids"] and len(entry["case_ids"]) < 5:
                    entry["case_ids"].append(case_id)

        self._entries = {
            k: v for k, v in agg.items() if v["occurrences"] >= self.MIN_OCCURRENCES
        }
        self._doc_count = docs
        self._built_at = time.time()

    def ensure(self) -> None:
        if self._entries and time.time() - self._built_at < self.TTL_SECONDS:
            return
        with self._lock:
            if self._entries and time.time() - self._built_at < self.TTL_SECONDS:
                return
            self._build()

    @property
    def stats(self) -> dict:
        self.ensure()
        return {
            "distinct_errors": len(self._entries),
            "source_calls": self._doc_count,
            "built_at": self._built_at,
            "total_observations": sum(e["occurrences"] for e in self._entries.values()),
        }

    def top(self, n: int = 25) -> list[tuple[str, int]]:
        self.ensure()
        rows = sorted(self._entries.items(), key=lambda kv: -kv[1]["occurrences"])
        return [(v["surfaces"].most_common(1)[0][0], v["occurrences"]) for _, v in rows[:n]]

    def lookup(self, text: str) -> tuple[dict | None, str]:
        """Exact key first, then containment. Never embeddings."""
        self.ensure()
        key = canonical_error(text)
        if not key:
            return None, "empty"
        entry = self._entries.get(key)
        if entry:
            return entry, "exact"
        # The customer pastes the dialog box, not the string: "Flashing failed:
        # The request is not supported (0x7F)". Longest known key contained in
        # the pasted text wins -- still a lookup, still deterministic.
        contained = [k for k in self._entries if k in key]
        if contained:
            best = max(contained, key=lambda k: (len(k), self._entries[k]["occurrences"]))
            return self._entries[best], "contains"
        # Reverse: the customer paraphrased down to a fragment of a known
        # string. Require the fragment to be substantial to avoid "cable"
        # matching every cable error.
        if len(key) >= 12:
            supersets = [k for k in self._entries if key in k]
            if supersets:
                best = max(supersets, key=lambda k: self._entries[k]["occurrences"])
                return self._entries[best], "fragment_of"
        return None, "no_match"


_INDEX = _ErrorIndex()


def _entry_view(entry: dict, match_type: str) -> dict:
    return {
        "matched": True,
        "match_type": match_type,
        "canonical_key": entry["key"],
        "error_string": entry["surfaces"].most_common(1)[0][0],
        "surface_variants": [s for s, _ in entry["surfaces"].most_common(6)],
        "occurrences": entry["occurrences"],
        "technical_categories": [c for c, _ in entry["categories"].most_common(3)],
        "canonical_problems": [p for p, _ in entry["problems"].most_common(3)],
        # Success-biased by construction (AGENT_PLAN.md §8.2): solution_steps
        # are populated on 66% of resolved calls and 19% of unresolved ones.
        # The counts are exposed so the model can see how thin the evidence is.
        "common_solution_steps": [
            {"step": s, "seen_on_calls": n} for s, n in entry["steps"].most_common(8)
        ],
        "resolved_calls": entry["resolved"],
        "unresolved_calls": entry["unresolved"],
        "languages": [l for l, _ in entry["languages"].most_common(2)],
        # Feed these to get_case for the failed attempts -- the negative
        # evidence that solution_steps does not carry.
        "related_case_ids": entry["case_ids"],
    }


def lookup_error_string(text: str) -> dict:
    if not (text or "").strip():
        raise ToolInputError("lookup_error_string needs the error text")
    entry, how = _INDEX.lookup(text)
    if not entry:
        return {
            "matched": False,
            "match_type": how,
            "query": text,
            "canonical_key": canonical_error(text),
            "note": (
                "Not in the known flashing-error vocabulary. Fall back to "
                "search_knowledge, and consider log_knowledge_gap if the customer "
                "quoted the string exactly."
            ),
            "index_stats": _INDEX.stats,
        }
    view = _entry_view(entry, how)
    view["query"] = text
    return view


def error_index_stats() -> dict:
    return _INDEX.stats


# --- Tool definitions --------------------------------------------------------

TOOLS = [
    Tool(
        name="search_knowledge",
        description=(
            "Semantic search over the Unitronic knowledge base built from support and "
            "sales calls. Use for procedures, troubleshooting, policy and product "
            "questions. Do NOT use it for prices (use get_fee_schedule), for stage "
            "availability (use check_stage_availability) or for a verbatim error "
            "message (use lookup_error_string first)."
        ),
        parameters=obj_schema(
            {
                "query": {"type": "string", "description": "The customer's question"},
                "kind": {
                    "type": "string",
                    "enum": _KINDS,
                    "description": "Restrict to one kind of knowledge unit",
                },
                "department": {
                    "type": "string",
                    "description": "e.g. technical_support, sales",
                },
                "language": {"type": "string", "enum": ["en", "fr"]},
            },
            ["query"],
        ),
        handler=search_knowledge,
        dependency="qdrant",
        injects=("persona",),
    ),
    Tool(
        name="get_case",
        description=(
            "Full history of a multi-call support case: chronology, everything that was "
            "tried INCLUDING the attempts that failed, and what finally worked. The "
            "failed attempts are the only record of what does not work -- check them "
            "before suggesting a fix, and use them to pre-empt ('before I suggest X, is "
            "Secure Boot enabled?'). Case ids come from lookup_error_string or "
            "search_knowledge results."
        ),
        parameters=obj_schema(
            {"case_id": {"type": "string", "description": "16-hex case id"}},
            ["case_id"],
        ),
        handler=get_case,
        dependency="mongo",
    ),
    Tool(
        name="lookup_error_string",
        description=(
            "Exact lookup of a verbatim flashing/software error message against the "
            "closed vocabulary of errors seen on real calls. ALWAYS try this first when "
            "the customer quotes an on-screen message such as 'the request is not "
            "supported', 'cable is not responding', 'interrupted flashing session "
            "detected', 'ECU file invalid' or 'data is invalid'. Paste the message as "
            "the customer wrote it; normalisation is handled here."
        ),
        parameters=obj_schema(
            {
                "text": {
                    "type": "string",
                    "description": "The error message verbatim, as the customer typed or read it",
                }
            },
            ["text"],
        ),
        handler=lookup_error_string,
        dependency="mongo",
    ),
]


def self_check() -> None:
    import json

    t0 = time.time()
    stats = error_index_stats()
    print(f"--- error index built in {time.time()-t0:.2f}s ---")
    print(json.dumps(stats, indent=1))
    assert stats["distinct_errors"] > 100

    print("--- top 15 error strings ---")
    for surface, n in _INDEX.top(15):
        print(f"  {n:5d}  {surface}")

    # The five strings AGENT_PLAN.md §8.2 names must all resolve.
    for probe in [
        "the request is not supported",
        "Interrupted flashing session detected",
        "cable is not responding",
        "ECU file invalid",
        "data is invalid",
        "Flashing failed: The request is not supported (0x7F)",
    ]:
        res = lookup_error_string(probe)
        print(
            f"  {probe!r:60s} -> matched={res['matched']} "
            f"type={res['match_type']} n={res.get('occurrences')} "
            f"key={res.get('canonical_key')!r}"
        )
        assert res["matched"], probe

    print("--- collapse check: cable variants share one key ---")
    keys = {canonical_error(v) for v in
            ["cable is not responding", "cable not responding", "cable s not responding",
             "Cable is not responding."]}
    print("  keys:", keys)
    assert len(keys) == 1

    print("--- full lookup_error_string('interrupted flashing session detected') ---")
    print(json.dumps(lookup_error_string("interrupted flashing session detected"), indent=1))

    print("--- no-match path ---")
    print(json.dumps(lookup_error_string("my toaster is on fire"), indent=1)[:400])

    print("--- get_case on a case with failed attempts ---")
    case = source_db()[config.COLL_CASES].find_one(
        {"training_safe": True, "attempts.result": "failed"}, {"case_id": 1}
    )
    view = get_case(case["case_id"])
    print(json.dumps(view, indent=1, default=str)[:2500])
    assert view["attempt_counts"]["failed"] >= 1
    blob = json.dumps(view, default=str)
    assert ".wav" not in blob and "phone_key" not in blob, "PII leaked into case view"

    try:
        get_case("nope")
        raise AssertionError("bad case id not rejected")
    except ToolInputError as exc:
        print("ToolInputError:", exc)

    print("--- search_knowledge ---")
    try:
        out = search_knowledge(
            "how do I install UniCONNECT+ drivers", persona="support"
        )
        if isinstance(out, Degraded):
            print("DEGRADED:", out.reason)
            out = out.data
        print("source:", out["source"], "results:", out["result_count"])
        print("retrieval meta:", json.dumps(out.get("retrieval", {}), default=str)[:400])
        for r in out["results"][:3]:
            print(
                f"  {r['score']:.4f} {str(r.get('title') or r.get('product_name'))[:60]!r} "
                f"<- {r['collections']}"
            )
        # Must be JSON-serialisable: it goes on the wire as a tool message.
        json.dumps(out, default=None)
    except ToolDependencyError as exc:
        print("ToolDependencyError (expected until the index is built):", exc)

    print("--- fallback path, exercised directly (it is a fuse, not the route) ---")
    try:
        rows = _fallback_search(
            "uniconnect driver install", kind=None, department=None, language=None, limit=3
        )
        print(f"  fallback returned {len(rows)} rows from the kb_units alias")
    except ToolDependencyError as exc:
        # Expected until Phase 4a builds unitronic_kb_units. The point of
        # running it is that the code path is exercised rather than assumed.
        print("  ToolDependencyError (kb_units not built yet):", exc)

    print("knowledge.py self-check OK")


if __name__ == "__main__":
    self_check()
