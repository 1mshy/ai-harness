"""The composed retrieval path.

    normalize -> encode once -> hybrid top-30 per collection -> RRF across
    collections -> dedupe on unit_id -> rerank -> supersession -> bounded
    occurrence boost + time_sensitive-gated decay -> top 5-8

Three properties in here are not tunables, and each one exists because the
stack this replaces got it wrong in a way that was invisible in testing.

**The ``training_safe`` floor is AND-ed server-side and cannot be replaced.**
``chat.py:110-111`` read ``collection_filters`` and ``metadata_filters``
straight off the request body and passed them to the query engine. A safety
filter expressed that way is a suggestion: whoever writes the next client
decides whether it applies. Here the caller's filter is *nested inside* the
must-clause of a filter the caller never sees. A caller passing
``training_safe=False`` gets ``training_safe == True AND training_safe ==
False`` -- an empty result set, not an unsafe one. There is no code path that
builds a query filter without :func:`_floor`.

**Occurrence is a bounded boost.** 28,423 of the units are true singletons and
they are the reason the corpus exists. Occurrence enters as a capped additive
nudge on a [0,1] relevance score, clamped so the total adjustment can only
reorder near-ties. It is never a filter, never a sort key, and a unit seen 400
times cannot outrank a unit that actually answers the question.

**Decay is gated on ``time_sensitive``.** 14.2% of units are time-sensitive;
the other 86% must not decay at all. A 2024 fix for a flashing error is still
the fix, and a global recency prior quietly deletes the corpus's long tail.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from qdrant_client import models as qm

from .. import config
from ..clients.qdrant import CollectionEmpty, Hit
from ..text.normalize import (
    expand_aliases,
    marker_tokens,
    normalize_text,
    strip_pii_markers,
)
from . import parallel, supersede
from .parallel import CollectionProfile, CollectionResult, QueryVectors
from .rerank import get_reranker, rerank_enabled

# --- routing -----------------------------------------------------------------
# Persona selects *which* collections are worth the round trip, nothing else.
# Content gates (emissions, safety, pricing) are pre-retrieval and post-
# generation concerns and live in guardrails/, not here -- a gate implemented
# as "do not retrieve it" is defeated by the model already knowing the answer.
_LEGACY = {name.rsplit("_", 2)[0]: name for name in config.LEGACY_COLLECTIONS}

ROUTES: dict[str, list[str]] = {
    "support": [
        config.ALIAS_KB_UNITS,
        config.ALIAS_CASE_NARRATIVES,
        config.ALIAS_CALL_RESIDUAL,
        # Compatibility is 11,696 support calls and is the one question type
        # the corpus must not arbitrate -- the platforms table is live truth and
        # 88 stage rows are currently unreleased. Routing support here does not
        # replace the compatibility oracle tool; it stops the agent answering
        # from a 2024 call that said "not out yet".
        config.ALIAS_PLATFORM_STAGES,
        _LEGACY["unitronic_comprehensive"],
        _LEGACY["unitronic_tuning"],
        _LEGACY["unitronic_products_tuning"],
    ],
    "sales": [
        config.ALIAS_KB_UNITS,
        config.ALIAS_PLATFORM_STAGES,
        _LEGACY["unitronic_products"],
        _LEGACY["unitronic_products_tuning"],
        _LEGACY["unitronic_comprehensive"],
        _LEGACY["unitronic_company_info"],
    ],
}
DEFAULT_ROUTE: list[str] = list(
    dict.fromkeys(config.NEW_ALIASES + config.LEGACY_COLLECTIONS)
)

# Hard blocklist. `unitronic_call_transcriptions_0_6b` carries `file_name`
# (the caller's phone number), `agent_name` and `caller_area_code`, both as
# payload keys and again serialised inside `_node_content`. It is never
# routable, including when a caller names it explicitly.
BLOCKED_COLLECTIONS = frozenset(config.QUARANTINED_COLLECTIONS)

# Routing is an allowlist, not a blocklist, for the same reason the payload is.
# A blocklist only excludes the leak someone already found: measured 2026-08-06,
# `unitronic_customer_service_training_0_6b` and
# `unitronic_customer_service_classification_0_6b` (and their `_8b` twins) sit
# on the live cluster carrying `thread_path` values of the form
# `inbox/<customer surname+given name>_<messenger user id>` plus the whole
# conversation inside `_node_content`. They are not in
# `config.QUARANTINED_COLLECTIONS`, so a blocklist let
# `search(..., collections=["unitronic_customer_service_training_0_6b"])`
# through and returned `thread_path='inbox/samylarrivee_10155364132362826'` to
# the caller. Naming a collection that is not on this list is now an error
# rather than a retrieval, so the next collection someone drops into the
# cluster is unroutable until it is deliberately added here.
ROUTABLE_COLLECTIONS = frozenset(config.NEW_ALIASES) | frozenset(
    config.LEGACY_COLLECTIONS
)


def is_routable(name: str) -> bool:
    """Alias on the allowlist, or a physical generation behind one of them.

    The generation form (``<alias>_<size>__<gen>``) is accepted so that a caller
    can pin a specific build -- which is how a rollback is verified, and how the
    self-check addresses its fixture -- without that becoming a way to name an
    arbitrary collection.
    """
    if name in BLOCKED_COLLECTIONS:
        return False
    if name in ROUTABLE_COLLECTIONS:
        return True
    return any(
        name.startswith(f"{alias}_{config.QWEN_SIZE}__")
        for alias in config.NEW_ALIASES
    )

# --- scoring constants -------------------------------------------------------
# All adjustments act on a relevance score already normalised into [0,1]
# (sigmoid of the cross-encoder logit, or normalised RRF when the reranker is
# off), and their sum is clamped. The clamp is the actual safety property: at
# 0.05 an adjustment can reorder units the reranker considers near-equivalent
# and cannot touch a decision it made with confidence.
ADJUSTMENT_CAP = float(os.environ.get("RETRIEVAL_ADJUSTMENT_CAP", "0.05"))
OCC_BOOST_MAX = float(os.environ.get("RETRIEVAL_OCC_BOOST", "0.03"))
OCC_SATURATION = float(os.environ.get("RETRIEVAL_OCC_SATURATION", "25"))
DECAY_MAX = float(os.environ.get("RETRIEVAL_DECAY_MAX", "0.04"))
DECAY_PER_YEAR = float(os.environ.get("RETRIEVAL_DECAY_PER_YEAR", "0.02"))

# How many merged candidates reach the cross-encoder, and the single biggest
# term in end-to-end latency: measured on the live cluster, a support-route
# query is ~1050 ms total of which ~900 ms is this rerank. Cost is linear in
# candidates and in tokens per candidate (424 ms at kb_unit length, 1024 ms at
# legacy-page length), so this is the knob to turn if the budget tightens --
# not max_length, which barely moves.
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", str(config.RETRIEVE_TOP_K)))
# Alias canonicals appended to the dense query text, capped. `dongle` ->
# `uniconnect plus cable` is a real rewrite mined from 52.9% of analysed calls;
# appending twenty of them would drown the customer's own wording.
MAX_ALIAS_EXPANSIONS = int(os.environ.get("RETRIEVAL_MAX_ALIASES", "4"))

# Fields a caller may not decide. Supplying one is not an error -- it is
# AND-ed and therefore harmless -- but it is recorded in the trace, because a
# client repeatedly trying to set `training_safe` is a thing worth seeing.
RESERVED_FILTER_KEYS = frozenset(
    {"training_safe", "internal_only", "status", "kind"}
)


# --- result types ------------------------------------------------------------


@dataclass
class RetrievedUnit:
    """One deduplicated unit, with its provenance across collections/roles."""

    unit_id: str
    payload: dict
    collections: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    point_ids: list[str] = field(default_factory=list)
    rrf_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float | None = None
    relevance: float = 0.0  # [0,1] pre-adjustment
    final_score: float = 0.0
    adjustments: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return unit_text(self.payload)

    def as_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "collections": self.collections,
            "roles": self.roles,
            "rrf_score": round(self.rrf_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "relevance": round(self.relevance, 6),
            "final_score": round(self.final_score, 6),
            "adjustments": {k: round(v, 6) for k, v in self.adjustments.items()},
            "payload": self.payload,
        }


@dataclass
class SearchResult:
    query: str
    normalized_query: str
    units: list[RetrievedUnit]
    dropped_superseded: list[supersede.Drop] = field(default_factory=list)
    collections_searched: list[str] = field(default_factory=list)
    collections_missing: list[str] = field(default_factory=list)
    collections_failed: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self):
        return iter(self.units)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "units": [u.as_dict() for u in self.units],
            "dropped_superseded": [
                {"unit_id": d.unit_id, "superseded_by": d.superseded_by, "reason": d.reason}
                for d in self.dropped_superseded
            ],
            "collections_searched": self.collections_searched,
            "collections_missing": self.collections_missing,
            "collections_failed": self.collections_failed,
            "trace": self.trace,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


# --- query normalization -----------------------------------------------------


@dataclass(frozen=True)
class NormalizedQuery:
    raw: str
    text: str  # what actually gets encoded
    aliases: list[str]
    markers: list[str]


def normalize_query(query: str) -> NormalizedQuery:
    """PII-strip, then additively expand colloquialisms.

    Additive, never substitutive: the colloquial form is usually also what the
    corpus says, so replacing ``dongle`` with ``uniconnect plus cable`` would
    lose a match as often as it gains one. The PII strip runs first because
    the query text reaches an embedding backend and a trace, and a phone number
    in either is the same leak as a phone number in a payload.
    """
    raw = (query or "").strip()
    text = strip_pii_markers(raw)
    aliases = [a for a in expand_aliases(text) if a.lower() not in text.lower()]
    aliases = aliases[:MAX_ALIAS_EXPANSIONS]
    if aliases:
        text = f"{text} {' '.join(aliases)}"
    return NormalizedQuery(
        raw=raw, text=text, aliases=aliases, markers=marker_tokens(text)
    )


# --- filters -----------------------------------------------------------------


def _facet(key: str, value: Any) -> qm.Filter:
    """``key == value`` OR ``key`` absent.

    A collection that predates the taxonomy is not evidence of a mismatch --
    a scraped product page has no ``language`` field and is still the right
    answer to a French question about that product. A point that *declares* a
    different value is excluded. This tolerance applies only to facets; the
    floor below is strict by construction.
    """
    match = (
        qm.MatchAny(any=list(value))
        if isinstance(value, (list, tuple, set))
        else qm.MatchValue(value=value)
    )
    return qm.Filter(
        should=[
            qm.FieldCondition(key=key, match=match),
            qm.IsEmptyCondition(is_empty=qm.PayloadField(key=key)),
        ]
    )


def _floor(
    profile: CollectionProfile,
    *,
    allow_internal: bool,
    facets: dict[str, Any],
    extra_filter: qm.Filter | None,
) -> qm.Filter:
    """The non-bypassable base filter, built server-side, per collection.

    ``extra_filter`` is nested one level down inside ``must``. Qdrant evaluates
    a nested filter as a single condition, so the caller's clause can only ever
    *narrow* the result set. There is no arrangement of caller input that
    removes a condition from this list, which is the whole point: the previous
    implementation let the request body supply the filter object outright.

    Gate-bearing collections get ``training_safe == True``. The legacy website
    and product collections do not carry the field at all -- requiring it there
    would return zero rows and quietly disable half the index -- so they get
    the defensive form, ``NOT (training_safe == False)``, which is exactly as
    strict for any point that ever gains the field.
    """
    must: list[Any] = []
    must_not: list[Any] = [
        qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=False)),
        # 9.4: revocation is a targeted delete, but a unit tombstoned in Mongo
        # between builds must stop being served immediately.
        qm.FieldCondition(key="status", match=qm.MatchValue(value="revoked")),
    ]
    if profile.gate_bearing:
        must.append(
            qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=True))
        )
    if not allow_internal:
        # 2,564 internal_process units. Both spellings are excluded because the
        # boolean is derived from the kind and a derivation can be skipped.
        must_not.append(
            qm.FieldCondition(key="internal_only", match=qm.MatchValue(value=True))
        )
        must_not.append(
            qm.FieldCondition(key="kind", match=qm.MatchValue(value="internal_process"))
        )
    for key, value in facets.items():
        if value is not None:
            must.append(_facet(key, value))
    if extra_filter is not None:
        must.append(extra_filter)
    return qm.Filter(must=must, must_not=must_not)


def _reserved_keys_touched(f: qm.Filter | None) -> list[str]:
    """Reserved payload keys named anywhere in a caller-supplied filter."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, qm.Filter):
            for group in (node.must, node.should, node.must_not, node.min_should):
                if group is None:
                    continue
                items = group.conditions if hasattr(group, "conditions") else group
                for c in items or []:
                    walk(c)
        else:
            key = getattr(node, "key", None) or getattr(
                getattr(node, "is_empty", None), "key", None
            )
            if key in RESERVED_FILTER_KEYS:
                found.add(key)

    walk(f)
    return sorted(found)


# --- payload -> text ---------------------------------------------------------

# `kb_units` answer-role points assemble from the first group. The legacy
# collections carry `original_text` / `context` *and* a `title`, so the
# assembled form is only used when the unit shape is actually present --
# keying off `title` alone reranks a scraped product page against the string
# "UniCONNECT+", which scores every product in the catalogue identically.
_TEXT_FIELDS = ("title", "question", "answer", "conditions")
_UNIT_SHAPE_FIELDS = ("question", "answer")
_LEGACY_TEXT_FIELDS = ("original_text", "context", "text", "content", "title")


# The payload that leaves this module is an allowlist, not whatever the
# collection happens to store. The index builder controls what goes into the
# collections this project owns, but retrieval also reads five pre-existing
# collections built by another stack, and *those* payloads are a projection of
# the source doc: every one of them carries `_node_content`, a 2-4 KB
# serialised LlamaIndex node that duplicates the text and carries the source
# metadata along with it. That is the exact field the contract names as the
# second half of the leak it is replacing ("once as payload keys and once again
# serialised inside `_node_content`"). Returning it verbatim puts it in
# `SearchResult.as_dict()`, which is what an SSE frame serialises.
#
# Anything not on this list is dropped. Adding a field is a deliberate edit
# here, so a new payload key on a collection someone rebuilds cannot reach a
# caller by default.
PAYLOAD_ALLOWLIST = frozenset({
    # kb_units identity + versioning (CONTRACT.md)
    "unit_id", "gen", "kb_version", "merge_version", "point_role",
    # content
    "kind", "title", "question", "answer", "conditions",
    "vehicles_applicable", "products_applicable", "hypothetical_questions",
    # provenance + supersession
    "evidence", "outranks_call_units", "superseded_unit_ids", "case_id",
    "confidence", "occurrences", "source_ids", "cluster_id", "merge_action",
    # gates and derived labels
    "training_safe", "time_sensitive", "emissions_risk", "safety_gated",
    "dealer_pricing", "internal_only", "contains_price",
    "requires_tool_confirmation",
    # facets and lifecycle
    "department", "language", "technical_category", "doc_type",
    "created_at", "updated_at", "status",
    # case/call derived fields carried by case_narratives and call_residual
    "issue_category", "root_cause", "resolution_status", "error_codes", "text",
    # platform_stages
    "platform_id", "platform_name", "makes", "model_names", "stage_id",
    "stage_label", "stage_family", "stage_plus", "stage_rank",
    "stage_released", "max_released_stage", "year_start", "year_end",
    "synced_at",
    # legacy website/product collections: the display and grounding fields.
    # `_node_content`, `_node_type`, `doc_id`, `document_id` and `ref_doc_id`
    # are deliberately absent.
    "original_text", "context", "content", "url", "source", "page_type",
    "content_type", "enhanced_content", "semantic_keywords", "quality_score",
    "brand", "sku", "primary_price", "stage", "product_name", "product_type",
    "product_url", "interaction_type",
})


def scrub_payload(payload: dict) -> dict:
    """Drop every key that is not explicitly allowed out of the retrieval layer."""
    return {k: v for k, v in payload.items() if k in PAYLOAD_ALLOWLIST}


def unit_text(payload: dict) -> str:
    if any(payload.get(f) for f in _UNIT_SHAPE_FIELDS):
        return "\n".join(
            str(payload[f]).strip() for f in _TEXT_FIELDS if payload.get(f)
        )
    for f in _LEGACY_TEXT_FIELDS:
        if payload.get(f):
            return str(payload[f]).strip()
    return ""


def dedupe_key(hit: Hit) -> str:
    """Two points per unit (answer + query roles) collapse to one unit.

    Legacy collections have no ``unit_id``, so identity there falls back to a
    hash of the normalised text. This is not the semantic merge -- that is
    Phase 4a's job and it needs an LLM to arbitrate -- it only collapses
    *verbatim* restatements. It is still load-bearing: measured on this cluster,
    an unfiltered top-5 for "does stage 1+ need a downpipe on a mk7 gti"
    returned the identical string ``Unitronic Stage 1+ : 254HP / 293LB-FT``
    twice under different point ids, spending two of five context slots on one
    fact. The point id is the last resort, for points with no usable text.
    """
    uid = hit.payload.get("unit_id")
    if uid:
        return str(uid)
    text = unit_text(hit.payload)
    if text:
        return "sha1:" + hashlib.sha1(normalize_text(text).encode()).hexdigest()[:20]
    return str(hit.id)


# --- scoring -----------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def occurrence_boost(occurrences: Any) -> float:
    """Capped, log-shaped, and worth at most ``OCC_BOOST_MAX``.

    Log-shaped because the corpus is Zipfian -- ``remote cable reset process``
    appears 152 times and the median unit once. A linear boost would hand the
    head an unbounded advantage; this one saturates at 25 occurrences, so the
    difference between 30 and 400 is nothing.
    """
    # OverflowError as well as TypeError/ValueError: a float payload value can
    # be an infinity, and `int(inf)` raises OverflowError, not ValueError. This
    # runs inside the request path over payloads the retrieval layer does not
    # build, so one malformed `occurrences` must degrade to "no boost" rather
    # than take the whole search down. (`_parse_ts` already guards this way.)
    try:
        n = int(occurrences or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if n <= 1:
        return 0.0
    n = min(n, int(OCC_SATURATION))
    return OCC_BOOST_MAX * math.log1p(n - 1) / math.log1p(OCC_SATURATION - 1)


def time_decay_penalty(payload: dict, *, now: datetime | None = None) -> float:
    """Zero unless the unit is flagged ``time_sensitive``.

    Not a global recency prior. 86% of the corpus should not decay at all, and
    a global prior is indistinguishable from deleting the long tail. Where it
    does apply -- pricing (53% of pricing units), stage availability -- the
    magnitude is deliberately small, because the real protection there is a
    tool result, not a rank.
    """
    if not payload.get("time_sensitive"):
        return 0.0
    ts = _parse_ts(payload.get("updated_at") or payload.get("created_at"))
    if ts is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    years = max(0.0, (now - ts).total_seconds() / (365.25 * 86400.0))
    return min(DECAY_MAX, DECAY_PER_YEAR * years)


def _apply_adjustments(units: Sequence[RetrievedUnit], *, now: datetime | None = None) -> None:
    for u in units:
        boost = occurrence_boost(u.payload.get("occurrences"))
        penalty = time_decay_penalty(u.payload, now=now)
        total = max(-ADJUSTMENT_CAP, min(ADJUSTMENT_CAP, boost - penalty))
        u.adjustments = {
            "occurrence_boost": boost,
            "time_decay": -penalty,
            "applied": total,
        }
        u.final_score = u.relevance + total


# --- the composed path -------------------------------------------------------


def resolve_collections(
    persona: str | None, collections: Sequence[str] | None
) -> list[str]:
    if collections is not None:
        wanted = list(dict.fromkeys(collections))
    elif persona is None:
        wanted = list(DEFAULT_ROUTE)
    elif persona in ROUTES:
        wanted = list(ROUTES[persona])
    else:
        raise ValueError(
            f"unknown persona {persona!r}; expected one of {sorted(ROUTES)} or None"
        )
    if collections is not None:
        # An explicitly named collection that is not routable is an error, not a
        # silent drop. A caller asking for a quarantined or unknown collection
        # has a bug or is probing; either way it should not look like a
        # successful search over the remainder.
        rejected = [c for c in wanted if not is_routable(c)]
        if rejected:
            raise ValueError(
                "collections not routable (not on the retrieval allowlist): "
                + ", ".join(rejected)
            )
    return [c for c in wanted if is_routable(c)]


def _merge(results: Iterable[CollectionResult]) -> tuple[list[RetrievedUnit], dict]:
    """RRF across collections, deduping on unit_id.

    RRF rather than score normalisation, for the same reason the hybrid halves
    are fused with RRF inside a single collection: a cosine score from a
    products collection and a cosine score from a knowledge collection are not
    comparable, and any normalisation that makes them comparable is a knob
    nobody will revisit.
    """
    from ..clients.sparse import fuse_rrf

    rankings: list[list[str]] = []
    units: dict[str, RetrievedUnit] = {}
    for res in results:
        if not res.ok:
            continue
        ranking: list[str] = []
        for hit in res.hits:
            key = dedupe_key(hit)
            ranking.append(key)
            unit = units.get(key)
            if unit is None:
                unit = RetrievedUnit(
                    unit_id=key, payload=scrub_payload(hit.payload), vector_score=hit.score
                )
                units[key] = unit
            else:
                # Keep the richer payload: an answer-role point carries the
                # answer text, a query-role point carries only questions.
                scrubbed = scrub_payload(hit.payload)
                if len(unit_text(scrubbed)) > len(unit_text(unit.payload)):
                    unit.payload = scrubbed
                unit.vector_score = max(unit.vector_score, hit.score)
            if res.collection not in unit.collections:
                unit.collections.append(res.collection)
            role = hit.payload.get("point_role")
            if role and role not in unit.roles:
                unit.roles.append(role)
            if str(hit.id) not in unit.point_ids:
                unit.point_ids.append(str(hit.id))
        rankings.append(ranking)

    scores = fuse_rrf(rankings, k=config.RRF_K)
    for key, score in scores.items():
        units[key].rrf_score = score
    merged = sorted(units.values(), key=lambda u: (-u.rrf_score, u.unit_id))
    return merged, {"rankings": len(rankings), "candidates": len(merged)}


def search(
    query: str,
    *,
    persona: str | None = None,
    kind: str | None = None,
    department: str | None = None,
    language: str | None = None,
    collections: Sequence[str] | None = None,
    top_k: int | None = None,
    extra_filter: qm.Filter | None = None,
    allow_internal: bool = False,
) -> SearchResult:
    """Retrieve, rerank, filter supersessions, and cut to ``top_k``.

    ``extra_filter`` narrows; it never replaces. See :func:`_floor`.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    final_k = top_k or config.FINAL_TOP_K

    norm = normalize_query(query)
    if not norm.text:
        return SearchResult(
            query=query,
            normalized_query="",
            units=[],
            trace={"skipped": "empty query"},
        )

    routed = resolve_collections(persona, collections)
    if not routed:
        raise ValueError("no collections to search after applying the blocklist")

    t0 = time.perf_counter()
    vectors: QueryVectors = parallel.encode_query(norm.text)
    timings["encode"] = (time.perf_counter() - t0) * 1000.0

    facets = {"kind": kind, "department": department, "language": language}
    reserved = _reserved_keys_touched(extra_filter)

    def filter_for(profile: CollectionProfile) -> qm.Filter:
        return _floor(
            profile,
            allow_internal=allow_internal,
            facets=facets,
            extra_filter=extra_filter,
        )

    t0 = time.perf_counter()
    results = parallel.fan_out(
        routed, vectors, limit=config.RETRIEVE_TOP_K, filter_for=filter_for
    )
    timings["fanout"] = (time.perf_counter() - t0) * 1000.0

    missing = [r.collection for r in results if r.status == "missing"]
    failed = [r.collection for r in results if r.status == "failed"]
    searched = [r.collection for r in results if r.status == "ok"]
    if not searched:
        raise CollectionEmpty(
            "every routed collection is missing or failed: "
            + ", ".join(f"{r.collection} ({r.error})" for r in results)
        )

    merged, merge_trace = _merge(results)

    # Supersession runs on the *whole* merged pool, before the rerank window is
    # cut. A case unit that ranked 47th still gets to delete the call unit that
    # ranked 2nd -- and it often will, because a unit phrased as "that does not
    # work when X" matches the query surface badly. The filter runs again after
    # reranking; it is idempotent, so the second pass is a free assertion.
    t0 = time.perf_counter()
    sup = supersede.apply_supersession(merged)
    candidates = sup.kept[:RERANK_CANDIDATES]
    timings["supersede"] = (time.perf_counter() - t0) * 1000.0

    reranked = False
    if rerank_enabled() and candidates:
        t0 = time.perf_counter()
        scores = get_reranker().score(norm.text, [u.text for u in candidates])
        for unit, raw in zip(candidates, scores):
            unit.rerank_score = raw
            unit.relevance = _sigmoid(raw)
        candidates.sort(key=lambda u: (-(u.rerank_score or 0.0), -u.rrf_score))
        timings["rerank"] = (time.perf_counter() - t0) * 1000.0
        reranked = True
    else:
        top = max((u.rrf_score for u in candidates), default=0.0) or 1.0
        for unit in candidates:
            unit.relevance = unit.rrf_score / top

    sup2 = supersede.apply_supersession(candidates)
    candidates = sup2.kept

    _apply_adjustments(candidates)
    candidates.sort(key=lambda u: (-u.final_score, -u.rrf_score, u.unit_id))
    units = candidates[:final_k]

    timings["total"] = (time.perf_counter() - t_start) * 1000.0
    return SearchResult(
        query=query,
        normalized_query=norm.text,
        units=units,
        dropped_superseded=sup.dropped + sup2.dropped,
        collections_searched=searched,
        collections_missing=missing,
        collections_failed=failed,
        trace={
            "persona": persona,
            "routed": routed,
            "alias_expansions": norm.aliases,
            "marker_tokens": norm.markers,
            "sparse_source": {
                r.collection: r.sparse_source for r in results if r.sparse_source
            },
            "modes": {r.collection: r.mode for r in results},
            "gate_bearing": {r.collection: r.profile.gate_bearing for r in results},
            "per_collection_hits": {r.collection: len(r.hits) for r in results},
            "retried": [r.collection for r in results if r.attempts > 1],
            "collection_errors": {r.collection: r.error for r in results if r.error},
            "reranked": reranked,
            "rerank_candidates": len(candidates),
            "training_safe_floor": "server_side_and",
            "caller_filter_reserved_keys": reserved,
            "allow_internal": allow_internal,
            **merge_trace,
        },
        timings_ms=timings,
    )


if __name__ == "__main__":  # self-check against the live cluster
    # 1. The floor is present in every filter, in both variants, and a hostile
    #    caller filter lands *inside* it rather than replacing it.
    gated = CollectionProfile("kb", "kb", True, True, True, True)
    legacy = CollectionProfile("old", "old", True, True, False, False)
    hostile = qm.Filter(
        must=[qm.FieldCondition(key="training_safe", match=qm.MatchValue(value=False))]
    )
    f = _floor(gated, allow_internal=False, facets={}, extra_filter=hostile)
    must_pairs = [(c.key, c.match.value) for c in f.must if hasattr(c, "key")]
    assert ("training_safe", True) in must_pairs, must_pairs
    assert hostile in f.must, "caller filter must be nested, not merged"
    assert any(c.key == "training_safe" and c.match.value is False for c in f.must_not)
    f2 = _floor(legacy, allow_internal=False, facets={}, extra_filter=None)
    assert not [c for c in f2.must if getattr(c, "key", None) == "training_safe"]
    assert any(c.key == "training_safe" and c.match.value is False for c in f2.must_not)
    assert _reserved_keys_touched(hostile) == ["training_safe"]
    print("1 floor: strict on gated, defensive on legacy, caller filter nested")

    # 2. Routing is an allowlist: the quarantined collection and every
    #    collection nobody put on the list are unroutable even when named.
    for bad in [
        *config.QUARANTINED_COLLECTIONS,
        "unitronic_customer_service_training_0_6b",
        "unitronic_customer_service_classification_0_6b",
        "unitronic_customer_service_training_8b",
        "unitronic_faq_0_6b",
    ]:
        assert not is_routable(bad), bad
        try:
            resolve_collections(None, [bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was routable")
    assert all(c not in resolve_collections("sales", None) for c in BLOCKED_COLLECTIONS)
    assert is_routable(f"{config.ALIAS_KB_UNITS}_{config.QWEN_SIZE}__abc123")
    print("2 routing allowlist: quarantined + customer_service_* -> unroutable "
          "(ValueError when named); pinned generations still routable")

    # 2b. The payload leaving retrieval is an allowlist. `_node_content` is the
    #     serialised-node half of the leak the contract names.
    dirty = {
        "unit_id": "u1", "title": "t", "answer": "a",
        "_node_content": "{...whole source doc...}", "_node_type": "TextNode",
        "doc_id": "d", "ref_doc_id": "r", "document_id": "dd",
        "thread_path": "inbox/samylarrivee_10155364132362826",
        "file_name": "3CX_5145551234.wav", "agent_name": "Marc",
        "caller_area_code": "514",
    }
    clean = scrub_payload(dirty)
    assert clean == {"unit_id": "u1", "title": "t", "answer": "a"}, clean
    print(f"2b payload allowlist: dropped {sorted(set(dirty) - set(clean))}")

    # 3. Occurrence is bounded and saturates.
    assert occurrence_boost(1) == 0.0
    assert occurrence_boost(400) == occurrence_boost(25) <= OCC_BOOST_MAX
    assert occurrence_boost(3) < occurrence_boost(20)
    print(f"3 occurrence: x1={occurrence_boost(1):.4f} x3={occurrence_boost(3):.4f} "
          f"x25={occurrence_boost(25):.4f} x400={occurrence_boost(400):.4f} "
          f"cap={OCC_BOOST_MAX}")

    # 4. Decay fires only on time_sensitive units.
    old = "2024-01-01T00:00:00Z"
    assert time_decay_penalty({"updated_at": old}) == 0.0
    assert time_decay_penalty({"time_sensitive": True, "updated_at": old}) > 0.0
    assert time_decay_penalty({"time_sensitive": True, "updated_at": "1999-01-01Z"}) == DECAY_MAX
    print(f"4 decay: non-sensitive=0.0  sensitive(2024)="
          f"{time_decay_penalty({'time_sensitive': True, 'updated_at': old}):.4f}  "
          f"max={DECAY_MAX}")

    # 5. Live end-to-end.
    for q in [
        "does stage 1+ need a downpipe on a mk7 gti",
        "mon cable uniconnect n'est pas reconnu par le logiciel",
    ]:
        res = search(q, persona="support", top_k=5)
        print(f"\n5 live: {q!r}")
        print(f"   missing={res.collections_missing} searched={len(res.collections_searched)} "
              f"candidates={res.trace['candidates']} timings={res.timings_ms}")
        for u in res.units:
            print(f"   {u.final_score:.4f} rr={u.rerank_score:+7.3f} "
                  f"[{','.join(u.collections)}] {u.text[:88]!r}")
        assert res.units, "live search returned nothing"
        assert config.ALIAS_KB_UNITS in res.collections_missing
        # No unit that leaves a live search may carry a non-allowlisted key.
        for u in res.units:
            leaked = sorted(set(u.payload) - PAYLOAD_ALLOWLIST)
            assert not leaked, (u.collections, leaked)
            assert u.text, "scrub removed the text the reranker scored"
    print("   live payloads carry no non-allowlisted keys")
    print("\npipeline.py self-check OK")
