"""The payload allowlist — the only door between Mongo and Qdrant.

AGENT_PLAN.md §3.1 measured what a *denylist* buys you: the collection this
build replaces (``unitronic_call_transcriptions_0_6b``, 24,760 live points)
carries ``file_name`` — a 3CX filename embedding the caller's phone number —
plus ``agent_name`` and ``caller_area_code``, **and carries them twice**: once
as payload keys and again inside a serialised ``_node_content`` blob. Dropping
the fields from the metadata dict that the loader passed to llama-index left
the second copy untouched. A sibling collection ships
``thread_path = "inbox/<instagram handle>"`` on 100% of sampled points, which a
ten-digit-phone regex sails straight past.

Both failures share one shape: the payload was a *projection of whatever the
source document happened to contain*. So this module inverts it.

* Only keys in :data:`ALLOWED_KEYS` can exist in a payload. There is no
  pass-through path, no ``**metadata``, no ``extra`` escape hatch.
* Values must be scalars or flat lists of scalars. **Nested dicts are refused**,
  which is what structurally kills the ``_node_content`` class of leak: there is
  no container left in which a second copy can hide.
* Every string is scrubbed on the way through anyway. Belt and braces — the
  allowlist is the gate, the scrub is defence in depth for the case where a
  PII value ends up inside a *permitted* field.

Adding a field to the index means editing this file, which means someone has to
look at the list of things we promised not to publish while they do it.

Nothing here calls an LLM or touches the network.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from functools import lru_cache
from typing import Any, Mapping

from ..config import DATA_DIR
from ..text.normalize import expand_aliases, strip_pii_markers

# Long enough for the longest measured answer (1,165 chars) with headroom;
# short enough that a pathological source document cannot bloat the payload
# store. Truncation is visible (ellipsis) rather than silent.
MAX_TEXT_CHARS = 6000
MAX_LIST_ITEMS = 32
MAX_ITEM_CHARS = 200


class PayloadLeak(RuntimeError):
    """Raised when a payload still looks like it carries an identifier.

    Fail-closed on purpose. Refusing to publish a unit costs one unit of recall;
    publishing a caller's phone number costs a disclosure notice.
    """


# --- the allowlist -----------------------------------------------------------
# Grouped by why each key exists, because "why is this published?" is the only
# question worth asking when the list grows.

_IDENTITY = (
    "unit_id",       # stable opaque id; retrieval dedupes on it, revoke.py deletes by it
    "point_role",    # "answer" | "query"
    "doc_type",      # kb_unit | case_narrative | call_residual | platform_stage
)

_CONTENT = (
    "title",
    "question",
    "answer",
    "conditions",
    "text",          # the exact string that was embedded; the reranker scores this
)

_FACETS = (
    "kind",
    "evidence",
    "confidence",
    "occurrences",   # bounded boost only (§ non-negotiable 5), never a sort key
    "language",
    "department",
    "technical_category",
    "issue_category",
    "root_cause",
    "resolution_status",
    "vehicles_applicable",
    "products_applicable",
    "tune_stages",
    "part_numbers",
    "error_codes",
)

_GATES = (
    "training_safe",   # AND-ed server-side into every query; never absent
    "time_sensitive",
    "emissions_risk",
    "safety_gated",
    "dealer_pricing",
    "internal_only",
    "contains_price",
    "requires_tool_confirmation",  # stage/price cards: retrieval may surface, never quote
)

_LINEAGE = (
    "source_ids",           # mongo _id strings -> targeted revocation + Law 25 deletion
    "case_id",
    "cluster_id",
    "superseded_unit_ids",  # remapped from supersedes_calls[]; opaque ids, never filenames
    "outranks_call_units",
    "kb_version",
    "merge_version",
    "index_version",
    "gen",
    "updated_at",
)

_PLATFORM = (
    "platform_id",
    "platform_name",
    "platform_code",
    "makes",
    "model_names",
    "year_start",
    "year_end",
    "stage_id",
    "stage_label",
    "stage_family",
    "stage_number",
    "stage_plus",
    "stage_rank",
    "stage_released",
    "max_released_stage",
    "synced_at",       # freshness; the compatibility oracle refuses on a stale snapshot
)

ALLOWED_KEYS: frozenset[str] = frozenset(
    _IDENTITY + _CONTENT + _FACETS + _GATES + _LINEAGE + _PLATFORM
)

# Measured on live collections 2026-08-05. Nothing consults this list to decide
# what to publish — the allowlist already did that. It exists so that a
# regression that re-adds one of these to ALLOWED_KEYS trips an assertion in the
# test suite instead of shipping.
KNOWN_LEAK_KEYS: frozenset[str] = frozenset(
    {
        "file_name", "filename", "file_path", "basename",
        "agent_name", "caller_area_code", "caller_phone_number", "phone",
        "phone_key", "thread_path", "_node_content", "_node_type",
        "customer_match", "caller_identity", "customer_name_mentioned",
        "email", "vehicle_context", "order_context", "members", "sessions",
        "conversation_turns", "full_transcription", "supersedes_calls",
        "pii_spoken_in_call", "customer_id", "vin",
    }
)

assert not (ALLOWED_KEYS & KNOWN_LEAK_KEYS), "a known-leak key is on the allowlist"

# Keys whose values this pipeline *mints* (or lifts verbatim from a Mongo `_id`)
# and which must survive byte-identical: a scrubbed ``source_ids`` entry is a
# Law 25 deletion request that cannot be served. They are exempt from the free
# text scrubber and from the digit-run heuristics -- a 24-hex ObjectId that
# happens to be all digits would otherwise be mangled -- and are instead held to
# a strict charset. Nothing in this set is ever populated from call content.
_OPAQUE_KEYS: frozenset[str] = frozenset(
    {
        "unit_id", "point_role", "doc_type", "gen", "cluster_id", "case_id",
        "source_ids", "superseded_unit_ids", "updated_at", "synced_at",
        "kb_version", "merge_version", "index_version",
        "platform_id", "stage_id", "stage_rank", "stage_number",
        "year_start", "year_end", "occurrences",
    }
)
_OPAQUE_RE = re.compile(r"^[0-9A-Za-z:_\-.+ ]{1,64}$")


# --- scrubbing ---------------------------------------------------------------
# 3CX recording filenames: "[Coles, Zoe]_124-8012307610_20250603202223(152).wav".
# strip_pii_markers() blanks the phone run but leaves the staff surname and the
# filename itself, which is a join key back into the recording store, so the
# whole token goes.
_WAV_RE = re.compile(r"\[?[^\s\[\]]*\]?_?\d{2,4}-\[?PHONE\]?[^\s]*\.wav", re.IGNORECASE)
# The bracketed prefix is part of the token and contains a staff name, which may
# hold a space -- "[Nick Pludowski]_127-4507128184_20240516193346(1338).wav" --
# so a plain \S* run would leave the first name stranded in the text.
_ANY_WAV_RE = re.compile(r"(?:\[[^\]\n]*\])?\S*\.wav\b", re.IGNORECASE)
_BRACKET_NAME_RE = re.compile(r"\[[A-Z][a-zà-ÿ'\-]+,\s*[A-Z][a-zà-ÿ'\-]+\]")
# A courtesy title followed by a capitalised word is the one *name* shape that
# survived everything else here. `problem`/`solution` are LLM-written prose over
# a transcript, so a surname can enter a permitted field with no filename and no
# digits attached to catch it -- measured: one genuine customer surname
# ("Mr. Jelinski") in 12,413 published points, on a call_residual card.
# Names have a known regex recall problem (text/normalize.strip_pii_markers says
# so in its own docstring); this closes the one shape actually observed rather
# than pretending to solve the general case, and it is listed in
# _RESIDUAL_LEAK_RES below so a *new* shape stops the build instead of shipping.
_TITLE_NAME_RE = re.compile(r"\b(?:M(?:r|rs|s|iss)|Dr|Prof)\.?\s+[A-Z][a-zà-ÿ'\-]+")
_ORDER_RE = re.compile(r"\b(?:sk\d:|SO-|PO-)\d{3,}\b", re.IGNORECASE)

# What a leak looks like *after* scrubbing. If any of these still match we have
# found a pattern the scrubber does not know about, and that is a stop-the-build
# event rather than something to log and continue past.
_RESIDUAL_LEAK_RES = (
    _ANY_WAV_RE,
    _TITLE_NAME_RE,
    re.compile(r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b"),
    re.compile(r"(?<!\d)\d{10,}(?!\d)"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"inbox/\S+"),
)


# Lookarounds rather than \b: \b does not fire between two digits, so a plain
# \b\d{10,15}\b silently misses a 24-digit run -- which is exactly the shape an
# ObjectId pasted into a transcript takes.
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")


def scrub(text: str) -> str:
    """Redact identifiers from a string that is otherwise allowed to publish."""
    if not text:
        return ""
    text = _WAV_RE.sub("[RECORDING]", text)
    text = _ANY_WAV_RE.sub("[RECORDING]", text)
    text = _BRACKET_NAME_RE.sub("[NAME]", text)
    text = _TITLE_NAME_RE.sub("[NAME]", text)
    text = strip_pii_markers(text)
    text = _ORDER_RE.sub("[ORDER]", text)
    # strip_pii_markers stops at exactly ten digits; international numbers and
    # account ids run longer, and a bare long digit run in a support corpus is
    # an identifier far more often than it is a fact worth keeping.
    text = _LONG_DIGITS_RE.sub("[NUM]", text)
    # strip_pii_markers can expose a bare wav token once the phone run inside it
    # is replaced, so run the filename pass again on the result.
    return _ANY_WAV_RE.sub("[RECORDING]", text)


def assert_no_leak(payload: Mapping[str, Any]) -> None:
    """Verify a finished payload. Raises :class:`PayloadLeak`."""
    stray = set(payload) - ALLOWED_KEYS
    if stray:
        raise PayloadLeak(f"keys not on the allowlist reached a payload: {sorted(stray)}")
    for key, value in payload.items():
        opaque = key in _OPAQUE_KEYS
        for text in _strings_in(value):
            if opaque:
                if not _OPAQUE_RE.match(text):
                    raise PayloadLeak(
                        f"opaque payload key {key!r} holds {text!r}, which is not an "
                        f"identifier shape. Opaque keys bypass the scrubber, so they "
                        f"may only ever be minted ids, not call content."
                    )
                continue
            for pattern in _RESIDUAL_LEAK_RES:
                if pattern.search(text):
                    raise PayloadLeak(
                        f"payload key {key!r} still matches {pattern.pattern!r} after "
                        f"scrubbing -- the scrubber does not know this shape yet"
                    )


def _strings_in(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


# --- coercion ----------------------------------------------------------------
def _coerce(key: str, value: Any) -> Any | None:
    """Scalar or flat list of scalars, scrubbed. ``None`` means "drop it"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    opaque = key in _OPAQUE_KEYS
    if isinstance(value, str):
        cleaned = value.strip() if opaque else scrub(value).strip()
        if not cleaned:
            return None
        return cleaned if len(cleaned) <= MAX_TEXT_CHARS else cleaned[:MAX_TEXT_CHARS] + "..."
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            if isinstance(item, bool) or isinstance(item, (int, float)):
                out.append(item)
            elif isinstance(item, str):
                cleaned = item.strip() if opaque else scrub(item).strip()
                if cleaned:
                    out.append(cleaned[:MAX_ITEM_CHARS])
            else:
                # A list of dicts is exactly how supersedes_calls[] and
                # conversation_turns[] carry PII. There is no safe flattening.
                raise PayloadLeak(
                    f"payload key {key!r} holds a {type(item).__name__} inside a list; "
                    f"only scalars may be published"
                )
            if len(out) >= MAX_LIST_ITEMS:
                break
        return out or None
    if isinstance(value, dict):
        raise PayloadLeak(
            f"payload key {key!r} holds a dict. Nested containers are refused: the "
            f"live leak this replaces hid a second copy of the caller's phone number "
            f"inside a serialised _node_content blob. Flatten it at the call site."
        )
    # datetimes and ObjectIds arrive from Mongo; render them, do not guess.
    rendered = str(value).strip()
    return (rendered if opaque else scrub(rendered).strip()) or None


def project(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist ``unit`` down to a publishable Qdrant payload.

    ``unit`` may be a raw Mongo document, a hand-built candidate dict, or
    anything in between — every key absent from :data:`ALLOWED_KEYS` is dropped
    without being inspected, so passing more than necessary is safe by
    construction rather than by review.
    """
    payload: dict[str, Any] = {}
    for key in unit:
        if key not in ALLOWED_KEYS:
            continue
        coerced = _coerce(key, unit[key])
        if coerced is not None:
            payload[key] = coerced

    # training_safe is the server-side floor for every query. A payload without
    # it is invisible to a `training_safe = true` filter, which fails safe, but
    # silently — so make the absence explicit and false.
    payload["training_safe"] = bool(unit.get("training_safe", False))

    assert_no_leak(payload)
    return payload


# --- identity ----------------------------------------------------------------
def stable_unit_id(prefix: str, *parts: str) -> str:
    """Opaque, deterministic id derived from source identity, never content.

    Content hashes are unstable here: a cluster gains members as the analyzer
    adds documents, so a content-derived id changes under the same logical unit
    and idempotent upsert becomes impossible (CONTRACT.md, ``kb_units.unit_id``).
    """
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def unit_id_for_call(source_id: str) -> str:
    """Residual whole-call card. Derived from the Mongo ``_id``, not the filename."""
    return stable_unit_id("c", str(source_id))


def unit_id_for_case(case_id: str) -> str:
    return stable_unit_id("n", str(case_id))


def unit_id_for_platform_stage(platform_id: int, stage_id: int) -> str:
    return stable_unit_id("p", str(platform_id), str(stage_id))


def point_uuid(unit_id: str, role: str) -> str:
    """Point id = uuid5(NAMESPACE_URL, "<unit_id>:<role>").

    Deterministic so that re-running a build upserts over the previous point
    instead of duplicating it, which is what makes the whole projection
    idempotent and resumable.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{unit_id}:{role}"))


# --- text composition --------------------------------------------------------
@lru_cache(maxsize=1)
def _reverse_alias_map() -> dict[str, list[str]]:
    """canonical -> colloquial forms, mined from ``calls_analysis.search_aliases``.

    ``text.normalize.expand_aliases`` runs the *forward* direction, which is what
    a query needs: the customer says "flash", we also search "ecu tune". The
    query point needs the mirror image — the unit says "ecu tune", so the point
    should also be findable by someone who typed "flash".
    """
    path = DATA_DIR / "aliases.json"
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    for entry in json.loads(path.read_text()):
        out.setdefault(entry["canonical"].lower(), []).append(entry["colloquial"])
    return out


def colloquial_forms(text: str, *, limit: int = 12) -> list[str]:
    """Colloquialisms that map onto canonical terms occurring in ``text``."""
    low = (text or "").lower()
    out: list[str] = []
    for canonical, colloquials in _reverse_alias_map().items():
        if canonical and canonical in low:
            for form in colloquials:
                if form not in out:
                    out.append(form)
                if len(out) >= limit:
                    return out
    return out


def _joined(*parts: str) -> str:
    return "\n".join(p.strip() for p in parts if p and p.strip())


def answer_text(unit: Mapping[str, Any]) -> str:
    """The ANSWER point: what the unit actually says."""
    return _joined(
        str(unit.get("title") or ""),
        str(unit.get("question") or ""),
        str(unit.get("answer") or ""),
        str(unit.get("conditions") or ""),
    )


def query_text(unit: Mapping[str, Any]) -> str:
    """The QUERY point: the shapes a customer's question actually takes.

    ``hypothetical_questions`` is a HyDE corpus of ~186k entries written offline
    *with the answer in hand* (CONTRACT.md). Embedding it is free recall; asking
    an LLM to regenerate it per query would be the same information at 400 ms a
    turn.
    """
    question = str(unit.get("question") or "")
    hypotheticals = [str(q) for q in (unit.get("hypothetical_questions") or []) if q]
    seed = _joined(str(unit.get("title") or ""), question, *hypotheticals)
    extras = colloquial_forms(seed) + expand_aliases(seed)
    return _joined(question, *hypotheticals, " ".join(dict.fromkeys(extras)))
