"""Vehicle resolution and the stage-availability oracle, over ``tuning_platforms``.

Two facts shape this module.

*The agent does not need to pin the platform.* It needs to know when the
candidates disagree. Of 7,957 resolved vehicle entries in the corpus, 57% have
every candidate agreeing on ``max_released_stage`` -- so in the majority case
the ambiguity is free and asking a clarifying question is pure friction.
``resolve_vehicle`` therefore returns candidates *plus* ``agree_on_stages``,
and only emits a clarifying question when the answer actually turns on it.
Vehicle field completeness on sales calls is make 96%, model 96%, year 63%,
chassis 28%, engine 20%, which is why year is the field that most often
splits the set.

*``tuning_platforms._id`` is the platform id.* This was believed broken. It is
not: 93.4% of customer vehicle rows join straight onto it, and the misses are
``platform_id: null`` on the vehicle side -- unmatched rows, not dangling
references. So ``check_stage_availability`` is a primary-key read, not a search.

The freshness gate is the reason this is a tool and not a prompt. ``sync_tuning_db``
is unscheduled and Mac-pinned; a snapshot that is a fortnight old must say so
rather than answer confidently. Thresholds live in ``config`` because they were
tuned against measured staleness (5.8 days on 2026-08-05) -- a 48h *refuse*
gate would have parked the oracle in permanent refusal.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .. import config
from ..clients.mongo import source_db
from .base import (
    Degraded,
    Tool,
    ToolInputError,
    ToolPolicyError,
    obj_schema,
)

# Ported from the retriever this replaces. Without a ceiling a bare
# ("Volkswagen", "Golf") resolves to double-digit candidates and the model
# spends its whole context reciting engine names.
MAX_CANDIDATES = 8

# Marques in the table, plus the colloquial forms customers actually say. The
# table holds 9 makes over 106 platforms; a full fuzzy matcher here would be
# machinery for a nine-element lookup.
_MAKE_ALIASES = {
    "vw": "volkswagen",
    "volkswagon": "volkswagen",
    "vdub": "volkswagen",
    "dub": "volkswagen",
    "audi": "audi",
    "porsche": "porsche",
    "porshe": "porsche",
    "seat": "seat",
    "skoda": "skoda",
    "škoda": "skoda",
    "cupra": "cupra",
    "lambo": "lamborghini",
    "lamborghini": "lamborghini",
    "bentley": "bentley",
    "opel": "opel",
    "vauxhall": "opel",
}

# Chassis and engine-family markers are not their own column -- they live in
# `engine_name`, `description` and `model_names` as free text. Matching them is
# a token containment test over that concatenation, which is exact enough
# because the vocabulary is closed and short.
_CHASSIS_RE = re.compile(
    r"\b(mk[1-8]|mqb(?:\s*evo)?|mlb(?:\s*evo)?|pq3[45]|pq46|b[5-9]|c[5-8]|"
    r"8[jlnpvy]|8[svp]|9[0-9]{2}|d[3-5]|4[fgmn]|5[fq]|3[cg]|1k|6[cr]|typ\s*\w+)\b",
    re.IGNORECASE,
)
_DISPLACEMENT_RE = re.compile(r"\b(\d)[.,](\d)\s*(?:l|t|litre|liter)?\b", re.IGNORECASE)
_ENGINE_FAMILY_RE = re.compile(
    r"\b(ea113|ea888|ea839|ea825|ea211|ea390|evo\s*[0-9]|gen\s*[0-9]|"
    r"tsi|tfsi|tdi|fsi|mpi|vr6|dsg|vp37|pd)\b",
    re.IGNORECASE,
)

_WS = re.compile(r"[^a-z0-9+]+")

# Rank order for max_released_stage comparison. `rank` exists per stage inside
# the platform doc but `max_released_stage` is a bare label, so the labels get
# their own ordering rather than a join back through `stages`.
_STAGE_ORDER = {
    "stock": 0.0,
    "stage 1": 1.0,
    "stage 1+": 1.5,
    "stage 2": 2.0,
    "stage 2+": 2.5,
    "stage 3": 3.0,
    "stage 3+": 3.5,
    "stage 4": 4.0,
}


def _norm(text: Any) -> str:
    if not text:
        return ""
    return _WS.sub(" ", str(text).lower()).strip()


def _tokens(text: Any) -> set[str]:
    return {t for t in _norm(text).split() if t}


def _canonical_make(make: str) -> str:
    n = _norm(make)
    return _MAKE_ALIASES.get(n, n)


def _canonical_stage(stage: str) -> str:
    """``stage2+`` / ``Stage 2 Plus`` / ``s2+`` -> ``stage 2+``."""
    n = _norm(stage).replace("plus", "+")
    n = re.sub(r"\bs(?=[0-9])", "stage ", n)
    n = re.sub(r"\bstage\s*", "stage ", n)
    n = re.sub(r"\s*\+\s*", "+", n)
    n = re.sub(r"\s+", " ", n).strip()
    if n in ("stock", "oem", "original"):
        return "stock"
    m = re.match(r"stage\s*([0-4])\s*(\+?)", n)
    if m:
        return f"stage {m.group(1)}{m.group(2)}"
    return n


def _year_band(platform: dict) -> tuple[int | None, int | None]:
    def _i(v: Any) -> int | None:
        try:
            return int(str(v).strip()[:4])
        except (TypeError, ValueError):
            return None

    return _i(platform.get("year_start")), _i(platform.get("year_end"))


# --- resolve_vehicle ---------------------------------------------------------


def _score_platform(
    platform: dict,
    *,
    make: str,
    model: str,
    year: int | None,
    engine: str | None,
    chassis: str | None,
) -> tuple[float, list[str], dict]:
    """Additive score plus the fields that contributed, for explainability."""
    matched: list[str] = []
    detail: dict[str, Any] = {}
    score = 0.0

    makes = {_canonical_make(m) for m in platform.get("makes") or []}
    if make and make not in makes:
        return 0.0, [], {}
    if make:
        matched.append("make")
        score += 1.0

    model_tokens = _tokens(model)
    best_model = None
    best_overlap = 0.0
    for candidate in platform.get("model_names") or []:
        ct = _tokens(candidate)
        if not ct or not model_tokens:
            continue
        overlap = len(ct & model_tokens)
        if not overlap:
            continue
        # Jaccard, not overlap-over-max: "Golf R" against "Golf SportWagen"
        # shares one token out of three, and scoring that as half a match puts
        # a wagon above the actual car. Exact normalised equality is boosted
        # past 1.0 so it outranks an in-band year on a wrong model.
        ratio = overlap / len(ct | model_tokens)
        if _norm(candidate) == _norm(model):
            ratio = 1.75
        if ratio > best_overlap:
            best_overlap, best_model = ratio, candidate
    if model_tokens and best_model is None:
        return 0.0, [], {}
    if best_model:
        matched.append("model")
        score += 2.0 * best_overlap
        detail["matched_model"] = best_model

    y0, y1 = _year_band(platform)
    if year is not None:
        if y0 is not None and y1 is not None and y0 <= year <= y1:
            matched.append("year")
            score += 2.0
            detail["year_in_band"] = True
        else:
            # Do not eliminate. `year_end` is a snapshot of what has been
            # released, so a current-model-year car legitimately falls past
            # the end of its own platform's band.
            detail["year_in_band"] = False
            score -= 0.75

    haystack = " ".join(
        [
            _norm(platform.get("engine_name")),
            _norm(platform.get("platform_code")),
            _norm(platform.get("description")),
            " ".join(_norm(m) for m in platform.get("model_names") or []),
        ]
    )

    if engine:
        hits = 0
        for m in _DISPLACEMENT_RE.finditer(engine):
            if f"{m.group(1)} {m.group(2)}" in haystack or f"{m.group(1)}{m.group(2)}" in haystack.replace(".", ""):
                hits += 1
        for m in _ENGINE_FAMILY_RE.finditer(engine):
            if _norm(m.group(1)) in haystack:
                hits += 1
        if hits:
            matched.append("engine")
            score += min(2.0, 0.8 * hits)
            detail["engine_hits"] = hits

    if chassis:
        found = [m.group(1).lower() for m in _CHASSIS_RE.finditer(chassis)]
        if any(c in haystack for c in found):
            matched.append("chassis")
            score += 1.5
            detail["chassis_hits"] = [c for c in found if c in haystack]

    return score, matched, detail


def _candidate_view(platform: dict, score: float, matched: list[str], detail: dict) -> dict:
    y0, y1 = _year_band(platform)
    return {
        "platform_id": platform["_id"],
        "engine_name": platform.get("engine_name"),
        "platform_code": platform.get("platform_code") or None,
        "year_start": y0,
        "year_end": y1,
        "makes": platform.get("makes") or [],
        "matched_model": detail.get("matched_model"),
        "max_released_stage": platform.get("max_released_stage"),
        "released_stage_labels": platform.get("released_stage_labels") or [],
        "unreleased_stage_labels": platform.get("unreleased_stage_labels") or [],
        "is_dsg_platform": bool(platform.get("is_dsg_platform")),
        "uniflex_supported": bool(platform.get("uniflex_supported")),
        "match_score": round(score, 3),
        "matched_on": matched,
        "year_in_band": detail.get("year_in_band"),
    }


def _clarifying_question(
    candidates: list[dict], *, have_year: bool, have_engine: bool, have_chassis: bool
) -> tuple[str | None, str | None]:
    """The single question worth asking, or None.

    Chosen by which missing field actually partitions the surviving candidate
    set. Asking for a chassis code when every candidate shares one is how a
    qualification flow loses a customer at question three.
    """
    if len(candidates) < 2:
        return None, None
    if not have_year:
        bands = {(c["year_start"], c["year_end"]) for c in candidates}
        if len(bands) > 1:
            return "What model year is the vehicle?", "year"
    if not have_engine:
        engines = {c["engine_name"] for c in candidates}
        if len(engines) > 1:
            listed = ", ".join(sorted(e for e in engines if e)[:4])
            return (
                f"Which engine does it have? I have {listed} on that model.",
                "engine",
            )
    if not have_chassis:
        codes = {c["matched_model"] for c in candidates}
        if len(codes) > 1:
            return (
                "Which exact trim or chassis is it? "
                + ", ".join(sorted(c for c in codes if c)[:4]),
                "chassis",
            )
    return None, None


def resolve_vehicle(
    make: str,
    model: str,
    year: int | None = None,
    engine: str | None = None,
    chassis: str | None = None,
) -> dict:
    if not _norm(make) and not _norm(model):
        raise ToolInputError("resolve_vehicle needs at least a make or a model")

    canon_make = _canonical_make(make) if make else ""
    if make and canon_make not in {
        "volkswagen", "audi", "porsche", "seat", "skoda", "cupra",
        "lamborghini", "bentley", "opel",
    }:
        # Not a marque Unitronic tunes at all. Saying so is a better answer
        # than an empty candidate list the model will try to explain away.
        return {
            "query": {"make": make, "model": model, "year": year},
            "candidates": [],
            "candidate_count": 0,
            "agree_on_stages": None,
            "clarifying_question": None,
            "unsupported_make": True,
            "note": f"{make} is not a marque in the Unitronic platform table.",
        }

    scored: list[tuple[float, dict, list[str], dict]] = []
    for platform in source_db()[config.COLL_PLATFORMS].find({}):
        score, matched, detail = _score_platform(
            platform,
            make=canon_make,
            model=model or "",
            year=year,
            engine=engine,
            chassis=chassis,
        )
        if score > 0:
            scored.append((score, platform, matched, detail))

    scored.sort(key=lambda row: (-row[0], row[1]["_id"]))
    total = len(scored)
    top = scored[:MAX_CANDIDATES]
    candidates = [_candidate_view(p, s, m, d) for s, p, m, d in top]

    stages = {c["max_released_stage"] for c in candidates}
    agree = None
    agreed = None
    if candidates:
        agree = len(stages) == 1
        if agree:
            agreed = candidates[0]["max_released_stage"]
        else:
            # Even when they disagree, the *floor* is safe to state: every
            # candidate supports at least this much.
            known = [s for s in stages if s]
            if known and len(known) == len(stages):
                agreed = min(known, key=lambda s: _STAGE_ORDER.get(_canonical_stage(s), 99))

    question, on_field = _clarifying_question(
        candidates,
        have_year=year is not None,
        have_engine=bool(engine),
        have_chassis=bool(chassis),
    )

    out = {
        "query": {
            "make": make,
            "model": model,
            "year": year,
            "engine": engine,
            "chassis": chassis,
        },
        "candidate_count": total,
        "truncated": total > MAX_CANDIDATES,
        "candidates": candidates,
        "agree_on_stages": agree,
        "agreed_max_released_stage": agreed if agree else None,
        "lowest_common_max_stage": None if agree else agreed,
        "clarifying_question": question,
        "clarify_on": on_field,
    }
    if year is not None and candidates and not any(c["year_in_band"] for c in candidates):
        out["note"] = (
            f"No platform band covers {year}. The table's year_end reflects what has "
            f"been released, so a current-model-year car can legitimately fall past it. "
            f"Confirm with check_stage_availability before quoting."
        )
    return out


# --- check_stage_availability ------------------------------------------------


def _sync_age_hours() -> tuple[float | None, str | None]:
    doc = source_db()[config.COLL_SYNC_STATE].find_one({"_id": "last_run"})
    if not doc:
        return None, None
    raw = doc.get("finished_at") or doc.get("started_at")
    if not raw:
        return None, None
    if isinstance(raw, datetime):
        when = raw
    else:
        try:
            when = datetime.fromisoformat(str(raw))
        except ValueError:
            return None, str(raw)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - when).total_seconds() / 3600.0
    return age, when.isoformat()


def _stage_view(stage: dict) -> dict:
    return {
        "label": stage.get("label"),
        "raw": stage.get("raw"),
        "variant": stage.get("variant") or None,
        "stage_id": stage.get("stage_id"),
        "rank": stage.get("rank"),
        "released": bool(stage.get("released")),
        "files_available": stage.get("files_available"),
        "files_total": stage.get("files_total"),
        "newest_available_at": stage.get("newest_available_at"),
        "power_figures": [
            {
                "reference": pf.get("reference"),
                "power_gain": pf.get("power_gain"),
                "fuel": pf.get("fuel") or None,
            }
            for pf in (stage.get("power_figures") or [])[:6]
        ],
    }


def check_stage_availability(platform_id: int, stage: str | None = None) -> dict | Degraded:
    age_h, synced_at = _sync_age_hours()
    if age_h is not None and age_h > config.PLATFORM_STALE_REFUSE_HOURS:
        raise ToolPolicyError(
            f"The platform table was last synced {age_h/24:.1f} days ago "
            f"(refuse threshold {config.PLATFORM_STALE_REFUSE_HOURS/24:.0f} days). "
            f"Stage availability cannot be quoted from a snapshot this old -- "
            f"escalate to a human or re-run sync_tuning_db."
        )

    platform = source_db()[config.COLL_PLATFORMS].find_one({"_id": platform_id})
    if not platform:
        raise ToolInputError(
            f"No platform with id {platform_id}. Platform ids come from "
            f"resolve_vehicle or from a customer vehicle record; they are not guessable."
        )

    all_stages = [s for s in platform.get("stages") or [] if not s.get("is_stock")]
    y0, y1 = _year_band(platform)
    data: dict[str, Any] = {
        "platform_id": platform["_id"],
        "engine_name": platform.get("engine_name"),
        "platform_code": platform.get("platform_code") or None,
        "makes": platform.get("makes") or [],
        "model_names": (platform.get("model_names") or [])[:25],
        "year_start": y0,
        "year_end": y1,
        "is_dsg_platform": bool(platform.get("is_dsg_platform")),
        "uniflex_supported": bool(platform.get("uniflex_supported")),
        "max_released_stage": platform.get("max_released_stage"),
        "released_stage_labels": platform.get("released_stage_labels") or [],
        "unreleased_stage_labels": platform.get("unreleased_stage_labels") or [],
        "platform_note": (platform.get("description") or "").strip()[:600] or None,
        "synced_at": synced_at,
        "sync_age_hours": None if age_h is None else round(age_h, 1),
    }

    if stage:
        want = _canonical_stage(stage)
        matches = [s for s in all_stages if _canonical_stage(s.get("label") or "") == want]
        released = [s for s in matches if s.get("released")]
        data["requested_stage"] = stage
        data["requested_stage_canonical"] = want
        data["available"] = bool(released)
        data["variants"] = [_stage_view(s) for s in matches]
        if not matches:
            data["available"] = False
            data["reason"] = (
                f"{want!r} does not exist on this platform at all "
                f"(released here: {data['released_stage_labels']})"
            )
        elif not released:
            data["reason"] = f"{want!r} exists on this platform but is not released."
        else:
            # files_available is the honest measure. A stage can be flagged
            # released while every file for the customer's specific ECU is
            # missing, which is the failure mode that produces a support call.
            data["files_available"] = sum(s.get("files_available") or 0 for s in released)
            data["files_total"] = sum(s.get("files_total") or 0 for s in released)
            if not data["files_available"]:
                data["available"] = False
                data["reason"] = (
                    f"{want!r} is flagged released but has 0 files available -- "
                    f"treat as unavailable and check with a human."
                )
    else:
        data["stages"] = [_stage_view(s) for s in all_stages]

    if age_h is not None and age_h > config.PLATFORM_STALE_WARN_HOURS:
        return Degraded(
            data,
            f"platform table last synced {age_h/24:.1f} days ago "
            f"(warn threshold {config.PLATFORM_STALE_WARN_HOURS/24:.0f} days); "
            f"state the availability as 'as of {synced_at}', not as current",
        )
    return data


# --- Tool definitions --------------------------------------------------------

TOOLS = [
    Tool(
        name="resolve_vehicle",
        description=(
            "Resolve a customer's vehicle to Unitronic tuning platform candidates. "
            "Returns every plausible platform, whether they all agree on the highest "
            "released stage, and the one clarifying question worth asking if they do "
            "not. Call this before check_stage_availability. Supported marques: "
            "Volkswagen, Audi, Porsche, Seat, Skoda, CUPRA, Lamborghini, Bentley, Opel."
        ),
        parameters=obj_schema(
            {
                "make": {"type": "string", "description": "e.g. Volkswagen, Audi, VW"},
                "model": {"type": "string", "description": "e.g. Golf R, S3, GTI, Macan"},
                "year": {"type": "integer", "description": "Model year, e.g. 2018"},
                "engine": {
                    "type": "string",
                    "description": "Engine as the customer says it, e.g. '2.0T EA888 Gen3', '3.0 TDI'",
                },
                "chassis": {
                    "type": "string",
                    "description": "Chassis or platform code, e.g. MK7, MQB, B8, 8V",
                },
            },
            ["make", "model"],
        ),
        handler=resolve_vehicle,
        dependency="mongo",
    ),
    Tool(
        name="check_stage_availability",
        description=(
            "Authoritative check of which tuning stages exist and are released for a "
            "platform id, including how many calibration files are actually available. "
            "This is the only acceptable source for a stage-availability claim; never "
            "answer one from memory or from retrieved call text. platform_id comes from "
            "resolve_vehicle or from a customer's own vehicle record."
        ),
        parameters=obj_schema(
            {
                "platform_id": {
                    "type": "integer",
                    "description": "Unitronic platform id, from resolve_vehicle",
                },
                "stage": {
                    "type": "string",
                    "description": "Optional single stage to check, e.g. 'Stage 2+'. Omit for all stages.",
                },
            },
            ["platform_id"],
        ),
        handler=check_stage_availability,
        dependency="mongo",
    ),
]


def self_check() -> None:
    import json

    print("--- resolve_vehicle('Volkswagen', 'Golf R', 2018) ---")
    r = resolve_vehicle("Volkswagen", "Golf R", 2018)
    print(json.dumps(r, indent=1, default=str))
    assert r["candidate_count"] >= 1
    assert all(c["platform_id"] for c in r["candidates"])
    assert len(r["candidates"]) <= MAX_CANDIDATES

    print("--- resolve_vehicle('VW', 'Golf') no year ---")
    r2 = resolve_vehicle("VW", "Golf")
    print(
        json.dumps(
            {
                "candidate_count": r2["candidate_count"],
                "truncated": r2["truncated"],
                "agree_on_stages": r2["agree_on_stages"],
                "clarifying_question": r2["clarifying_question"],
                "clarify_on": r2["clarify_on"],
                "platform_ids": [c["platform_id"] for c in r2["candidates"]],
            },
            indent=1,
        )
    )
    assert len(r2["candidates"]) <= MAX_CANDIDATES

    print("--- resolve_vehicle('Toyota', 'Supra') ---")
    print(json.dumps(resolve_vehicle("Toyota", "Supra"), indent=1))

    pid = r["candidates"][0]["platform_id"]
    print(f"--- check_stage_availability({pid}, 'Stage 2') ---")
    out = check_stage_availability(pid, "Stage 2")
    if isinstance(out, Degraded):
        print("DEGRADED:", out.reason)
        out = out.data
    print(json.dumps(out, indent=1, default=str))
    assert out["platform_id"] == pid

    print(f"--- check_stage_availability({pid}) all stages ---")
    allout = check_stage_availability(pid)
    body = allout.data if isinstance(allout, Degraded) else allout
    print("stage labels:", [s["label"] for s in body["stages"]])
    print("sync_age_hours:", body["sync_age_hours"])

    print("--- check_stage_availability(999999) must raise ---")
    try:
        check_stage_availability(999999)
        raise AssertionError("bad platform id not rejected")
    except ToolInputError as exc:
        print("ToolInputError:", exc)

    assert _canonical_stage("stage2+") == "stage 2+"
    assert _canonical_stage("Stage 1 Plus") == "stage 1+"
    print("vehicle.py self-check OK")


if __name__ == "__main__":
    self_check()
