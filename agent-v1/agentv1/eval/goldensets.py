"""Golden sets built from data that already exists. Nothing here is annotated.

Every one of the five sets is a projection of a field the analyzer already
wrote with the full transcript in hand. That matters more than it sounds:
hand-annotating an eval set for this domain would need a tuning technician, and
the set would be ~200 rows. These are 5,776 / 2,471 / 563 / 268 rows and they
were labelled by someone who could hear the call.

The five sets, and what each one actually tests:

``known_gaps``
    ``calls_analysis.agent_unanswered_questions`` -- questions a *human* agent
    could not answer on the call. They are absent from the corpus **by
    construction**: the analyzer only records a question here when the call
    ended without an answer, so no downstream knowledge unit contains one. This
    is the hard-negative set, and the only correct behaviour is to say so.
    Measured category mix (deterministic keyword rules, see ``classify_gap``):
    hardware spec/fitment 21.6%, software compatibility 24.8% -- the union,
    "will this work with that", is **46.4%**, which is where the ~46% figure
    quoted in the plan comes from. There is no data source anywhere in this
    system that answers it: ``tuning_platforms`` knows stages per platform, not
    whether a CTS K04X hybrid turbo clears the charge pipe.

``ground_truth``
    ``calls_cases.case_knowledge_units`` restricted to cases that carry
    ``what_finally_worked``. The distinction this set exists to measure is
    *plausible* vs *what actually worked* -- so each row also carries the
    failed attempts from ``attempts[]`` as named distractors. An answer that
    recommends a documented failed attempt scores zero even though it will read
    perfectly well.

``must_never_say``
    ``calls_analysis.review.incorrect_statements``. Query it as
    ``{'review.incorrect_statements.0': {'$exists': True}}`` -> 535 docs / 563
    items. Queried as a boolean (``{'review.incorrect_statements': True}``) it
    returns **0**, because Mongo compares the array to the literal ``true``.
    Any dashboard written that way is silently reporting zero wrong statements
    on a corpus that has 563 of them.

``emissions``
    ``review.emissions_tampering_request: true``, 268 calls. Refusal recall.
    Human baseline measured on this same set: 199 ``refused_correctly`` /
    268 = **74.3%**, i.e. the ~75% in the plan. 32 calls are
    ``complied_improperly``. Beating a 74% human baseline is the low bar; the
    gate is that this number never goes down.

``french``
    The ``language == "fr"`` slice of the four sets above, joined for cases via
    ``file_name`` -> member filename (read-only; the filename never leaves this
    module -- it embeds the caller's phone number). It is deliberately the same
    four axes and not a fifth kind of question, so the French and English
    numbers are comparable. ANALYZER.md records an unexplained 11.7-point
    French/English scoring gap; until that is explained, French is reported as
    its own column and parity is never assumed. Note the analyzer's own output
    is English even on French calls, so the *reference* answers here are
    English while the *expected response language* is French -- the judge is
    told about this explicitly rather than penalising the mismatch.

PII: source docs carry ``file_name`` / ``basename`` / ``members[]`` (3CX WAV
names embedding the caller's phone), ``caller_phone_number``, ``phone_key``,
``agent_name``, ``caller_area_code`` and ``vehicle_context.name`` /
``.customer_id`` / ``.match_key``. This module is an allowlist -- fields are
copied in by name, never by projection-minus-blocklist -- and then every string
is scrubbed and every finished record is re-scanned before it is written. Two
independent mechanisms, because the allowlist protects against fields we know
about and the scan protects against free text that quotes a phone number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .. import config
from ..clients.mongo import source_db
from ..text.normalize import strip_pii_markers

EVAL_DATA_DIR = config.REPO_ROOT / "eval_data"

SET_NAMES = ("known_gaps", "ground_truth", "must_never_say", "emissions", "french")

# What a correct response looks like, per set. The judge prompt is selected off
# this, so it is data rather than a branch in the runner.
BEHAVIOUR_ABSTAIN = "abstain"
BEHAVIOUR_MATCH = "match_verified_outcome"
BEHAVIOUR_AVOID = "avoid_incorrect_claim"
BEHAVIOUR_REFUSE = "refuse"


# --- PII defence -------------------------------------------------------------
# strip_pii_markers() rewrites the known shapes; this second pass is the
# assertion that it worked. A record that trips it is a bug in the allowlist,
# not something to filter out silently, so it raises.
# Canadian postal codes are not covered by text/normalize.strip_pii_markers and
# do occur in this data: two `attempts[]` rows describe using a specific postal
# code to get a better UPS rate. A postal code plus a vehicle is close enough to
# an identifier in a customer base this size, so it is rewritten here rather
# than in normalize.py, which the retrieval path shares and which is owned
# elsewhere.
_POSTAL_CA = re.compile(r"(?<![\w])[A-Z]\d[A-Z][ \-]?\d[A-Z]\d(?![\w])")

_RESIDUAL_PII = (
    ("phone", re.compile(r"(?<![\[\w])\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}(?![\w\]])")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("wav_filename", re.compile(r"\.wav\b", re.IGNORECASE)),
    ("dm_handle", re.compile(r"inbox/\S+")),
    ("postal_ca", _POSTAL_CA),
)

# Fields that must never be copied out of a source doc, listed explicitly so
# that adding one to the allowlist below trips review rather than slipping in.
FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "file_name",
        "file_path",
        "filename",
        "basename",
        "members",
        "sessions",
        "withheld_members",
        "supersedes_calls",
        "caller_phone_number",
        "caller_area_code",
        "caller_identity",
        "customer_name_mentioned",
        "agent_name",
        "phone_key",
        "full_transcription",
        "customer_match",
        "urls_or_emails_mentioned",
        "location_mentioned",
        "locations_mentioned",
    }
)


def scrub(value: Any) -> Any:
    """Recursively apply the PII rewrites to every string in a structure."""
    if isinstance(value, str):
        return _POSTAL_CA.sub("[POSTAL]", strip_pii_markers(value))
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def assert_pii_free(record: dict) -> None:
    """Raise if any string in ``record`` still matches a known PII shape."""
    hits: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            for label, rx in _RESIDUAL_PII:
                if rx.search(node):
                    hits.append(f"{path}: {label}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, dict):
            for k, v in node.items():
                if k in FORBIDDEN_SOURCE_FIELDS:
                    hits.append(f"{path}.{k}: forbidden field")
                walk(v, f"{path}.{k}")

    walk(record, record.get("example_id", "?"))
    if hits:
        raise ValueError(f"PII survived into golden-set record: {hits[:5]}")


# --- Gap taxonomy ------------------------------------------------------------
# First match wins, so the order is the taxonomy. Hardware is checked before
# software because "does the K04X fit" and "is the K04X supported" are the same
# customer question and the hardware wording is the more specific signal.
# Deterministic on purpose: a categoriser that drifts makes the per-category
# abstention rates incomparable across runs, which is the only thing they are
# for.
_GAP_RULES: tuple[tuple[str, str], ...] = (
    (
        "hardware_spec_fitment",
        r"\b(fit|fits|fitment|bolt[- ]?on|clearance|thread|torque|dimension|diameter|"
        r"size|sizing|length|width|spec|specification|part number|hardware|bracket|"
        r"hose|clamp|bolt|gasket|flange|inlet|outlet|piping|intercooler|downpipe|"
        r"turbo|injector|pulley|manifold|charge pipe|catch can|exhaust|intake|"
        r"coilover|clutch|wheel|adapter|physical|material|weight|cast|billet)\b",
    ),
    (
        "availability_eta",
        r"\b(when will|release date|eta|timeline|available|availability|in development|"
        r"coming|back ?in stock|stock|restock|lead time|when is|when are)\b",
    ),
    (
        "pricing_promo",
        r"\b(price|pricing|cost|how much|discount|promo|sale|rebate|credit|"
        r"refund amount|msrp|quote)\b",
    ),
    (
        "order_logistics",
        r"\b(order|shipping|ship|tracking|delivery|customs|duty|invoice|rma|return|"
        r"warranty claim|dealer|distributor)\b",
    ),
    (
        "software_compat",
        r"\b(software|firmware|driver|version|revision|update|compatib|support(ed|s)?|"
        r"works with|tune for|calibration|map|ecu|tcu|dsg|flash)\b",
    ),
    (
        "diagnostic_procedure",
        r"\b(why|cause|error|fault|code|diagnos|troubleshoot|fix|resolve|misfire|limp|"
        r"check engine|dtc|log|datalog|symptom)\b",
    ),
    (
        "policy_legal",
        r"\b(warranty|policy|legal|emission|carb|epa|inspection|insurance|liabilit|terms)\b",
    ),
)
_GAP_COMPILED = tuple((name, re.compile(pat, re.IGNORECASE)) for name, pat in _GAP_RULES)

# The two categories that together form "will this thing work with that thing",
# which is 46.4% of the gap set and has no data source in this system.
_FITMENT_FAMILY = frozenset({"hardware_spec_fitment", "software_compat"})


def classify_gap(question: str) -> str:
    for name, rx in _GAP_COMPILED:
        if rx.search(question):
            return name
    return "other"


def gap_family(category: str) -> str:
    return "compatibility_fitment" if category in _FITMENT_FAMILY else category


# --- Record construction -----------------------------------------------------


def _example_id(set_name: str, source_id: str, ordinal: int) -> str:
    """Stable across rebuilds so run-over-run comparison survives re-materialising.

    Keyed on (set, source doc, ordinal) rather than on question text, because
    the analyzer rewrites its own output when it re-analyses a call and a
    content hash would silently fork the row.
    """
    h = hashlib.sha1(f"{set_name}|{source_id}|{ordinal}".encode()).hexdigest()
    return f"{set_name[:2]}_{h[:16]}"


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat()
    if isinstance(value, str) and value:
        return value
    return None


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _record(
    set_name: str,
    source_id: str,
    ordinal: int,
    *,
    question: str,
    behaviour: str,
    language: str,
    axis: str | None = None,
    **fields: Any,
) -> dict:
    rec = {
        "example_id": _example_id(set_name, source_id, ordinal),
        "set": set_name,
        "axis": axis or set_name,
        "question": question,
        "language": language,
        "expect": {
            "behaviour": behaviour,
            # French turns must answer in French even though the analyzer's
            # reference text is English. Kept as data so the judge prompt does
            # not have to special-case the set name.
            "answer_language": language,
        },
        "source": {"collection": None, "id": source_id, "ordinal": ordinal},
        **fields,
    }
    rec = scrub(rec)
    assert_pii_free(rec)
    return rec


# --- Set builders ------------------------------------------------------------
# Each builder is a generator over live Mongo. No set is small enough to be
# worth caching in memory and the largest (known_gaps) is 5,776 rows.


_ANALYSIS_FIELDS = {
    "_id": 1,
    "canonical_problem": 1,
    "problem": 1,
    "agent_unanswered_questions": 1,
    "review.incorrect_statements": 1,
    "review.agent_knowledge_gaps": 1,
    "review.emissions_tampering_request": 1,
    "review.emissions_handling": 1,
    "review.safety_issue": 1,
    "knowledge_units": 1,
    "language": 1,
    "secondary_language": 1,
    "department": 1,
    "technical_category": 1,
    "vehicle_make": 1,
    "vehicle_model": 1,
    "vehicle_year": 1,
    "vehicle_platform": 1,
    "products_mentioned": 1,
    "tune_stages": 1,
    "training_safe": 1,
    "useful_content": 1,
    "call_ts": 1,
    "outcome": 1,
    "resolution_status": 1,
}


def _vehicle(doc: dict) -> dict:
    """Vehicle identity, allowlisted. Never the customer's name or match key."""
    return {
        "year": doc.get("vehicle_year"),
        "make": _clean_str(doc.get("vehicle_make")),
        "model": _clean_str(doc.get("vehicle_model")),
        "platform": _clean_str(doc.get("vehicle_platform")),
    }


def _context(doc: dict) -> str:
    """The situation the question was asked in, for the model to answer against."""
    return _clean_str(doc.get("canonical_problem")) or _clean_str(doc.get("problem"))


def build_known_gaps(
    db=None, *, language: str | None = None, set_name: str = "known_gaps"
) -> Iterator[dict]:
    db = source_db() if db is None else db
    query: dict[str, Any] = {"agent_unanswered_questions.0": {"$exists": True}}
    if language:
        query["language"] = language
    for doc in db[config.COLL_ANALYSIS].find(query, _ANALYSIS_FIELDS):
        sid = str(doc["_id"])
        for i, raw in enumerate(doc.get("agent_unanswered_questions") or []):
            question = _clean_str(raw)
            if not question:
                continue
            category = classify_gap(question)
            rec = _record(
                set_name,
                sid,
                i,
                question=question,
                behaviour=BEHAVIOUR_ABSTAIN,
                language=doc.get("language") or "en",
                axis="known_gaps",
                context=_context(doc),
                gap_category=category,
                gap_family=gap_family(category),
                vehicle=_vehicle(doc),
                meta={
                    "department": _clean_str(doc.get("department")),
                    "technical_category": _clean_str(doc.get("technical_category")),
                    "training_safe": bool(doc.get("training_safe")),
                    "call_ts": _iso(doc.get("call_ts")),
                    # Present because the human could not answer either -- the
                    # analyzer's own gap note is the closest thing to a reason.
                    "agent_knowledge_gaps": [
                        _clean_str(g)
                        for g in (doc.get("review") or {}).get("agent_knowledge_gaps") or []
                    ][:3],
                },
            )
            rec["source"]["collection"] = config.COLL_ANALYSIS
            yield rec


_CASE_FIELDS = {
    "_id": 1,
    "case_id": 1,
    "case_knowledge_units": 1,
    "what_finally_worked": 1,
    "attempts": 1,
    "issue": 1,
    "issue_category": 1,
    "root_cause": 1,
    "root_cause_detail": 1,
    "final_resolution": 1,
    "training_safe": 1,
    "vehicle_context.vehicles.display_name": 1,
    "vehicle_context.vehicles.platform_name": 1,
    "case_resolution_status": 1,
    "first_call_ts": 1,
}


def _french_call_filenames(db) -> set[str]:
    """3CX filenames of French calls, used only as a join key.

    ``calls_cases`` has no language field; its ``members[]`` are WAV filenames
    that embed the caller's phone number. Reading them to *label* a case is
    fine -- the set is built here, consulted here, and never written anywhere.
    Nothing derived from it leaves this function except the boolean.
    """
    return {
        d["file_name"]
        for d in db[config.COLL_ANALYSIS].find({"language": "fr"}, {"file_name": 1})
        if d.get("file_name")
    }


def build_ground_truth(
    db=None, *, language: str | None = None, set_name: str = "ground_truth"
) -> Iterator[dict]:
    db = source_db() if db is None else db
    query = {
        "what_finally_worked": {"$nin": [None, ""]},
        "case_knowledge_units.0": {"$exists": True},
    }
    fields = dict(_CASE_FIELDS)
    fr_names: set[str] = set()
    if language == "fr":
        fr_names = _french_call_filenames(db)
        fields["members"] = 1

    for doc in db[config.COLL_CASES].find(query, fields):
        if language == "fr" and not any(m in fr_names for m in (doc.get("members") or [])):
            continue
        sid = str(doc["_id"])
        attempts = doc.get("attempts") or []
        # The whole point of this set: things that were tried and did not work
        # read exactly as plausibly as the thing that did.
        failed = [
            _clean_str(a.get("attempt"))
            for a in attempts
            if isinstance(a, dict) and a.get("result") in ("failed", "partial")
        ]
        worked = [
            _clean_str(a.get("attempt"))
            for a in attempts
            if isinstance(a, dict) and a.get("result") == "worked"
        ]
        vehicles = [
            _clean_str(v.get("display_name"))
            for v in ((doc.get("vehicle_context") or {}).get("vehicles") or [])
            if _clean_str(v.get("display_name"))
        ]
        for i, unit in enumerate(doc.get("case_knowledge_units") or []):
            question = _clean_str(unit.get("question"))
            if not question:
                continue
            rec = _record(
                set_name,
                sid,
                i,
                question=question,
                behaviour=BEHAVIOUR_MATCH,
                language="fr" if language == "fr" else "en",
                axis="ground_truth",
                context=_clean_str(doc.get("issue")),
                reference_answer=_clean_str(unit.get("answer")),
                verified_outcome=_clean_str(doc.get("what_finally_worked")),
                worked_attempts=[a for a in worked if a],
                distractors=[a for a in failed if a],
                vehicles=vehicles[:4],
                meta={
                    "case_id": _clean_str(doc.get("case_id")),
                    "kind": _clean_str(unit.get("kind")),
                    "conditions": _clean_str(unit.get("conditions")),
                    "confidence": _clean_str(unit.get("confidence")),
                    "evidence_contacts": unit.get("evidence_contacts"),
                    "issue_category": _clean_str(doc.get("issue_category")),
                    "root_cause": _clean_str(doc.get("root_cause")),
                    "training_safe": bool(doc.get("training_safe")),
                    "call_ts": _iso(doc.get("first_call_ts")),
                },
            )
            rec["source"]["collection"] = config.COLL_CASES
            yield rec


def build_must_never_say(
    db=None, *, language: str | None = None, set_name: str = "must_never_say"
) -> Iterator[dict]:
    db = source_db() if db is None else db
    # The array-index form. As a boolean this returns 0 -- see module docstring.
    query: dict[str, Any] = {"review.incorrect_statements.0": {"$exists": True}}
    if language:
        query["language"] = language
    for doc in db[config.COLL_ANALYSIS].find(query, _ANALYSIS_FIELDS):
        sid = str(doc["_id"])
        review = doc.get("review") or {}
        units = doc.get("knowledge_units") or []
        # Prefer the analyzer's own question for the call: it is what a customer
        # would actually type. canonical_problem is the fallback and exists on
        # all 535.
        unit_q = _clean_str(units[0].get("question")) if units else ""
        question = unit_q or _context(doc)
        if not question:
            continue
        for i, raw in enumerate(review.get("incorrect_statements") or []):
            claim = _clean_str(raw)
            if not claim:
                continue
            rec = _record(
                set_name,
                sid,
                i,
                question=question,
                behaviour=BEHAVIOUR_AVOID,
                language=doc.get("language") or "en",
                axis="must_never_say",
                context=_context(doc),
                forbidden_claim=claim,
                # What the corpus says is true, where the analyzer captured it.
                corrected_answer=_clean_str(units[0].get("answer")) if units else "",
                vehicle=_vehicle(doc),
                meta={
                    "department": _clean_str(doc.get("department")),
                    "technical_category": _clean_str(doc.get("technical_category")),
                    "training_safe": bool(doc.get("training_safe")),
                    "call_ts": _iso(doc.get("call_ts")),
                },
            )
            rec["source"]["collection"] = config.COLL_ANALYSIS
            yield rec


def build_emissions(
    db=None, *, language: str | None = None, set_name: str = "emissions"
) -> Iterator[dict]:
    db = source_db() if db is None else db
    query: dict[str, Any] = {"review.emissions_tampering_request": True}
    if language:
        query["language"] = language
    for doc in db[config.COLL_ANALYSIS].find(query, _ANALYSIS_FIELDS):
        sid = str(doc["_id"])
        review = doc.get("review") or {}
        question = _context(doc)
        if not question:
            continue
        handling = _clean_str(review.get("emissions_handling"))
        rec = _record(
            set_name,
            sid,
            0,
            question=question,
            behaviour=BEHAVIOUR_REFUSE,
            language=doc.get("language") or "en",
            axis="emissions",
            context=_context(doc),
            # The human's own outcome on this exact call. The set carries its
            # own baseline so a run never has to be compared against a number
            # from a document.
            human_handling=handling,
            human_refused=handling == "refused_correctly",
            vehicle=_vehicle(doc),
            meta={
                "department": _clean_str(doc.get("department")),
                "technical_category": _clean_str(doc.get("technical_category")),
                "safety_issue": bool(review.get("safety_issue")),
                "training_safe": bool(doc.get("training_safe")),
                "call_ts": _iso(doc.get("call_ts")),
            },
        )
        rec["source"]["collection"] = config.COLL_ANALYSIS
        yield rec


def build_french(db=None) -> Iterator[dict]:
    """The fr slice of the other four axes, tagged with the axis it came from.

    Same four questions, different language -- which is the only way the
    11.7-point gap in ANALYZER.md can be attributed to language rather than to
    a different question mix.
    """
    db = source_db() if db is None else db
    for builder in (build_known_gaps, build_must_never_say, build_emissions):
        yield from builder(db, language="fr", set_name="french")
    yield from build_ground_truth(db, language="fr", set_name="french")


BUILDERS = {
    "known_gaps": build_known_gaps,
    "ground_truth": build_ground_truth,
    "must_never_say": build_must_never_say,
    "emissions": build_emissions,
    "french": build_french,
}


# --- Materialisation ---------------------------------------------------------


def path_for(set_name: str, out_dir: Path | None = None) -> Path:
    return (out_dir or EVAL_DATA_DIR) / f"{set_name}.jsonl"


def materialise(
    set_name: str, *, out_dir: Path | None = None, limit: int | None = None, db=None
) -> dict:
    """Write one set to JSONL and return its stats. Overwrites atomically-ish.

    Written via a temp file and renamed so a crash mid-build cannot leave a
    half-set that a runner would happily evaluate against and report a number
    for.
    """
    if set_name not in BUILDERS:
        raise KeyError(f"unknown set {set_name!r}; known: {sorted(BUILDERS)}")
    out = path_for(set_name, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")

    n = 0
    by_lang: Counter[str] = Counter()
    by_axis: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    sources: set[str] = set()
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in BUILDERS[set_name](db):
            if limit is not None and n >= limit:
                break
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
            by_lang[rec["language"]] += 1
            by_axis[rec["axis"]] += 1
            sources.add(rec["source"]["id"])
            if "gap_category" in rec:
                by_category[rec["gap_category"]] += 1
    tmp.replace(out)

    stats = {
        "set": set_name,
        "path": str(out),
        "rows": n,
        "source_docs": len(sources),
        "by_language": dict(by_lang),
        "by_axis": dict(by_axis),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if by_category:
        stats["by_gap_category"] = dict(by_category.most_common())
        # Denominator is the categorised rows, not the whole set: in `french`
        # only the known_gaps axis carries a category, and dividing by 723
        # would report a share of a population that was never classified.
        categorised = sum(by_category.values())
        fam = sum(v for k, v in by_category.items() if k in _FITMENT_FAMILY)
        stats["categorised_rows"] = categorised
        stats["compatibility_fitment_share"] = round(fam / categorised, 4)
    return stats


def materialise_all(*, out_dir: Path | None = None, limit: int | None = None) -> list[dict]:
    db = source_db()
    return [materialise(name, out_dir=out_dir, limit=limit, db=db) for name in SET_NAMES]


def load(set_name: str, *, out_dir: Path | None = None) -> list[dict]:
    """Read a materialised set. Builds it first if it is missing."""
    p = path_for(set_name, out_dir)
    if not p.exists():
        materialise(set_name, out_dir=out_dir)
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_sets(names: Iterable[str]) -> Iterator[tuple[str, list[dict]]]:
    for name in names:
        yield name, load(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Materialise the eval golden sets.")
    ap.add_argument("--set", default="all", help=f"one of {', '.join(SET_NAMES)}, or all")
    ap.add_argument("--out", default=str(EVAL_DATA_DIR))
    ap.add_argument("--limit", type=int, default=None, help="cap rows per set (smoke runs)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    names = SET_NAMES if args.set == "all" else (args.set,)
    db = source_db()
    total = 0
    for name in names:
        stats = materialise(name, out_dir=out_dir, limit=args.limit, db=db)
        total += stats["rows"]
        print(json.dumps(stats, ensure_ascii=False))
    print(json.dumps({"total_rows": total, "sets": len(names)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
