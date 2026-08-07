#!/usr/bin/env python3
"""Enumerate every live Qdrant collection and scan sampled payloads for PII.

AGENT_PLAN.md §3.1 is explicit that a phone-only test is theatre: it passes on
``unitronic_customer_service_training_0_6b`` while that collection still ships
``thread_path = "inbox/<instagram handle>"`` on essentially every point. So the
pattern set here covers handles, emails, VINs, order numbers and staff names as
well as phone runs, and every text is scanned in **two loci** -- the payload
keys themselves, and the ``_node_content`` string, where LlamaIndex serialised
the identical metadata dict a second time. Dropping a field from the writer's
``create_metadata()`` fixes only the first locus; that is precisely why §3.1
concludes the collection has to be deleted rather than patched.

Two false-positive sources are designed out, because an audit that cries wolf
gets muted:

* Point ids and ``doc_id``/``ref_doc_id`` UUIDs contain hyphenated digit runs
  that a naive dashed-phone regex reads as phone numbers. UUIDs and long
  hex/digit runs are masked before value scanning.
* Instagram thread ids are 17 digits, exactly a VIN's length. VIN candidates
  must pass the ISO 3779 check digit (or sit next to the literal word "VIN")
  before they are reported.

The inverse bias is also handled: ``866.341.2447`` is Unitronic's published
support line and appears ~900 times per sampled page of the product corpus.
Reporting it as a customer phone number would bury the four real ones. It is
classified INFO, not CRITICAL.

Raw matches are never printed. Every example in the report is redacted -- an
audit artefact that is itself a PII spill helps nobody.

Usage:
    python ops/qdrant_pii_audit.py                     # all collections, text report
    python ops/qdrant_pii_audit.py --json              # machine-readable
    python ops/qdrant_pii_audit.py -c unitronic_faq_0_6b --sample 1000
    python ops/qdrant_pii_audit.py --selftest          # offline detector check

Exit codes: 0 clean or INFO/HIGH only, 1 at least one CRITICAL finding,
2 could not reach Qdrant / bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentv1 import config  # noqa: E402
from agentv1.clients.qdrant import get_client  # noqa: E402

CRITICAL = "CRITICAL"
HIGH = "HIGH"
INFO = "INFO"
_SEVERITY_RANK = {INFO: 0, HIGH: 1, CRITICAL: 2}

LOCUS_PAYLOAD = "payload"
LOCUS_NODE = "_node_content"


# --------------------------------------------------------------------------
# Masking -- applied before value detectors run
# --------------------------------------------------------------------------

# Point ids, doc_id, ref_doc_id and every relationship node_id in _node_content
# are lowercase UUIDs. Their hyphenated hex groups are the single largest
# source of phantom "dashed phone number" hits.
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
# Lowercase hex and pure-digit runs only. Uppercase is deliberately left alone:
# a VIN is 17 uppercase alphanumerics and could be masked away by a
# case-insensitive rule, and the VIN check digit is a better filter than
# blanket masking.
_LONG_TOKEN = re.compile(r"\b(?:[0-9a-f]{12,}|\d{12,})\b")


def mask_identifiers(text: str) -> str:
    return _LONG_TOKEN.sub(" ", _UUID.sub(" ", text))


# --------------------------------------------------------------------------
# Redaction -- what actually reaches the report
# --------------------------------------------------------------------------


def redact(value: str, keep: int = 3) -> str:
    """Keep a short prefix so a finding is traceable, drop the rest."""
    value = " ".join(str(value).split())
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "*" * min(len(value) - keep, 12)


def redact_email(value: str) -> str:
    local, _, domain = value.partition("@")
    return f"{redact(local, 2)}@{domain}"


def redact_wav(value: str) -> str:
    """3CX filename: hide the staff name in brackets and the caller's digits.

    The generic ``*.wav`` detector can start mid-token, so the match often
    begins *inside* the bracket -- ``Nick]_127-...``. Stripping only balanced
    brackets left the staff first name in the report; anything up to the first
    closing bracket goes too.
    """
    out = re.sub(r"\[[^\]]*\]", "[***]", value)
    out = re.sub(r"^[^\[\]]*\]", "[***]", out)
    out = re.sub(r"(\d{3})\d{4,}", r"\1****", out)
    return out[:48]


# --------------------------------------------------------------------------
# Value detectors
# --------------------------------------------------------------------------

# Unitronic's published support line. Verified live: 912 occurrences in a
# 144-point sample of unitronic_company_info_0_6b, and it is printed on the
# website. Toll-free NANP prefixes are treated the same way -- a customer's
# personal number is never toll-free.
_TOLL_FREE_NPA = {"800", "833", "844", "855", "866", "877", "888"}
CORPORATE_PHONE_DIGITS = {"8663412447", "18663412447"}
CORPORATE_EMAIL_DOMAINS = {"getunitronic.com", "unitronic.com", "unitronic-inc.com"}

_PHONE_BARE = re.compile(r"(?<![0-9A-Za-z])([2-9]\d{2})([2-9]\d{2})(\d{4})(?![0-9A-Za-z])")
_PHONE_FORMATTED = re.compile(
    r"(?<![0-9A-Za-z])(?:\+?1[\s.\-]?)?\(?([2-9]\d{2})\)?[\s.\-]\s?([2-9]\d{2})[\s.\-](\d{4})"
    r"(?![0-9A-Za-z])"
)
# [Coles, Zoe]_124-8012307610_20250603202223(152).wav -- staff name AND the
# caller's 10-digit number in one token.
_WAV_3CX = re.compile(r"\[[^\]\n]{1,60}\]_\d{2,4}-\d{7,15}_\d{8,14}(?:\(\d+\))?(?:\.wav)?", re.I)
_WAV_ANY = re.compile(r"[^\s\"\\]{4,90}\.wav", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# thread_path is literally "inbox/<handle>_<thread id>" on the Instagram
# collections; the handle resolves to a real person in one click.
_INSTAGRAM = re.compile(r"\binbox/([A-Za-z0-9._]{2,60})")
_VIN_CANDIDATE = re.compile(r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])")
_VIN_CONTEXT = re.compile(r"\bvin\b", re.I)
# Bare integers are useless as a signal (5.4% of callers quote an order number
# per §9.3, and every price and part number is also digits), so the order
# detector is anchored on the surrounding word.
_ORDER = re.compile(
    r"\b(?:order|purchase order|p\.?o\.?|invoice|rma|ticket)\s*"
    r"(?:#|no\.?|number|num|id)?\s*[:#]?\s*(\d{4,10})\b",
    re.I,
)

# Cardholder data. Measured live: one Luhn-valid Visa PAN, read aloud by a
# customer and transcribed verbatim with its expiry date, appears in
# `solution` and `full_transcript_ref` on unitronic_call_transcriptions_0_6b
# and again in a DM on unitronic_customer_service_classification_0_6b. This is
# PCI-DSS scope, not merely Law 25, so it is its own CRITICAL category rather
# than being folded into the digit-run detectors.
#
# The rule is deliberately narrow. A bare "13-19 digits + Luhn" test matches 281
# times per 500 sampled points here, and essentially all of it is the 13/14-digit
# 3CX timestamp inside a WAV filename passing Luhn by chance. Requiring 15-16
# digits AND a real issuer prefix drops that to zero false positives across
# 1,500 sampled product/tuning/comprehensive points while still catching both
# real cards.
_CARD_CANDIDATE = re.compile(r"(?<![\d\w])(\d(?:[ -]?\d){14,15})(?![\d\w])")
_CARD_IIN = re.compile(
    r"^(?:4\d{15}"                                  # Visa
    r"|5[1-5]\d{14}"                                # Mastercard
    r"|2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)\d{12}"   # Mastercard 2-series
    r"|3[47]\d{13}"                                 # Amex
    r"|6(?:011|5\d\d|4[4-9]\d)\d{12})$"             # Discover
)

# A full 6-character Canadian postal code resolves to roughly one city block,
# which makes it a direct locator rather than the coarse `caller_area_code`
# already covered by GEO_FIELDS. Measured: 56 hits per 500 sampled points of
# unitronic_customer_service_training_0_6b, and 0 across 1,500 sampled
# product/comprehensive/tuning points, so it does not collide with SKUs.
# The alphabet is the real Canada Post one -- D, F, I, O, Q, U never appear in
# the leading position and I/O never appear at all -- which is what keeps
# strings like "G4T2R5"-shaped part codes from matching by accident.
_POSTAL_CA = re.compile(
    r"\b[ABCEGHJKLMNPRSTVXY]\d[ABCEGHJKLMNPRSTVWXYZ][ -]?\d[ABCEGHJKLMNPRSTVWXYZ]\d\b"
)
# ZIP+4 is shaped exactly like the year range in a product slug --
# ".../Audi-Q5-20T-TSI-20015-2016-220hp-stage1" contains "20015-2016", which a
# plain \b\d{5}-\d{4}\b reads as a Chevy Chase ZIP on 5 of every 500 tuning
# points. Rejecting a candidate that touches a hyphen or word character on
# either side removes it without weakening the real case, where a ZIP+4 is
# delimited by whitespace or punctuation.
_POSTAL_US = re.compile(r"(?<![\w-])\d{5}-\d{4}(?![\w-])")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
_VIN_VALUES = {
    **{str(d): d for d in range(10)},
    **dict(
        zip(
            "ABCDEFGHJKLMNPRSTUVWXYZ",
            (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9),
        )
    ),
}


def vin_check_digit_ok(vin: str) -> bool:
    """ISO 3779 position-9 check digit. Mandatory on North American VINs."""
    if len(vin) != 17 or vin.isdigit():
        return False
    letters = sum(c.isalpha() for c in vin)
    digits = sum(c.isdigit() for c in vin)
    if letters < 3 or digits < 4:
        return False
    try:
        total = sum(_VIN_VALUES[c] * w for c, w in zip(vin, _VIN_WEIGHTS))
    except KeyError:
        return False
    return "0123456789X"[total % 11] == vin[8]


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    locus: str
    field: str
    value: str          # raw; aggregated for distinct counts, never printed
    example: str        # redacted; safe to print


def _phone_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    for pattern, category in ((_PHONE_BARE, "phone_10digit"), (_PHONE_FORMATTED, "phone_formatted")):
        for m in pattern.finditer(text):
            digits = "".join(m.groups())
            if digits[:3] in _TOLL_FREE_NPA or digits in CORPORATE_PHONE_DIGITS:
                yield Finding("corporate_contact", INFO, locus, fieldname, digits, m.group(0))
            else:
                yield Finding(category, CRITICAL, locus, fieldname, digits, redact(m.group(0), 3))


def _email_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    for m in _EMAIL.finditer(text):
        addr = m.group(0)
        domain = addr.rpartition("@")[2].lower().rstrip(".")
        if domain in CORPORATE_EMAIL_DOMAINS:
            yield Finding("corporate_contact", INFO, locus, fieldname, addr.lower(), addr)
        else:
            yield Finding("email", CRITICAL, locus, fieldname, addr.lower(), redact_email(addr))


def _wav_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    seen: set[str] = set()
    for m in _WAV_3CX.finditer(text):
        seen.add(m.group(0))
        yield Finding("wav_filename_3cx", CRITICAL, locus, fieldname, m.group(0), redact_wav(m.group(0)))
    for m in _WAV_ANY.finditer(text):
        if any(m.group(0) in s for s in seen):
            continue
        yield Finding("wav_filename_other", HIGH, locus, fieldname, m.group(0), redact_wav(m.group(0)))


def _instagram_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    for m in _INSTAGRAM.finditer(text):
        handle = m.group(1)
        # "facebookuser_1015..." is a platform placeholder, not a chosen handle;
        # still an account id, so it is reported, just not as a named person.
        category = "instagram_thread_id" if handle.lower().startswith("facebookuser") else "instagram_handle"
        yield Finding(category, CRITICAL, locus, fieldname, handle.lower(), redact(handle, 3))


def _vin_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    has_context = bool(_VIN_CONTEXT.search(text))
    for m in _VIN_CANDIDATE.finditer(text.upper()):
        vin = m.group(0)
        if vin_check_digit_ok(vin):
            yield Finding("vin", CRITICAL, locus, fieldname, vin, redact(vin, 3))
        elif has_context and not vin.isdigit():
            yield Finding("vin_suspected", HIGH, locus, fieldname, vin, redact(vin, 3))


def _order_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    for m in _ORDER.finditer(text):
        yield Finding("order_number", HIGH, locus, fieldname, m.group(1), redact(m.group(1), 2))


def _card_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    for m in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        if luhn_ok(digits) and _CARD_IIN.match(digits):
            # Never echo more than the issuer nibble: the report itself would
            # otherwise become the cardholder-data spill it is reporting.
            yield Finding("payment_card", CRITICAL, locus, fieldname, digits, digits[:2] + "*" * 14)


def _postal_findings(text: str, locus: str, fieldname: str) -> Iterator[Finding]:
    for pattern in (_POSTAL_CA, _POSTAL_US):
        for m in pattern.finditer(text):
            code = m.group(0)
            yield Finding("postal_code", HIGH, locus, fieldname, code.upper().replace(" ", ""),
                          redact(code, 2))


# mask_identifiers() blanks every 12+ digit run and every 12+ character
# lowercase-hex run. That is what keeps UUIDs from reading as dashed phone
# numbers, but it also erases a 16-digit card number before it can be tested,
# and eats an address whose local part happens to be hex-shaped
# (``deadbeefcafe@gmail.com`` scanned masked yields nothing at all). So the two
# detectors that masking would blind are run against the original text instead.
MASKED_DETECTORS: tuple[Callable[[str, str, str], Iterator[Finding]], ...] = (
    _wav_findings,
    _phone_findings,
    _instagram_findings,
    _vin_findings,
    _order_findings,
    _postal_findings,
)
RAW_DETECTORS: tuple[Callable[[str, str, str], Iterator[Finding]], ...] = (
    _card_findings,
    _email_findings,
)
VALUE_DETECTORS: tuple[Callable[[str, str, str], Iterator[Finding]], ...] = (
    MASKED_DETECTORS + RAW_DETECTORS
)


# --------------------------------------------------------------------------
# Field-name detectors -- a key whose very presence is the leak
# --------------------------------------------------------------------------

PERSON_NAME_FIELDS = {
    "agent_name", "agent", "staff_name", "rep_name", "representative",
    "handled_by", "assigned_to", "caller_name", "customer_name",
    "contact_name", "sender_name", "recipient_name", "first_name",
    "last_name", "full_name", "user_name", "author",
}
# *_name keys that are about a thing, not a person.
NON_PERSON_NAME_FIELDS = {
    "file_name", "product_name", "brand_name", "model_name", "platform_name",
    "collection_name", "field_name", "page_name", "vehicle_name", "stage_name",
    "part_name", "category_name", "company_name", "dealer_name",
}
GEO_FIELDS = {"caller_area_code", "area_code", "postal_code", "zip_code", "location_mentioned"}

_PERSON_NAME_SHAPE = re.compile(r"^[A-Z][a-zà-öø-ÿ'\-]{1,19}(?:[ ,]+[A-Z][a-zà-öø-ÿ'\-]{1,19}){0,3}$")


def looks_like_person_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    return 2 <= len(v) <= 48 and bool(_PERSON_NAME_SHAPE.match(v))


def scan_fields(mapping: dict, locus: str, _depth: int = 0, _prefix: str = "") -> Iterator[Finding]:
    """Field-name detectors, applied at every depth.

    The live payloads here are flat, so a one-level scan happens to be
    sufficient *today*. It is still the wrong shape: a payload that nests the
    caller under ``{"caller": {"customer_name": ..., "caller_area_code": ...}}``
    produced zero findings, because the value detectors see only a JSON blob
    with no phone or email in it and the field detectors never looked inside.
    An audit whose premise is that partial checks are theatre should not itself
    stop at the first level.
    """
    for key, value in mapping.items():
        low = key.lower()
        name = f"{_prefix}{key}"
        if low in GEO_FIELDS and value not in (None, "", [], {}):
            yield Finding("geo_identifier", HIGH, locus, name, f"{name}={value}",
                          f"{name}={redact(str(value), 1)}")
        elif low not in NON_PERSON_NAME_FIELDS and (
            low in PERSON_NAME_FIELDS or low.endswith("_name")
        ):
            if looks_like_person_name(value):
                yield Finding("person_name_field", CRITICAL, locus, name, f"{name}={value}",
                              f"{name}={redact(str(value), 1)}")

        if _depth < 4:
            if isinstance(value, dict):
                yield from scan_fields(value, locus, _depth + 1, f"{name}.")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield from scan_fields(item, locus, _depth + 1, f"{name}[].")


# --------------------------------------------------------------------------
# Point scanning
# --------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def scan_text(text: str, locus: str, fieldname: str) -> list[Finding]:
    if not text:
        return []
    masked = mask_identifiers(text)
    out: list[Finding] = []
    for detector in MASKED_DETECTORS:
        out.extend(detector(masked, locus, fieldname))
    for detector in RAW_DETECTORS:
        out.extend(detector(text, locus, fieldname))
    return out


def scan_payload(payload: dict) -> list[Finding]:
    """Scan a single point in both loci. Order matters only for readability."""
    findings: list[Finding] = []
    shallow = {k: v for k, v in payload.items() if k != LOCUS_NODE}

    findings.extend(scan_fields(shallow, LOCUS_PAYLOAD))
    for key, value in shallow.items():
        findings.extend(scan_text(_stringify(value), LOCUS_PAYLOAD, key))

    node_raw = payload.get(LOCUS_NODE)
    if isinstance(node_raw, str) and node_raw:
        findings.extend(scan_text(node_raw, LOCUS_NODE, LOCUS_NODE))
        try:
            node = json.loads(node_raw)
        except (ValueError, TypeError):
            node = None
        if isinstance(node, dict) and isinstance(node.get("metadata"), dict):
            findings.extend(scan_fields(node["metadata"], LOCUS_NODE))
    return findings


# --------------------------------------------------------------------------
# Collection audit
# --------------------------------------------------------------------------


@dataclass
class CategoryReport:
    category: str
    severity: str
    points_affected: int = 0
    occurrences: int = 0
    distinct_values: set = field(default_factory=set)
    by_locus: dict = field(default_factory=lambda: defaultdict(int))
    fields: set = field(default_factory=set)
    examples: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "points_affected": self.points_affected,
            "occurrences": self.occurrences,
            "distinct_values": len(self.distinct_values),
            "by_locus": dict(self.by_locus),
            "fields": sorted(self.fields),
            "examples_redacted": self.examples[:5],
        }


def audit_collection(client, name: str, sample: int, page: int = 128) -> dict:
    info = client.get_collection(name)
    total = info.points_count or 0

    reports: dict[str, CategoryReport] = {}
    payload_fields: set[str] = set()
    node_fields: set[str] = set()
    sampled = 0
    with_node_content = 0
    offset = None

    while total and (sample == 0 or sampled < sample):
        limit = page if sample == 0 else min(page, sample - sampled)
        points, offset = client.scroll(
            collection_name=name, limit=limit, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            sampled += 1
            payload_fields.update(payload.keys())
            node_raw = payload.get(LOCUS_NODE)
            if isinstance(node_raw, str) and node_raw:
                with_node_content += 1
                try:
                    node = json.loads(node_raw)
                    if isinstance(node.get("metadata"), dict):
                        node_fields.update(node["metadata"].keys())
                except (ValueError, TypeError, AttributeError):
                    pass

            seen_here: set[str] = set()
            for f in scan_payload(payload):
                rep = reports.get(f.category)
                if rep is None:
                    rep = reports[f.category] = CategoryReport(f.category, f.severity)
                # A category's severity is the worst seen for it.
                if _SEVERITY_RANK[f.severity] > _SEVERITY_RANK[rep.severity]:
                    rep.severity = f.severity
                rep.occurrences += 1
                rep.distinct_values.add(f.value)
                rep.by_locus[f.locus] += 1
                rep.fields.add(f.field)
                if f.category not in seen_here:
                    seen_here.add(f.category)
                    rep.points_affected += 1
                if len(rep.examples) < 5 and f.example not in rep.examples:
                    rep.examples.append(f.example)
        if offset is None:
            break

    severities = [r.severity for r in reports.values()]
    if total == 0:
        verdict, reason = "EMPTY", "collection holds 0 points; nothing to leak, nothing to serve"
    elif CRITICAL in severities:
        cats = sorted(c for c, r in reports.items() if r.severity == CRITICAL)
        verdict, reason = CRITICAL, "direct personal identifiers present: " + ", ".join(cats)
    elif HIGH in severities:
        cats = sorted(c for c, r in reports.items() if r.severity == HIGH)
        verdict, reason = "WARN", "indirect identifiers present: " + ", ".join(cats)
    else:
        verdict, reason = "CLEAN", "no personal identifiers in the sampled points"

    return {
        "collection": name,
        "points": total,
        "sampled": sampled,
        "coverage_pct": round(100.0 * sampled / total, 2) if total else 0.0,
        "points_with_node_content": with_node_content,
        "payload_fields": sorted(payload_fields),
        "node_content_metadata_fields": sorted(node_fields),
        "findings": {c: r.as_dict() for c, r in sorted(
            reports.items(), key=lambda kv: (-_SEVERITY_RANK[kv[1].severity], kv[0])
        )},
        "verdict": verdict,
        "verdict_reason": reason,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_BAR = "-" * 78


def render(reports: list[dict]) -> str:
    lines = [_BAR, "QDRANT PII AUDIT -- AGENT_PLAN.md 3.1", _BAR, ""]
    for r in reports:
        lines.append(f"{r['collection']}")
        lines.append(
            f"  points {r['points']:>7,}   sampled {r['sampled']:>5,}"
            f"  ({r['coverage_pct']:.2f}%)   _node_content on"
            f" {r['points_with_node_content']}/{r['sampled']}"
        )
        fields = r["payload_fields"]
        lines.append(f"  payload fields ({len(fields)}): {', '.join(fields) if fields else '(none)'}")
        nf = r["node_content_metadata_fields"]
        if nf:
            lines.append(f"  _node_content metadata fields ({len(nf)}): {', '.join(nf)}")
        if r["findings"]:
            lines.append("  findings:")
            for cat, f in r["findings"].items():
                loci = " + ".join(f"{k} x{v}" for k, v in sorted(f["by_locus"].items()))
                pct = 100.0 * f["points_affected"] / r["sampled"] if r["sampled"] else 0.0
                lines.append(
                    f"    {f['severity']:<8} {cat:<22} pts {f['points_affected']:>4}/"
                    f"{r['sampled']} ({pct:5.1f}%)  distinct {f['distinct_values']:>4}  [{loci}]"
                )
                lines.append(f"             fields: {', '.join(f['fields'])}")
                lines.append(f"             e.g. {' | '.join(f['examples_redacted'])}")
        else:
            lines.append("  findings: none")
        lines.append(f"  VERDICT: {r['verdict']} -- {r['verdict_reason']}")
        lines.append("")

    crit = [r["collection"] for r in reports if r["verdict"] == CRITICAL]
    warn = [r["collection"] for r in reports if r["verdict"] == "WARN"]
    clean = [r["collection"] for r in reports if r["verdict"] in ("CLEAN", "EMPTY")]
    lines += [_BAR, "SUMMARY", _BAR]
    lines.append(f"  CRITICAL  {len(crit):>2}: {', '.join(crit) if crit else '-'}")
    lines.append(f"  WARN      {len(warn):>2}: {', '.join(warn) if warn else '-'}")
    lines.append(f"  CLEAN     {len(clean):>2}: {', '.join(clean) if clean else '-'}")
    lines.append("")
    if crit:
        lines.append("  Remediate with: python ops/p0_remediate.py quarantine <collection> --yes")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-check -- offline, no Qdrant required
# --------------------------------------------------------------------------

_SELFTEST_POSITIVE = [
    ("wav_filename_3cx", {"file_name": "[Coles, Zoe]_124-8012307610_20250603202223(152).wav"}),
    ("phone_10digit", {"summary": "call back on 3064913593 tomorrow"}),
    ("phone_formatted", {"summary": "reach them at 801-710-3169"}),
    ("email", {"summary": "sent to kylestephenson3@gmail.com"}),
    ("instagram_handle", {"thread_path": "inbox/dustinulichney_10153446946432182"}),
    ("vin", {"context": "WAUDGAFL3BA016563 on the work order"}),
    ("order_number", {"summary": "order #74657 shipped"}),
    ("person_name_field", {"agent_name": "Dustin"}),
    ("geo_identifier", {"caller_area_code": "306"}),
    ("corporate_contact", {"original_text": "technical support: 866.341.2447"}),
    # Luhn-valid test PANs (never issued). Cardholder data is live in the
    # transcripts corpus, so this category has to exist.
    ("payment_card", {"solution": "customer gave Visa 4539578763621486, expiry 06/26"}),
    ("payment_card", {"solution": "card 5425 2334 3010 9903 on file"}),
    ("payment_card", {"full_transcript_ref": "AGENT: ready. CUSTOMER: 3782 822463 10005."}),
    ("postal_code", {"summary": "ships to Laval H7T 2P5"}),
    ("postal_code", {"summary": "billing zip 90210-1234"}),
    # mask_identifiers() blanks 12+ char lowercase-hex runs; the email detector
    # must therefore see the unmasked text or this address vanishes.
    ("email", {"summary": "write to deadbeefcafe@gmail.com"}),
]

_SELFTEST_NEGATIVE = [
    # A UUID's hyphenated hex groups must not read as a dashed phone number.
    ("phone_formatted", {"doc_id": "0000a73b-a1a0-4723-ab54-112b0d95e0a7"}),
    ("phone_10digit", {"doc_id": "0000a73b-a1a0-4723-ab54-112b0d95e0a7"}),
    # A 17-digit Instagram thread id is not a VIN.
    ("vin", {"thread_path": "inbox/x_10153446946432182"}),
    # A SKU is not an order number.
    ("order_number", {"sku": "UH016-GR4"}),
    # A product title is not a person.
    ("person_name_field", {"product_name": "Unitronic Signature Badge"}),
    # A corporate address is not a customer's.
    ("email", {"original_text": "write to info@getunitronic.com"}),
    # The 3CX timestamp inside a WAV filename is 14 digits and passes Luhn by
    # chance ~10% of the time. Requiring 15-16 digits plus a real issuer prefix
    # is what stops the card detector firing ~281 times per 500 points.
    ("payment_card", {"file_name": "[Coles, Zoe]_124-8012307610_20250603202223(152).wav"}),
    ("payment_card", {"summary": "reference 1234567890123456 in the ticket"}),
    # A part number is not a postal code.
    ("postal_code", {"sku": "UH016-GR4"}),
]

# The exact scenario §3.1 calls out: a phone-only audit passes here, so the
# broader detector set must not.
_SELFTEST_THEATRE_PAYLOAD = {
    "interaction_type": "customer_service_training",
    "thread_path": "inbox/dustinulichney_10153446946432182",
    "_node_content": json.dumps(
        {"id_": "0000a73b-a1a0-4723-ab54-112b0d95e0a7",
         "metadata": {"thread_path": "inbox/dustinulichney_10153446946432182"}}
    ),
}


def selftest() -> int:
    failures = []
    for expected, payload in _SELFTEST_POSITIVE:
        cats = {f.category for f in scan_payload(payload)}
        if expected not in cats:
            failures.append(f"expected {expected} in {payload!r}, got {sorted(cats)}")
    for forbidden, payload in _SELFTEST_NEGATIVE:
        cats = {f.category for f in scan_payload(payload)}
        if forbidden in cats:
            failures.append(f"false positive {forbidden} on {payload!r}")

    phone_only = {c for c in (f.category for f in scan_payload(_SELFTEST_THEATRE_PAYLOAD))
                  if c.startswith("phone")}
    if phone_only:
        failures.append(f"phone detectors should be silent on the theatre payload, got {phone_only}")
    theatre = scan_payload(_SELFTEST_THEATRE_PAYLOAD)
    cats = {f.category for f in theatre}
    if "instagram_handle" not in cats:
        failures.append("broad detectors missed the Instagram handle -- audit would be theatre")
    loci = {f.locus for f in theatre if f.category == "instagram_handle"}
    if loci != {LOCUS_PAYLOAD, LOCUS_NODE}:
        failures.append(f"handle must be found in BOTH loci, found {sorted(loci)}")

    if not vin_check_digit_ok("WAUDGAFL3BA016563"):
        failures.append("VIN check digit rejected a known-good VIN")
    if vin_check_digit_ok("WAUDGAFL3BA016564"):
        failures.append("VIN check digit accepted a corrupted VIN")
    if "8012307610" in redact_wav("[Coles, Zoe]_124-8012307610_20250603202223(152).wav"):
        failures.append("redaction leaked the caller phone number")
    if "Coles" in redact_wav("[Coles, Zoe]_124-8012307610_20250603202223(152).wav"):
        failures.append("redaction leaked the staff name")
    # The generic .wav detector starts mid-bracket; that path leaked names once.
    if "Zoe" in redact_wav("Zoe]_124-anonymous_20250603202223(348).wav"):
        failures.append("redaction leaked the staff name from a partial bracket")
    # Field detectors must reach nested payloads, not just the top level.
    nested = {"caller": {"customer_name": "Marc Tremblay", "caller_area_code": "514"}}
    nested_cats = {f.category for f in scan_payload(nested)}
    if not {"person_name_field", "geo_identifier"} <= nested_cats:
        failures.append(f"field detectors did not recurse into a nested payload, got {sorted(nested_cats)}")
    # The report must never echo a full PAN.
    card_examples = [f.example for f in scan_payload(
        {"solution": "Visa 4539578763621486"}) if f.category == "payment_card"]
    if any("4539578763621486" in e for e in card_examples):
        failures.append("redaction leaked a full card number")

    for f in failures:
        print(f"FAIL  {f}")
    checks = len(_SELFTEST_POSITIVE) + len(_SELFTEST_NEGATIVE) + 10
    print(f"{checks - len(failures)}/{checks} detector checks passed")
    return 1 if failures else 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--collection", action="append", default=[],
                    help="restrict to this collection (repeatable); default is every live collection")
    ap.add_argument("-n", "--sample", type=int, default=500,
                    help="points to sample per collection, 0 for all (default 500)")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--selftest", action="store_true", help="run offline detector checks and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        client = get_client()
        live = [c.name for c in client.get_collections().collections]
    except Exception as exc:  # network / auth -- distinguish from a PII finding
        print(f"could not reach Qdrant: {exc}", file=sys.stderr)
        return 2

    targets = args.collection or sorted(live)
    missing = [t for t in targets if t not in live]
    if missing:
        print(f"no such collection(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    reports = [audit_collection(client, name, args.sample) for name in targets]

    if args.json:
        print(json.dumps({
            "qdrant_url": config.QDRANT_URL,
            "collections_live": len(live),
            "sample_per_collection": args.sample,
            "reports": reports,
        }, indent=2))
    else:
        print(render(reports))

    return 1 if any(r["verdict"] == CRITICAL for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
