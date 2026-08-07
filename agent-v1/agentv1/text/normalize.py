"""Surface-form normalization, applied identically at index time and query time.

This is the higher-leverage half of the hybrid-retrieval work and it is not
where you would expect. The classic "you need BM25 for error codes" argument is
weak on this corpus -- DTC codes appear in ~1% of units and part numbers in
~1.4%. The real lexical failure is the plus sign:

    measured over 25,613 knowledge units --
      "stage 2"  10241    "stage 1+"  987
      "stage 1"   6767    "stage 2+"  438
      "stage 3"   3167

Dense embeddings blur ``Stage 1`` against ``Stage 1+``, and a default BM25
tokenizer *deletes* the ``+`` outright, so neither half of a hybrid retriever
can tell them apart. They are different products at different prices.
UniCONNECT has seven measured surface forms in the same corpus, collapsing to
two distinct products (``uniconnect`` and ``uniconnect+``).

The fix is to emit explicit marker tokens -- ``stage_1_plus``, ``uniconnect_plus``
-- on both the document and the query side. Because both sides run the same
function, the tokens either match or they do not; there is no asymmetry to
debug later.

Everything here is deterministic. None of it is an LLM call.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ..config import DATA_DIR

# --- Stage forms -------------------------------------------------------------
# "Stage 1+", "stage 1 +", "stage1plus", "Stage One Plus" all reach stage_1_plus.
_STAGE_RE = re.compile(
    r"\bstage\s*[\-_]?\s*(?P<num>[0-4]|one|two|three|four)\s*(?P<plus>\+|\bplus\b)?",
    re.IGNORECASE,
)
_WORD_NUM = {"one": "1", "two": "2", "three": "3", "four": "4"}

# --- UniCONNECT forms --------------------------------------------------------
# Ordered longest-first: "uni connect plus" must not be consumed by the
# "uni connect" branch before the "plus" is seen.
# The `plus` branch must not require a word boundary before "plus": the
# measured corpus contains `uniconnectplus` as one token, and `\b` between two
# word characters never matches.
_UNICONNECT_RE = re.compile(
    r"\buni[\s\-_]?connect[\s\-_]*(?P<plus>\+|plus\b)?", re.IGNORECASE
)

# --- Codes -------------------------------------------------------------------
_DTC_RE = re.compile(r"\b([PBCU][0-2][0-9A-F]{3})\b", re.IGNORECASE)
_PART_RE = re.compile(r"\b([A-Z]{2,4}[\-_]?\d{3,6}(?:[\-_][A-Z0-9]{1,4})?)\b")
_ECU_BOX_RE = re.compile(r"\b(0[A-Z0-9]{2}\s?\d{3}\s?\d{3}\s?[A-Z]{0,2})\b")
_VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")

_PUNCT_RE = re.compile(r"[^\w\s+]+")
_WS_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    """colloquial -> canonical, mined from ``calls_analysis.search_aliases``.

    52.9% of analysed calls carry this field; the analyzer wrote it with the
    full transcript in hand. It is a deterministic query-rewrite table and
    costs nothing at query time, so there is no reason to pay an LLM to
    rediscover it.
    """
    path = DATA_DIR / "aliases.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {a["colloquial"]: a["canonical"] for a in raw}


def stage_tokens(text: str) -> list[str]:
    """Marker tokens for every stage mention. ``Stage 1+`` -> ``stage_1_plus``."""
    out: list[str] = []
    for m in _STAGE_RE.finditer(text):
        num = m.group("num").lower()
        num = _WORD_NUM.get(num, num)
        out.append(f"stage_{num}_plus" if m.group("plus") else f"stage_{num}")
    return out


def uniconnect_tokens(text: str) -> list[str]:
    """``UniCONNECT+`` / ``uni-connect plus`` / ``uniconnectplus`` -> one token."""
    return [
        "uniconnect_plus" if m.group("plus") else "uniconnect"
        for m in _UNICONNECT_RE.finditer(text)
    ]


def code_tokens(text: str) -> list[str]:
    """DTCs, part numbers and ECU box codes as single opaque tokens."""
    out: list[str] = []
    out += [f"dtc_{m.group(1).upper()}" for m in _DTC_RE.finditer(text)]
    out += [
        f"pn_{m.group(1).upper().replace('-', '').replace('_', '')}"
        for m in _PART_RE.finditer(text)
    ]
    for m in _ECU_BOX_RE.finditer(text):
        out.append("ecubox_" + _WS_RE.sub("", m.group(1)).upper())
    return out


def marker_tokens(text: str) -> list[str]:
    """Every deterministic marker for a piece of text, deduped, order-stable."""
    seen: dict[str, None] = {}
    for tok in stage_tokens(text) + uniconnect_tokens(text) + code_tokens(text):
        seen.setdefault(tok, None)
    return list(seen)


def expand_aliases(text: str) -> list[str]:
    """Canonical terms implied by colloquialisms present in ``text``.

    Additive only -- we never replace the customer's wording, because the
    colloquial form is often also what the corpus says.
    """
    low = text.lower()
    out: dict[str, None] = {}
    for colloquial, canonical in _alias_map().items():
        # Word-boundary check without building 15k regexes.
        idx = low.find(colloquial)
        while idx != -1:
            before_ok = idx == 0 or not low[idx - 1].isalnum()
            end = idx + len(colloquial)
            after_ok = end == len(low) or not low[end].isalnum()
            if before_ok and after_ok:
                out.setdefault(canonical, None)
                break
            idx = low.find(colloquial, idx + 1)
    return list(out)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation except ``+``, collapse whitespace."""
    if not text:
        return ""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def lexical_terms(text: str, *, with_aliases: bool = True) -> list[str]:
    """The full lexical bag for BM25: words + markers + alias expansions.

    Called by both the indexer and the query encoder. Any change here is a
    reindex, which is why the sparse encoder stamps the analyzer version.
    """
    if not text:
        return []
    terms = normalize_text(text).split()
    terms += marker_tokens(text)
    if with_aliases:
        # Alias canonicals are phrases ("ecu tune installation"); BM25 scores
        # terms, so split them. The phrase itself is preserved by the dense
        # side, which does understand multi-word meaning.
        for canonical in expand_aliases(text):
            terms += normalize_text(canonical).split()
    return terms


# Transaction identifiers, matched only in an explicit context word. A bare
# 5-6 digit run is far more often a part number ("part number 853671"), an ECU
# identifier ("last six digits 011358"), an RPM figure or a year, and redacting
# those would damage genuinely useful answers. Requiring the context word makes
# this precise rather than merely aggressive.
#
# Found by the PII audit against the first residual/case-narrative build:
# whole-call and case summaries carry "Order #854916", "Invoice 71520",
# "cable UCP 58647". These are indirect identifiers -- an order number resolves
# to one customer's transaction -- and, unlike a payload key, they sit in the
# body text that gets returned to *other* customers as a retrieval result.
_TXN_RE = re.compile(
    r"\b(order|invoice|inv|p\.?o\.?|purchase order|transaction|txn|ticket|rma|receipt|"
    r"confirmation|tracking|serial|cable\s+ucp|ucp)\b"
    r"(\s*(?:#|no\.?|num(?:ber)?|id)?\s*[:\-]?\s*\(?\s*)"
    r"(?P<id>[A-Z]{0,3}\d{4,12})\b",
    re.IGNORECASE,
)


def redact_transaction_ids(text: str) -> str:
    """Redact order/invoice/serial numbers that appear with a context word."""
    return _TXN_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[ORDER_REF]", text)


def strip_pii_markers(text: str) -> str:
    """Redact identifiers that must never reach a payload or an SSE frame.

    Defence in depth only. The real gate is the payload allowlist in
    ``index/payload.py`` -- an allowlist cannot be defeated by a regex miss,
    and hand-rolled regex has a known recall problem on names and addresses,
    which is why ``guardrails/pii.py`` exists as well.
    """
    text = re.sub(r"\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    text = re.sub(r"inbox/\S+", "[HANDLE]", text)
    text = _VIN_RE.sub("[VIN]", text)
    text = redact_transaction_ids(text)
    return text


def wav_filename_phone(filename: str) -> str | None:
    """Extract the phone key a 3CX recording filename embeds.

    ``[Coles, Zoe]_124-8012307610_20250603202223(152).wav`` -> ``8012307610``.
    Used only to *remap away from* these filenames: ``supersedes_calls[]``
    expresses the supersession relation -- the entire point of case units -- as
    a list of these, so the relation is PII until it is remapped to opaque
    unit ids.
    """
    m = re.search(r"_(\d{3,4})-(\d{7,15})_", filename)
    if m:
        return m.group(2)
    m = re.search(r"_(\d{7,15})-(\d{3,4})_", filename)
    if m:
        return m.group(1)
    return None
