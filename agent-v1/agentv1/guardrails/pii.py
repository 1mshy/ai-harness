"""PII egress scanning, behind an interface Presidio drops into.

AGENT_PLAN.md 9.8. Every other guardrail in this package is specified as
custom regex, and that is correct for the emissions lexicon -- it is genuinely
domain-specific and needs compliance sign-off, not a general safety model.
It is *not* correct here. The gate on PII egress is set at absolute zero, and
hand-rolled regex has a known recall problem on exactly the entity types this
corpus is full of: names, addresses, and anything written in a non-standard
format.

So this module ships the regex backend, and ships the seam. ``PIIBackend`` is
the contract; ``RegexBackend`` implements it today; ``PresidioBackend``
implements it against ``presidio-analyzer`` and is selected by setting
``PII_BACKEND=presidio`` once the package is installed. No call site changes.

HOW BIG THE REGEX GAP ACTUALLY IS -- measured 2026-08-06, live Mongo,
400-call sample. Reproduce with ``.venv/bin/python -m agentv1.guardrails.pii``:

    3CX recording filenames detected        400/400   100.0%
    bare 10-digit caller numbers detected   388/400    97.0%
    PERSON names detected                     0/180     0.0%

    The 12 phone misses are all internal extensions ("123"), which are not
    customer identifiers. The structured side is essentially solved.

    The zero is the point. 180 of the sampled calls record the customer's
    spoken name in ``customer_name_mentioned`` AND contain that exact string
    in the transcript, so the ground truth is unambiguous and the position is
    known. The regex backend finds none of them, because there is no regex
    for a name, and no amount of pattern work changes that. That is the 9.8
    argument stated as a measurement rather than an opinion.

    Until Presidio lands, the real control is the payload allowlist in
    ``index/payload.py`` -- an allowlist cannot be defeated by a regex miss.
    This module is defence in depth on the way out, not the primary gate, and
    describing it as the primary gate would be the mistake.

WHAT "EGRESS" MEANS HERE
    Anything leaving the process: a Qdrant payload about to be returned, an
    SSE frame about to be written, the final answer. :func:`scan_egress`
    walks arbitrary nested structures because an SSE frame is a dict of dicts
    and the leak that motivated all of this appeared twice in the same
    document -- once as payload keys (``file_name``, ``agent_name``,
    ``caller_area_code``) and once again serialised inside ``_node_content``.
    A scanner that only looked at top-level string fields would have found
    neither.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Protocol, runtime_checkable

from ..text.normalize import wav_filename_phone


class PIIEgressError(RuntimeError):
    """Raised by :func:`assert_clean`. Never caught inside this package."""


@dataclass(frozen=True)
class PIIEntity:
    entity_type: str
    start: int
    end: int
    text: str
    score: float
    backend: str
    # Dotted path into the scanned object, e.g. "payload.metadata.file_name".
    path: str = ""

    def __repr__(self) -> str:
        """Never render the matched text.

        The default dataclass repr would print ``text=`` -- the PII itself --
        and findings end up in log lines, exception messages and structured
        detail dicts by way of ``str(entity)``. A finding that carries its own
        payload is a second copy of the leak in a place with a longer
        retention than the first. ``.text`` stays available on the object for
        :func:`redact`, which needs it and does not serialise it.
        """
        return (
            f"PIIEntity({self.entity_type} @{self.path or '<root>'}"
            f"[{self.start}:{self.end}] score={self.score:.2f} "
            f"backend={self.backend})"
        )

    __str__ = __repr__

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            # The matched text is NOT included. A PII finding that carries the
            # PII is a second copy of the leak, and these reports go to logs.
            "score": self.score,
            "backend": self.backend,
            "path": self.path,
        }


@runtime_checkable
class PIIBackend(Protocol):
    """The seam. Presidio satisfies this without any call site changing."""

    name: str

    def analyze(self, text: str, *, language: str = "en") -> list[PIIEntity]:
        """Return every entity found in ``text``. Offsets index ``text``."""

    def supported_entities(self) -> tuple[str, ...]:
        ...


# ---------------------------------------------------------------------------
# Regex backend
# ---------------------------------------------------------------------------
def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Confidence values are honest rather than flattering. 0.99 means "this is
# checksummed or structurally unambiguous"; 0.5 means "this fires on things
# that are not PII and you should expect it to".
_REGEX_RULES: tuple[tuple[str, str, float], ...] = (
    (
        "EMAIL_ADDRESS",
        r"[\w.+-]+@[\w-]+\.[\w.]{2,}",
        0.95,
    ),
    (
        "PHONE_NUMBER",
        # NANP, plus a bare international form. The corpus contains Swedish
        # and other +CC numbers that a NANP-only pattern reports as clean.
        r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)"
        r"|\+\d{8,15}(?!\d)",
        0.85,
    ),
    (
        "VIN",
        # 17 chars, no I/O/Q. The (?<![A-Z0-9]) guard stops it matching the
        # tail of a longer alphanumeric blob such as a build hash.
        r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])",
        0.8,
    ),
    (
        "CREDIT_CARD",
        r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)",
        0.99,  # post-filtered by Luhn below; survivors are near-certain
    ),
    (
        "CA_SIN",
        r"(?<!\d)\d{3}[\s\-]\d{3}[\s\-]\d{3}(?!\d)",
        0.6,
    ),
    (
        "US_SSN",
        r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
        0.9,
    ),
    (
        "CA_POSTAL_CODE",
        r"\b[ABCEGHJ-NPRSTVXY]\d[A-Z][ \-]?\d[A-Z]\d\b",
        0.8,
    ),
    (
        "US_ZIP",
        r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)",
        0.35,  # collides with part numbers and years; deliberately low
    ),
    (
        "IP_ADDRESS",
        r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
        0.9,
    ),
    (
        "DATE_OF_BIRTH",
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date|date\s+de\s+naissance)\b"
        r"[^\n]{0,20}?\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",
        0.9,
    ),
    (
        "STREET_ADDRESS",
        # Weak by construction. Street-type suffix list only; it will not find
        # "123 Rue Principale" without the suffix, and it will not find an
        # address written across two fields. See the module docstring.
        r"\b\d{1,6}[\w\-]*\s+(?:[A-Za-z'\-]+\s+){0,3}"
        r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|ct|court"
        r"|pl|place|cres|crescent|hwy|highway|rue|chemin|ch\.|boul|mont[eé]e|rang)\b\.?",
        0.5,
    ),
    (
        "PBX_RECORDING_FILENAME",
        # 3CX recording filenames embed the caller's phone number AND the
        # agent's name. This is the leak that quarantined an entire Qdrant
        # collection, so it gets its own rule rather than relying on the
        # phone rule happening to fire inside it.
        #
        # Four shapes measured in a 400-call live sample that a tighter
        # pattern misses, 66 filenames in total:
        #   `[]_123-2484461900_...`        empty name bracket
        #   `[Angrignon, Dustin]_115-123_` internal extension-to-extension,
        #                                  "number" is three digits
        #   `[...]_123-+46734100034_`      international, leading +
        #   `[EN%3AGS%3ACAI TAN]_6479967388-123_`
        #                                  inbound: number FIRST, extension
        #                                  second, and a URL-encoded routing
        #                                  prefix inside the bracket
        # The extension-only ones still carry the agent's name, which is 9.7
        # material even when no customer number is present, so they are
        # matched too rather than being treated as safe.
        r"\[[^\]\n]{0,80}\]_\+?\d{2,15}-\+?\d{2,15}_\d{8,14}(?:\(\d+\))?(?:\.wav)?",
        0.99,
    ),
)

_REGEX_COMPILED = tuple(
    (name, re.compile(pat, re.IGNORECASE), score) for name, pat, score in _REGEX_RULES
)


class RegexBackend:
    """Deterministic, dependency-free, and knowingly incomplete on names."""

    name = "regex"

    def supported_entities(self) -> tuple[str, ...]:
        return tuple(n for n, _, _ in _REGEX_RULES)

    def analyze(self, text: str, *, language: str = "en") -> list[PIIEntity]:
        if not text:
            return []
        out: list[PIIEntity] = []
        for etype, rx, score in _REGEX_COMPILED:
            for m in rx.finditer(text):
                frag = m.group(0)
                if etype == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", frag)
                    # Without Luhn this rule fires on every long digit run --
                    # VINs, order numbers, phone-plus-extension. With it, a
                    # false positive needs to be a checksum collision.
                    if not (13 <= len(digits) <= 19 and _luhn(digits)):
                        continue
                out.append(
                    PIIEntity(
                        entity_type=etype,
                        start=m.start(),
                        end=m.end(),
                        text=frag,
                        score=score,
                        backend=self.name,
                    )
                )
        # A recording filename contains a phone number; reporting both is
        # noise. Drop any entity fully covered by a higher-scoring one.
        out.sort(key=lambda e: (e.start, -(e.end - e.start), -e.score))
        kept: list[PIIEntity] = []
        for e in out:
            if any(k.start <= e.start and e.end <= k.end and k.score >= e.score
                   for k in kept):
                continue
            kept.append(e)
        return kept


class PresidioBackend:
    """Microsoft Presidio, self-hosted. Selected with ``PII_BACKEND=presidio``.

    Not a stub: the calls below are the real ``presidio-analyzer`` API. The
    import is lazy because ``presidio-analyzer`` pulls spaCy and a language
    model, which is a real install decision and not one this package makes on
    anybody's behalf. :meth:`available` answers whether it can be used, and
    :func:`get_backend` degrades to regex with a stated reason rather than
    raising at import time.
    """

    name = "presidio"

    # Presidio's own names for the things we care about. `PERSON` and
    # `LOCATION` are the two the regex backend cannot do at all and are the
    # entire reason for 9.8.
    ENTITIES = (
        "PERSON",
        "LOCATION",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "IBAN_CODE",
        "IP_ADDRESS",
        "DATE_TIME",
        "US_SSN",
        "CA_SIN",
        "US_DRIVER_LICENSE",
        "NRP",
    )

    def __init__(self, *, score_threshold: float = 0.4) -> None:
        from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
        from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: PLC0415

        # Both languages, one engine. The corpus is 8.5% French and a
        # French-blind PII scanner on a Quebec call centre is not a scanner.
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_lg"},
                    {"lang_code": "fr", "model_name": "fr_core_news_lg"},
                ],
            }
        )
        self._engine = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["en", "fr"]
        )
        self._threshold = score_threshold

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import presidio_analyzer  # noqa: F401,PLC0415
        except Exception as exc:  # noqa: BLE001
            return False, f"presidio-analyzer not importable: {exc}"
        return True, "presidio-analyzer importable"

    def supported_entities(self) -> tuple[str, ...]:
        return self.ENTITIES

    def analyze(self, text: str, *, language: str = "en") -> list[PIIEntity]:
        if not text:
            return []
        lang = "fr" if language.lower().startswith("fr") else "en"
        results = self._engine.analyze(
            text=text,
            language=lang,
            entities=list(self.ENTITIES),
            score_threshold=self._threshold,
        )
        return [
            PIIEntity(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start : r.end],
                score=float(r.score),
                backend=self.name,
            )
            for r in results
        ]


@lru_cache(maxsize=1)
def get_backend() -> PIIBackend:
    """The configured backend, or regex with the reason it fell back."""
    choice = os.environ.get("PII_BACKEND", "regex").strip().lower()
    if choice == "presidio":
        ok, reason = PresidioBackend.available()
        if ok:
            return PresidioBackend()
        # Loud, but not fatal. A failed PII upgrade must not take the service
        # down; it must be obvious in the log and must leave the weaker
        # scanner running rather than none.
        print(f"[guardrails.pii] PII_BACKEND=presidio requested but {reason}; "
              f"falling back to regex")
    return RegexBackend()


# ---------------------------------------------------------------------------
# Egress
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PIIReport:
    clean: bool
    entities: tuple[PIIEntity, ...]
    backend: str
    scanned_chars: int
    # Entity types the active backend cannot detect at all. Carried in the
    # report so a caller cannot mistake "clean" for "no PII present".
    blind_spots: tuple[str, ...] = ()

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.entity_type for e in self.entities))

    def as_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "entity_count": len(self.entities),
            "types": list(self.types),
            "backend": self.backend,
            "scanned_chars": self.scanned_chars,
            "blind_spots": list(self.blind_spots),
            "entities": [e.as_dict() for e in self.entities],
        }


# Types no backend in this package can find. Stated rather than implied.
_REGEX_BLIND_SPOTS = ("PERSON", "LOCATION", "ORGANIZATION", "AGE", "NRP")


def _blind_spots(backend: PIIBackend) -> tuple[str, ...]:
    supported = set(backend.supported_entities())
    return tuple(t for t in _REGEX_BLIND_SPOTS if t not in supported)


def scan_text(text: str, *, language: str = "en") -> PIIReport:
    # Our own opaque ids are not free text -- see _is_opaque_id. scan_text is
    # the per-string entry point the eval gate uses, so the exemption has to
    # live on both paths or the gate fails on every turn that cites a unit.
    if _is_opaque_id(text):
        return PIIReport(clean=True, entities=(), backend=get_backend().name,
                         scanned_chars=len(text or ""), blind_spots=_blind_spots(get_backend()))
    backend = get_backend()
    ents = backend.analyze(text or "", language=language)
    return PIIReport(
        clean=not ents,
        entities=tuple(ents),
        backend=backend.name,
        scanned_chars=len(text or ""),
        blind_spots=_blind_spots(backend),
    )


# Identifiers this project mints itself. A `unit_id` is `u_` plus 16 hex
# characters, and a hex run like `4489100490` is ten digits, so every one of
# them looks like a phone number to a regex and roughly half look like a US ZIP.
# Scanning our own opaque ids as if they were free text is a category error, and
# left alone it makes the PII gate fail on every single turn -- which is worse
# than useless, because a gate that always fires gets switched off.
#
# The match is deliberately exact and anchored, not a field-name check: a real
# phone number cannot match `^u_[0-9a-f]{16}$`, so this cannot be used to smuggle
# anything past the scanner by naming a field `unit_id`.
#
# The deeper fix is to mint ids that cannot collide with a phone shape at all
# (a leading non-hex letter in every group). That is a KB rebuild, so it is
# recorded here rather than done silently.
_OPAQUE_ID_RE = re.compile(r"^(?:u_[0-9a-f]{16}|c_\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")


def _is_opaque_id(text: str) -> bool:
    return bool(_OPAQUE_ID_RE.match(text or ""))


def _iter_strings(node: Any, path: str = "") -> Iterable[tuple[str, str]]:
    """Every string leaf in an arbitrary structure, with its dotted path.

    Dict *keys* are yielded too. The quarantined collection leaked
    ``caller_area_code`` as a key name and the value beside it; a scanner that
    only read values would report the object clean while the key told you what
    it was.
    """
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            kp = f"{path}.{k}" if path else str(k)
            if isinstance(k, str):
                yield f"{kp}#key", k
            yield from _iter_strings(v, kp)
    elif isinstance(node, (list, tuple, set)):
        for i, v in enumerate(node):
            yield from _iter_strings(v, f"{path}[{i}]")
    elif node is not None and not isinstance(node, (bool, int, float)):
        yield path, str(node)


def scan_egress(obj: Any, *, language: str = "en") -> PIIReport:
    """Scan anything about to leave the process.

    Payload dict, SSE frame, final answer string -- all the same call. There
    is no separate ``scan_payload`` / ``scan_frame`` / ``scan_answer``,
    because three functions is three chances to call the wrong one.
    """
    backend = get_backend()
    ents: list[PIIEntity] = []
    total = 0
    for path, text in _iter_strings(obj):
        total += len(text)
        if _is_opaque_id(text):
            continue
        for e in backend.analyze(text, language=language):
            ents.append(
                PIIEntity(
                    entity_type=e.entity_type,
                    start=e.start,
                    end=e.end,
                    text=e.text,
                    score=e.score,
                    backend=e.backend,
                    path=path,
                )
            )
        # A 3CX recording filename can appear without matching the filename
        # regex when a field holds only the basename. The phone key is the
        # thing that matters, so it is extracted structurally as well.
        if wav_filename_phone(text):
            ents.append(
                PIIEntity(
                    entity_type="PBX_RECORDING_FILENAME",
                    start=0,
                    end=len(text),
                    text=text,
                    score=0.99,
                    backend="structural",
                    path=path,
                )
            )
    # Dedupe on (path, span, type): the filename rule and the structural check
    # will agree on the obvious cases.
    seen: set[tuple[str, int, int, str]] = set()
    uniq: list[PIIEntity] = []
    for e in ents:
        key = (e.path, e.start, e.end, e.entity_type)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return PIIReport(
        clean=not uniq,
        entities=tuple(uniq),
        backend=backend.name,
        scanned_chars=total,
        blind_spots=_blind_spots(backend),
    )


def redact(text: str, *, language: str = "en", min_score: float = 0.5) -> str:
    """Replace findings with their type. Right-to-left so offsets stay valid."""
    ents = [
        e
        for e in get_backend().analyze(text or "", language=language)
        if e.score >= min_score
    ]
    out = text or ""
    for e in sorted(ents, key=lambda x: x.start, reverse=True):
        out = out[: e.start] + f"[{e.entity_type}]" + out[e.end :]
    return out


def assert_clean(obj: Any, *, language: str = "en", min_score: float = 0.5) -> None:
    """Raise rather than return. For paths where a leak is not recoverable.

    The exception message names the entity types and the paths and carries no
    matched text -- an exception that logs the PII it caught has leaked it to
    the log.
    """
    report = scan_egress(obj, language=language)
    bad = [e for e in report.entities if e.score >= min_score]
    if bad:
        detail = ", ".join(sorted({f"{e.entity_type}@{e.path or '<root>'}" for e in bad}))
        raise PIIEgressError(f"PII would have left the process: {detail}")


# ---------------------------------------------------------------------------
# Self-check:  .venv/bin/python -m agentv1.guardrails.pii
# ---------------------------------------------------------------------------
def _self_check() -> int:
    from ..clients.mongo import source_db

    fails = 0

    def check(cond: bool, label: str, detail: str = "") -> None:
        nonlocal fails
        if cond:
            print(f"  ok   {label}")
        else:
            fails += 1
            print(f"  FAIL {label} {detail}")

    backend = get_backend()
    print(f"backend = {backend.name}")
    ok, reason = PresidioBackend.available()
    print(f"presidio available = {ok} ({reason})")
    print(f"blind spots of the active backend: {_blind_spots(backend)}")

    # --- fixed cases --------------------------------------------------------
    cases = [
        ("call me at 514-555-0142", "PHONE_NUMBER"),
        ("email is j.staffi@getunitronic.com", "EMAIL_ADDRESS"),
        ("VIN WAUZZZ8V1JA123456 on the door jamb", "VIN"),
        ("card 4111 1111 1111 1111 expires soon", "CREDIT_CARD"),
        ("ship to 1234 Sherbrooke Street", "STREET_ADDRESS"),
        ("postal code H2X 1Y4", "CA_POSTAL_CODE"),
        ("date of birth 1985-03-11", "DATE_OF_BIRTH"),
        ("[Coles, Zoe]_124-4032388633_20260730161447(280).wav", "PBX_RECORDING_FILENAME"),
    ]
    for text, expect in cases:
        got = {e.entity_type for e in backend.analyze(text)}
        check(expect in got, f"{expect} found in {text[:44]!r}", f"got {sorted(got)}")

    clean_cases = [
        "the Stage 2 tune adds 90 horsepower",
        "box code 8V0 906 259 K software 0001",
        "je voudrais un stage 1 pour ma Golf",
    ]
    for text in clean_cases:
        r = scan_text(text)
        check(r.clean, f"no findings in {text[:44]!r}", str(r.types))

    # Luhn actually filters.
    check(
        not any(e.entity_type == "CREDIT_CARD" for e in backend.analyze("order 4111111111111112")),
        "a 16-digit number failing Luhn is not reported as a card",
    )

    # --- nested egress ------------------------------------------------------
    frame = {
        "event": "citation",
        "data": {
            "unit_id": "u_0123456789abcdef",
            "meta": {"note": "customer reached at (403) 238-8633"},
            "sources": ["[Coles, Zoe]_124-4032388633_20260730161447(280).wav"],
        },
    }
    r = scan_egress(frame)
    check(not r.clean, "nested SSE frame with PII is not clean")
    check(
        any(e.path.startswith("data.meta.note") for e in r.entities),
        "finding is attributed to its dotted path",
        str([e.path for e in r.entities]),
    )
    check(
        "PBX_RECORDING_FILENAME" in r.types,
        "3CX recording filename detected inside a list",
        str(r.types),
    )
    check(
        all("matched_text" not in e.as_dict() and "text" not in e.as_dict()
            for e in r.entities),
        "as_dict() findings do not carry the matched PII",
    )
    check(
        all("4032388633" not in str(e) and "4032388633" not in repr(e)
            for e in r.entities),
        "str()/repr() of a finding does not carry the matched PII",
        str([str(e) for e in r.entities]),
    )
    try:
        assert_clean(frame)
        check(False, "assert_clean raises on a dirty frame")
    except PIIEgressError as exc:
        check("PBX_RECORDING_FILENAME" in str(exc), "assert_clean names the type",
              str(exc))
        check("4032388633" not in str(exc), "assert_clean message carries no PII",
              str(exc))
    check(scan_egress({"event": "token", "data": "Stage 2 is a good fit"}).clean,
          "a clean frame is clean")

    check(
        redact("call 514-555-0142 or mail bob@example.com")
        == "call [PHONE_NUMBER] or mail [EMAIL_ADDRESS]",
        "redact substitutes type names",
        redact("call 514-555-0142 or mail bob@example.com"),
    )

    # --- the recall gap, measured on live data -----------------------------
    db = source_db()
    coll = db["calls_analysis"]

    # Structured identifiers: the recording filename is on every document and
    # embeds the caller's number, so it is ground truth we can count.
    docs = list(
        coll.find(
            {"call_id": {"$mod": [97, 5]}},
            {
                "file_name": 1,
                "caller_phone_number": 1,
                "agent_name": 1,
                "customer_name_mentioned": 1,
                "full_transcription": 1,
                "language": 1,
            },
        ).limit(400)
    )
    fn_hit = sum(
        1
        for d in docs
        if d.get("file_name")
        and any(
            e.entity_type == "PBX_RECORDING_FILENAME"
            for e in backend.analyze(d["file_name"])
        )
    )
    with_fn = sum(1 for d in docs if d.get("file_name"))
    print(f"\nstructured identifiers over {with_fn} live recording filenames")
    print(f"  detected            {fn_hit}  ({100.0 * fn_hit / max(with_fn, 1):.1f}%)")

    phone_hit = sum(
        1
        for d in docs
        if d.get("caller_phone_number")
        and any(
            e.entity_type == "PHONE_NUMBER"
            for e in backend.analyze(str(d["caller_phone_number"]))
        )
    )
    with_phone = sum(1 for d in docs if d.get("caller_phone_number"))
    print(f"  bare 10-digit phone numbers detected  {phone_hit}/{with_phone}"
          f"  ({100.0 * phone_hit / max(with_phone, 1):.1f}%)")

    # Names: the corpus tells us the name and where it was said.
    named = [
        d
        for d in docs
        if d.get("customer_name_mentioned")
        and len(str(d["customer_name_mentioned"]).strip()) >= 3
        and str(d["customer_name_mentioned"]) in (d.get("full_transcription") or "")
    ]
    name_found = 0
    for d in named:
        name = str(d["customer_name_mentioned"])
        idx = d["full_transcription"].index(name)
        window = d["full_transcription"][max(0, idx - 200) : idx + 200]
        offset = min(idx, 200)
        if any(
            e.start <= offset < e.end and e.entity_type in ("PERSON", "NRP")
            for e in backend.analyze(window, language=d.get("language") or "en")
        ):
            name_found += 1
    print(f"\nPERSON recall over {len(named)} calls where the corpus records the")
    print("customer's spoken name AND that exact name appears in the transcript")
    print(f"  detected            {name_found}  "
          f"({100.0 * name_found / max(len(named), 1):.1f}%)")
    if backend.name == "regex":
        print("  This is the 9.8 argument, measured. A regex backend cannot find a")
        print("  name, so it finds none, and no amount of pattern work changes")
        print("  that. Until PII_BACKEND=presidio, the payload allowlist in")
        print("  index/payload.py is the control that matters and this module")
        print("  is defence in depth behind it.")

    agent_named = [d for d in docs if d.get("agent_name") and d.get("full_transcription")]
    leaks = sum(
        1
        for d in agent_named
        if str(d["agent_name"]).split()[0] in d["full_transcription"]
    )
    print(f"\n  for scale: the agent's own first name appears in the transcript on")
    print(f"  {leaks} of {len(agent_named)} sampled calls, and review.agent_performance,")
    print("  coaching_note and incorrect_statements name staff directly (9.7).")

    print(f"\nself-check {'PASS' if fails == 0 else 'FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
