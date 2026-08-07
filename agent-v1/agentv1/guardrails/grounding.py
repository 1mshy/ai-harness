"""Post-generation grounding. Makes an ungrounded price claim unspeakable.

AGENT_PLAN.md 9.5 and CONTRACT.md non-negotiable 4. Prices, compatibility and
stage availability are the three places a wrong answer costs money, and they
are the three that go stale: 6,113 knowledge units contain a literal dollar
amount, 2,063 mention Stage 3, 453 say "coming soon" or "not yet released",
over a corpus spanning 2024-2026, against a platforms table whose stage rows
change under it.

The weak version of this control is a prompt sentence telling the model not to
quote prices. ``templates.py:38`` already is that sentence, and the corpus it
retrieves from is full of 2024 prices, so the model is being asked to ignore
the most quotable thing in its own context window. This module is the strong
version: after generation, before the answer leaves the process, every
currency amount and every stage/availability claim in the draft must be
traceable to a tool result **produced in this turn**. If it is not, the answer
does not ship -- it is replaced and the turn routes to handoff.

WHY A PROVENANCE TOKEN AND NOT "DID A TOOL RUN"
    Two failure modes that "a tool ran this turn" does not catch:

    * A tool ran and returned $1,299; the model wrote $1,499. Checking that a
      tool ran passes this. Checking the *amount* against the tool's payload
      does not.
    * The conversation object still holds tool results from turn 3 and the
      model is answering turn 7. Prices move between turns; a stale result is
      exactly the ungrounded claim this exists to stop.

    So every tool result is minted with an HMAC over ``(turn_id, tool_name,
    call_id)`` under a process secret. The secret never enters a prompt, so a
    token cannot be produced by generation -- only by having actually called
    the tool, in this turn. Validation re-derives the HMAC; a forged or
    replayed token fails :func:`verify`.

WHAT COUNTS AS A MATCH
    Amounts match numerically, not textually: the draft's "$1,299" is grounded
    by a payload containing ``1299``, ``1299.00`` or ``"$1,299.00"``. Stage
    availability matches on the marker token from ``text.normalize`` --
    ``stage_2`` and ``stage_2_plus`` are different products at different prices
    and a substring comparison collapses them, which is the entire reason
    ``text/normalize.py`` exists.

FAIL-CLOSED, ALWAYS
    Every error path in this module returns ``ok=False``. An exception while
    parsing a tool payload, an unparseable amount, an empty evidence set -- all
    of them block. A grounding validator that fails open is decoration.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Iterable, Literal, Sequence

from ..text.normalize import stage_tokens

Action = Literal["allow", "block_and_handoff"]

# Per-process secret. Deliberately not read from config and not persisted: a
# token must not survive a restart, because a token that survives a restart is
# a token that can be replayed from a log. Overridable only for tests that
# need two processes to agree.
_SECRET = os.environ.get("AGENTV1_PROVENANCE_SECRET") or secrets.token_hex(32)


def mint_provenance(turn_id: str, tool_name: str, call_id: str) -> str:
    """Token stamped onto a tool result at the moment the tool returns."""
    msg = f"{turn_id}\x00{tool_name}\x00{call_id}".encode()
    return hmac.new(_SECRET.encode(), msg, sha256).hexdigest()[:32]


def verify(token: str, turn_id: str, tool_name: str, call_id: str) -> bool:
    return hmac.compare_digest(token or "", mint_provenance(turn_id, tool_name, call_id))


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
_NUM_IN_TEXT = re.compile(r"(?<![\w.])(\d{1,3}(?:[ ,]\d{3})+|\d+)(?:[.,](\d{1,2}))?(?![\w])")


def _to_decimal(whole: str, frac: str | None) -> Decimal | None:
    try:
        return Decimal(
            re.sub(r"[ ,\u00a0\u202f]", "", whole) + ("." + frac if frac else "")
        )
    except (InvalidOperation, ValueError):
        return None


def _walk(node: Any) -> Iterable[Any]:
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, (list, tuple, set)):
        for v in node:
            yield from _walk(v)
    else:
        yield node


def _amounts_in(payload: Any) -> set[Decimal]:
    """Every number anywhere in a tool payload, as a Decimal.

    Deliberately generous. This is the *supporting* side of the check, and a
    tool that returns ``{"price_cents": 129900}`` should still ground "$1,299"
    -- so cents are folded in as well. Being generous here only ever lets a
    number through that the tool really did return; the restrictive half of
    the check is the draft side, which extracts only things written as money.
    """
    out: set[Decimal] = set()
    for leaf in _walk(payload):
        if isinstance(leaf, bool) or leaf is None:
            continue
        if isinstance(leaf, (int, float, Decimal)):
            try:
                d = Decimal(str(leaf))
            except InvalidOperation:
                continue
            out.add(d)
            if d == d.to_integral_value() and abs(d) >= 100:
                out.add(d / 100)  # price_cents -> dollars
        elif isinstance(leaf, str):
            for m in _NUM_IN_TEXT.finditer(leaf):
                d = _to_decimal(m.group(1), m.group(2))
                if d is not None:
                    out.add(d)
                    if d == d.to_integral_value() and abs(d) >= 100:
                        out.add(d / 100)
    return out


# Stage availability keys a tool may declare. Matches the live
# `tuning_platforms` document shape (verified 2026-08-06): `stages` is a list
# of {label, released, ...}, plus the flattened `released_stage_labels` /
# `unreleased_stage_labels` / `max_released_stage`.
def _stage_availability_in(payload: Any) -> dict[str, bool]:
    """stage marker token -> released?, read out of a tool payload."""
    out: dict[str, bool] = {}

    def note(label: Any, released: Any) -> None:
        for tok in stage_tokens(str(label or "")):
            # False beats True: if any row says a stage is unreleased for this
            # platform, the agent may not call it available.
            out[tok] = bool(released) and out.get(tok, True)

    def rec(node: Any) -> None:
        if isinstance(node, dict):
            if "label" in node and "released" in node:
                note(node["label"], node["released"])
            for key in ("released_stage_labels", "max_released_stage"):
                v = node.get(key)
                for lbl in v if isinstance(v, list) else ([v] if v else []):
                    note(lbl, True)
            for lbl in node.get("unreleased_stage_labels") or []:
                note(lbl, False)
            for v in node.values():
                rec(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                rec(v)

    rec(payload)
    return out


@dataclass(frozen=True)
class ToolEvidence:
    """A tool result, bound to the turn that produced it."""

    turn_id: str
    tool_name: str
    call_id: str
    provenance: str
    payload: Any = None
    amounts: frozenset[Decimal] = field(default_factory=frozenset)
    stage_availability: dict[str, bool] = field(default_factory=dict)

    def is_valid_for(self, turn_id: str) -> bool:
        return self.turn_id == turn_id and verify(
            self.provenance, turn_id, self.tool_name, self.call_id
        )


def record_tool_result(
    turn_id: str, tool_name: str, call_id: str, payload: Any
) -> ToolEvidence:
    """Call this on EVERY tool return. The token is minted here and nowhere else."""
    return ToolEvidence(
        turn_id=turn_id,
        tool_name=tool_name,
        call_id=call_id,
        provenance=mint_provenance(turn_id, tool_name, call_id),
        payload=payload,
        amounts=frozenset(_amounts_in(payload)),
        stage_availability=_stage_availability_in(payload),
    )


# ---------------------------------------------------------------------------
# Claim extraction from the draft
# ---------------------------------------------------------------------------
# Money as a human writes it, in both languages. French Canadian puts the sign
# after the number and uses a comma decimal and a narrow/regular space
# thousands separator: "1 299,99 $".
# `_NUM` allows the space, comma, no-break space and narrow no-break space
# thousands separators, because "1 299,99 $" arrives from a French web form
# with U+202F in it and from an agent's keyboard with a plain space.
# The separated alternative takes `+`, not `*`. With `*` it matches "149"
# out of "1499" -- the first branch succeeds on three digits, the separator
# group matches empty, and the regex never reaches `\\d+`.
_NUM = r"\d{1,3}(?:[ ,\u00a0\u202f]\d{3})+|\d+"
_MONEY_RE = re.compile(
    # $1,299.99  /  US$1299  /  CA$1299
    rf"(?:US|CA)?\$\s*(?P<w1>{_NUM})(?:[.,](?P<f1>\d{{1,2}}))?"
    # USD 1499.00
    rf"|\b(?:USD|CAD|EUR)\s*(?P<w2>{_NUM})(?:[.,](?P<f2>\d{{1,2}}))?"
    # 1 299,99 $   /   850 dollars   /   1299 CAD
    rf"|(?P<w3>{_NUM})(?:[.,](?P<f3>\d{{1,2}}))?\s*(?:\$|\b(?:dollars?|CAD|USD|bucks)\b)",
    re.IGNORECASE,
)

# Availability / release assertions. Kept separate from the stage regex because
# "coming soon" is a claim about the future with no stage attached and is just
# as unquotable without a tool.
_AVAILABILITY_RE = re.compile(
    r"\b(?:is|are|it'?s|we\s+have|there'?s|currently)?\s*"
    r"(?:available|released|out\s+now|in\s+stock|shipping\s+now|ready\s+to\s+order"
    r"|coming\s+soon|not\s+yet\s+released|unreleased|out\s+of\s+stock|back\s*ordered"
    r"|disponible|en\s+stock|bient[oô]t\s+disponible|pas\s+encore\s+(?:sorti|disponible))\b",
    re.IGNORECASE,
)

_NEGATIVE_AVAIL = re.compile(
    r"\b(?:coming\s+soon|not\s+yet\s+released|unreleased|out\s+of\s+stock|back\s*ordered"
    r"|pas\s+encore\s+(?:sorti|disponible)|bient[oô]t\s+disponible)\b",
    re.IGNORECASE,
)

# Sentences that ask rather than assert. "Would you like me to check whether
# Stage 3 is available?" is not a claim and blocking it would make the agent
# unable to offer to do the lookup that would ground it.
_HEDGE_RE = re.compile(
    r"\b(?:can\s+i\s+check|let\s+me\s+check|i\s+can\s+check|i'?ll\s+check|would\s+you\s+like"
    r"|i\s+don'?t\s+have|i\s+can'?t\s+confirm|i\s+am\s+not\s+able\s+to\s+confirm"
    r"|je\s+peux\s+v[eé]rifier|je\s+ne\s+peux\s+pas\s+confirmer)\b",
    re.IGNORECASE,
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")

ClaimKind = Literal["price", "stage_availability", "availability"]


@dataclass(frozen=True)
class Claim:
    kind: ClaimKind
    text: str
    span: tuple[int, int]
    amount: Decimal | None = None
    stage_token: str | None = None
    asserts_available: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "span": list(self.span),
            "amount": str(self.amount) if self.amount is not None else None,
            "stage_token": self.stage_token,
            "asserts_available": self.asserts_available,
        }


def extract_claims(draft: str) -> list[Claim]:
    """Every money amount and availability assertion in a draft answer."""
    claims: list[Claim] = []
    if not draft:
        return claims

    for m in _MONEY_RE.finditer(draft):
        whole = m.group("w1") or m.group("w2") or m.group("w3")
        frac = m.group("f1") or m.group("f2") or m.group("f3")
        amount = _to_decimal(whole, frac) if whole else None
        if amount is None:
            continue
        claims.append(
            Claim(kind="price", text=m.group(0).strip(), span=m.span(), amount=amount)
        )

    pos = 0
    for sm in _SENTENCE_RE.finditer(draft):
        sent = sm.group(0)
        pos = sm.start()
        if not sent.strip():
            continue
        if _HEDGE_RE.search(sent):
            continue
        avail = _AVAILABILITY_RE.search(sent)
        if not avail:
            continue
        asserts_available = _NEGATIVE_AVAIL.search(sent) is None
        toks = stage_tokens(sent)
        if toks:
            for tok in dict.fromkeys(toks):
                claims.append(
                    Claim(
                        kind="stage_availability",
                        text=sent.strip(),
                        span=(pos, sm.end()),
                        stage_token=tok,
                        asserts_available=asserts_available,
                    )
                )
        else:
            claims.append(
                Claim(
                    kind="availability",
                    text=sent.strip(),
                    span=(pos, sm.end()),
                    asserts_available=asserts_available,
                )
            )
    claims.sort(key=lambda c: (c.span[0], c.kind))
    return claims


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Violation:
    claim: Claim
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim.as_dict(), "reason": self.reason}


HANDOFF_MESSAGE = {
    "en": (
        "I don't want to give you a number I can't stand behind. Pricing and "
        "stage availability change, and I wasn't able to confirm this one "
        "against our live system just now -- so I'm passing you to someone on "
        "the team who can quote it properly."
    ),
    "fr": (
        "Je ne veux pas vous donner un chiffre que je ne peux pas confirmer. "
        "Les prix et la disponibilite des stages changent, et je n'ai pas pu "
        "verifier celui-ci dans notre systeme a l'instant -- je vous transfere "
        "donc a un membre de l'equipe qui pourra vous faire une soumission "
        "exacte."
    ),
}


@dataclass(frozen=True)
class GroundingVerdict:
    ok: bool
    action: Action
    claims: tuple[Claim, ...] = ()
    violations: tuple[Violation, ...] = ()
    answer: str | None = None
    handoff_reason: str | None = None
    evidence_used: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "claims": [c.as_dict() for c in self.claims],
            "violations": [v.as_dict() for v in self.violations],
            "answer": self.answer,
            "handoff_reason": self.handoff_reason,
            "evidence_used": list(self.evidence_used),
        }


def validate(
    draft: str,
    evidence: Sequence[ToolEvidence],
    *,
    turn_id: str,
    language: str = "en",
) -> GroundingVerdict:
    """Gate a draft answer. Returns the answer to send, or a handoff.

    ``evidence`` may contain results from other turns; they are filtered out
    here rather than trusted, because the caller holding a conversation-scoped
    list and passing all of it is the normal and expected mistake.
    """
    lang = "fr" if language.lower().startswith("fr") else "en"

    try:
        claims = extract_claims(draft)
    except Exception as exc:  # noqa: BLE001 -- fail closed, always
        return GroundingVerdict(
            ok=False,
            action="block_and_handoff",
            answer=HANDOFF_MESSAGE[lang],
            handoff_reason=f"claim extraction failed: {exc!r}",
        )

    live = [e for e in evidence if e.is_valid_for(turn_id)]
    amounts: set[Decimal] = set()
    stages: dict[str, bool] = {}
    for e in live:
        amounts |= set(e.amounts)
        for tok, released in e.stage_availability.items():
            stages[tok] = released and stages.get(tok, True)

    violations: list[Violation] = []
    for c in claims:
        if c.kind == "price":
            if c.amount not in amounts:
                violations.append(
                    Violation(
                        claim=c,
                        reason=(
                            "no tool result in this turn contains the amount "
                            f"{c.amount}"
                            if live
                            else "no valid tool result in this turn"
                        ),
                    )
                )
        elif c.kind == "stage_availability":
            known = stages.get(c.stage_token)
            if known is None:
                violations.append(
                    Violation(
                        claim=c,
                        reason=(
                            f"no tool result in this turn reports availability for "
                            f"{c.stage_token}"
                        ),
                    )
                )
            elif known != c.asserts_available:
                violations.append(
                    Violation(
                        claim=c,
                        reason=(
                            f"draft asserts available={c.asserts_available} for "
                            f"{c.stage_token}; tool reports available={known}"
                        ),
                    )
                )
        else:  # bare availability claim, no stage named
            if not live:
                violations.append(
                    Violation(
                        claim=c,
                        reason="availability asserted with no valid tool result in this turn",
                    )
                )
            elif not stages and not amounts:
                violations.append(
                    Violation(
                        claim=c,
                        reason=(
                            "availability asserted; tool results in this turn carry "
                            "no availability or pricing facts to support it"
                        ),
                    )
                )

    if violations:
        return GroundingVerdict(
            ok=False,
            action="block_and_handoff",
            claims=tuple(claims),
            violations=tuple(violations),
            answer=HANDOFF_MESSAGE[lang],
            handoff_reason="; ".join(v.reason for v in violations[:3]),
            evidence_used=tuple(f"{e.tool_name}:{e.call_id}" for e in live),
        )
    return GroundingVerdict(
        ok=True,
        action="allow",
        claims=tuple(claims),
        answer=draft,
        evidence_used=tuple(f"{e.tool_name}:{e.call_id}" for e in live),
    )


# ---------------------------------------------------------------------------
# Self-check:  .venv/bin/python -m agentv1.guardrails.grounding
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

    # --- provenance ---------------------------------------------------------
    tok = mint_provenance("t1", "price_lookup", "c1")
    check(verify(tok, "t1", "price_lookup", "c1"), "token verifies for its own turn")
    check(not verify(tok, "t2", "price_lookup", "c1"), "token rejected on another turn")
    check(not verify(tok, "t1", "order_lookup", "c1"), "token rejected for another tool")
    check(not verify("0" * 32, "t1", "price_lookup", "c1"), "forged token rejected")

    # --- claim extraction ---------------------------------------------------
    money_drafts = [
        ("The Stage 2 is $1,299 plus tax.", Decimal("1299")),
        ("It runs 1 299,99 $ au total.", Decimal("1299.99")),
        ("That'll be 850 dollars.", Decimal("850")),
        ("Around USD 1499.00 shipped.", Decimal("1499.00")),
    ]
    for txt, expect in money_drafts:
        got = [c.amount for c in extract_claims(txt) if c.kind == "price"]
        check(expect in got, f"money extracted from {txt!r}", f"got {got}")

    stage_draft = "Stage 3 is available for your car right now."
    cl = [c for c in extract_claims(stage_draft) if c.kind == "stage_availability"]
    check(
        len(cl) == 1 and cl[0].stage_token == "stage_3" and cl[0].asserts_available,
        "stage availability claim extracted",
        f"got {[c.as_dict() for c in cl]}",
    )
    plus = extract_claims("Stage 2+ is not yet released for that platform.")
    check(
        any(c.stage_token == "stage_2_plus" and c.asserts_available is False for c in plus),
        "Stage 2+ distinguished from Stage 2 and polarity read",
        f"got {[c.as_dict() for c in plus]}",
    )
    check(
        not extract_claims("Let me check whether Stage 3 is available for you."),
        "offer-to-check is not a claim",
    )
    check(not extract_claims("The Stage 2 tune adds about 90 horsepower."), "non-money number ignored")

    # --- validation against a REAL platform document ------------------------
    db = source_db()
    # Most `unreleased_stage_labels` entries are big-turbo injector sizes
    # ("BT 415CC"), which carry no stage marker token. Pick a row whose
    # unreleased label really is a Stage, so the false-availability path is
    # exercised against production data rather than a fixture.
    plat = db["tuning_platforms"].find_one(
        {"unreleased_stage_labels": {"$regex": "^Stage", "$options": "i"}},
        sort=[("_id", 1)],
    )
    check(plat is not None, "found a live platform row with an unreleased Stage")
    assert plat is not None
    unreleased = next(
        l for l in plat["unreleased_stage_labels"] if stage_tokens(l)
    )
    released = next(
        l
        for l in plat["released_stage_labels"]
        if stage_tokens(l) and stage_tokens(l) != stage_tokens(unreleased)
    )
    print(f"  using platform _id={plat['_id']} released={released!r} "
          f"unreleased={unreleased!r}")

    ev = record_tool_result("turn-1", "platform_lookup", "call-1", plat)
    print(f"  evidence: {len(ev.amounts)} numeric facts, "
          f"{len(ev.stage_availability)} stage facts -> {ev.stage_availability}")

    v = validate(f"{released} is available for that platform.", [ev], turn_id="turn-1")
    check(v.ok, f"true claim about released {released!r} passes", v.handoff_reason or "")

    v = validate(f"{unreleased} is available for that platform.", [ev], turn_id="turn-1")
    check(
        not v.ok and v.action == "block_and_handoff",
        f"false claim about unreleased {unreleased!r} is blocked",
        str(v.violations),
    )
    v = validate(f"{unreleased} is not yet released.", [ev], turn_id="turn-1")
    check(v.ok, f"true negative claim about {unreleased!r} passes", v.handoff_reason or "")

    # Real conflict: platform _id=2 lists "Stage 2+" in BOTH released and
    # unreleased. The rule is that false wins -- if any row on the platform
    # says a stage is unreleased, the agent may not call it available.
    conflicted = db["tuning_platforms"].find_one(
        {
            "released_stage_labels": "Stage 2+",
            "unreleased_stage_labels": "Stage 2+",
        }
    )
    if conflicted is not None:
        cev = record_tool_result("turn-1", "platform_lookup", "call-9", conflicted)
        check(
            cev.stage_availability.get("stage_2_plus") is False,
            f"platform _id={conflicted['_id']} lists Stage 2+ as both released and "
            f"unreleased; resolved to unavailable",
            str(cev.stage_availability),
        )
        check(
            not validate("Stage 2+ is available.", [cev], turn_id="turn-1").ok,
            "conflicting availability blocks the claim",
        )
    else:
        print("  note: no platform currently lists the same stage as both "
              "released and unreleased; the conflict path was not exercised")

    # price paths
    priced = record_tool_result("turn-1", "price_lookup", "call-2", {"sku": "S2", "price": 1299})
    v = validate("The Stage 2 file is $1,299.", [priced], turn_id="turn-1")
    check(v.ok, "price matching a tool result passes", v.handoff_reason or "")
    v = validate("The Stage 2 file is $1,499.", [priced], turn_id="turn-1")
    check(not v.ok, "price NOT in the tool result is blocked")
    v = validate("The Stage 2 file is $1,299.", [], turn_id="turn-1")
    check(not v.ok, "price with no tool result at all is blocked")
    stale = record_tool_result("turn-0", "price_lookup", "call-2", {"price": 1299})
    v = validate("The Stage 2 file is $1,299.", [stale], turn_id="turn-1")
    check(not v.ok, "price grounded only by a PREVIOUS turn's tool result is blocked")
    check(
        v.answer == HANDOFF_MESSAGE["en"],
        "blocked answer is replaced by the handoff message",
    )
    v = validate("Le fichier Stage 2 coute 1 299 $.", [priced], turn_id="turn-1", language="fr")
    check(v.ok, "French money form matches the same tool result", v.handoff_reason or "")
    v = validate("Le fichier coute 1 499 $.", [priced], turn_id="turn-1", language="fr")
    check(
        not v.ok and v.answer == HANDOFF_MESSAGE["fr"],
        "French block returns the French handoff message",
    )
    check(
        validate("I can look up the current price for you.", [], turn_id="turn-1").ok,
        "an answer with no claims needs no evidence",
    )
    cents = record_tool_result("turn-1", "price_lookup", "call-3", {"price_cents": 129900})
    check(
        validate("It's $1,299.", [cents], turn_id="turn-1").ok,
        "cents-denominated tool payload grounds a dollar claim",
    )

    # --- what the corpus would do to an ungrounded agent --------------------
    coll = db["calls_analysis"]
    n_units = 0
    n_money = 0
    n_avail = 0
    for d in coll.find({"knowledge_units.0": {"$exists": True}},
                       {"knowledge_units": 1}).limit(4000):
        for u in d.get("knowledge_units") or []:
            n_units += 1
            text = " ".join(
                str(u.get(f) or "") for f in ("title", "question", "answer", "conditions")
            )
            cs = extract_claims(text)
            if any(c.kind == "price" for c in cs):
                n_money += 1
            if any(c.kind != "price" for c in cs):
                n_avail += 1
    print(f"\nover {n_units} knowledge units from 4000 calls:")
    print(f"  units carrying a currency amount        {n_money}"
          f"  ({100.0 * n_money / max(n_units, 1):.1f}%)")
    print(f"  units carrying an availability claim    {n_avail}"
          f"  ({100.0 * n_avail / max(n_units, 1):.1f}%)")
    print("  Every one of these is quotable by a model with the unit in "
          "context and\n  no tool call. That is the population this module "
          "exists to intercept.")

    unrel_rows = sum(
        1
        for p in db["tuning_platforms"].find({}, {"stages": 1})
        for s in (p.get("stages") or [])
        if s.get("released") is False
    )
    print(f"  currently-unreleased stage rows in tuning_platforms: {unrel_rows}")

    print(f"\nself-check {'PASS' if fails == 0 else 'FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
