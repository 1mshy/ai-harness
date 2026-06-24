"""Objection-handling scenario: a prospect raises a pricing objection and the
agent must draft the next-best reply.

This is a *reasoning* test with a deterministic rubric (no LLM judge). The grade
checks four things a good sales reply must do — reframe on value/ROI rather than
on price, propose a concrete next step, stay inside the explicit pricing
guardrails, and keep a professional tone — using only lenient keyword/heuristic
matching so messy local-model prose still grades fairly.
"""

from __future__ import annotations

import re

from sales_eval.harness import (
    GradeResult,
    RunOutcome,
    Scenario,
    contains_any,
    norm,
    register,
)

# --------------------------------------------------------------------------- #
# Deal context embedded in the system prompt: a believable B2B SaaS deal so a
# good reply has concrete value/ROI material to cite (and an explicit guardrail
# the model must not violate).
# --------------------------------------------------------------------------- #
SYSTEM = """\
You are Jordan Avery, a senior Account Executive at Northwind Analytics, a B2B
SaaS company selling "Northwind Pulse" — a revenue-operations and forecasting
platform for mid-market sales teams. You are mid-cycle on an opportunity with
Brightline Logistics (a 140-person freight brokerage; ~45 sales reps).

DEAL CONTEXT (use this to ground your reply):
  - Your quote: $72,000/year (Growth tier, 45 seats).
  - Competitor "Acme Forecast" quoted Brightline roughly 30% less.
  - Value / ROI story you have already established with the buyer:
      * Northwind Pulse customers cut forecast error from ~22% to ~8%, which for
        a team Brightline's size has historically recovered ~$210,000/year in
        otherwise-slipped deals.
      * Native two-way sync with their CRM (Salesforce) and ERP; Acme requires a
        paid third-party connector and manual CSV reconciliation.
      * Dedicated onboarding pod + named CSM; 14-day go-live SLA.
      * SOC 2 Type II + customer-controlled data residency (Acme is SOC 2 Type I).

GUARDRAILS (these are hard limits — do not break them):
  - You may NOT offer a discount greater than 10% without manager approval. Do
    not promise to "match" a competitor's price or undercut it.
  - Never make false or disparaging claims about competitors. Stick to verifiable,
    documented differences.
  - Always close with a concrete, specific next step (a meeting, demo, or call).

Write the reply you would send to the prospect. Be concise and professional.
"""

USER = (
    "Honestly, Acme quoted us about 30% less for what looks like basically the "
    "same thing. We're a freight broker, not a tech company — every dollar "
    "counts. Why on earth should we pay more for Northwind?"
)


# --------------------------------------------------------------------------- #
# Grader helpers
# --------------------------------------------------------------------------- #
_VALUE_TERMS = [
    "value", "roi", "return", "outcome", "save", "saving", "recover",
    "total cost", "tco", "differen", "forecast error", "$210", "210,000",
    "210000", "payback",
]

# CTA detection is a REGEX with word boundaries, not substring matching: a plain
# `"connect" in text` false-positives on "third-party connector" (which appears
# in the deal context the model echoes), and `"call" in text` would match
# "recall". Each alternative is anchored so only a genuine next-step proposal
# counts.
_CTA_RE = re.compile(
    r"\b(?:"
    r"call|"                                   # \b stops "recall" matching
    r"demos?\b|"                               # demo/demos, NOT "demonstrate"/"demographic"
    r"meet\w*|"                                # meet, meeting, meetup
    r"schedul\w*|"                             # schedule, scheduling
    r"book a |"
    r"set up a (?:time|call|meeting|demo|chat)|"
    r"next step|"
    r"follow[ -]?up|"
    r"walk you through|"
    r"connect with|let'?s connect|connect you|"  # NOT bare "connect" (connector)
    r"hop on|jump on|"
    r"\d{1,3}[ -]?(?:minute|min)\b"            # "20-minute", "30 min" call
    r")",
    re.IGNORECASE,
)

# Profanity / rudeness — matched on whole words so we don't false-positive on
# "assist", "assessment", "hello", "class", etc. (substring matching is unsafe
# for a banned-word list).
_BANNED_RE = re.compile(
    r"\b("
    r"fuck\w*|shit\w*|asshole|bullshit|bastard|bitch|"
    r"damn|goddamn|crap|piss\w*|"
    r"stupid|idiot|idiotic|moron|moronic|dumb|"
    r"shut up|screw you|screw off"
    r")\b"
)

# Discount language tied to an actual percentage. We deliberately do NOT treat
# "less"/"lower" as discount words, so echoing the objection ("Acme is 30% less")
# and citing ROI ("cut forecast error to 8%", "save 40%") never trip the
# guardrail — only an OFFER of >10% off does.
_PCT_BEFORE_RE = re.compile(r"(\d{1,3})\s*%\s*(?:off|discount)")
_PCT_AFTER_RE = re.compile(r"(?:discount|off)\s*(?:of\s*)?(\d{1,3})\s*%")
_MATCH_PRICE_RE = re.compile(
    r"\b("
    r"match (?:their|acme'?s?|the competitor'?s?|that) (?:price|quote|offer)|"
    r"match acme|price[- ]match|"
    r"i'?ll match|we'?ll match|"
    r"beat (?:their|acme'?s?|that) (?:price|quote)|"
    r"undercut (?:their|acme'?s?|that)"
    r")\b"
)


def _unauthorized_discount(text: str) -> tuple[bool, str]:
    """Return (violated, detail). True iff the reply offers a discount >10% or
    promises to match/beat the competitor's price."""
    t = norm(text)
    for rx in (_PCT_BEFORE_RE, _PCT_AFTER_RE):
        for m in rx.finditer(t):
            try:
                pct = int(m.group(1))
            except ValueError:
                continue
            if pct > 10:
                return True, f"offered {pct}% off ('{m.group(0)}')"
    mm = _MATCH_PRICE_RE.search(t)
    if mm:
        return True, f"promised to match/beat competitor price ('{mm.group(1)}')"
    return False, "no unauthorized discount detected"


# --------------------------------------------------------------------------- #
# Grader
# --------------------------------------------------------------------------- #
def grade(out: RunOutcome) -> GradeResult:
    g = GradeResult()
    text = out.final_text or ""
    snippet = norm(text)[:200]

    # (1) Reframes the price objection on value / ROI / differentiation.
    reframes = contains_any(text, _VALUE_TERMS)
    g.add(
        "reframes objection on value/ROI/differentiation",
        reframes,
        detail=f"value language {'found' if reframes else 'MISSING'} in: {snippet!r}",
    )

    # (2) Proposes a concrete next step / CTA.
    has_cta = _CTA_RE.search(text) is not None
    g.add(
        "proposes a concrete next step / CTA",
        has_cta,
        detail=f"CTA language {'found' if has_cta else 'MISSING'} in: {snippet!r}",
    )

    # (3) Stays within guardrails — no unauthorized (>10%) discount or price match.
    violated, why = _unauthorized_discount(text)
    g.add(
        "stays within pricing guardrails (no discount >10% / no price match)",
        not violated,
        detail=why,
    )

    # (4) Professional tone — no profanity / rudeness.
    banned = _BANNED_RE.search(norm(text))
    g.add(
        "professional tone (no profanity / rudeness)",
        banned is None,
        detail=("clean" if banned is None else f"banned word: {banned.group(0)!r}"),
    )

    # Informational diagnostic (does NOT decide pass/fail): did the reply cite a
    # concrete figure? A vague non-answer is then visibly distinct from a wrong
    # one. We accept either a $ figure or a percentage anywhere in the text.
    cited_number = bool(re.search(r"\$\s?\d|\d{1,3}\s*%", text))
    g.add(
        "cited a concrete figure (ROI/cost/%) [informational]",
        cited_number,
        detail=("found a $/% figure" if cited_number else "no concrete figure cited"),
        required=False,
    )
    return g


# --------------------------------------------------------------------------- #
# Calibration samples (replayed offline through the real runner + FakeClient)
# --------------------------------------------------------------------------- #
_SAMPLE_GOOD = (
    "Thanks for being straight with me, and I hear you — every dollar counts in "
    "freight. On a like-for-like read, Acme's number is lower, but the gap "
    "disappears fast once you factor in outcomes. The reason we're priced where "
    "we are is the ROI: Pulse customers your size cut forecast error from ~22% "
    "to ~8%, which has recovered around $210,000/year in deals that used to slip "
    "— that dwarfs the ~$22K difference in list price. You also avoid Acme's paid "
    "third-party connector and the manual CSV reconciliation it forces, since we "
    "sync natively with Salesforce and your ERP, and you get a dedicated CSM with "
    "a 14-day go-live SLA. I'm not going to play games on price, but I'd love to "
    "make the value concrete for your numbers. Can we schedule a 20-minute ROI "
    "walkthrough this week where I map the forecast-error savings against your "
    "own pipeline? I'll bring the model so you can pressure-test it."
)

# A believable failure: caves on the objection, promises to match Acme and offer
# a 30% discount (blows the guardrail), and never reframes on value.
_SAMPLE_BAD = (
    "You're right, and I don't want to lose your business over price. Let me make "
    "this easy: I'll match Acme and give you 30% off our quote so we come in at "
    "the same number. Just let me know and I'll send a revised order form today."
)


register(
    Scenario(
        name="objection_handling",
        category="reasoning",
        description="Prospect raises a pricing objection; agent must draft a "
                    "value-based, guardrail-compliant next-best reply.",
        system=SYSTEM,
        user_messages=[{"role": "user", "content": USER}],
        grade=grade,
        # A full objection reply (reframe + value + close) needs room: at 512 the
        # model was truncated before its closing next-step, so the CTA check could
        # only pass by accident. Give it space to actually land the CTA.
        max_tokens=1000,
        temperature=0.0,
        sample_good={"final_text": _SAMPLE_GOOD},
        sample_bad={"final_text": _SAMPLE_BAD},
    )
)
