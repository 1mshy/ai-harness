"""Grounded pricing Q&A — answer strictly from an embedded price sheet (RAG-style).

Tests two things sales agents fail at constantly: (1) reading the *right row* of a
price/feature matrix instead of a neighboring tier, and (2) not hallucinating a
feature's availability. The system prompt carries a complete, unambiguous B2B SaaS
price sheet; the user asks a question that is fully answerable from it but requires
one small arithmetic step (per-seat price x seats x 12 months, minus the annual
discount). A correct answer must quote the right computed total AND correctly state
that SSO is included on Pro.
"""

from __future__ import annotations

from sales_eval.harness import (
    GradeResult,
    RunOutcome,
    Scenario,
    contains_any,
    has_number,
    norm,
    register,
)

# --------------------------------------------------------------------------- #
# Fixture: the price sheet the model must ground on.
#
#   Pro:  $49 / seat / month.  Annual billing = 20% off.
#   Question: 25 seats, annual, Pro.
#     monthly list  = 49 * 25      = $1,225 / month
#     annual list   = 1,225 * 12   = $14,700 / year   (the "no discount" distractor)
#     annual w/ 20% = 14,700 * 0.8 = $11,760 / year   (the CORRECT total)
#   SSO is included on Pro and Enterprise (NOT on Starter).
# --------------------------------------------------------------------------- #
PRICE_SHEET = """\
ACME Analytics — Pricing & Plans (effective 2026, USD)

All prices are PER SEAT, PER MONTH, billed monthly by default.
ANNUAL BILLING: pay for 12 months up front and receive a 20% discount on the
total. (i.e. annual total = list monthly price x seats x 12 x 0.80.)

------------------------------------------------------------------------------
TIER         PRICE/SEAT/MO   SSO     SLA           API CALLS/MO   SUPPORT
------------------------------------------------------------------------------
Starter      $19             No      None          50,000         Email (48h)
Pro          $49             Yes     99.5% uptime   1,000,000      Email + Chat (8h)
Enterprise   $99             Yes     99.9% uptime   Unlimited      Dedicated CSM (1h)
------------------------------------------------------------------------------

Notes:
- SSO (SAML / OIDC single sign-on) is bundled at no extra cost on Pro and
  Enterprise. It is NOT available on Starter.
- The 99.9% uptime SLA with financial credits is Enterprise-only.
- Seat minimums: Starter 1, Pro 5, Enterprise 25.
- Volume discounts beyond the annual 20% require a custom Enterprise quote.
"""

SYSTEM = (
    "You are a sales engineer for ACME Analytics. Answer the prospect's question "
    "STRICTLY and ONLY from the price sheet below. Do not invent prices, discounts, "
    "or features. If something is not in the sheet, say so. Show the arithmetic for "
    "any total you quote.\n\n" + PRICE_SHEET
)

USER = (
    "Hi — we're a team of 25 people and we'd like annual billing on the Pro plan. "
    "Two questions: what's our total for the year, and is SSO included on Pro or is "
    "it an add-on?"
)

# Gold values derived from the sheet.
CORRECT_ANNUAL_TOTAL = 11760.0   # 49 * 25 * 12 * 0.80
DISTRACTOR_NO_DISCOUNT = 14700.0  # forgot the 20% annual discount

# Affirmative phrasings for "SSO is included on Pro". Chosen so that NONE of them
# is a contiguous substring of a *negative* statement like "SSO is not included on
# Pro" — the "not" breaks the contiguity, so a wrong answer can't sneak through.
SSO_YES_PHRASES = [
    "sso is included",
    "sso is bundled",
    "yes, sso",
    "yes — sso",
    "yes, it is included",
    "sso comes with pro",
    "sso is part of pro",
    "includes sso",
    "with sso included",
    "sso (yes)",
]

# Tightly-bound distractor claims that SSO is unavailable ON PRO. The first group
# is Pro-bound on purpose: a correct, thorough answer may legitimately note that
# Starter does NOT include SSO ("Starter does not include SSO"), so tier-agnostic
# negatives like "does not include sso" would false-fail real grounded output. The
# second group is genuinely-wrong regardless of tier (SSO is bundled on Pro at no
# cost per the sheet), so it stays tier-agnostic. A wrong answer that just says
# "No, SSO isn't included" without naming Pro already fails check 2 (no affirmative
# phrase), so it still fails overall.
SSO_WRONG_PHRASES = [
    "sso is not included on pro",
    "not included on pro",
    "sso is not available on pro",
    "pro does not include sso",
    "no sso on pro",
    "sso is an add-on",
    "sso is an addon",
    "sso costs extra",
    "sso requires enterprise",
    "sso is enterprise-only",
    "sso is enterprise only",
]


def grade(out: RunOutcome) -> GradeResult:
    g = GradeResult()
    text = out.final_text or ""
    snippet = text[:400]

    # (1 / required) Correct computed annual total appears. has_number tolerates
    #     $ and thousands separators, so "$11,760" / "11760" / "$ 11,760.00" match.
    g.add(
        "quotes correct annual total ($11,760)",
        has_number(text, CORRECT_ANNUAL_TOTAL),
        detail=f"expected {CORRECT_ANNUAL_TOTAL:.0f} (49*25*12*0.80). answer: {snippet!r}",
        required=True,
    )

    # (2 / required) Correctly states SSO IS included on Pro.
    g.add(
        "states SSO is included on Pro",
        contains_any(text, SSO_YES_PHRASES),
        detail=f"looked for an affirmative SSO phrase. answer: {snippet!r}",
        required=True,
    )

    # (3 / required) Does NOT assert the distractor that SSO is unavailable on Pro.
    asserted_wrong = contains_any(text, SSO_WRONG_PHRASES)
    g.add(
        "does NOT wrongly claim SSO is unavailable on Pro",
        not asserted_wrong,
        detail=(
            "found a 'SSO unavailable' claim — Pro includes SSO per the sheet"
            if asserted_wrong
            else "no wrong SSO-unavailable claim present"
        ),
        required=True,
    )

    # (informational) Did the model quote the no-discount list price instead? Helps
    #     diagnose "forgot the annual discount" vs "right answer". Not pass/fail.
    g.add(
        "did not stop at the no-discount list price ($14,700)",
        not (has_number(text, DISTRACTOR_NO_DISCOUNT) and not has_number(text, CORRECT_ANNUAL_TOTAL)),
        detail="informational: $14,700 is the pre-discount total (annual discount omitted)",
        required=False,
    )

    # (informational) Stayed grounded / did not refuse with an empty answer.
    g.add(
        "produced a substantive grounded answer",
        bool(norm(text)) and len(norm(text)) > 20,
        detail=f"answer length (normalized): {len(norm(text))}",
        required=False,
    )

    return g


register(
    Scenario(
        name="grounded_qa_pricing",
        category="grounding",
        description="Answer a prospect's pricing/SSO question strictly from an embedded price sheet (no hallucination, right row).",
        system=SYSTEM,
        user_messages=[{"role": "user", "content": USER}],
        grade=grade,
        max_tokens=400,
        temperature=0.0,
        sample_good={
            "final_text": (
                "Great — here's the breakdown for 25 seats on Pro with annual billing:\n\n"
                "Pro is $49 per seat per month. For 25 seats that's $49 x 25 = $1,225 per month, "
                "or $1,225 x 12 = $14,700 per year at the monthly list rate. Annual billing applies "
                "a 20% discount, so your total for the year is $14,700 x 0.80 = $11,760.\n\n"
                "On your second question: yes, SSO is included on Pro at no extra cost (SAML / OIDC "
                "single sign-on is bundled on both Pro and Enterprise)."
            )
        },
        sample_bad={
            # Believable misread: did the per-seat-per-month and 12-month math but
            # forgot to apply the 20% annual discount, quoting $14,700. SSO answer is
            # correct, so this fails ONLY the total check — clean discrimination.
            "final_text": (
                "For 25 seats on Pro: $49 per seat per month x 25 seats = $1,225/month, "
                "which comes to $1,225 x 12 = $14,700 for the year. And yes, SSO is included "
                "on the Pro plan."
            )
        },
    )
)
