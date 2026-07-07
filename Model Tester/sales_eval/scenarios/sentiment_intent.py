"""Sentiment / churn-risk / intent classification for a sales-ops pipeline.

A B2B SaaS account team triages inbound customer messages by tagging each one
with sentiment, churn risk, and a primary intent so it can route at-risk
accounts to a CSM before they cancel. This scenario hands the model one clearly
unhappy message (repeated outages, "looking at alternatives") and asks for a
single JSON object. The grader parses leniently and matches each field against
a synonym-tolerant label set.
"""

from __future__ import annotations

from sales_eval.harness import (
    GradeResult,
    RunOutcome,
    Scenario,
    contains_any,
    extract_json,
    label_match,
    register,
)

SYSTEM = (
    "You are a sales-operations classifier. You read a single customer message "
    "and tag it for the CRM. Reply with ONLY a JSON object and nothing else — "
    "no prose, no markdown, no explanation. The object MUST have exactly these "
    "keys:\n"
    '  "sentiment": one of "positive", "neutral", "negative"\n'
    '  "churn_risk": one of "low", "medium", "high"\n'
    '  "primary_intent": a short label (1-4 words) describing what the customer '
    "wants (e.g. \"billing question\", \"feature request\", \"cancellation\").\n"
    "Base every tag strictly on the message content."
)

# A realistic, frustrated B2B SaaS customer hinting hard at churning.
CUSTOMER_MESSAGE = (
    "This is the third time in two weeks our team has been locked out by an "
    "outage during business hours — yesterday it took down checkout for almost "
    "an hour and we lost real revenue. We pay for the Enterprise tier and were "
    "promised 99.9% uptime in our contract, but the reliability has been "
    "nothing like that. Honestly my leadership has asked me to start evaluating "
    "other vendors, and we're already in conversations with one of your "
    "competitors. Unless something changes fast, I don't see us renewing in "
    "Q3. I need someone senior to explain what's actually being done here."
)


def grade(out: RunOutcome) -> GradeResult:
    g = GradeResult()

    # Informational: did anything come back at all?
    g.add(
        "produced output",
        bool(out.final_text and out.final_text.strip()),
        detail=f"final_text[:160]={out.final_text[:160]!r}",
        required=False,
    )

    data = extract_json(out.final_text)

    # Required: must be a parseable JSON object.
    is_obj = isinstance(data, dict)
    g.add(
        "emitted parseable JSON object",
        is_obj,
        detail=(
            f"parsed type={type(data).__name__}; final_text[:200]="
            f"{out.final_text[:200]!r}"
        ),
    )

    if not is_obj:
        # No object -> the remaining field checks can't pass; record them as
        # failed with a clear reason so the report is diagnostic.
        for field_name in ("sentiment=negative", "churn_risk=high", "primary_intent~churn"):
            g.add(
                f"classified {field_name}",
                False,
                detail="no JSON object to read fields from",
            )
        return g

    sentiment = data.get("sentiment")
    churn = data.get("churn_risk")
    intent = data.get("primary_intent")

    # Required: sentiment is negative (synonym-tolerant).
    g.add(
        "classified sentiment=negative",
        label_match(
            sentiment,
            "negative",
            {"negative": ["frustrated", "angry", "unhappy", "dissatisfied", "upset"]},
        ),
        detail=f"sentiment={sentiment!r}",
    )

    # Required: churn risk is high (synonym-tolerant).
    g.add(
        "classified churn_risk=high",
        label_match(
            churn,
            "high",
            {"high": ["elevated", "severe", "critical", "very high"]},
        ),
        detail=f"churn_risk={churn!r}",
    )

    # Required: primary intent reflects an at-risk / dissatisfied customer.
    # "primary_intent" is a free-text field and this message legitimately expresses
    # BOTH a reliability complaint AND churn risk (and a request to escalate), so
    # accept the whole valid space — a label like "reliability complaint",
    # "support escalation", or "cancellation" all correctly read this message.
    # The churn dimension specifically is already pinned by the churn_risk=high
    # check above, so this stays meaningful: "billing question", "feature request",
    # "upsell", "positive feedback", "general inquiry" would all (correctly) fail.
    intent_ok = intent is not None and contains_any(
        str(intent),
        [
            "cancel", "churn", "leav", "escalat", "switch", "renew", "retention",
            "at risk", "at-risk", "complaint", "reliab", "outage", "downtime",
            "uptime", "incident", "dissatisf", "unhapp", "frustrat", "problem",
            "issue", "support", "sla",
        ],
    )
    g.add(
        "classified primary_intent~dissatisfaction/complaint/churn/escalation",
        intent_ok,
        detail=f"primary_intent={intent!r}",
    )

    return g


register(
    Scenario(
        name="sentiment_intent",
        category="sentiment",
        description="Tag a frustrated customer message with sentiment, churn risk, and intent (JSON).",
        system=SYSTEM,
        user_messages=[{"role": "user", "content": CUSTOMER_MESSAGE}],
        grade=grade,
        response_format={"type": "json_object"},
        # Reasoning models (deepseek-class) often "think out loud" in the content
        # and emit the JSON last; at 256 tokens they get truncated mid-reasoning
        # before any JSON appears. Give room so the test measures classification,
        # not whether reasoning+answer fit the cap. extract_json picks the final
        # JSON object out of the surrounding prose. The cap is only spent when the
        # model actually reasons long (the JSON answer itself is tiny), so this is
        # cheap headroom against the run-to-run reasoning-dump variance seen live.
        max_tokens=1536,
        temperature=0.0,
        # Good output: gold labels, deliberately wrapped in a ```json fence to
        # prove extract_json strips fences.
        sample_good={
            "final_text": (
                "```json\n"
                "{\n"
                '  "sentiment": "negative",\n'
                '  "churn_risk": "high",\n'
                '  "primary_intent": "cancellation / escalation"\n'
                "}\n"
                "```"
            )
        },
        # Bad output: well-formed JSON, plausible-looking, but BOTH sentiment
        # (positive) and primary_intent (an off-target "billing question") are
        # wrong — so the offline gate exercises the failure path of the intent
        # check too, not just the sentiment check.
        sample_bad={
            "final_text": (
                '{"sentiment": "positive", "churn_risk": "high", '
                '"primary_intent": "billing question"}'
            )
        },
    )
)
