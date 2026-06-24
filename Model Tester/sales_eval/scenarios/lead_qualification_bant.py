"""BANT extraction from a discovery-call transcript.

A sales engineer ran a discovery call with a prospect. The model must read the
transcript and emit the four BANT fields (Budget, Authority, Need, Timeline)
as a flat JSON object for the CRM. This is a field-level information-extraction
test: every BANT element is stated or strongly implied in the dialogue, and the
grader checks that each extracted value captures the gold fact — leniently, so
paraphrase and messy formatting still pass.
"""

from __future__ import annotations

from sales_eval.harness import (
    GradeResult,
    RunOutcome,
    Scenario,
    contains_any,
    extract_json,
    has_number,
    norm,
    register,
)

# --------------------------------------------------------------------------- #
# Fixture: a realistic ~20-turn discovery call.
# --------------------------------------------------------------------------- #
TRANSCRIPT = """\
[Discovery call — Northwind Logistics x Cadence (developer-onboarding platform)]

AE (Cadence): Thanks for making the time, Priya. To start — what pulled you
    into looking at an onboarding platform now?
Priya (Northwind): We've doubled engineering headcount in the last year and our
    onboarding is just not keeping up. It currently takes a new backend hire
    about three weeks before they ship their first real PR. We want that down to
    one week.
AE: That's a big gap. What's the cost of those extra two weeks for you?
Priya: Honestly it's the senior engineers babysitting new folks instead of
    building. We're losing real velocity, and a couple of new hires got
    frustrated and almost left in their first month.
AE: Understood. So the core problem is ramp time — getting a new engineer
    productive fast and taking that load off your seniors.
Priya: Exactly. If we can cut ramp from three weeks to one, that's the win.
AE: Got it. Have you scoped what you'd be able to invest in solving this?
Priya: We've set aside roughly fifty thousand a year for tooling in this area —
    call it $50k/year as a working number. I don't want to go much past that
    without a clear ROI story.
AE: That's helpful, thank you. And on a deal like this, who's typically
    involved in the decision?
Priya: I'm the VP of Engineering, and I own the tooling budget — I sign off on
    purchases in this range myself. For anything north of six figures I'd loop
    in our CFO, but at this level it's my call.
AE: Perfect. Is there a date you're working back from?
Priya: Yeah — we're hiring a big cohort in the second half of the year, so we
    want this rolled out before Q3 starts. If it slips past that, we miss the
    window and have to onboard the cohort the old way.
AE: So the goal is to be live before Q3, cut ramp from three weeks to one,
    inside about a fifty-k-a-year budget, and you're the sign-off.
Priya: You've got it.
AE: Last thing — anything that would block a yes if the product fits?
Priya: As long as it integrates with our GitHub and SSO, and the price holds,
    I'm ready to move. I just need to see it work for backend onboarding
    specifically.
AE: Great. I'll put together a tailored pilot plan and send it over today.
"""

SYSTEM = (
    "You are a sales-operations assistant. You read discovery-call transcripts "
    "and extract structured BANT data for the CRM. BANT = Budget, Authority, "
    "Need, Timeline.\n\n"
    "Output ONLY a single JSON object, no prose, no markdown fences, with "
    "exactly these keys:\n"
    '  {"budget": "...", "authority": "...", "need": "...", "timeline": "..."}\n'
    "Each value is a short string summarizing what the transcript states or "
    "strongly implies for that field. If a field is truly absent, use an empty "
    "string."
)

USER = (
    "Extract the BANT fields from this discovery call and return the JSON "
    "object described.\n\n" + TRANSCRIPT
)


# --------------------------------------------------------------------------- #
# Grader
# --------------------------------------------------------------------------- #
def grade(out: RunOutcome) -> GradeResult:
    g = GradeResult()

    data = out.parsed_json if isinstance(out.parsed_json, dict) else extract_json(out.final_text)
    parsed_ok = isinstance(data, dict)
    g.add(
        "output is a JSON object",
        parsed_ok,
        detail=f"parsed type={type(data).__name__}; raw={out.final_text[:160]!r}",
    )

    if not parsed_ok:
        # Nothing more can be checked field-by-field; make each field a clear FAIL
        # rather than crashing, so the failure mode is visible per-field.
        for field in ("budget", "authority", "need", "timeline"):
            g.add(f"{field} captured", False, detail="no JSON object to read fields from")
        return g

    # Informational: did the model use the expected key set at all? Diagnoses a
    # model that emits JSON but ignores the requested schema, vs. one that gets
    # the values wrong.
    keys = {norm(k) for k in data.keys()}
    expected_keys = {"budget", "authority", "need", "timeline"}
    g.add(
        "uses expected BANT keys",
        expected_keys.issubset(keys),
        detail=f"keys present: {sorted(data.keys())}",
        required=False,
    )

    def fv(key: str) -> str:
        # Case-insensitive key fetch; tolerate stringified non-strings.
        for k, v in data.items():
            if norm(k) == key:
                return "" if v is None else str(v)
        return ""

    budget = fv("budget")
    g.add(
        "budget captures ~$50k/year",
        has_number(budget, 50000) or contains_any(budget, ["50k", "50 k", "fifty thousand", "fifty k"]),
        detail=f"budget={budget!r}",
    )

    authority = fv("authority")
    g.add(
        "authority captures decision-maker (VP / sign-off)",
        contains_any(authority, ["vp", "vice president", "decision", "decision-maker",
                                 "decision maker", "sign off", "sign-off", "signs off",
                                 "owns the budget", "her call", "his call", "their call"]),
        detail=f"authority={authority!r}",
    )

    need = fv("need")
    g.add(
        "need captures faster onboarding / ramp time",
        contains_any(need, ["onboard", "onboarding", "ramp", "3 weeks", "three weeks",
                            "one week", "1 week", "time to", "productive", "ship"]),
        detail=f"need={need!r}",
    )

    timeline = fv("timeline")
    g.add(
        "timeline captures before Q3",
        contains_any(timeline, ["q3", "quarter", "before q3", "second half", "h2",
                               "this year"]),
        detail=f"timeline={timeline!r}",
    )

    return g


register(Scenario(
    name="lead_qualification_bant",
    category="extraction",
    description="Extract BANT (Budget/Authority/Need/Timeline) from a discovery call into CRM JSON.",
    system=SYSTEM,
    user_messages=[{"role": "user", "content": USER}],
    grade=grade,
    response_format={"type": "json_object"},
    # Headroom for reasoning models that think out loud before emitting the JSON
    # (extract_json pulls the final object from the prose). Generous because the
    # cap is only spent when the model reasons long; the JSON answer is tiny.
    max_tokens=1536,
    temperature=0.0,
    sample_good={
        "final_text": (
            '{"budget": "~$50k/year set aside for onboarding tooling; needs clear ROI to exceed", '
            '"authority": "Priya, VP of Engineering, owns the tooling budget and signs off on '
            'purchases in this range herself", '
            '"need": "Cut new backend engineer ramp/onboarding from 3 weeks to 1 week and free up '
            'senior engineers", '
            '"timeline": "Wants the platform rolled out before Q3 to support the H2 hiring cohort"}'
        ),
    },
    sample_bad={
        # Believable partial extraction: gets need/timeline/authority right, but
        # whiffs on budget by marking it unknown (the $50k was clearly stated).
        "final_text": (
            '{"budget": "unknown", '
            '"authority": "Priya is the VP of Engineering and signs off on this purchase", '
            '"need": "Reduce new-hire onboarding time from three weeks to one week", '
            '"timeline": "Roll out before Q3"}'
        ),
    },
))
