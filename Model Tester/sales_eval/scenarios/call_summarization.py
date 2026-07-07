"""Long-context capability test: summarize a long discovery call transcript.

The model is handed a ~45-turn sales discovery call between an account exec and
a prospect, and asked to produce a structured recap (Key Points, Objections/
Risks, Action Items with owners, Next Steps). Several concrete "needle" facts
are planted across the call:

    * a rep commitment        -> "I'll get the SOC 2 report over by Thursday"
    * a budget objection      -> "budget isn't approved until Q3"
    * a competitor            -> "we're also evaluating Acme"
    * a dated next step        -> "follow-up demo on the 14th"
    * a named stakeholder      -> CFO "Diane Park" owns the budget sign-off

The grader checks (leniently) that the recap captured each needle plus has some
structure and a sane length. The classic failure mode — a vague recap that
drops the SOC 2 commitment — is the calibrated sample_bad.
"""

from __future__ import annotations

from sales_eval.harness import (
    GradeResult,
    RunOutcome,
    Scenario,
    contains_all,
    contains_any,
    register,
)

# --------------------------------------------------------------------------- #
# Fixture: a long, realistic B2B SaaS discovery-call transcript.
# --------------------------------------------------------------------------- #
TRANSCRIPT = """\
[Discovery call — NorthBridge Analytics (AE: Marcus Hale) x Cortado Logistics]

Marcus (AE): Thanks for making the time today, Priya. Before we dive in — how's
the rollout of the new warehouse going on your end?

Priya (Ops Lead): Busy. We opened the Reno facility last month, so everything's
a bit on fire right now. That's actually part of why we're talking to you.

Marcus: Makes sense. So I know you came in through the website, you downloaded
the routing-optimization one-pager. What kicked that off?

Priya: Our delivery promise dates keep slipping. We quote two-day, we hit it
maybe seventy percent of the time. Customers are noticing, and our CX team is
drowning in "where is my order" tickets.

Marcus: Seventy percent. And what's the target you're being held to?

Priya: Leadership wants ninety-five. It's a board-level metric now, honestly.

Marcus: Got it. Walk me through how routing decisions get made today.

Priya: Mostly manual. Our dispatch team uses a spreadsheet and tribal knowledge.
We have a legacy TMS but it doesn't do dynamic re-routing, so once a route is
locked in the morning, that's it for the day.

Marcus: So no re-optimization when a truck breaks down or a rush order lands.

Priya: Exactly. And we're growing — three new facilities planned next year — so
the manual approach just doesn't scale.

Marcus: That's the pattern we see a lot. Quick question: how many shipments a
day are we talking, roughly?

Priya: Right now about twelve thousand a day across all sites. Reno adds maybe
another three thousand once it's at full capacity.

Marcus: Okay, fifteen thousand at steady state. That's well within what our
engine handles. Who else is involved in evaluating a solution like this?

Priya: So it's me on the ops side. Our VP of Engineering, Sam Okafor, has to
sign off on anything that touches our data pipeline. And then budget — that
goes through our CFO, Diane Park. Diane's the one who ultimately approves spend
at this level.

Marcus: Good to know. Is Diane bought into solving this, or is this still an
ops-level initiative?

Priya: She's aware. She hasn't seen a number yet. That's where it gets tricky —
honestly, the budget isn't approved until Q3. We're in a spending freeze through
the end of this quarter, so even if I loved the product tomorrow I couldn't cut
a PO until July.

Marcus: That's helpful context, and it's fine — a lot of our customers run a
pilot or a paid POC before the full budget lands. We can structure around the
Q3 timeline. Let me ask about the technical side. Sam — what's he going to want
to see?

Priya: Integration story, mostly. Our data's in Snowflake, our TMS exposes a
REST API but the docs are... aspirational. He's going to ask how clean your
integration is and whether it's going to eat six months of his team's time.

Marcus: Fair. We have a native Snowflake connector and a pre-built adapter for
the three most common TMS platforms. Which TMS are you on?

Priya: We're on Manhattan, an older version.

Marcus: We support Manhattan. Older versions sometimes need a light shim but
it's days, not months. I can have a solutions engineer walk Sam through the
integration architecture in detail.

Priya: That would help. He's skeptical — last vendor promised a two-week
integration and it took five months.

Marcus: Understood, and I won't promise two weeks. Let me be straight with you
on something else: are you evaluating other vendors?

Priya: Yeah, I'll be honest, we're also evaluating Acme — Acme Routing. They
came in cheaper on the initial quote.

Marcus: Acme's a real product, I won't pretend otherwise. Where I'd push back
is on dynamic re-optimization at your volume — that's historically where they
struggle past about ten thousand shipments a day. But you should test that
yourself, don't take my word for it.

Priya: That's exactly the kind of thing I want to see in a head-to-head.

Marcus: Then let's make the demo a real one — your data, your volume, both
scenarios. On security: since you're touching customer data, your team will
almost certainly want our compliance posture. We're SOC 2 Type II certified.

Priya: Yes — Sam will absolutely ask for that. Can you send the SOC 2 report?

Marcus: Absolutely. I'll get the SOC 2 report over to you by Thursday so Sam
has it before our next conversation.

Priya: Perfect, that unblocks his review.

Marcus: What about data residency — any requirements there?

Priya: Everything has to stay in US regions. We have some Canadian volume but
even that has to be processed in US-East per our legal team.

Marcus: That's our default deployment, so no issue. Let me make sure I
understand priorities. If I had to rank them: hitting the ninety-five percent
promise-date target, scaling to the new facilities, and a clean integration that
doesn't burn Sam's team. Is that the order?

Priya: I'd actually put integration risk first. If it blows up Sam's roadmap,
the project dies internally regardless of how good the routing is.

Marcus: Noted — integration confidence is the gating factor. That's good for me
to know going into the demo. Let me talk through what a rollout could look like.

Priya: Please.

Marcus: Phase one is a read-only shadow mode — we ingest your shipment data and
run our optimization in parallel without touching live dispatch. You see what we
*would* have routed versus what actually happened. No risk, pure comparison.

Priya: I like that. That's an easy yes internally — nothing's live.

Marcus: Phase two, once you trust the recommendations, we flip on assisted mode:
dispatch sees our suggestions but still has final say. Phase three is full
automation with human override. Most customers spend four to six weeks in each
phase.

Priya: So a few months to full automation. That actually lines up with the Q3
budget — we could do shadow mode now and have the business case ready when the
freeze lifts.

Marcus: Exactly the play I'd recommend. Does shadow mode need budget?

Priya: A small POC fee might slip under my discretionary spend. I'd have to
check the threshold. What's the POC cost?

Marcus: For a 30-day paid POC at your volume it's typically around eight
thousand dollars, and we credit it fully against an annual contract if you move
forward. I'll confirm the exact figure in writing.

Priya: Eight thousand I can probably get approved without Diane. Send me that in
writing too.

Marcus: Will do. Let me also flag a risk on my side so we're aligned: if Sam's
integration review finds your Manhattan version is too old for the standard
adapter, we'd need a custom shim and that adds two to three weeks. I don't think
it'll happen, but I'd rather you hear it now than be surprised.

Priya: I appreciate the honesty. Flag it to Sam directly when you meet.

Marcus: I will. Okay — next steps. I'd like to get our solutions engineer in
front of Sam for the integration deep-dive, and run the head-to-head demo with
your real data. When works?

Priya: Sam's traveling next week. The week after is better. Can we target the
fourteenth for the follow-up demo? That gives me time to pull the data extract.

Marcus: The 14th works on my end. I'll lock in a follow-up demo on the 14th and
send a calendar invite to you and Sam. I'll bring the solutions engineer.

Priya: Great. And you'll have the SOC 2 report to us before then?

Marcus: Yes — SOC 2 by Thursday, demo on the 14th. I'll also send the POC
pricing in writing and a short shadow-mode overview Sam can skim.

Priya: That covers it. One more thing — please loop in Diane on the eventual
proposal, even though she's not in the demo. She'll want lead time before Q3.

Marcus: Good call. I'll prep a one-pager for Diane on ROI so she's warm before
the budget conversation. Anything else on your mind before we wrap?

Priya: No, I think that's everything. This was useful — thank you for being
straight about Acme and the integration risk.

Marcus: My pleasure, Priya. To recap my commitments: SOC 2 report by Thursday,
POC pricing in writing, and the follow-up demo locked for the 14th with Sam and
our solutions engineer. Talk soon.

Priya: Sounds good. Bye, Marcus.
"""


SYSTEM = (
    "You are a sales operations assistant. You produce accurate, well-structured "
    "recaps of sales calls for the account team. Capture commitments, owners, "
    "and dates precisely — the team acts on your recap, so do not drop or invent "
    "details."
)

USER = (
    "Summarize the following discovery call transcript into a structured recap "
    "with these sections: Key Points, Objections / Risks, Action Items (with an "
    "owner for each), and Next Steps. Be specific about commitments, dates, "
    "competitors, and the people involved.\n\n"
    f"{TRANSCRIPT}"
)


# --------------------------------------------------------------------------- #
# Grader
# --------------------------------------------------------------------------- #
def grade(out: RunOutcome) -> GradeResult:
    g = GradeResult()
    text = out.final_text or ""
    snippet = text[:240]

    # Needle 1: the rep's SOC 2 commitment, with its deadline.
    soc2 = contains_all(text, ["soc 2", "thursday"])
    g.add(
        "captured SOC 2 commitment by Thursday",
        soc2,
        detail=f"need 'SOC 2' + 'Thursday'. text[:240]={snippet!r}",
    )

    # Needle 2: budget objection tied to Q3.
    q3 = contains_any(text, ["q3", "third quarter", "july", "spending freeze"])
    g.add(
        "captured Q3 budget / spending-freeze objection",
        q3,
        detail=f"need 'Q3'/'third quarter'/'July'/'freeze'. text[:240]={snippet!r}",
    )

    # Needle 3: the competitor.
    competitor = contains_any(text, ["acme"])
    g.add(
        "captured competitor (Acme)",
        competitor,
        detail=f"need 'Acme'. text[:240]={snippet!r}",
    )

    # Needle 4: the dated next step. Anchored so it can't match a bare '14'.
    date = contains_any(text, ["14th", "the 14th", "fourteenth", "demo on the 14"])
    g.add(
        "captured follow-up demo date (the 14th)",
        date,
        detail=f"need '14th'/'fourteenth'. text[:240]={snippet!r}",
    )

    # Needle 5: a named decision-maker / process detail. CFO Diane Park owns budget.
    stakeholder = contains_any(text, ["diane", "park", "cfo"])
    g.add(
        "captured budget decision-maker (CFO Diane Park)",
        stakeholder,
        detail=f"need 'Diane'/'CFO'. text[:240]={snippet!r}",
    )

    # Structure: the recap is actually organized, not a blob.
    structured = contains_any(
        text,
        ["action item", "action items", "next step", "next steps",
         "objection", "key point", "owner"],
    )
    g.add(
        "produced a structured recap (sections / action items)",
        structured,
        detail=f"need a section header or 'action'/'owner'. text[:240]={snippet!r}",
    )

    # Sanity length bounds: not empty/trivial, and a summary, not a re-paste.
    length_ok = 150 < len(text) < len(TRANSCRIPT)
    g.add(
        "recap length is sane (summary, not empty, not the full transcript)",
        length_ok,
        detail=f"len(text)={len(text)} ; bounds=(150, {len(TRANSCRIPT)})",
    )

    # Informational: did the model produce JSON? Not required — prose recap is fine.
    g.add(
        "emitted parseable JSON (informational)",
        out.parsed_json is not None,
        detail="prose recaps are accepted; this is diagnostic only",
        required=False,
    )

    return g


# --------------------------------------------------------------------------- #
# Calibration samples (replayed offline by the self-test)
# --------------------------------------------------------------------------- #
_SAMPLE_GOOD = """\
Call Recap — Cortado Logistics (AE: Marcus Hale)

Key Points
- Cortado is missing delivery promise dates (hitting ~70% vs a board-level 95% target).
- Routing is manual today on an older Manhattan TMS; ~15,000 shipments/day at steady state.
- Native Snowflake connector + Manhattan adapter; US-only data residency is supported.
- Proposed phased rollout: shadow mode -> assisted -> full automation.

Objections / Risks
- Budget is NOT approved until Q3 — they are in a spending freeze through this quarter.
- Integration risk is the gating factor; VP Eng Sam Okafor is skeptical after a prior 5-month integration.
- They are also evaluating Acme Routing, which quoted lower.
- An older Manhattan version may need a custom shim (adds 2-3 weeks).

Action Items (owner)
- Send the SOC 2 Type II report by Thursday — Owner: Marcus.
- Send 30-day POC pricing (~$8,000, credited to annual) in writing — Owner: Marcus.
- Pull the data extract for the head-to-head demo — Owner: Priya.

Next Steps
- Follow-up demo locked for the 14th with Sam and a solutions engineer (calendar invite to follow).
- Loop CFO Diane Park into the eventual proposal ahead of the Q3 budget conversation.
"""

_SAMPLE_BAD = """\
Call Recap — Cortado Logistics

Key Points
- The prospect is struggling to hit their delivery promise dates and wants to improve.
- Their routing process is manual and they are growing, so it doesn't scale.
- We talked through a phased rollout starting with a low-risk shadow mode.

Objections / Risks
- Budget isn't approved until Q3 and they're in a spending freeze right now.
- They're also evaluating Acme, which came in cheaper.
- VP Eng Diane has concerns about integration risk after a bad prior experience.

Next Steps
- Run a head-to-head demo with their real data on the 14th.
- Send over POC pricing in writing.
"""

register(
    Scenario(
        name="call_summarization",
        category="summarization",
        description="Summarize a long discovery-call transcript into a structured recap, capturing planted needles (SOC 2 commitment, Q3 budget, competitor, follow-up date, stakeholder).",
        system=SYSTEM,
        user_messages=[{"role": "user", "content": USER}],
        grade=grade,
        # A 4-section recap of a ~45-turn call needs room: at 800 tokens the model
        # was truncated mid-"Key Points" and never reached Action Items / Next
        # Steps (where the SOC 2 deadline and the demo date live). Give it space
        # so the test measures recall, not the token cap.
        max_tokens=2000,
        temperature=0.0,
        sample_good={"final_text": _SAMPLE_GOOD},
        sample_bad={"final_text": _SAMPLE_BAD},
    )
)
