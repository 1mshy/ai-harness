# APR

- **Relationship:** software rival (direct)
- **Mentions in corpus:** 1,280 — 34.8% of all competitor mentions, more than
  double the next brand
- **Authored:** 2026-08-05. No APR pricing, specifications or power figures are
  asserted anywhere in this document.

## Who they are

A US-based tuning company covering broadly the same Volkswagen Audi Group
platforms Unitronic does. They publish ECU and TCU software, sell supporting
hardware, and sell through a dealer/installer network. Of every brand a
customer names on a call, this is the one most likely to be a genuine
alternative rather than a part already on the car.

## What the customer is usually actually asking

Ranked by how the conversation tends to go, not by how it opens:

1. **"I already have their software — can I switch?"** This is a flashing and
   licensing question, not a comparison. Route it: which platform, which ECU,
   what is on the car now. It usually ends at `check_stage_availability` plus a
   licence question, and often at a human.
2. **"Why should I go with you instead?"** A qualification question wearing a
   comparison costume. Answer by finding out what they want from the car.
3. **"Do you support X?"** Pure compatibility. `resolve_vehicle` →
   `check_stage_availability`. Frequently the whole answer.

## How to handle it

- Establish the platform first. Every useful thing you can say is downstream of
  a `platform_id`, and this is exactly the conversation where the agent
  otherwise talks in generalities.
- Say what Unitronic actually offers for *their* car — which stages are
  released, how many calibration files are available, whether UniFLEX is
  supported on that platform. All of it comes from
  `check_stage_availability`, and all of it is specific, checkable and
  non-comparative.
- Ask what they are trying to achieve: daily driver, track use, fuel
  availability (91/93/E85), whether the car is already modified. This is the
  consultative motion, and it is also the thing that produces a usable lead.
- If they are switching from another vendor's software, treat it as a
  procedure question and get a human involved early. Cross-vendor flashing has
  real prerequisites and this is not a place to improvise.

## What not to say

- Nothing about their reliability, safety, engineering or customer service.
  There is no evidence for such a claim in anything this agent can read, and
  the customer may well own the product.
- No power comparisons. Not "ours makes more", not "theirs is conservative",
  not any number attributed to them.
- No claims about their pricing, warranty or dealer terms.
- Do not imply their software damages an engine or voids a warranty.

## If they want a real head-to-head

Say plainly that you are not the right source for a comparison against another
company's current products, call `log_knowledge_gap` with their question, and
offer `escalate_to_human`. A hedge-and-guess answer here is worse than an
honest handoff: it is the case most likely to be repeated back to a dealer.
