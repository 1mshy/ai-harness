# CTS Turbo

- **Relationship:** **COMPLEMENTARY HARDWARE.** They also publish software, so
  a minority of mentions are a genuine software comparison.
- **Mentions in corpus:** 357 — 9.7%
- **Authored:** 2026-08-05. No CTS Turbo pricing, specifications or power
  figures are asserted anywhere in this document.

## Who they are

A Canadian VAG-focused manufacturer known for turbocharger upgrade kits,
intakes, intercoolers, charge pipes and exhaust hardware. They also publish
software for some platforms.

## Read this before responding

**Assume the part is on the car.** CTS hardware — particularly intercoolers and
charge pipes — shows up constantly in build lists on higher-stage cars, and
those cars are exactly the ones calling about calibration. An agent that treats
"I've got the CTS intercooler" as a competitive objection has misread a
compatibility question, and it will be obvious to the customer within one
sentence.

Being a fellow Canadian company also means the two brands share dealers and
share customers. Framing them as an enemy is wrong on the facts as well as
tactically.

## The hardware conversation (the common one)

- Turbo upgrade kits are the important case. A different turbo is not a bolt-on
  in calibration terms — it determines whether an off-the-shelf stage applies
  at all or whether the customer is into custom-calibration territory.
- Do **not** infer that a released stage covers an upgraded turbo. Check the
  platform with `check_stage_availability`, check for a documented procedure
  with `search_knowledge`, and if neither is conclusive, `log_knowledge_gap`
  and hand to a human. This is a case where a confident wrong answer has
  mechanical consequences.
- Intercoolers, charge pipes and intakes are supportive: they help a
  heat-limited higher stage hold its numbers. Say so neutrally if it is
  relevant; do not turn it into a pitch.

## The software conversation (the rare one)

Same handling as APR: platform first, state what Unitronic offers from tools,
ask what they want, no comparative claims.

## What not to say

- Nothing about CTS quality, fitment or support.
- No prices, specifications, power figures or warranty terms.
- Never claim an off-the-shelf calibration is safe on non-stock turbo hardware
  without a tool result or a human saying so.
