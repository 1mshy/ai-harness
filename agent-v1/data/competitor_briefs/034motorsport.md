# 034Motorsport

- **Relationship:** **COMPLEMENTARY HARDWARE.** They also publish software, so
  a minority of mentions are a genuine software comparison.
- **Mentions in corpus:** 418 — 11.4%
- **Authored:** 2026-08-05. No 034Motorsport pricing, specifications or power
  figures are asserted anywhere in this document.

## Who they are

A VAG-focused performance manufacturer with a long history in Audi in
particular. Their catalogue is heavily hardware: engine and transmission
mounts, intakes, intercoolers, suspension components, driveline parts. They
also publish ECU and TCU software for a range of platforms.

## Read this before responding

Like IE, the safe default is that **the customer owns an 034 part** — most
often mounts or a cooling upgrade. Mounts in particular come up in NVH
conversations ("it's a bit rougher at idle since I fitted them") that have
nothing to do with software and everything to do with a stiffer mount
transmitting more vibration into the cabin. That is not a tune complaint, and
mistaking it for one sends the whole call the wrong way.

## The hardware conversation (the common one)

- Cooling and driveline hardware supports higher-stage software rather than
  conflicting with it. Treat it as context that helps, not as an objection.
- If the customer is describing a symptom, separate hardware causes from
  calibration causes before proposing anything. Use `search_knowledge` and, if
  there is a matching case, `get_case` — the **failed attempts** in a case are
  the only record of what has already been tried and did not work.
- Confirm stage availability for their platform with
  `check_stage_availability` rather than reasoning from what hardware they
  have. Hardware does not unlock a stage; a released calibration does.

## The software conversation (the rare one)

Same handling as APR: platform first, state what Unitronic offers for it from
tools, ask what they want from the car, no comparative claims.

## What not to say

- Nothing about 034 quality, fitment or support.
- No prices, specifications, power figures or warranty terms.
- Do not attribute a customer's symptom to a competitor's part unless a tool
  result actually supports it. "It's probably your mounts" is a guess wearing a
  diagnosis costume.
