# COBB Tuning

- **Relationship:** software rival (direct, on overlapping platforms)
- **Mentions in corpus:** 177 — 4.8%
- **Authored:** 2026-08-05. No COBB pricing, specifications or power figures
  are asserted anywhere in this document.

## Who they are

A US tuning company best known for the Accessport, a handheld flashing device
that stores and loads maps on the vehicle. Their platform coverage is broader
than VAG — Subaru, Ford, Porsche among others — with some overlap onto the
platforms Unitronic covers. Where they overlap, they are a direct alternative
for both the software and the flashing hardware.

## What the customer is usually actually asking

The distinguishing feature of these calls is that the comparison is often about
**the device and the workflow**, not the calibration. The customer already
knows how they want to flash their car.

1. **"How does your cable compare to the Accessport?"** A workflow question.
   Describe how UniCONNECT+ actually works for them — what it is for, that a
   cable is tied to one vehicle at a time — and get the price from
   `get_fee_schedule` (`uniconnect_cable`), never from memory or a product
   page.
2. **"I have an Accessport, can I use it with your software?"** Answer the
   compatibility question directly and route to a human if you cannot confirm
   it from a tool. Do not speculate about another vendor's device behaviour.
3. **"Do you cover my car?"** `resolve_vehicle` → `check_stage_availability`.
   Note that a customer arriving from COBB may well be on a platform Unitronic
   does not cover at all — the supported marques are Volkswagen, Audi, Porsche,
   Seat, Skoda, CUPRA, Lamborghini, Bentley and Opel. If the car is a Subaru,
   say so kindly and immediately. That is a real answer and it saves everyone
   time.

## How to handle it

- Platform first, as always. On this brand more than any other, the
  conversation can end at "we don't cover that vehicle", and finding that out
  in the first exchange is the courteous outcome.
- Cable and licence questions are fee-schedule questions. `get_fee_schedule`
  covers UniCONNECT+, licence transfer, and the remote and mail-in resets.
  Quote those exactly as written and nothing else.
- If they are moving from another vendor's device, treat it as a procedure
  question with prerequisites and get a human involved.

## What not to say

- No claims about Accessport reliability, its map handling, or their support.
- No prices, specifications, power figures or warranty terms of theirs.
- Do not assert what happens to a car flashed by another vendor's device unless
  a tool result says so.
