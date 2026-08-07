# Integrated Engineering (IE)

- **Relationship:** **COMPLEMENTARY HARDWARE.** Not a rival in most
  conversations. They also publish software, so the software conversation is a
  separate and much rarer one.
- **Mentions in corpus:** 553 — 15.0%, second only to APR
- **Authored:** 2026-08-05. No IE pricing, specifications or power figures are
  asserted anywhere in this document.

## Who they are

A VAG-focused performance manufacturer best known for hardware: intake
manifolds, intakes, turbo kits, connecting rods and other internals, exhaust
components. They also publish ECU software for some platforms.

## Read this before responding

**The default assumption must be that the customer owns an IE part.** When
somebody says "I've got the IE manifold and an IE intake", they are telling you
what is bolted to their car so that you can answer a compatibility question.
Treating that as a competitive objection is the specific failure AGENT_PLAN.md
§8.1 calls out: the agent ends up arguing against the customer's own build,
which reads as either ignorance or hostility and is unrecoverable in one turn.

Only treat IE as a competitor when the customer explicitly says they are
choosing between IE *software* and Unitronic software. That is a small
fraction of these 553 mentions.

## The hardware conversation (the common one)

- The question is nearly always: does Unitronic software account for this part,
  is there a calibration that suits it, does it change what stage I can run?
- Answer from tools. `resolve_vehicle` → `check_stage_availability` tells you
  what exists for the platform; `search_knowledge` covers documented
  hardware-specific procedures and conditions.
- Forged internals and larger turbo hardware push the conversation toward the
  upper stages and toward custom work. That is a real conversation and it is
  frequently one for a human — say so rather than approximating.
- **Never suggest replacing working hardware with a Unitronic equivalent
  unprompted.** They did not ask, and it converts a support call into a sales
  pitch they did not want.
- If you cannot confirm from a tool whether a specific part is accounted for in
  a given calibration, say so, `log_knowledge_gap`, and offer a human. Guessing
  at hardware compatibility is how a customer ends up with a lean condition.

## The software conversation (the rare one)

Handle exactly as the APR brief describes: establish the platform, state what
Unitronic offers for it from `check_stage_availability`, ask what they want
from the car, and make no comparative claim of any kind.

## What not to say

- No claim that IE hardware is inferior, unsupported or a problem. The customer
  owns it.
- No IE prices, specifications, power figures or warranty terms.
- No implication that mixing brands is inherently unsafe. Say what is verified
  and hand off what is not.
