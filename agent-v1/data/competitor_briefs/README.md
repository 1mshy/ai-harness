# Competitor briefs

## Why these exist

3,680 competitor mentions were measured across the call corpus. Five brands
account for 2,785 of them:

| Brand | Mentions | Share |
|---|---:|---:|
| APR | 1,280 | 34.8% |
| Integrated Engineering | 553 | 15.0% |
| 034Motorsport | 418 | 11.4% |
| CTS Turbo | 357 | 9.7% |
| COBB | 177 | 4.8% |
| **Total** | **2,785** | **75.7%** |

(AGENT_PLAN.md §8.1 rounds this to "~74%"; the arithmetic above is the exact
figure. Either way, five short documents cover three quarters of the surface.)

Competitor objections end in `needs_info` **53%** of the time, and `sale_made`
**twice out of 286**. That is not a skill gap or a persuasion problem — the
agent had nothing factual to say and went quiet. It is a five-page content fix,
which is what this directory is.

## The rule that matters most

**Six of the brands customers name are not rivals.** CTS Turbo, Integrated
Engineering, ECS Tuning, AWE Tuning, Eventuri and CSF sell *hardware that
Unitronic software runs on top of*. When a customer says "I have a CTS
intercooler and an IE intake", they are describing their build, not comparing
vendors. An agent that treats that as an objection argues against the
customer's own car — the single worst failure mode available on this path, and
the reason the classification in `_index.yaml` is machine-readable rather than
left to the model's judgement.

Only APR and COBB are primarily software rivals. IE, 034Motorsport and CTS also
publish software, so those three are complementary on hardware *and* competing
on software at the same time. Each brief says which conversation is which.

## House rules for every one of these

1. **Never disparage.** No claims about another company's reliability, safety,
   support quality or engineering. The corpus contains no evidence for such
   claims and a customer who owns the part hears an insult.
2. **Never state a competitor's specifications, prices, warranty terms or
   power figures.** They change without notice and nobody here owns them. Say
   "you'd want to confirm that with them".
3. **Answer with what Unitronic does, verified by a tool.** Stage availability
   comes from `check_stage_availability`. Prices come from `get_fee_schedule`.
   Nothing on a competitor page is a source for either.
4. **The usual question underneath is compatibility, not comparison.** "Will
   your tune work with my IE intake?" is a hardware question with a factual
   answer. Answer that.
5. **Emissions equipment is a hard stop regardless of who made the part.** A
   competitor's downpipe being catless does not change the answer; escalate
   per the emissions policy.
6. **When the customer wants a head-to-head that you cannot source, say so and
   offer a human.** `log_knowledge_gap` first, then `escalate_to_human`.

## Files

- `_index.yaml` — classification, aliases and mention counts. Machine-readable.
- `apr.md`, `integrated-engineering.md`, `034motorsport.md`, `cts-turbo.md`,
  `cobb.md`

## Provenance

Hand-authored 2026-08-05 from public brand positioning plus the measured
mention counts above. **No pricing, no specifications and no performance
figures are asserted for any competitor.** Nothing here is mined from call
transcripts: `objection_detail` records what the customer worried about and
there is no rebuttal field anywhere in the schema, so retrieving five of them
returns a taxonomy of anxiety and nothing to say.
