# agent-v1 — one agent platform, two personas

Implementation of `AGENT_PLAN.md`. Sales and support share one index set, one
retrieval layer, one tool executor and one runtime, and differ only by system
prompt, default filters and tool allowlist.

The persona triggers on **purchase intent, not department**. Re-measured against
live Mongo on 2026-08-06: of 12,835 calls carrying active purchase intent,
**3,616 (28.2%) are handled by `technical_support`**, 1,310 of them
`ready_to_buy`. Routing on department alone sends more than a quarter of buyers
to a persona told not to sell to them.

## Layout

```
agentv1/
  config.py            pinned settings; QWEN_SIZE has no default, dims follow it
  text/normalize.py    surface-form normalization, applied at index AND query time
  clients/             mongo · qdrant · embeddings · sparse (BM25) · llm
  kb/                  PHASE 4a — canonical unit identity in Mongo (no Qdrant)
  index/               PHASE 4b — Mongo -> Qdrant projection (no LLM)
  retrieval/           hybrid -> rerank -> supersession -> training_safe floor
  tools/               15 tools, native in-process, read-only, executor-enforced
  guardrails/          emissions · safety · grounding · pii · escalation
  agent/               personas · session · bounded loop · FastAPI :5124
  eval/                golden sets built from data that already exists
ops/                   P0 remediation: PII audit, snapshot-then-quarantine
data/                  aliases · fee schedule · emissions lexicon · competitor briefs
tests/                 cross-module invariants
```

## Run it

```bash
uv venv --python 3.11 .venv && uv pip install -e .

# P0 — audit first, remediate deliberately
.venv/bin/python ops/qdrant_pii_audit.py
.venv/bin/python ops/p0_remediate.py quarantine <collection> --yes   # snapshots first

# Phase 4a: gates -> dedup -> LLM merge -> supersession -> kb_units
LLM_MAX_CONCURRENCY=48 .venv/bin/python -m agentv1.kb.build_kb --refresh

# Phase 4b: project into Qdrant behind aliases (build one alias per process)
.venv/bin/python -m agentv1.index.build_kb_index --refresh --alias unitronic_kb_units

# serve
.venv/bin/python -m agentv1.agent.server        # :5124
.venv/bin/python -m pytest tests/ -q
```

## Corrections to `AGENT_PLAN.md`, measured on 2026-08-06

The plan asks for every figure to be re-measured before it is relied on. These
came back different, and two of them change design decisions.

| Plan says | Measured | Consequence |
|---|---|---|
| The reasoner cannot do native tool calling; expect free-form JSON + code-side validation (§10) | **False.** The endpoint runs `--enable-auto-tool-choice --tool-call-parser gemma4` and returns `finish_reason="tool_calls"` with well-formed arguments | The agent loop uses native tool calling. The separate `json_schema` guided-decoding failure *is* real, so KB merge still uses free-form JSON + validation |
| `calls_cases` is 914 and climbing | **6,771** cases, 3,716 case units, 2,516 recorded *failed* attempts | The negative-evidence layer is fully available now, not a future dependency |
| Embeddings come from LM Studio on `:1234` | LM Studio is gone; `:1234` is vLLM serving Gemma-4-31B and has no embeddings route | Added a two-backend embedder. Local MPS measured **256 texts/s** vs **121/s** for a DGX vLLM embedding server; cross-backend cosine is 1.0000 |
| PII exposure is 2 collections (§3.1) | **5 collections** are CRITICAL — both `customer_service_classification` twins and the `customer_service_training_8b` twin were not named | The audit enumerates all 15 rather than checking the two that were known |
| ~1,200 LLM merge calls, about an hour | **4,272** clusters after 3-stage dedup; the box is aggregate-throughput-bound at ~227 tok/s, so ~3 h | Concurrency past ~48 does not help; the limit is total generation throughput, not queue depth |
| 5,353-call residual | **5,361** | — |
| 61,426 knowledge units | **61,501** over 29,574 docs | — |

Confirmed as stated: the 28.2% support-queue buying intent (§1), the
`incorrect_statements` array trap (535 vs 0 as a boolean), 32 documents
`complied_improperly`, `unitronic_faq_0_6b` at 0 points while routed, and
`supersedes_calls[]` resolving 100% against `calls_analysis.file_name`.

## Things found during implementation that the plan does not mention

**Order and invoice numbers in body text.** The plan's PII work targets payload
*keys*. The first residual/case-narrative build was clean by that standard and
still shipped `Order #854916`, `Invoice 71520`, `cable UCP 58647` inside the
answer text — an indirect identifier that resolves to one customer's
transaction and would be returned to a different customer as a retrieval
result. `normalize.redact_transaction_ids` now redacts these, but only with an
explicit context word, so part numbers, ECU identifiers, RPM and horsepower
survive intact.

**Union-find over-merges.** Stages 1–3 of the dedup all key on the *question*
side, and single-link chaining collapsed 1,087 units into one cluster titled
"UniConnect Plus Cable Reset Process". Refining on the *answer* side split it
into the facts it actually contained — reset options, scheduling, the $150 fee,
the fee waiver. Without that, a $150 unit and a $300 unit would have been handed
to the merge model as one topic.

**Model channel markers leak into content.** Gemma-4 emits
`<|channel>thought<channel|>` into `content` once a conversation contains
tool calls — never on a single-turn completion, so it survives every smoke
test. Stripped at the client boundary.

**`swap_alias` was broken.** `qm.AliasOperations` is a `typing.Union`, not a
model class, so instantiating it raised `TypeError`. The generation swap — the
thing that makes a bad build a seconds-long rollback — did not work until fixed.

**Build one alias per process.** Running two projections in one process
produced a 404 against a collection that had just been created. Not yet root
caused; the workaround is one alias per invocation.

## What is enforced by mechanism rather than by prompt

1. **`training_safe` floor** is AND-ed server-side in the retrieval layer. A
   caller-supplied filter is intersected with it, never substituted for it.
2. **Payload is an allowlist.** Adding a field means editing
   `index/payload.py`. Verified: no `_node_content` on any new collection, so
   the double-serialisation leak that made the old fix insufficient cannot recur.
3. **Tier-2 identity is inexpressible.** `get_my_*` schemas expose empty
   `properties` with `additionalProperties: false`; `customer_id` is injected by
   the executor from server-side session state. A phone match sets context, never
   authorisation — `name_agrees` holds on only 31.2% of phone matches and
   `1111111111` maps to 374 accounts.
4. **Grounding.** A price or stage-availability claim requires a matching tool
   result in the same turn, matched by provenance token. Verified: `"A license
   transfer costs $150"` is blocked with no evidence and passes once
   `get_fee_schedule` has run.
5. **Emissions runs pre-retrieval**, so a refusal is not a refusal with the
   refused content already in the context window. Measured **229/268 (85.4%)**
   refusal recall against the historically flagged calls, versus a 74.3% human
   baseline.
6. **Occurrence is a bounded boost**, never a sort key. 42,209 singletons are
   the reason the corpus exists.

## What was actually built and measured

```
Phase 4a  63,011 gated units -> 56,029 canonical      65 min, gen 1ef22e879d97
          merged 3,011 · SPLIT 1,197 · verbatim 50,286 · failed 64 (1.5%, fell
          back to the best member rather than losing the cluster)
          supersession: 9,617/9,617 filenames resolved, 18,709 edges, 0 misses
Phase 4b  unitronic_kb_units        112,019 pts   (56,029 units x answer+query)
          unitronic_case_narratives   6,664 pts
          unitronic_call_residual     5,339 pts
          unitronic_platform_stages     410 pts
          all four behind aliases; all four audited CLEAN
```

Eval, against the golden sets built from data that already existed (small
samples — treat as smoke tests, not as the number):

| Set | n | Pass | Note |
|---|---|---|---|
| emissions (agent) | 40 | **80.0%** | human baseline on the same 40 is 80.0% |
| emissions (raw model, no guardrails) | 40 | 82.5% | the pre-retrieval gate is not what carries this |
| must_never_say | 20 | 85.0% | 3/20 repeated a claim a human got wrong |
| known_gaps | 20 | **50.0%** | 10 abstained, **9 invented specifics** |

**`known_gaps` at 50% is the headline weakness.** These are questions no human
could answer, so the corpus contains no answer by construction, and the agent
invents specifics roughly half the time. The abstention instruction in the
persona prompt is not sufficient on its own. Fixing it properly means a
retrieval-confidence threshold that forces abstention when nothing clears the
reranker — that is a real piece of work and it is not done.

The emissions comparison is also worth reading carefully: the raw model scores
*higher* than the agent on this sample. The pre-retrieval lexicon gate catches
the requests that name a defeat device, but these eval questions are
paraphrases of what a caller said across a long call, so many never contain a
lexicon term. Lexicon recall against the *units* is 85.4%; recall against
*paraphrased questions* is materially lower, and that gap is the honest number
for a chat surface.

## Known gaps

- **Escalation has no sink.** `agents` (40 rows) carries only
  name/email/active/userId, `departments` (3 rows) only name/headUserIds, and
  there is no agent→department edge. "Route French escalations to a
  French-capable human" is unbuildable as specified. Every escalation record
  says so explicitly rather than implying a queue exists.
- **PII name recall is 0%** with the regex backend. `guardrails/pii.py` defines
  the backend interface and ships a Presidio implementation behind it; that
  swap is the fix, not more regex.
- **`sync_tuning_db.py` is unscheduled.** Measured 152 h stale, so
  `check_stage_availability` returns `Degraded` today. That is correct behaviour
  and also means the freshness gate is load-bearing from day one.
- **§3.2 (`POST /seed` running `deleteMany({})` unauthenticated) and the
  `chat.py` / `qdrant_config.py` fixes are in a repository that is not checked
  out here.** They are recorded in `ops/README.md` as external actions so they
  are not silently lost.
- **The 5 CRITICAL collections are snapshotted, not deleted.** `ops/snapshots/`
  holds 2.85 GB of SHA-256-verified Qdrant snapshots. Quarantine is one command
  and it is left to an operator on purpose: `unitronic_call_transcriptions_0_6b`
  is still read by the running Flask RAG path, and
  `unitronic_customer_service_training_0_6b` is the only source of
  natively-written French in the whole system. Deleting either has a
  consequence outside this repo.
  **New since the plan: the audit found Luhn-valid payment card numbers with
  expiry dates** in `unitronic_call_transcriptions_0_6b` and
  `unitronic_customer_service_classification_0_6b`, transcribed from customers
  reading cards aloud. That is PCI-DSS scope and it raises the urgency of the
  quarantine well above what §3.1 assumed.
- **Unit ids can look like phone numbers.** `u_` + 16 hex contains ten-digit
  runs, so the PII scanner flagged every citation until an anchored exemption
  was added. The exemption cannot be used to smuggle anything (a raw phone in a
  field *named* `unit_id` is still caught, and there is a test for it), but the
  real fix is an id format that cannot collide with a phone shape. That is a KB
  rebuild, so it is recorded rather than done.
- No A/B is promised. Four agents carry 97.1% of volume, so agent-level
  randomisation gives two clusters per arm and treatment is confounded with the
  person; quote→sale needs ~2,300 per arm to detect 3pp and four weeks yields
  ~60. Use within-agent, within-intent before/after on operational measures.
