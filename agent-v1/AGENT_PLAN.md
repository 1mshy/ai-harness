# Plan: sales and support agents on the call corpus

**Status:** proposal. **As of 2026-08-05** — every number below was measured on that
date against live Mongo (`10.150.0.30/transcribing`) and live Qdrant (`:6333`).
Several move daily; the ones that move fastest are flagged. Re-measure before any
decision that depends on a figure.

---

## 1. The decision in one page

Build **one agent platform with two personas**, not two agents. Sales and support
differ only by system prompt, default retrieval filters and tool allowlist — they
share one index set, one retrieval layer, one tool executor, one runtime.

The reason is measured, not aesthetic: **29% of buying intent arrives through the
support queue.** 4,091 calls with active purchase intent (1,309 `ready_to_buy`) are
handled by `technical_support`, while only 137 of 5,219 `support_only` calls sit in
sales. The leak is one-directional and large, so crossing the sales/support boundary
is the common case, not an edge case. Two separate services would need a
mid-conversation state-handoff contract for the majority of conversations. Trigger
the persona on `purchase_intent`, not on `department`.

**Ship internal-copilot first, customer-facing second.** Staff traffic is real,
the audience is forgiving, and it is the only source of eval signal that isn't
contaminated by the corpus the agent was built from.

**The honest prize.** Not "half the call centre". Measured on 2025, the last full
year: 17,129 calls / 1,115 talk-hours, of which **7,838 calls / ~430 talk-hours**
were both `bot_automatable` and already `resolved`. Automatable calls are shorter
than average (3.3 min vs 4.8), so 53.4% of *calls* is only 44.2% of *talk time*,
and the >10-minute bucket is 31.5% of all labour at 24% automatable. Quote the 430
hours.

**But the bigger prize is the blocker, not the knowledge.** Of the 15,450
conversations judged *not* automatable, **6,975 (497 hours) were blocked solely by
an account or data lookup** — the single most common `bot_blockers` string is the
bare `"needed account lookup"` at 3,189 occurrences; account/customer lookup is
6,832 of 23,800 blocker items (28.9%). Emotional or judgement blockers are 193
items (0.8%). A tool-using agent with read access to `tuning_customers` converts
most of that without a single new document. Adding it to the base gives a total
addressable **25,885 of 34,360 conversations (75.3%) and 1,583 of 2,418 talk-hours
(65.5%)** — the ceiling, not the target.

**Three things must be fixed before any agent work**, because an agent amplifies
all three. They are in §3.

---

## 2. Corrections to premises — things that were checked and are wrong

These were asserted during research and then falsified by querying the systems.
They are listed first because each one would have produced wasted work.

| Claim | Reality |
|---|---|
| "The `platform_id` join from a caller's car to `tuning_platforms` is broken; 73 of 75 resolve to nothing." | **False.** `tuning_platforms._id` *is* the platform id. All 75 distinct `tuning_customers.vehicles[].platform_id` values resolve; **83,747 of 89,632 vehicle rows join (93.4%)**, and the 5,885 misses are `platform_id: null`, not dangling references. No `sync_tuning_db.py` change is needed. The compatibility oracle is unblocked. |
| "`connect_calls` is 7% built and weeks of GPU away — the long pole." | **False and moving fast.** It is running now. `calls_cases` went **409 → 914** and `calls_cases_state` **25 → 105 keys** during the few hours this plan was written. At the observed rate the queue drains in roughly a day or two, not weeks. Do not sequence anything around it. |
| "The DGX vLLM endpoint is saturated by a ~128-request third-party load; its p99 belongs to someone else." | **Does not reproduce.** Five consecutive trivial completions against `:1234` *while `connect_calls` was consuming it* returned in 1.29–1.56 s. The single 99.9 s measurement in `CONNECT_CALLS.md` was an anomaly, not a steady state. Hosting the reasoner off the shared box is still defensible on *variance* grounds, but the premise as written is not evidence. Also: a large share of that load is likely **first-party** — `transcribing.settings.defaultModel` is `nvidia/Gemma-4-31B-IT-NVFP4` with `autoAnalysis` and `autoImport` enabled, i.e. an internal app is driving the same endpoint. That makes it schedulable. |
| "`'sensitivity' not in c` at `embed_classified_calls.py:438` is a latent bug because Mongo stores the field as null." | **Not a bug.** `count({'sensitivity': {'$exists': False}}) = 1083`, `count({'sensitivity': {'$type':'null'}}) = 0`. The confusion is that Mongo's `{'sensitivity': None}` also matches *missing* fields. No fix needed. |
| "There are 40 agents to randomise an A/B across." | **There are effectively 4.** `agents` has 40 rows, but 2026 volume is 3,405 / 2,660 / 2,319 / 1,372 across four people — **97.1% of 10,050 calls**. Any agent-level A/B gives 2 clusters per arm, where treatment and agent are perfectly confounded. |
| "The corpus has zero French answer text, so all French must be authored." | **Narrower than that.** True of the *call* corpus (the analyzer prompts force English output). But `unitronic_customer_service_training_0_6b` — 19,872 live points, guaranteed-included on every routed query — is Unitronic's own agents writing French to customers on a written channel. 40 sampled payloads carried `bonjour` 38×, `merci` 46×, `vous` 50×. That is exactly the register the chat and DM surfaces need. |
| "Compatibility closes at 0.4% because the lookup didn't happen on the call — fix the lookup and it closes." | **Contradicted.** Of 2,212 calls carrying a `compatibility` objection, **1,646 (74.4%) already end `resolved`** and 96.8% have a populated `vehicles[]`. The subpopulation where the lookup *succeeded* closes at 9/1,646 = **0.5%** — statistically identical to the 0.4% headline. Build the compatibility oracle for correctness and handle time, not on a conversion business case. |
| "Quote→sale is 14.6%." | **Under-counts by ~2×.** `sales_outcome: sale_made` means a sale was recorded *during a phone call*. Cross-joining quote callers to `tuning_customers.orders[].created_at`, **361 of 1,252 (28.8%) placed a real order within 90 days**. The dominant conversion channel is invisible to a call corpus, so quote→sale cannot be an A/B endpoint. |
| "Fix the Qdrant PII by dropping fields from `create_metadata()`." | **Insufficient for existing points.** The metadata dict is serialised a *second* time inside `_node_content`. Current code already omits `file_name` (see the comment at `embed_classified_calls.py:232-235`) — the 24,760 live points predate that fix and still carry it. Only delete-and-rebuild removes it. |

---

## 3. P0 — three things to fix before any agent work

### 3.1 Live PII in a served Qdrant payload

`unitronic_call_transcriptions_0_6b` holds 24,760 points whose payloads carry:

- `file_name` — a 3CX filename that **embeds the caller's phone number**
  (verified: `[<StaffName>]_124-<10 digits>_...wav`),
- `agent_name` — a real staff first name,
- `caller_area_code`,
- and the same values again inside `_node_content`.

A filtered count on `training_safe = true` in that collection returns **0** — the
publication gate is not represented at all. These payloads reach API clients via
`rag_core.py:924` → `chat.py:163` (`sources[].metadata`) and via SSE `source`
events. This is the same class of exposure as the 2026-05 git-history purge.

Separately, `unitronic_customer_service_training_0_6b` (19,872 points, **guaranteed
included on every routed query**) carries `thread_path = "inbox/<instagram handle>"`
on 100% of sampled points. A 10-digit-phone regex test would pass while that
identifier still ships.

**Action:** delete `unitronic_call_transcriptions_0_6b` outright (it is superseded
by §5 anyway); enumerate all 15 live collections against a pattern set covering
handles, emails, VINs and order numbers, not just phone runs; replace the
`sources[]` projection with an explicit allowlist rather than `node.metadata`
pass-through.

Note the pipeline side of this is already in motion — `analyze_calls
--backfill-scrub` / `connect_calls --rescrub` and the new `name_gazetteer` close the
measured **1.79% third-party-name leak** (B2B calls where the shop is redacted and
*their* customer is not). That work is a prerequisite for §5, not a parallel track.

### 3.2 `POST /seed` on the MCP server

`MCP/server/server.ts:1184` exposes an **unauthenticated** `POST /seed` that runs
`deleteMany({})` at `:1198` before inserting 5 fake records. If anyone repoints
that server at real customer data without deleting the route, one stray request
destroys the collection. `server.ts:186` additionally does
`customersCollection.find({}).toArray()` — a full-collection scan per phone lookup.

**Action:** delete the route, bind to localhost. Keep only its two genuinely-real
tools (`search_unitronic_parts`, `lookup_product_by_sku`, backed by scraped product
data); the other 8 resolve against a 9 KB `data/customers.json` of John Doe /
+1-555 records.

### 3.3 Retrieval is configured such that "the agent hallucinates" would be a lie

- `similarity_top_k: 2` (`config/qdrant_config.py:83`), reduced from 4 for latency.
  Two chunks of median 336 chars is ~200 words of evidence for the whole answer.
- **No reranker anywhere** — repo-wide grep for `rerank|node_postprocessor` returns
  zero real hits.
- **Hybrid search is built, populated and switched off.** `comprehensive` and
  `products` carry real sparse vectors alongside dense; `enable_hybrid_search`
  defaults to `False`.
- `unitronic_faq_0_6b` has **0 points** and is a live member of
  `DEFAULT_COLLECTIONS` whose selector summary owns *"Troubleshooting ('how to
  fix', 'problem with')"* — the highest-volume intent in the corpus is routed to an
  empty collection today.
- `QWEN_SIZE` is unset in `.env`, so prod runs the dev-sized 1024-dim default. Three
  collections have no `_8b` twin, and `qdrant_config.py` **creates collections on a
  read path** — so flipping to 8b silently produces empty retrieval rather than an
  error.

**Action:** pin `QWEN_SIZE` explicitly; make `get_vector_store(create=False)` the
default; add a startup assertion that every routed collection has
`points_count > 0` (it will refuse to boot today — that is correct); add the
reranker and raise top-k as in §6.

Fixing §3.3 alone gives the *existing* RAG path a large quality jump with no agent
present — and a clean before/after baseline for whether the agent adds anything.

---

## 4. What the corpus actually supports

### 4.1 Volume and shape (`calls_analysis`, 38,522 docs, 0 errors)

| Dimension | Distribution |
|---|---|
| department | technical_support 22,297 · sales 10,492 · customer_service 3,005 · unknown 1,676 · other 1,042 |
| caller_type | end_customer 26,267 · **dealer_installer 8,750** · vendor 457 · distributor 210 |
| technical_category | compatibility 11,696 · flashing_error 5,668 · shipping_order 3,731 · software_download 2,977 · account_portal 2,716 · performance_issue 2,587 · billing 2,271 · installation 1,641 · hardware 1,580 · returns_rma 779 |
| resolution_status | resolved 25,416 · pending 9,642 · unresolved 2,670 · escalated 784 |
| language | en 35,225 · **fr 3,283 (8.5%; 11.6% of sales)** |
| training_safe | 38,311 true / 201 false (the gate costs the sales corpus 1.05%) |

### 4.2 The knowledge asset

- **61,426 knowledge units** across 29,536 calls — `{kind, title, question, answer,
  conditions, vehicles_applicable, products_applicable, hypothetical_questions[],
  agent_uncertain, time_sensitive, confidence, confidence_reason}`.
  Kinds: product_info 15,538 · procedure 13,145 · troubleshooting 10,340 ·
  compatibility 7,808 · policy 5,391 · pricing 4,969 · internal_process 2,564 ·
  faq 1,671. Confidence: high 40,902 / medium 20,321 / low 203.
- **~186,000 `hypothetical_questions`** — a HyDE corpus written offline *with the
  real answer in hand*. Query-time HyDE is redundant; embed these instead.
- **`search_aliases`** on 52.9% of calls (`flash→ecu tune` 618, `cel→check engine
  light` 363, `dongle→uniconnect plus cable` 98) — a deterministic query-rewrite
  table, not an LLM call.
- **`calls_cases`** — 914 cases and climbing. The differentiator: `chronology[]`,
  `attempts[{attempt, made_by, result}]` where result is `worked`/`failed`, and
  `what_finally_worked`. Of 857 attempts recorded at the 409-case mark, 141 were
  `failed` and 62 cases paired a failed attempt with the fix that worked. This is
  the corpus's **only negative evidence** and nothing else in the system has it.
  The worked example: a per-call unit says *"disable Windows Driver Signature
  Enforcement"*; the case unit says that **fails when Secure Boot is enabled in
  BIOS**. Retrieval that can't express supersession recommends the thing that
  didn't work.
- **`agent_unanswered_questions`** — 5,775 items on 5,260 calls. Questions no human
  could answer, so they are *absent* from the corpus by construction. An agent built
  only on calls fails all of them identically. Largest cluster is hardware
  spec/fitment at 2,633 (45.7%) — **which has no data source anywhere in this
  system.** Roughly half the backlog is unanswerable by any component today.

### 4.3 What must never be answered from the corpus

**Pricing.** 13,359 `prices_discussed` observations over 4,862 distinct free-text
labels. `"license transfer"` spans **$2 to $150,000** (median $150). ASR mangles
amounts — `review.incorrect_statements` catches agents quoting *"$1.50 in line [12]
and then $150 in line [17]"* and *"total is $23.60 but individual prices give
$2,360"*. 442 items quoted in more than one year moved >15%. 53% of the 4,969
pricing units are `time_sensitive`, and **215 pricing units contain dealer cost or
margin** — 17 of them from `end_customer` calls. With `similarity_top_k: 2` there is
no majority vote to protect you: one bad chunk is either a wrong quote or a margin
leak.

The **flat fee schedule** is the exception and is genuinely stable: license transfer
$150 / $150 / $150 across 2024-2026 (n=206/335/258), UniCONNECT cable $165 ×3,
remote reset $150 ×3, mail-in reset $50 ×3. That is ~10 line items that should be
**hand-authored**, not mined. (Even here, UniFlex moved $500 → $400.)

**Stage availability.** `tuning_platforms` is live truth: 106 platforms, 410 stage
rows, **88 currently `released: false`**, and 90 released rows with their newest
file in 2025/26. The call corpus holds 2,437 Stage-3 units, many from 2024 saying
"not out yet" — now false. Retrieval cannot arbitrate this; the structured table
must win.

**Power figures.** `tuning_platforms.stages[].power_figures[]` is authoritative.
Agents self-contradict on power live (*"400 hp at [43], then 600, usually 450 at
[55-56]"*) — 534 calls carry `review.incorrect_statements`, dominated by prices and
power numbers.

### 4.4 Deduplication is a correctness requirement, not an optimisation

Measured three ways over 61,429 units: exact normalised-title match collapses
15.4%; title token-set Jaccard collapses 25–45% depending on threshold; **40.6%
share a verbatim hypothetical-question token set** with another unit. Union:
**53.7% have at least one near-duplicate; 28,423 are true singletons.**

Head examples: `remote cable reset process` ×152, `custom tuning availability` ×120,
and six spellings of "license transfer fee/process" totalling 411 units.

Indexed raw, top-k is routinely five restatements of the same fact, the model reads
repetition as corroboration, and the 28,423 singletons — the actual reason to have
this corpus — never surface.

The merge is affordable: the cluster-size distribution is 28,423 singletons, 2,355
pairs, and only **1,190 clusters of size ≥3**. That is ~1,200 LLM merge calls, about
an hour, not 61,429.

**Do not use recency to arbitrate.** Price disagreement within a topic is context
noise, not drift: `license transfer fee` reads $150 in 17/29/24 units across
2024/25/26, and the $300 outliers co-occur with *cable-reset* scenarios — a
different question with a similar title. Use **consensus within a trailing window**,
with recency as a tiebreak and only on `time_sensitive` units (14.2% of the corpus;
86% should not decay at all — a 2024 fix for a flashing error is still the fix).

The merge prompt must be allowed to answer **"these are two units, split them"**,
or $150 and $300 get averaged into a canonical unit that is wrong in a way no
metric catches.

---

## 5. Architecture

```
════════ SOURCES — PII-bearing, never leave the LAN ════════
  $DATA_ROOT/all_calls/*.wav        tuning-platform MySQL      unitronic_scraper/
  (filenames embed phone numbers)   (restore is Mac-only)      scraped_data_enhanced/
        │                                  │ sync_tuning_db.py        │
╔═══════▼══════════════════════════════════▼════════════════╗         │
║ DGX SPARK 10.150.0.30                                     ║         │
║  gateway :8000  WhisperX + pyannote                       ║         │
║  vLLM    :1234  nvidia/Gemma-4-31B-IT-NVFP4, ctx 32768    ║         │
║  mongod  :27017 db `transcribing`  ⚠ SHARED with an       ║         │
║                 internal staff LLM app (see §11)          ║         │
╚═══════════════════════════════════════════════════════════╝         │
   PHASE 1        PHASE 2         PHASE 3        PHASE 4a ◄NEW        │
   transcribe   analyze_calls   connect_calls    build_kb             │
        ▼            ▼               ▼               ▼                │
     calls    calls_analysis   calls_cases      kb_units  ◄NEW        │
     38,628   38,522 / 61k KU  914 & climbing   kb_units_state        │
                                                kb_revocations        │
   tuning_customers 91,754 ─┐                                         │
   tuning_platforms    106  ├─ authoritative, read at TOOL time,      │
   crm_contacts     22,093 ─┘  NEVER embedded                         │
════════════════════════════════════════════════════════════▼═════════▼══
  PHASE 4b — embedding/build_kb_index.py — pure projection, NO LLM,
             idempotent, alias-fronted.  LM Studio Qwen3 (LOCAL :1234)
        ▼
╔═══════════════════ QDRANT :6333 ═══════════════════════════════════╗
║ NEW (hybrid dense+sparse from birth, behind aliases):              ║
║   unitronic_kb_units{sfx}        call AND case units, ONE set      ║
║   unitronic_case_narratives{sfx}                                   ║
║   unitronic_call_residual{sfx}   the measured 5,353-call residual  ║
║   unitronic_platform_stages{sfx} 410 platform×stage cards          ║
║ KEEP: comprehensive / products / tuning / products_tuning /        ║
║       company_info / customer_service_training (+classification)   ║
║ DELETE: unitronic_call_transcriptions_0_6b  (P0 — §3.1)            ║
║ FIX:    unitronic_faq_0_6b (0 points, routed to today)             ║
╚════════════════════════════════════════════════════════════════════╝
        ▼
  RETRIEVAL  hybrid top-30 → bge-reranker-v2-m3 → top 5-8
             → supersession filter → server-side training_safe floor
        ▼
  TOOL LAYER  native in-process Python · read-only creds
              customer_id injected by executor, never an argument
        ▼
  AGENT RUNTIME  FastAPI/uvicorn :5124 · POST /api/agent/stream
                 bounded tool-calling loop · Flask :5123 untouched
        ▼
  CHANNELS  internal copilot → web chat → email → IG DM
```

### 5.1 The knowledge pipeline (Phase 4), and why it splits in two

`embed_classified_calls.py` **cannot be adapted**. It globs `*_classified.json`
from disk (`:133`) while the analyzer writes `{basename}.json`; it has no pymongo
import at all; its checkpoint is a **positional index into a glob-ordered list**
(`:363-390`), which is only valid if the input never changes — and the analyzer adds
documents continuously. This is a new builder.

Split it along the deployment boundary that already exists:

| | Path | Deps | Owns |
|---|---|---|---|
| **4a** | `dgx_pipeline/build_kb/` | stdlib + `pymongo` | canonical unit **identity** in `transcribing.kb_units`: gates, 3-stage dedup, LLM merge, generation swap, tombstones |
| **4b** | `embedding/build_kb_index.py` | `qdrant-client`, `llama-index` | the **projection** Mongo → Qdrant. No LLM. |

A Qdrant rebuild never re-runs the merge; a merge re-run never needs Qdrant. 4a
follows the conventions this repo already enforces — atomic writes, resumability,
`KB_VERSION`/`MERGE_VERSION` stamps, `--refresh` vs `--redo-outdated`, and the
generation swap that `test_connect_store.py` already pins (write new gen → delete
old gen → stamp state last).

**Gates applied at 4a**, in order:

1. `training_safe` — document-level, **non-bypassable** (matching the pipeline's own
   discipline where `--no-filter` deliberately does not bypass it).
2. Unscreened — fail-closed for cases.
3. `review.emissions_handling == "complied_improperly"` — drops 32 docs / 76 units.
   These are `training_safe: true` *and* `useful_content: true` and would otherwise
   embed alongside the 201 correct refusals. `training_safe` is a PII gate; it
   screens nothing about behavioural correctness.
4. `review.incorrect_statements.0` exists — drops 534 docs.
   **Note the trap:** this field is an *array*. `{review.incorrect_statements: true}`
   returns 0; `{'review.incorrect_statements.0': {$exists: true}}` returns 534. The
   same applies to `threat_flags`, `agent_knowledge_gaps`, `pii_spoken_in_call` —
   any dashboard querying these as booleans is silently reporting zero.
5. Derived labels: `emissions_risk`, `safety_gated`, `dealer_pricing`.

**The PII fix that unblocks everything:** `case_knowledge_units[].supersedes_calls[]`
is currently a list of **3CX WAV filenames that embed the caller's phone number**.
The supersession relation — the entire point of case units — is expressed in PII
today and must be remapped to opaque `unit_id`s before anything is embedded.

**Why call units and case units share one collection.** A case unit's whole value is
outranking a call unit, and `router_factory.py` does *per-collection* selection.
~2,900 projected case units in their own collection would essentially never be
picked by the `LLMMultiSelector`, destroying exactly the property that makes them
valuable. Identical schemas — merge them, add an `evidence` field, and supersession
becomes a rerank-time filter over one result set.

**One unit = one point, no chunking.** Median unit is 336 chars, p95 591, longest
answer 1,165 — the `SentenceSplitter(chunk_size=1024)` at `:556` never fires. Two
points per canonical unit: an ANSWER point (title/question/answer/conditions) and a
QUERY point (question + `hypothetical_questions` + alias colloquial forms), deduped
on `unit_id` at retrieval.

**Whole-call summaries are mostly redundant.** 29,384 of the 34,737 safe+useful
calls already have at least one knowledge unit. Index only the measured **5,353-call
residual** — all of which have both `problem` and `solution` populated. That is a
6.5× reduction in duplicated content for zero coverage loss.

---

## 6. Retrieval

In strict priority order:

1. **Hybrid dense+sparse, created hybrid from birth.** Do **not** flip
   `enable_hybrid_search` on an existing collection — `qdrant_config.py:258-273`
   *deletes and recreates* any collection lacking sparse vectors.
2. **Surface-form normalization, applied identically at index and query time.** This
   is the higher-leverage half and it is not what you'd expect. DTC codes appear in
   1.0% of units and part numbers in 1.4% — the classic "BM25 for error codes"
   argument is weak here. The real lexical failure is `Stage 1` (5,236 units) vs
   `Stage 1+` (3,130): dense embeddings blur the `+` and a default BM25 tokenizer
   *strips* it. Emit `stage_1_plus`, `uniconnect_plus` tokens on both sides.
   UniConnect alone has 10 measured surface forms.
3. **`bge-reranker-v2-m3`, local, CPU/MPS.** Retrieve 30 → rerank → 5-8. Multilingual
   is a requirement, not a nicety (French is 8.5%). Rejected Cohere Rerank: ships
   customer-call text to a third party and adds a hop inside the latency budget.
   Rejected a large reranker on the DGX: this is a synchronous user-blocking call.
4. **Supersession filter — deterministic code, not model judgement.** If a case unit
   is retrieved, hard-drop every unit in its `superseded_unit_ids` before context
   assembly.
5. **Server-side `training_safe` floor.** `chat.py:110-111` reads
   `collection_filters` and `metadata_filters` **straight off the request body** and
   passes them through. Any safety filter expressed that way is a suggestion. The
   floor must be AND-ed in inside `create_filtered_query_engine` and never replaced
   by the caller's filters.
6. **Occurrence as a bounded boost, never a filter or primary sort.** The 28,423
   singletons are the reason the corpus exists; if occurrence becomes a sort key the
   index degrades into an FAQ of things everyone already knows.
7. **`time_sensitive`-gated decay only.** Never global.

**Rejected:** query-time HyDE and multi-query (the corpus ships ~186k hypothetical
questions written with the answer in hand — both techniques collapse into index-time
work already done); **GraphRAG** (the entity graph it would infer — vehicle →
platform → stage → availability — already exists as a normalised 106-row relational
table; querying it is faster, exact and auditable).

**Promote `ParallelRetriever` from dead code** (`parallel_retriever.py:69`; zero
callers outside itself). It is the right primitive — embed once, fan out, no LLM
synthesis. Two defects to fix on adoption: `:121` calls `get_text_embedding` (the
*document* path) for a query, skipping the Qwen3 instruct prefix at
`qwen3_embeddings.py:80`; and `qwen3_embeddings` uses blocking `requests` with an
`_aget_query_embedding` that just calls the sync version.

**Delete `GuaranteedIndexQueryEngine` / `MinimumPullQueryEngine`.**
`custom_query_engine.py:51` and `:107` run a *full LLM synthesis* per guaranteed
collection and then read only `source_nodes`, discarding the generated text. That is
N+1 paid generations per query, and it is why streaming had to be collapsed.

---

## 7. Tools

15 tools, native in-process Python. MCP is retained only where a second client (the
internal dashboard sharing this Mongo) needs the same surface — a tool call inside a
loop making 2-6 per turn should cost microseconds, not an HTTP round trip.

**Tier 0 — no identity required**

| Tool | Backing | Notes |
|---|---|---|
| `search_knowledge(query, kind?, department?, language?)` | Qdrant `kb_units` | training_safe floor server-side |
| `get_case(case_id)` | Qdrant `case_narratives` + Mongo `calls_cases` | chronology, attempts (incl. **failed**), what_finally_worked |
| `search_products(query)` / `lookup_product_by_sku(sku)` | existing MCP product tools (genuinely real) | |
| `resolve_vehicle(make, model, year?, engine?, chassis?)` | `tuning_platforms` | returns candidates + `agree_on_stages` + a clarifying question |
| `check_stage_availability(platform_id, stage?)` | `tuning_platforms._id` | **the compatibility oracle** |
| `get_fee_schedule()` | hand-authored ~10 items | never mined from calls |
| `lookup_error_string(text)` | exact-match index | see §8.2 |

**Tier 2 — authenticated, customer-scoped**

These take **no customer identifier in their signature**. The executor injects
`customer_id` from server-side session state, which makes "look up someone else's
order because the user asked nicely" *inexpressible* rather than merely discouraged.

`get_my_vehicles()` · `get_my_orders()` · `get_my_tune_history()` ·
`get_my_open_case()`

**Control:** `escalate_to_human(reason, summary)` · `record_lead(...)` ·
`log_knowledge_gap(question)` · `request_approval(...)`

**Enforcement is four mechanisms, none of them prompts:**

1. **Read-only at the credential** — a Mongo user with `read` on `transcribing`, a
   MySQL user with `SELECT` only. An agent cannot write what the connection string
   cannot write.
2. **Scoping by omission** (above).
3. **Budgets** — `Flask-Limiter` (already a dependency) per-session tool budgets plus
   a hard per-turn cap.
4. **Breakers** — `health_monitor.CircuitBreaker` is a complete 705-line
   implementation **instantiated nowhere**. Wire it per dependency rather than
   growing a second retry stack; adopt the DGX pipeline's tested
   transient-vs-permanent split (`test_resilience.py`).

`developer.py:225-392` already contains a working human-in-the-loop approval state
machine (pending → approve/deny → execute, persisted to Mongo). Move it out of the
blueprint into the tool executor so it gates real traffic — today it gates only the
developer console, its flag is a module global at `:27` that won't survive multiple
workers, and its hardcoded 6-tool list has already drifted from the 12 registered.

---

## 8. The two personas

### 8.1 Sales

**Ships as:** internal sales copilot first (live call assist, post-call follow-up
drafting), customer-facing consultative chat second.

`templates.py:222` already contains a consultative sales prompt that mandates
questioning. The delta between that mode and an agent is stated concretely by the
code: it can ask *"what engine code?"* and then **lose the answer** — conversation
memory is `self.conversation_memory[session_id][-5:]` with each message truncated to
200 chars and a literal `'...'` appended (`rag_core.py:1585-1587`). A mode is
instruction text injected once with no state and no actuators; an agent is a loop
with tools and durable slots.

**What it must do:**

- **Qualify to a `platform_id`.** That is the pivot. Vehicle field completeness on
  sales calls: make 96%, model 96%, **year 63%**, chassis 28%, engine 20%. Year is
  where most multi-candidate spread comes from. But the ambiguity mostly doesn't
  matter — of 7,957 resolved vehicle entries, **57% have all candidates agreeing on
  `max_released_stage`**. The agent doesn't need to pin the platform; it needs to
  know *when the candidates disagree*. That is a cheap test and it defines the one
  clarifying question worth asking.
- **Never quote a price it did not get from a tool.** §4.3.
- **Serve dealers differently.** 8,750 `dealer_installer` calls are a different
  motion, resolve to a CRM contact only 8.1% of the time, and comply-improperly on
  emissions at nearly twice the end-customer rate (19% vs 10%).
- **Fire on the support path too.** §1.

**Carry the prior quote forward.** 2,915 quote calls; 35% never generate another
call; 77% of returns happen inside 30 days; and **267 returning callers were
re-quoted**, proving the second agent did not have the first quote. The payload
already exists — `products_quoted` on 99% of quote calls, `prices_discussed` 84%,
`agent_promises` 46%, `dates_promised` 33%. This is a lightweight phone-key join,
**not** a reason to add a commercial dimension to `connect_calls` (case prompts
produce zero pricing/sales units today, and changing them costs a `CASE_VERSION`
bump plus `--redo-outdated`).

**Upsell — size it honestly.** 72,262 of 91,754 accounts (78.8%) own a stage below
what is released for their platform. That number is inflated by decade-old flashes —
`owned_stage.at` reaches back to 2007, and only 20.4% of vehicles were flashed since
2024. Filtered to phone-reachable accounts with 2024+ activity, the real campaign is
**~16,500 accounts**, mostly Stage 2 → Stage 3 on the two big EA888 platforms. Pure
structured query, zero LLM cost. Contacting the full 72k is a large-scale outreach to
dormant records with CASL exposure.

**Competitors:** five briefs cover ~74% of 3,680 mentions — APR 1,280, Integrated
Engineering 553, 034Motorsport 418, CTS Turbo 357, COBB 177. Competitor objections
end `needs_info` 53% of the time and `sale_made` twice out of 286: the agent had
nothing to say. This is a five-page content fix, not a skill gap. **Tag CTS, IE, ECS,
AWE, Eventuri, CSF as complementary hardware, not rivals** — otherwise the agent
argues against the customer's own build.

**Do not build a sales-objection index.** `objection_detail` records what the
customer worried about and there is **no rebuttal field anywhere in the schema**.
Retrieving five gives a taxonomy of anxiety and nothing to say. If objection handling
is genuinely wanted, that is a ninth analyzer pass capturing the agent's rebuttal
plus an `ANALYSIS_VERSION` bump.

### 8.2 Support

**Ships as:** internal agent-assist copilot first. The data argues for it strongly —
`bot_automatable` is the analysis model's own judgement about what a
knowledgebase-backed AI could do, never validated against a bot actually attempting
these calls. The copilot is how you validate it without exposing a customer.

**Three answer paths, chosen by question type.** This is the load-bearing design
decision, and it is measured:

- **Deterministic lookup.** Compatibility is 11,696 calls and is *not* a
  troubleshooting problem — its top verbatim error string appears 11 times (vs 159
  for flashing_error), `solution_steps` population is the lowest of any category
  (48%, avg 1.74 steps) because there is no procedure to give, and its top-10
  canonical problems are two questions: *"does a tune exist for this year/model"*
  and *"does this VIN already have a Unitronic tune"*. Both are database reads.
- **Exact-match error-string index.** `flashing_error` is a **closed vocabulary** —
  ~25 verbatim strings cover most of 5,663 calls: *"the request is not supported"*
  159, *"interrupted flashing session detected"* 130, *"cable is not responding"*
  129+71, *"ecu file invalid"* 115, *"data is invalid"* 111+47. This is a lookup key,
  not a semantic-search problem, and it is the cheapest high-volume win available.
- **Semantic retrieval** over `kb_units` for the residue.

**Use the failed attempts.** Two ways: as a **pre-check** ("before I suggest X — is
Secure Boot enabled?") and as an **anti-answer** that suppresses the superseded
per-call unit.

**Required-slots protocol.** Of 61 measured avoidable-repeat cases, the largest theme
(19) is *"agent did not collect required data upfront"*, followed by *"lacked the
fact"* (18), *"lacked tools/authority"* (17) and *"deferred instead of acting"* (16).
Three of the top four are fixable by protocol, not documents: enforce a slot
checklist per issue type (VIN / ECU box code / TCU ID + revision / cable serial /
fuel octane) *before* answering, and never close a flashing interaction without
confirming the vehicle started.

**Answer, don't promise.** The largest single follow-up reason is
`callback_promised` at 4,270 — a manufactured repeat driver. An agent that answers
synchronously removes it regardless of whether its answer quality beats a human's.

**Where humans already lose, the agent will too.** Scope automation to the tractable
head. Failure rates by category: compatibility 24.5%, flashing_error 28.8% — but
hardware 49.6%, performance_issue 48.4%, returns_rma 53.2%. By root cause,
`docs_unclear` fails only 13.4% and is 79.5% automatable — the single most
agent-addressable root cause, because it is a documentation gap humans closed by
talking.

**Phase-1 scope:** `compatibility_question` (84.8% automatable, n=4,037),
`product_question` (69.7%, n=8,581), `pricing_quote` (69.7%, n=910) — 13,528 calls,
36% of corpus. Everything account-touching (`billing` 21.0%, `shipping_order` 22.5%,
`returns_rma` 29.0%, `file_request` 11.5%) is bottom-quartile and comes later,
behind real tooling.

**Solution steps have two caveats.** They are **success-biased** — 66% populated on
resolved calls, 19% on unresolved — so the corpus teaches what worked and is nearly
silent on what to do when it doesn't. And a visible minority are **backend actions
the agent cannot perform** (*"Unlink the serial number from the previous client's
account in the backend system"*); emitting those verbatim gives the customer an
instruction they cannot follow and looks like a hallucination while being faithful to
the source.

---

## 9. Guardrails, PII and compliance

### 9.1 Emissions — the gate everyone builds in the wrong place

`review.emissions_tampering_request` is true on 268 calls; handling is
`refused_correctly` 201 (75%), `discussed_unclear` 35, **`complied_improperly` 32**.
Technical support accounts for 25 of the 32.

But the flag fires on the **request**. The exposure is in the *knowledge units*:
**432 units mention cat-delete / catless / de-cat, only 11 (2.5%) carry any refusal
language, and 399 (92%) sit on calls the review pass did not flag** — they are
written as neutral product or compatibility facts. Also 467 units match P0420/21/22
and 784 match smog/inspection/readiness. A gate built on
`review.emissions_tampering_request` misses 92% of it.

Today, enforcement is **one sentence in a prompt** (`templates.py:386/424/455`)
against a corpus containing 432 cat-delete units. Prompt-only controls do not survive
prompt injection, a model swap, or an OpenRouter fallback to a different model.

**Mechanism:** a deterministic **pre-retrieval** classifier over a compliance-reviewed
lexicon (cat/DPF/EGR/GPF delete, O2 spacer, readiness defeat, "off-road only"),
returning a canned refusal with no retrieval attempted. Measure recall against the
268 historically flagged calls — the human baseline is 75%, so the bar is well above
it.

### 9.2 Safety

`review.safety_issue` on 1,534 calls (5.0% of the 30,584 reviewed — note the review
block is present on only 79.4% of documents, so every review-derived rate understates
by ~21%). The dominant pattern is **a vehicle disabled by a failed flash**: engine
damage 301, drivability/limp mode 205, clutch/transmission 149, fuel/fire/EGT 123,
turbo overpressure 108. The safety surface and the #1 technical category are the same
population — the agent walking someone through a flash is one wrong file away from an
immobilised or thermally unsafe car.

**Mechanism:** pre-condition checks before *any* flashing instruction (correct file
for this ECU revision, required hardware present, fuel matches the file) plus an
unconditional stop-and-escalate on "won't start", "limp mode", "glowing", "smoke",
"towed".

### 9.3 Identity — a phone match is evidence, not identity

`vehicle_context.matched` on 61.9% of calls, but of those **`name_agrees` is true on
only 31.2%**, false on 22.4%, null on 46.4%. **`candidates > 1` on 12.8% of all
calls.** Placeholder keys are real: `1111111111` maps to 374 accounts, `1234567890`
to 162, `0000000000` to 134. Only 5.4% of callers quote an order number, so an
order-number prompt is the wrong front door.

**Policy:** caller-ID resolves *context*, never *disclosure*. Tier-2 reads require
explicit identity proof. `candidates > 1` forces Tier 0. Port `order_lookup`'s
`MAX_CANDIDATES` ceiling into the vehicle lookup, which currently has none.

### 9.4 The publication gate at query time

Today `training_safe` is enforced **at ingestion only**, and `ANALYZER.md:263` says
it plainly: *"re-screening does not un-embed anything. If a re-screen flips calls to
unsafe, the collection has to be rebuilt."*

`kb_units` owning canonical identity in Mongo converts that into a minutes-long
targeted delete: index `training_safe` as a Qdrant payload field, AND it into the
retriever's base filter server-side, and run a `--revoke` cycle driven by a
`kb_revocations` ledger. At case level `training_safe` is an AND over members, so
honour `withheld_members` and `unscreened_members` too.

**The same missing field — a stable `source_id` in the payload — is what makes a
Law 25 / PIPEDA deletion request unservable against the vector store today.**

### 9.5 Grounding

Prices, compatibility and stage availability are the three places a wrong answer
costs money, and they are exactly the three that go stale: 6,113 units contain a
literal dollar amount, 2,063 mention Stage 3, 453 say "coming soon"/"not yet
released" — over a corpus spanning 2024-2026, against a platforms table with 88
currently-unreleased stage rows.

**Mechanism — make the claim unspeakable without the tool.** Tool results carry a
provenance token; a post-generation validator regex-detects currency amounts and
stage/availability claims and requires a matching tool result **in the same turn**;
on failure, block and route to handoff.

### 9.6 Escalation — build the destination before the trigger

The triggers are measurable and mostly **do not look like anger**. Union of handoff
signals is 10,990 calls (28.5%); `needs_human_agent` alone is 10,192. The emotional
signals are tiny: `escalation_requested` 369, `abusive_language` 242, `threat_flags`
on 26 documents total, angry sentiment 78. **Building the rule off sentiment alone
catches 1,085 of 10,990 — 96% of handoff need is capability, not emotion.**
`churn_risk: high` (217) resolves only 27% of the time.

Two tiers: **hard stop** (any safety issue, emissions request, threat flag, abusive
language, refund/RMA/payment action, or a required account write) and **soft offer**
(churn risk high, frustrated/angry sentiment).

**But there is no sink.** `NEED_HUMAN_INTERVENTION` is dead code — its only call site
is commented out at `rag_core.py:937`, so `chat.py:167`, `rag_logger_wrapper.py:103`
and `n8n.json:103` have always read `False`. And the uncertainty detector cannot
replace it: `uncertainty_detector.py:35` flags `"please contact"` which
`templates.py:34` *mandates*, and flags a missing `$` while `templates.py:38`
*forbids* quoting prices. Its `min_sources_threshold: 2` against
`similarity_top_k: 2` means it fires whenever any collection returns one node.

Worse, **the routing substrate does not exist**: `agents` (40 rows) carries only
name/email/active/userId — no language, no skills, no availability — and
`departments` (3 rows) carries only `name` and `headUserIds` (managers). There is no
agent→department edge. So *"French escalations route to a French-capable human"* is
unbuildable as specified. Escalation is the highest-value agent behaviour and its
destination is the unsolved half. **Build the substrate first**, or ship an agent
that promises a callback nobody makes.

### 9.7 Employee data

`review.agent_performance`, `coaching_note`, `agent_knowledge_gaps`,
`incorrect_statements` and `case_metrics.agents_involved` name real staff.
`ANALYZER.md` documents an unexplained **11.7-point French/English scoring gap**
(69.0% vs 80.7% "good") that within-agent swings suggest is partly real skill and
partly reviewer bias — *"the data cannot separate the two"*.

Keep it out of every payload and prompt (all designs already do). But storage rules
do not discharge the obligation to the people: the copilot's whole eval strategy
depends on **four** agents honestly labelling suggestions accepted/edited/rejected,
and that signal is the first thing to degrade if the tool reads as surveillance. A
briefing, a written commitment on use, and a notice/correction path are
**prerequisites to the pilot**, not follow-ups. Quebec Law 25's automated-decision
provisions plausibly apply.

### 9.8 Evaluate Presidio rather than hand-rolling the PII scanner

Every guardrail here is specified as custom regex. That is right for the emissions
lexicon (genuinely domain-specific) and wrong for PII egress, where the gate is set
at absolute zero and hand-rolled regex has a known recall problem on names,
addresses and non-standard formats — and the corpus is documented to contain
`credit_card` (459 docs) and `dob` mentions. Microsoft Presidio self-hosts and would
be a two-day integration behind the same interface.

---

## 10. Technologies

| Layer | Choice | Rejected, and why |
|---|---|---|
| **Agent loop** | Own it. A bounded tool-calling loop against a first-party SDK, with per-turn hooks for approval, error interception and retries. Max 4 iterations, max 6 tool calls, 45 s hard wall clock. | **LangGraph** — a second state model parallel to the working `conversation_memory.py` + `MultiUserRAGWrapper`, and it drags the LangChain tree into a repo that deliberately has none; graph-shaped debugging when a customer-facing agent loops. **LlamaIndex AgentWorkflow** — couples agent control flow to the pinned `llama_index==0.12.44` that gates the Qdrant integration, so a retrieval bugfix becomes an agent regression. **Claude Agent SDK** — built around Read/Write/Edit/Bash for a sandboxed coding agent; wrong shape for 15 read-only business tools and a 3-sentence reply. **OpenAI Agents SDK** — a third provider SDK for no capability gain. **CrewAI / AutoGen** — multiply LLM calls and latency with role-play crews to solve what is one reasoner plus tools. |
| **Reasoner model** | A hosted frontier model on a **dedicated** endpoint. The rule: *the DGX gets everything with nobody waiting; the hosted API gets everything with a user waiting.* Justify it on **variance**, not saturation (§2). | Routing the reasoner to DGX vLLM. Even at 1.5 s measured, its queue depth is shared with batch phases and an internal app. Also note the DGX model has a hard invariant against `json_schema` constrained decoding (returns empty content), which would force free-form JSON + code-side validation for tool calls. |
| **Retrieval orchestration** | Keep **LlamaIndex** — but for retrieval only. There is no LlamaIndex agent code in the repo, so "we already use LlamaIndex" buys nothing for the loop and a lot for retrieval. | Replacing it. `VectorStoreIndex` / `QdrantVectorStore` / `RouterFactory` work. |
| **Vector DB** | **Qdrant**, named dense + sparse, **behind collection aliases** so a bad build is a seconds-long flip back rather than a re-embed. | Anything else — it is already production, and the hybrid machinery is built and populated. |
| **Embeddings** | **Qwen3 via LM Studio**, 1024-dim (`0.6b`), pinned. | Hosted Voyage/OpenAI/Cohere — the corpus is PII-adjacent and a model change means rebuilding 15 collections. `8b` for v1 — three collections have no `_8b` twin and prod already runs `0.6b` by default; 4× the cost for a marginal gain on 336-char atomic units the reranker recovers more cheaply. |
| **Reranker** | **bge-reranker-v2-m3**, local, ~568M params, tens of ms on CPU/MPS. Multilingual. | Cohere Rerank (third-party data egress + a network hop); a large reranker on the DGX (contended box, synchronous call). |
| **Canonical store** | **Mongo `transcribing.kb_units`** owns unit ids, merge state, tombstones, occurrences and the build queue. | Content-hash ids derived at embed time — dedup cluster membership moves as the analyzer adds documents, so those ids are unstable and idempotent upsert becomes impossible. |
| **Tool layer** | **Native in-process Python.** MCP retained only as an external boundary. | Repointing the existing Node MCP server at real data (§3.2). |
| **Serving** | **FastAPI + uvicorn** on `:5124`, Flask `:5123` untouched and serving as the degradation target. | Rewriting Flask — the blueprints work and the React SSE client change is purely additive. But an agent turn is multi-second, multi-tool and bound on waiting; ASGI is the right runtime for many concurrent slow streams, and `chat.py:280-291` creates a **fresh event loop per request**, which breaks any pooled async client the tool layer needs. |
| **Streaming** | **Reuse the SSE transport as-is.** `chat.py:277` is a schema-less `json.dumps` passthrough and the frontend `switch` has no throwing default — `tool_call` / `tool_result` / `thinking` / `citation` / `handoff` are purely additive, frontend-only. | Rebuilding it. Note there is nothing to preserve on the *token* side: no query engine is constructed with `streaming=True` anywhere in the repo, so `time_to_first_token` at `rag_core.py:1335` currently measures total generation time. |
| **Observability** | **Langfuse, self-hosted** via the existing docker-compose, instrumented through **OpenTelemetry** so `perf_tracer.py` and `llm_timing_handler.py` spans land in the same trace. | **LangSmith** — hosted-first, self-hosting enterprise-gated, ergonomics assume the LangChain stack this design rejects. **Arize Phoenix** as primary — its strength is embedding/retrieval drift; a small team needs prompt versioning and an annotation queue first. Traces will carry names, phones, VINs and order numbers; the PII constraint decides this, not the feature matrix. |
| **Eval** | **Extend `analytics.py`** (`:296` examples, `:330` LLM-judge scoring, `:423` batch runs, `:472` per-example persistence). The hard asset — real human-agent reference answers — already exists. Drive it with **promptfoo** for assertion-style regression. | **Ragas / DeepEval** as primary — generic RAG metrics via an LLM judge cannot distinguish "told a customer Stage 3 was available when `released` is false" (a business incident) from clumsier phrasing. Useful later as a secondary faithfulness signal. |
| **Guardrails** | Deterministic classifiers + **Microsoft Presidio** for PII egress (§9.8). | **NeMo Guardrails / Llama Guard** as the primary emissions control — the lexicon is domain-specific and needs compliance sign-off, not a general safety model. |
| **Resilience** | Wire the existing `health_monitor.CircuitBreaker`; adopt the pipeline's transient-vs-permanent split. | A second retry stack. |
| **Secrets** | **Needs a decision.** The Mongo credential is in a plaintext repo-root `.env`, and this plan adds a FastAPI service, read-only roles, a hosted API key and a Langfuse instance. | — |

---

## 11. Prerequisites nobody has costed

**An internal staff LLM assistant already exists in the same database.**
`transcribing` holds `users` (5, with `passwordHash`, `role` ∈ {dev, admin, agent},
`allowedIps`, `lastSeenAt`), `sessions`, `banned_ips`, `ip_approval_requests`,
`ip_sightings`, `settings` (`chatSystemPrompt`, `analysisPrompt`, `summaryPrompt`,
`defaultModel = nvidia/Gemma-4-31B-IT-NVFP4`, `autoAnalysis`, `autoImport`), `chats`
(69 threads), `analyses`, `user_transcriptions`. **Zero references anywhere in this
repo.**

That is a deployed staff-facing LLM chat product with three-role RBAC, per-user IP
allowlists, configurable prompts and live chat history — i.e. substantially the thing
this plan proposes to ship first, plus the authentication it budgets weeks to build.
**Spend two days inventorying it before writing any auth code**: who owns it, where
the source lives, how much DGX load it generates, and whether the agent service
should extend it, federate with it, or sit beside it. Write down the answer.

**`pbx_recordings` (39,039 docs) is unexploited** and is the only source of
deterministic per-call routing: `extension` (1,588 distinct values), `direction`,
`recordingId`, `startedAt`. `calls_analysis` has no `extension`. `agent_name` is an
LLM-extracted first name from a transcript and is PII; `extension` is a system fact.
Also unreconciled: 39,039 recordings vs 38,522 analyses — ~500 recordings never
reached analysis and nobody has looked at why.

**`crm_call_reports` is not a wrap-up store.** All 817 documents have `summary`,
`transcription` and `sentiment` **empty**; `subject` is the constant
`'3CX PhoneSystem Call'` and `description` is a 75-char auto-generated log line
(*"7/22/2026 11:03 a.m.: Answered incoming call from <number> to 127 (02:43)"*).
There is **no human-written disposition text anywhere in this system**, so a
wrap-up-drafting feature has neither a destination nor a reference corpus. It is also
a PII store the inventory omits: raw caller digits in free text plus `agentEmail`.

**`sync_tuning_db.py` is unscheduled and Mac-pinned.** `tuning_sync_state.last_run`
finished 2026-07-30 — **5.8 days stale**. Every design specifies a freshness gate
(`stale_hours > 48` or `staleness_days > 7`) that would put the compatibility oracle
in permanent refusal mode on day one. Either cron it with alerting, or the refusal
path is the default path.

**Nothing in the serving stack is running.** Port probe: Flask `:5123` down, MCP
`:3030` down, **LM Studio `:1234` down**, frontend down. Only Qdrant responds. LM
Studio is the acknowledged unreplicated SPOF for the entire retrieval path and it is
already down in the steady state. That is direct evidence about the operational
discipline a multi-month programme would assume.

**No non-functional requirements exist.** Zero mentions of SLA, on-call, DR, RTO/RPO
or a retention schedule anywhere in `docs/` or the root docs. The production
entrypoint is `app.py:425 app.run()` — the Flask dev server. Nothing in the pipeline
deletes anything (39,039 recordings, ~37k transcripts, 38,522 analyses) and a
`delete_subject()` path does not exist — which is the same mechanism the
`training_safe` revocation job needs, so it pays for itself twice.

**No frontend i18n.** `insta-front/src` has no i18n library. French cannot be a
first-class channel while every button, error state and consent string around a
natively-authored French refusal is in English.

---

## 12. Build-versus-buy

**This has to be answered before Phase 1, not retrospectively.** No vendor is named
anywhere in this repo or in the research behind this plan.

The programme is roughly 12-16 weeks of engineering against a measured ~430
automatable talk-hours/year, while repeatedly identifying missing prerequisites —
auth, a ticketing sink, escalation routing, a live order API, consent tooling, i18n —
that Intercom Fin, Zendesk AI, Salesforce Agentforce, Sierra and Decagon ship as
standard.

The genuinely defensible build case is **narrow and specific**, and it is exactly two
things, both of which depend on proprietary data no vendor can replicate:

1. **The compatibility oracle** over `tuning_platforms` — 106 rows that authoritatively
   answer the #1 question class.
2. **The case-based negative-evidence layer** over `calls_cases` — "this was tried and
   it failed" is not something a generic deflection product can learn.

The strongest position is almost certainly **buy the commodity deflection surface,
build the differentiated 20% behind it as tools/MCP**. That is a much stronger plan
than one that never asks the question — and "why did we analyse 38,000 phone calls to
build a web chat widget" is the first thing a reviewer will say.

Related: **voice is scoped out by assertion, not by argument.** 100% of the corpus is
phone calls; 100% of the proposed surfaces are text. There is zero telephony or
speech technology in the repo (`3CX` appears only as a *filename-parsing convention*
in `dgx_pipeline`), and `blueprints/calls.py` is a 158-line file browser, not an
integration. Scoping voice out may well be right — but it needs a quantified
statement of what share of the 430 hours a text surface can actually reach.

---

## 13. Evaluation

Build the golden sets from data that already exists; none of them need annotation:

| Set | Source | Tests |
|---|---|---|
| **Known gaps / hard negatives** | 5,775 `agent_unanswered_questions` | Does the agent say "I don't know" instead of inventing? |
| **Ground truth with verified outcomes** | `case_knowledge_units` + `what_finally_worked` | Does it give the answer that actually worked? |
| **Must-never-say** | 534 calls with `review.incorrect_statements` (562 items) | Regression suite of real wrong answers |
| **Emissions** | 268 flagged calls | Refusal recall; human baseline is 75% |
| **Retrieval quality** | The gap set, run against the deduped index vs raw | **Set the bar before the merge runs, not after** |
| **French** | The Instagram DM corpus (§2) | Style, tone and a French eval partition — measure French turns separately, never assume |

**Hard gates at zero:** PII in any payload or SSE frame; unsourced price or stage
claims surviving the groundedness validator; any Tier-2 data served from cache to a
different session.

**On A/B design — do not promise one.** With four agents carrying 97% of volume, an
agent-level randomisation gives two clusters per arm and treatment is perfectly
confounded with the person. And the commercial endpoints are unattainable at this
volume: quote→sale at ~121 quotes/month would need ~2,300 per arm to detect a 3pp
lift; four weeks yields ~60 per arm — short by ~38×. Even a full year is 3× short.

Use **within-agent, within-intent before/after** on operational measures instead:
handle time on matched intents, assist acceptance rate, first-contact resolution on
the scoped intents, and containment on the customer-facing surface. Report them as
operational improvements with confidence intervals, not as significance tests on
revenue.

---

## 14. Sequence

```
P0  Stop the bleeding ................. days      CRITICAL PATH
    delete the PII collection · sources allowlist · delete POST /seed
    pin QWEN_SIZE + create=False + points_count assertion
    (faq_0_6b has 0 points and is routed to today — this will refuse to boot)
P1  Phase 4a build_kb ................. 3-4 weeks CRITICAL PATH
    gates → 3-stage dedup → ~1,200 LLM merges → kb_units + generation swap
    supersedes_calls[] remapped from WAV filenames to opaque unit_ids
P2  Phase 4b + collections ............ 2 weeks   (tail overlaps P1)
P3  Retrieval layer ................... 1-2 weeks (can START during P1)
    ParallelRetriever · reranker · hybrid · supersession · server-side gate
P4  Tool layer ........................ 3 weeks   (design starts immediately)
P5  Agent runtime ..................... 3-4 weeks
P6  Guardrails + handoff .............. 2 weeks   (overlaps P5)
P7  Channels .......................... 2 weeks/channel

PARALLEL — none of these block P1
[A] Build-vs-buy decision ............. BEFORE P1 commits engineering time
[B] Internal-dashboard inventory ...... 2 days — gates the auth prerequisite
[C] Escalation routing substrate ...... language/skills/dept on `agents`; pick a sink
[D] cron sync_tuning_db.py ............ hours, Mac-only, gates the oracle's freshness
[E] Counsel: recording consent for AI use · Bill 96 / Law 25 · cross-border
    generation · employee consent for the QA data
[F] Change management for the 4-9 people whose work this changes
[G] Content: 5 competitor briefs · ~10-line fee schedule
[H] Mongo coordination with the dashboard owner (4 new collections + an index)
[I] connect_calls — already running, ~1-2 days out. Do NOT sequence around it.
```

**First value at ~week 6, with no agent.** P0 + P2 + P3 give the *existing* RAG path
the real corpus, hybrid search and a reranker. That is a large quality jump and the
clean baseline for whether the agent adds anything.

**The most likely sequencing error is building the agent first.** It would answer
order and vehicle questions from a 9 KB seed file; "the agent hallucinates" would
actually be `similarity_top_k=2` with no reranker; and its entire knowledge advantage
— 61,426 units, the case units that outrank them, the department and purchase-intent
dimensions — is **invisible to the current retrieval stack**, which reads 8,601
legacy transcripts from disk.

---

## 15. What not to build

1. **A sales-objection index** — no rebuttal field exists in the schema (§8.1).
2. **A separate case-unit collection** — destroys the outranking property (§5.1).
3. **A general whole-call index** — 6.5× duplication for zero coverage gain (§5.1).
4. **GraphRAG** — the graph is already a 106-row relational table (§6).
5. **Query-time HyDE / multi-query** — already done at index time (§6).
6. **An agent framework** — own the loop (§10).
7. **Reviving `NEED_HUMAN_INTERVENTION` or the uncertainty detector** — both are
   unusable foundations (§9.6). Build escalation properly.
8. **Semantic search over past conversations** — cross-session *case* memory is the
   version worth having and it already exists.
9. **Repointing the Node MCP server at real customer data** (§3.2).
10. **A commercial dimension in `connect_calls`** — the quote analysis is a phone-key
    join that costs seconds and no GPU (§8.1).
11. **Transacting** — payment links, promos, order modification. ~242 `bot_blockers`
    are payment-link creation, so the demand is real, but it changes the effort by an
    order of magnitude and carries approval implications. v1 informs; it does not
    transact.
12. **Rebuilding the SSE transport** — new chunk types are additive (§10).
13. **A new eval harness** — extend `analytics.py` (§10).
14. **Flipping `enable_hybrid_search` on a live collection** — it deletes and
    recreates (§6).
15. **Per-agent performance dashboards** — 12 identifiable people, an unexplained
    11.7-point language gap, and an unanswered counsel question (§9.7).

---

## 16. Open questions

**Blocking a decision:**

1. **Build or buy?** (§12) — must be answered before P1 commits engineering time.
2. **Who owns the internal dashboard app, and do we extend it or build beside it?**
   (§11) — gates the auth prerequisite of the surface that ships first.
3. **Is voice in or out?** If out, what share of the ~430 talk-hours does a text
   surface actually reach?
4. **Where do escalations go?** No ticketing system, chat queue or callback queue
   exists anywhere in the repo, and `agents` has no routing metadata.
5. **What is the programme's 12-month objective, budget and kill criterion?** Without
   one there is no basis for choosing a first surface and no trigger to stop.

**Blocking a build item:**

6. **What is the live source of retail price?** Nothing measured here is a current
   price list — `orders[].items[].unit_price` is historical transaction data. Without
   one, the agent cannot quote a platform-keyed price at all.
7. **What is the live source of order status?** `tuning_customers` is a point-in-time
   restore showing only **73 open orders across 91,754 accounts**. Any "where is my
   order" tool on that snapshot reports stale state.
8. **Where does hardware fitment data live?** The largest KB-gap cluster (2,633 of
   5,775 unanswered questions) is "what hardware do I need / does this part fit", and
   it has **no source in this Mongo at all**. MySQL? The scraped data? An ERP? Or does
   it need authoring?
9. **Where do release ETAs live?** "8Y.2 RS3 release date", "Mark 8.5 software",
   "E85 release" are among the most common unanswered sales questions and are
   structurally unanswerable from a call corpus. `tuning_platforms.stages[]` has
   `released` / `files_available` / `newest_available_at` but nothing forward-looking.
10. **Does `owned_stage` mean "purchased and entitled" or "eligible"?** The entire
    76,515-vehicle upsell gap hinges on it. 6,722 vehicles sit at-or-above top, which
    suggests a real entitlement, but it was not confirmed against the MySQL schema.
11. **Should `internal_process` units (2,564) be retrievable?** They are correct but
    not customer-facing, and `internal_process` is a *kind*, not a visibility flag.
12. **What is the acceptable false-answer rate on compatibility?** Humans are graded
    "poor" on 204 compatibility calls and self-contradict 534 times. "Better than the
    human baseline" and "never wrong" imply different architectures.
13. **Retention schedule, and who signs it off?** (§11)
