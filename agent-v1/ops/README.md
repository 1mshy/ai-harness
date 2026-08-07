# ops/ — P0 remediation

Tooling for AGENT_PLAN.md §3 (P0) and §14 `P0 Stop the bleeding`.

Everything here runs against live Qdrant at `http://localhost:6333`. Nothing
here writes to Mongo.

```
ops/qdrant_pii_audit.py   read-only. Enumerates every live collection and scans
                          sampled payloads for PII. Exit 1 on any CRITICAL.
ops/p0_remediate.py       the destructive half. Snapshot / quarantine / restore.
                          Dry run by default; --yes required to mutate anything.
ops/snapshots/            downloaded snapshots + manifests. git-ignored.
```

Run both self-checks before trusting either tool. Neither needs Qdrant:

```
.venv/bin/python ops/qdrant_pii_audit.py --selftest     # 35 detector checks
.venv/bin/python ops/p0_remediate.py     selftest       #  6 verification checks
```

---

## Audit

```
.venv/bin/python ops/qdrant_pii_audit.py                # every live collection, 500 pts each
.venv/bin/python ops/qdrant_pii_audit.py --json         # machine-readable
.venv/bin/python ops/qdrant_pii_audit.py -c <coll> -n 0 # one collection, all pts
```

Two loci are scanned per point: the payload keys, and the `_node_content`
string where LlamaIndex serialised the same metadata dict a second time.
§2 of the plan already notes that the writer stopped emitting `file_name`;
the live points predate that and still carry it in **both** places, which is
why the report prints a per-locus breakdown. A fix applied to the writer moves
only the `payload` column.

Categories: `wav_filename_3cx`, `wav_filename_other`, `phone_10digit`,
`phone_formatted`, `email`, `payment_card`, `instagram_handle`,
`instagram_thread_id`, `vin`, `vin_suspected`, `order_number`, `postal_code`,
`person_name_field`, `geo_identifier`, `corporate_contact`.

Value detectors run against the masked text (UUIDs and 12+ character hex/digit
runs blanked); `payment_card` and `email` run against the **original** text,
because masking would erase a 16-digit PAN before it could be tested and would
eat an address whose local part is hex-shaped. Field-name detectors
(`person_name_field`, `geo_identifier`) recurse to depth 4, so a payload that
nests the caller under `{"caller": {...}}` is not invisible to them.

`corporate_contact` (INFO) is the published support line `866.341.2447` and
`@getunitronic.com` addresses. It fires on 100% of `unitronic_tuning_*` and is
not a leak — it is scraped from the company's own website. Classifying it as
CRITICAL would bury the four real customer numbers in the same corpus.

**Result 2026-08-06, 500 points sampled per collection, exit code 1:**

| collection | points | verdict | worst categories |
|---|---:|---|---|
| unitronic_call_transcriptions_0_6b | 24,760 | **CRITICAL** | wav_filename_3cx 99.4% · phone_10digit 98.2% · person_name_field 100% · email · vin |
| unitronic_customer_service_classification_0_6b | 18,333 | **CRITICAL** | instagram_handle 94.6% · vin · phone_formatted |
| unitronic_customer_service_classification_8b | 8,192 | **CRITICAL** | instagram_handle 92.8% · vin |
| unitronic_customer_service_training_0_6b | 19,872 | **CRITICAL** | instagram_handle 93.6% · vin 2.0% · phone_formatted |
| unitronic_customer_service_training_8b | 19,881 | **CRITICAL** | instagram_handle 94.0% · vin 2.4% · phone_formatted |
| unitronic_company_info_0_6b | 144 | CLEAN | corporate_contact only |
| unitronic_comprehensive_0_6b | 28,472 | CLEAN | corporate_contact only |
| unitronic_comprehensive_8b | 53,964 | CLEAN | corporate_contact only |
| unitronic_products_0_6b | 11,781 | CLEAN | — |
| unitronic_products_8b | 11,321 | CLEAN | — |
| unitronic_products_tuning_0_6b | 13,114 | CLEAN | corporate_contact only |
| unitronic_products_tuning_8b | 12,566 | CLEAN | corporate_contact only |
| unitronic_tuning_0_6b | 1,333 | CLEAN | corporate_contact only |
| unitronic_tuning_8b | 1,245 | CLEAN | corporate_contact only |
| unitronic_faq_0_6b | 0 | EMPTY | routed to today, holds nothing (§3.3) |

The table above is the 15 pre-existing collections. The `kb_units` builders have
since added their own (`*__g<hash>`), and the audit picks them up automatically
— it enumerates whatever is live rather than working from a fixed list. Point
counts move while a build is running, so re-run rather than trusting a
transcript.

**The new collections are not clean either.** `unitronic_call_residual_0_6b__*`
and `unitronic_case_narratives_0_6b__*` are WARN: real customer order numbers
survive into the `answer` and `text` payload fields
(`"Verify order number 81xxx in the system"`, `"CUSTOMER placed an order (Order
#85xxx)"`). CONTRACT.md non-negotiable 2 is "no PII in any payload, allowlist
only" — the field allowlist is being honoured, but the *free text inside*
allowlisted fields is not scrubbed, so an order number reaches a client through
`answer` even though no `order_id` key exists. That is a `kb/` and
`index/payload.py` fix, not an `ops/` one; this tool can only report it.

### Cardholder data is in the corpus — PCI-DSS scope, not just Law 25

A Luhn-valid Visa PAN with its expiry date, read aloud by a customer and
transcribed verbatim, is live in `solution` and `full_transcript_ref` on
`unitronic_call_transcriptions_0_6b`, and a second PAN sits in a DM on
`unitronic_customer_service_classification_0_6b`. Both collections were already
CRITICAL for other reasons, so the verdicts do not move — but the remediation
urgency and the legal posture do. Storing a PAN in an unencrypted vector
payload is a PCI-DSS problem regardless of what the retrieval layer filters,
and it is an argument for quarantining rather than rebuilding.

The detector is deliberately narrow: 15–16 digits, Luhn-valid, **and** a real
issuer prefix. A bare "13–19 digits + Luhn" rule fires 281 times per 500 points
here, almost all of it the 13/14-digit 3CX timestamp inside a WAV filename
passing Luhn by chance. The narrow rule finds both real cards with **zero**
false positives across 4,000 sampled product/tuning/comprehensive points.

`postal_code` was added on the same evidence: full 6-character Canadian postal
codes (a city block, not a region) are extracted into the `technical_terms`
payload field on the Instagram collections at up to 2.4% of sampled points,
including French-language Quebec threads. It too was screened to zero false
positives on the product corpus — the first draft of the ZIP+4 half of that
pattern matched `20015-2016` inside product URL slugs and had to be tightened.

**Five collections are CRITICAL, not one.** §3.1 names
`unitronic_call_transcriptions_0_6b` and flags
`unitronic_customer_service_training_0_6b`; the audit adds the `_8b` twins of
both Instagram collections and `customer_service_classification_0_6b`. It also
finds check-digit-valid VINs inside the Instagram DM text — an identifier §3.1
predicted would be there and which no phone regex would have found.

Verified false positives that a naive pattern set produces here, and which this
one suppresses: hyphenated UUID fragments in `doc_id` / `ref_doc_id` /
`_node_content` relationship ids read as dashed phone numbers (~8 phantom
"phones" per collection), and 17-digit Instagram thread ids read as VINs (~270
phantom VINs per 300 points). VIN candidates must pass the ISO 3779 check
digit; phone candidates must be NANP-shaped and survive UUID masking.

Report examples are redacted. The audit output is safe to paste into a ticket.

---

## Remediation

```
.venv/bin/python ops/p0_remediate.py list
.venv/bin/python ops/p0_remediate.py snapshot   <collection>              # dry run
.venv/bin/python ops/p0_remediate.py snapshot   <collection>   --yes
.venv/bin/python ops/p0_remediate.py verify     ops/snapshots/<file>
.venv/bin/python ops/p0_remediate.py quarantine <collection>   --yes
.venv/bin/python ops/p0_remediate.py restore    ops/snapshots/<file> --yes
```

`quarantine` = snapshot → download → verify (exists, size matches the server's
own figure, SHA-256 matches the server's checksum, file is a readable tar) →
re-verify from disk → delete. Any failure aborts **before** the delete. Add
`--prune-server-snapshot` to also drop Qdrant's internal copy once the local
file has verified; without it the server keeps a second copy of the payloads
you just quarantined.

`--yes` must come **after** the subcommand. `p0_remediate.py --yes quarantine X`
is rejected with `unrecognized arguments` rather than silently dry-running.

Snapshots are ~71 MiB even for a 0-point collection, so budget disk before
quarantining the 24,760-point one.

### Recommended P0 order

1. `qdrant_pii_audit.py --json > audit-before.json`
2. `quarantine unitronic_call_transcriptions_0_6b --yes --prune-server-snapshot`
   — §3.1 says delete it outright; it is superseded by the §5 `kb_units` build
   and is not in `config.LEGACY_COLLECTIONS`.
3. The four Instagram collections are **not** a straight delete.
   `unitronic_customer_service_training_0_6b` is guaranteed-included on every
   routed query and, per §2, is the only French customer-facing register in the
   corpus. Quarantining it removes a real asset. The correct move is a rebuild
   that drops `thread_path` from both loci — which is P2 work, not P0. Until
   then the P0-scoped mitigation is the `sources[]` allowlist (below), which
   stops the handle reaching a client even while the point still carries it.
4. Re-run the audit and diff.

### What was actually executed here

Only non-destructive paths. `snapshot --yes` and `--prune-server-snapshot` were
run against `unitronic_faq_0_6b` (0 points). `quarantine` and `restore` were
exercised in dry-run only. **No collection was deleted and no snapshot has been
restored into a live Qdrant, so the `restore` upload path has not been executed
against this server.** Its inputs were verified (tar integrity, checksum,
manifest, overwrite warning, create-if-missing branch); the upload itself is
unproven until an operator runs it.

---

## P0 items that are NOT actionable in this repository

The plan's P0 spans two codebases. This one — `agent-v1` — contains
`agentv1/`, `ops/` and `tests/`. The serving stack the plan critiques
(`chat.py`, `rag_core.py`, `config/qdrant_config.py`, `MCP/server/server.ts`)
is a **different repository and is not checked out here.** Verified by
inspection of this working tree: none of those paths exist.

These are carried forward as external actions so they are not silently lost.
File:line references are as written in AGENT_PLAN.md.

| # | Plan ref | Action | File:line (other repo) | Why it cannot be done here |
|---|---|---|---|---|
| E1 | §3.2 | Delete the unauthenticated `POST /seed` route. It runs `deleteMany({})` before inserting 5 fake records; one unauthenticated request destroys the collection if that server is ever repointed at real customer data. | `MCP/server/server.ts:1184`, `deleteMany` at `:1198` | `MCP/` does not exist in this repo |
| E2 | §3.2 | Bind the MCP server to localhost. | `MCP/server/server.ts` | same |
| E3 | §3.2 | Replace the per-phone-lookup full-collection scan. | `MCP/server/server.ts:186` (`customersCollection.find({}).toArray()`) | same |
| E4 | §3.2 | Keep only `search_unitronic_parts` and `lookup_product_by_sku`; the other 8 tools resolve against a 9 KB `data/customers.json` of John Doe / +1-555 records. | `MCP/` + `data/customers.json` | same |
| E5 | §3.1 | Replace the `sources[].metadata` pass-through with an explicit allowlist. This is the mitigation that stops a leaked `thread_path` reaching an API client. | `rag_core.py:924` → `chat.py:163`, plus the SSE `source` event | neither file exists here |
| E6 | §3.3 | `similarity_top_k: 2` → raise, and add a reranker. | `config/qdrant_config.py:83` | not in this repo |
| E7 | §3.3 | `get_vector_store(create=False)` — stop creating collections on a read path. | `config/qdrant_config.py` | not in this repo |
| E8 | §3.3 | Pin `QWEN_SIZE` in that repo's `.env`; unset today, so prod silently runs the 1024-dim dev default. | that repo's `.env` | not in this repo |
| E9 | §3.3 | Add the startup assertion that every routed collection has `points_count > 0`. | that repo's boot path | not in this repo |
| E10 | §3.1 | `analyze_calls --backfill-scrub` / `connect_calls --rescrub` + `name_gazetteer`, closing the measured 1.79% third-party-name leak. | the pipeline repo | pipeline is not in this repo |

E6–E9 have **already been implemented in this repo's own stack** and are listed
only because the legacy serving path still runs the old code:

* `agentv1/config.py:52` — `QWEN_SIZE` explicit, `EMBED_DIM` derived, no silent default.
* `agentv1/config.py:85-86` — `RETRIEVE_TOP_K=30`, `FINAL_TOP_K=8`.
* `agentv1/config.py:78-80` — reranker pinned to `BAAI/bge-reranker-v2-m3`.
* `agentv1/clients/qdrant.py:68` — `open_collection(..., create=False)` is the default and raises `CollectionMissing`.
* `agentv1/clients/qdrant.py:179` — `assert_routed_collections_populated`, which will refuse to boot while `unitronic_faq_0_6b` holds 0 points. That is the intended behaviour.

E5 has a partial counterpart here: `CONTRACT.md` fixes the Qdrant payload to an
allowlist edited in `index/payload.py`. That governs new points only; it does
not change what the legacy serving stack projects out of the legacy
collections.

E1–E4 and E10 have **no counterpart in this repository at all** and need an
owner elsewhere.
