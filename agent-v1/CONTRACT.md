# Internal contract — schemas and module interfaces

Everything below is already implemented and smoke-tested against live systems
unless marked TODO. Code against these signatures; do not re-derive them.

## Environment (measured 2026-08-05)

| Thing | Where | Notes |
|---|---|---|
| Mongo | `10.150.0.30:27017` db `transcribing` | 24 collections. Shared with an internal staff LLM app — do not write to anything you did not create. |
| Qdrant | `http://localhost:6333` (Docker, local) | 15 pre-existing collections. Only `comprehensive/products/products_tuning/tuning/company_info` have named dense+sparse vectors; the rest are legacy single unnamed vector. |
| Reasoner | `http://10.150.0.30:1234/v1`, `nvidia/Gemma-4-31B-IT-NVFP4` | **Native tool calling works** (`--enable-auto-tool-choice --tool-call-parser gemma4`). `response_format=json_schema` still returns empty content — use `LLMClient.json_call`. ~0.4–1.6 s/call. |
| Embeddings | local MPS (default) or `http://10.150.0.30:1235/v1` model `qwen3-embed` | Qwen3-Embedding-0.6B, 1024-dim. Cross-backend cosine 1.0000. Local 256/s, remote 121/s. |
| Reranker | `BAAI/bge-reranker-v2-m3` local | downloaded and verified. |

## Live corpus counts (2026-08-05, re-measured — several differ from AGENT_PLAN.md)

```
calls_analysis      38,563   knowledge_units 61,501 over 29,574 docs
  training_safe true 38,362 / false 201
  useful_content     34,944
  review present     30,623   (79.4% — every review-derived rate understates)
  emissions complied_improperly 32 · refused_correctly 201
  review.incorrect_statements.0 → 535   (as a bool → 0; the array trap is real)
  safety_issue 1,534 · threat_flags.0 26 · agent_unanswered.0 5,261
  language fr 3,287 · sales block present 21,057
calls_cases          6,771   (AGENT_PLAN.md measured 914 — connect_calls drained)
  case_knowledge_units 3,716 over 3,214 cases
  training_safe true 6,664 / false 107 · unscreened_members>0 → 0
  attempts: worked 7,480 · failed 2,516 · untested 2,227 · unknown 1,498 · partial 866
tuning_platforms       106 · tuning_customers 91,754 · crm_contacts 22,093
```

## `transcribing.kb_units` — canonical unit identity (Phase 4a owns this)

```jsonc
{
  "unit_id":      "u_<16 hex>",   // stable; derived from the FIRST source unit's
                                  // (source_id, ordinal). NOT a content hash —
                                  // cluster membership moves as the analyzer adds
                                  // documents, so content-hash ids are unstable and
                                  // idempotent upsert becomes impossible.
  "gen":          "<12 hex>",     // build generation
  "kb_version": 1, "merge_version": 1,

  "kind":     "product_info|procedure|troubleshooting|compatibility|policy|pricing|internal_process|faq",
  "title":    "str",
  "question": "str",
  "answer":   "str",
  "conditions": "str",
  "vehicles_applicable":  ["str"],
  "products_applicable":  ["str"],
  "hypothetical_questions": ["str"],   // ~186k of these exist corpus-wide; they are a
                                       // HyDE corpus written offline WITH the answer in
                                       // hand. Embed them; never generate them at query time.

  "evidence":   "single_call|multi_call_case",
  "outranks_call_units": bool,
  "superseded_unit_ids": ["u_..."],    // REMAPPED from case_knowledge_units[].supersedes_calls[],
                                       // which is a list of 3CX WAV filenames embedding the
                                       // caller's phone number. The supersession relation is
                                       // PII until this remap runs.
  "case_id":    "str|null",

  "confidence": "high|medium|low",
  "occurrences": int,                  // cluster size; a BOUNDED BOOST only, never a sort key
  "source_ids": ["<mongo _id str>"],   // enables targeted revocation + Law 25 deletion
  "cluster_id": "str|null",
  "merge_action": "singleton|merged|split_child|verbatim",

  // gates + derived labels, all computed in kb/gates.py
  "training_safe": bool,               // non-bypassable
  "time_sensitive": bool,
  "emissions_risk": bool,
  "safety_gated":  bool,
  "dealer_pricing": bool,
  "internal_only": bool,               // kind == internal_process
  "contains_price": bool,

  "department": "str|null", "language": "en|fr", "technical_category": "str|null",
  "created_at": "iso", "updated_at": "iso", "status": "active|revoked"
}
```

## Qdrant point layout

Two points per unit — **no chunking**. Median unit is 336 chars, p95 591; a
1024-token splitter never fires.

* `point_role: "answer"` — text is `title / question / answer / conditions`
* `point_role: "query"`  — text is `question + hypothetical_questions + alias colloquial forms`

Dedupe on `unit_id` at retrieval time. Point id = uuid5 of `(unit_id, role)`.

**Payload is an allowlist, not a projection of the source doc.** Adding a field
means editing `index/payload.py`. The collection this replaces leaked
`file_name` (caller's phone), `agent_name` and `caller_area_code`, and did it
twice — once as payload keys and once again serialised inside `_node_content`.

## Module interfaces (implemented)

```python
from agentv1 import config                       # pinned; QWEN_SIZE has no default
from agentv1.clients.mongo import source_db, kb_db, get_state, put_state, ensure_kb_indexes
from agentv1.clients.qdrant import (
    get_client, open_collection, create_hybrid_collection, ensure_payload_indexes,
    swap_alias, upsert, points_count, assert_routed_collections_populated,
    hybrid_search, dense_search, has_named_vectors, delete_by_unit_ids,
    Hit, CollectionMissing, CollectionEmpty)      # Hit = {id, score, payload}
from agentv1.clients.embeddings import get_embedder
    # .embed_documents(list[str]) -> (n,1024)  — NO instruct prefix
    # .embed_query(str) -> (1024,)             — instruct prefix applied
from agentv1.clients.sparse import Bm25Encoder, SparseVector, fuse_rrf, term_id
    # .fit(corpus) / .encode_document(str) / .encode_query(str) / .save(p) / .load(p)
from agentv1.clients.llm import get_llm, ChatResult, ToolCall, extract_json, \
     LLMTransientError, LLMPermanentError
    # .chat(messages, tools=..., tool_choice=...) -> ChatResult(content, tool_calls, finish_reason)
    # .json_call(messages, validator=...)  — free-form JSON + repair retry
    # .map_concurrent(items, fn)           — bounded concurrency, failures -> None
from agentv1.text.normalize import (
    lexical_terms, marker_tokens, expand_aliases, normalize_text,
    strip_pii_markers, wav_filename_phone, stage_tokens, uniconnect_tokens)
```

`open_collection(name, create=False)` is the default and raises
`CollectionMissing` rather than creating. Only the builders create.

## Non-negotiables

1. **`training_safe` floor is AND-ed server-side** inside the retrieval layer and
   is never taken from a request body. Any filter a caller supplies is
   intersected with it, never substituted for it.
2. **No PII in any payload or SSE frame.** Allowlist only.
3. **Tier-2 tools take no customer identifier in their signature.** The executor
   injects `customer_id` from server-side session state, which makes "look up
   someone else's order because the user asked nicely" inexpressible rather
   than merely discouraged.
4. **Never quote a price or a stage availability without a tool result in the
   same turn.** `guardrails/grounding.py` enforces this after generation.
5. **Occurrence is a bounded boost.** 28,423 singletons are the reason the
   corpus exists; if occurrence becomes a sort key the index degrades into an
   FAQ of things everyone already knows.
6. Read-only Mongo/MySQL for tool credentials. Writes only to collections this
   project owns (`kb_*`, `agent_*`).
