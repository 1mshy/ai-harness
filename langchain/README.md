# Unitronic sales rep (LangChain)

A tool-calling sales agent grounded in the local Unitronic MariaDB
(`unidb` docker container, database `CONTENT_MGMT_SYS`). It resolves the
customer's vehicle, quotes software stages with regional pricing, and specs
hardware packages — all from live catalog data, never from model memory.

Talks to the OpenAI-compatible model box (vLLM / LM Studio); the model id is
discovered from `/v1/models` at startup because the lineup changes often.

## Run it

```bash
uv run main.py                      # chat REPL
uv run main.py --verbose            # also print tool results
uv run main.py "I have a 2018 Golf R, what's a full Stage 2 cost?"
```

## Configuration (env vars, all optional)

| var | default |
|---|---|
| `MODEL_BASE_URL` | `http://10.150.0.30:1234/v1` |
| `MODEL_ID` | discovered from `/v1/models` |
| `UNIDB_HOST` / `UNIDB_PORT` | `127.0.0.1` / `3306` |
| `UNIDB_USER` / `UNIDB_PASSWORD` | `root` / empty |
| `UNIDB_NAME` | `CONTENT_MGMT_SYS` |

The default base URL is the IP, not `dgx.local` — Python's resolver can't do
mDNS on this Mac (curl can; httpx hangs ~35s then fails).

## Layout

```
agent.py   persona system prompt + agent construction, model discovery
tools.py   six catalog tools (see below)
db.py      read-only PyMySQL access + Editor.js rich-text flattening
main.py    REPL / one-shot CLI, streams tool activity as it happens
```

## The tools

| tool | grounds |
|---|---|
| `find_vehicle` | customer's car → supported engine variants (the entry point) |
| `list_supported_vehicles` | full make/model coverage with year ranges |
| `software_for_vehicle` | stages + per-region MSRP + required/recommended hardware |
| `hardware_for_vehicle` | hardware that fits a model/engine, filterable by year & category |
| `search_hardware` | keyword search over live products |
| `hardware_details` | one product: pricing (incl. active sales), stock, fitment, add-ons |

Guardrails live in two places: the system prompt (no discounts, no dealer
pricing, no invented specs) and the tools themselves — the model is never
handed raw SQL, only parameterized SELECTs, and wholesale price columns are
never selected.

## Notes on the data

- Stage prices come from `Price` filtered to `continent LIKE '%United States
  (USD)%'` by default; pass a region (e.g. `Canada (CAD)`) for others.
  `suggested_price` is customer MSRP; dealer/distributor tiers exist in the
  table but are deliberately not exposed.
- Hardware US pricing is `HardwareProducts.usdMsrp`, with scheduled sales
  honored when `NOW()` falls in the schedule window.
- Vehicle fitment: `Stage → EngineVariant ←(M2M)→ Model ←(M2M)→ Manufacturer`
  for software; `HardwareFitment` (product × model × engine variant × year
  range) for hardware.
- Rich text fields are Editor.js block JSON; `db.rich_text` flattens them.

## Known model quirks (Gemma-4-31B on vLLM)

- Leaks `<|channel>thought` markers into replies — `main.clean` strips them.
- Occasionally fumbles multi-item price arithmetic by a few dollars; item
  prices themselves are always tool-grounded. A calculator tool would close
  this gap if it matters.
