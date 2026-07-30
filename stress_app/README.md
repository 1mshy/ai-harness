# Stress Rig

A Tauri desktop app for stress-testing an OpenAI-compatible vLLM server. Built
against `http://10.150.0.30:1234/v1` serving `nvidia/Gemma-4-31B-IT-NVFP4`
(vLLM 0.23.1, 32768-token context).

```bash
pnpm install
pnpm app          # dev
pnpm app:build    # bundle a .app
```

## Why the load generator is in Rust

Chromium caps concurrent connections per host at roughly six on HTTP/1.1. A
`fetch`-based tester would silently flatline at concurrency 6 and report numbers
that look fine and mean nothing. All request dispatch, SSE parsing, and timing
lives in `src-tauri/src/runner.rs` on `reqwest` + `tokio`, which has no such cap.
The webview only renders.

Token deltas are coalesced in Rust and flushed to the UI on a fixed cadence
(default 60 ms). One IPC message per token would drown the webview long before
the server broke a sweat.

## Pages

| Page | What it shows |
|---|---|
| **Console** | One card per in-flight request, filling with tokens live. Per-card TTFT, token count, decode rate. Event log alongside. 1–4 column density. |
| **Metrics** | Throughput, TTFT p50/p99, server pressure, KV-cache — sampled once a second. Per-difficulty latency bars. vLLM's own `/metrics` counters. |
| **Corpus** | All 350 prompts, searchable and filterable, individually armable. |
| **Results** | Every completed request with timings and full response body. Export JSON or CSV. |

## Settings (⌘,)

Five tabs, all persisted to `localStorage`:

- **Connection** — base URL, model (populated by probing `/v1/models`), API key,
  live connection test reporting engine version and `/metrics` availability.
- **Load shape** — concurrency 1–256, stop condition (fixed count / fixed
  duration / until stopped), ramp-up, think time, request timeout, retries,
  abort-on-first-failure.
- **Sampling** — max tokens, temperature, top-p, top-k, min-p, repetition /
  presence / frequency penalty, seed, `ignore_eos`, streaming toggle, system
  prompt.
- **Prompt mix** — per-difficulty weighting, sequential vs random selection,
  prefix-cache busting.
- **Advanced** — metrics poll interval, stream flush cadence, restore defaults.

Two knobs deserve a note because they change what the numbers *mean*:

- **`ignore_eos`** pins every completion to `max_tokens`, isolating decode
  throughput from the model's natural stopping behaviour. Good for comparing
  runs against each other; not representative of real traffic.
- **Prefix-cache busting** prepends a unique nonce per request. vLLM caches
  shared prompt prefixes, so re-running the same corpus without this reports
  throughput well above what cold traffic would see.

## The corpus

350 prompts under `src/data/prompts/`, bundled at build time via
`import.meta.glob`:

- **100 easy** — short, fast, minimal reasoning. Measures raw throughput and TTFT.
- **150 medium** — multi-step reasoning and structured generation.
- **100 hard** — long chains, heavy output, several carrying 800–2000 words of
  inline context to stress prefill and hold KV-cache blocks.

Drop another `{ "difficulty": …, "prompts": [ … ] }` file in that directory and
it is picked up on the next build; no loader changes needed.

## Measurement notes

- **TTFT** is measured at the first non-empty content delta. With streaming off
  there is no first-token signal, so TTFT degenerates to total latency — the
  Sampling tab says so.
- **Decode rate** is `completion_tokens / (total − TTFT)`, i.e. excludes prefill.
  The aggregate figure on the Metrics page is total tokens over wall-clock and
  will be lower.
- **Percentiles** on the time-series charts are computed over a trailing
  60-request window so degradation shows up as it happens rather than being
  smoothed into a lifetime average. The Results page reports whole-run
  percentiles.
- If the server omits `usage`, completion tokens are estimated at 4 chars/token
  and the figure is approximate.

## Design

Hallmark, atmospheric genre, Workbench macrostructure, Terminal theme. Tokens in
`tokens.css`; nothing in `src/styles/app.css` inlines a colour or font value.

Chart marks use their own lightness steps (`--chart-1..3`), separate from the UI
accents, validated against the dark chart surface with the dataviz palette
validator — lightness band, chroma floor, CVD separation, and contrast all pass
on adjacent and all-pairs. The UI accent steps sit above the dark band and are
deliberately not used as chart marks.

## Tests

```bash
cd src-tauri && cargo test        # SSE frame decoding, Prometheus parsing, prompt selection
pnpm build                        # typecheck + bundle
```
