# LM Studio Stress Tester

Load / stress testing for an OpenAI-compatible **LM Studio** endpoint (built for an
LM Studio server on a DGX Spark, but works against any `/v1` endpoint).

> **Looking for capability tests, not throughput?** See
> [`sales_eval/`](sales_eval/README.md) — a sales-agent eval suite (tool calling,
> grounded Q&A, sentiment, long-call summarization, lead qualification, objection
> handling) run with `python run_evals.py`.

It measures the things that matter for a local inference box:

- **TTFT** (time to first token) — streaming
- **Latency** — mean / p50 / p90 / p99
- **Decode speed** — output tokens/sec per request
- **Aggregate throughput** — total output tokens/sec across a run (the headline)
- **Success / failure** rates with per-error breakdown

## Setup

```bash
pip install -r requirements.txt
```

Default endpoint is `http://10.150.0.30:1234/v1`. Override with `--url`.

> **mDNS note (macOS):** Python can't resolve `.local` names that `curl` can.
> The tool detects this and auto-resolves `10.150.0.30` to its IP via the system
> resolver — no action needed. If it ever can't, pass the IP directly:
> `--url http://10.150.0.30:1234/v1`.

## Interactive mode

Run with no arguments for a menu:

```bash
python stress_test.py
```

```
1) Single request (smoke test)
2) Sequential run   (N requests, one at a time)
3) Concurrent batch (N requests @ concurrency C)
4) Ramp test        (throughput vs concurrency)   <- headline
5) Soak test        (sustained load for D seconds)
6) Settings         (url, model, max_tokens, temperature, prompt, timeouts)
7) Pick model
8) Warm up model
```

## Scripting mode (subcommands)

```bash
python stress_test.py models                       # list models
python stress_test.py single                       # one request, prints the reply
python stress_test.py sequential -n 20             # 20 requests, one at a time
python stress_test.py batch -n 32 -c 8             # 32 requests at concurrency 8
python stress_test.py ramp --levels 1,2,4,8,16     # throughput vs concurrency
python stress_test.py soak -d 60 -c 8              # 60s sustained at concurrency 8
```

Global flags (before the subcommand):

| flag | default | meaning |
|------|---------|---------|
| `--url` | `http://10.150.0.30:1234/v1` | endpoint |
| `--model` | first non-embedding model | model id |
| `--prompt` | rotating varied prompts | fix a single prompt for all requests |
| `--max-tokens` | 256 | output cap per request |
| `--temperature` | 0.7 | sampling temperature |
| `--timeout` | 120 | per-request timeout (s) for measured runs |
| `--warmup-timeout` | 300 | warm-up timeout (s); first request loads the model |
| `--no-warmup` | off | skip the warm-up request |
| `--json FILE` | — | write raw results + summary to JSON |

## The point of `ramp`

This is the most useful experiment for a local box. LM Studio may **serialize**
concurrent requests (a queue) or do **continuous batching** — you can't tell
without measuring. `ramp` runs batches at increasing concurrency and reports
aggregate tokens/sec at each level:

- throughput **scales up** with concurrency → the server is batching
- throughput stays **flat** → the server is serializing (your "batch" is a queue)

The tool prints which one it sees at the end of the run.

## Notes

- Requests **stream** so TTFT and decode speed are measured separately.
- Reasoning models (e.g. `nemotron`) emit `reasoning_content`; those tokens are
  counted as output and shown in the response.
- Client retries are **off** (`max_retries=0`) and the connection pool is sized to
  the run's concurrency, so the numbers reflect the server, not the client.
- A **warm-up** request runs before each measured run (excluded from stats) so
  model-load time doesn't pollute the first measurement. Disable with `--no-warmup`.
- The `gpt-oss-120b` model currently returns HTTP 500 on this server (a load-side
  issue); the tool reports it as a failure instead of crashing. `nemotron-3-super`
  works.
