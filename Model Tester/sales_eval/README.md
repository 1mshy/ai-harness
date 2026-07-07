# Sales-agent capability evals

Companion to `stress_test.py`. Where the stress tester measures **how fast** the
LM Studio box serves tokens, this measures **whether the model can actually do
the job of a sales agent**: call CRM tools, answer from a price sheet, read a
customer's sentiment, summarize a long phone call, qualify a lead, and handle an
objection within guardrails.

It talks to the same endpoint (`http://dgx.local:1234/v1` by default, with the
same `.local` mDNS auto-resolution as the stress tester).

## Run it

```bash
python run_evals.py list                 # show registered scenarios
python run_evals.py selftest             # offline grader gate — NO model needed
python run_evals.py run                   # run every scenario against the model
python run_evals.py run --only sentiment_intent call_summarization
python run_evals.py run --category tool_use
python run_evals.py run --model deepseek-v4-flash --json results.json
python run_evals.py run --url http://10.150.0.154:1234/v1
```

Run `run` **without `--model`** on a terminal and it fetches `/v1/models` and lets
you pick interactively:

```
Available models (from /v1/models):
  0) deepseek-v4-flash  [default]
  1) deepseek-v4-pro
  2) text-embedding-nomic-v1.5  [embedding]

Pick a model number (enter for default 'deepseek-v4-flash'):
```

Pass `--model <id>` to skip the prompt (scriptable). When stdin isn't a TTY
(CI / background), it auto-selects the first non-embedding model so unattended
runs don't block.

> **Reasoning models need a bigger token budget.** Models like `qwen3.6` or
> `deepseek-r1`-class spend most of their output budget on `reasoning_content`
> *before* emitting the answer — at the default caps the answer can come back
> empty (`finish_reason=length`). The suite reports this clearly ("reasoning ate
> the budget; raise --max-tokens"); give them room with the global override:
> `python run_evals.py --max-tokens 4000 run`. (Observed: `qwen/qwen3.6-35b-a3b`
> needed ~800 reasoning tokens before answering even a simple grounded question.)
>
> If the **server** itself caps output (LM Studio's per-model "max output tokens"),
> `--max-tokens` can't override it — the suite detects this and says so
> ("server returned only ~N of M requested tokens; raise the model's max output
> in LM Studio"). On the DGX box this caps `qwen3.6` at ~2000 completion tokens,
> which its reasoning alone exhausts on `call_summarization` (the long-transcript
> scenario) — so that one can't pass on qwen until the server limit is raised. The
> other five pass at `--max-tokens 4000`.

Exit code is non-zero if any scenario fails, so `run` and `selftest` both work in
CI. Global flags (before the subcommand): `--url`, `--model`, `--timeout`,
`--verbose` (prints tool calls as they happen).

## The scenarios

| scenario | category | what it exercises |
|---|---|---|
| `tool_calling_crm` | tool_use | Multi-step OpenAI function calling: extract an email + order id from a customer message, call `get_order_status`, answer grounded on the returned data. |
| `grounded_qa_pricing` | grounding | RAG-style: answer a pricing/SSO question strictly from an embedded price sheet, no hallucination, correct tier + arithmetic. |
| `sentiment_intent` | sentiment | Classify a frustrated customer message → JSON `{sentiment, churn_risk, primary_intent}`. |
| `call_summarization` | summarization | Long-context: summarize a ~45-turn discovery call, recovering planted "needle" facts (a SOC 2 commitment, a Q3 budget freeze, a competitor, a follow-up date, a stakeholder). |
| `lead_qualification_bant` | extraction | Extract BANT (Budget / Authority / Need / Timeline) from a discovery call into CRM JSON. |
| `objection_handling` | reasoning | Draft a value-based reply to a price objection **within guardrails** (no unauthorized discount, propose a next step). |

## How grading works

Each scenario ships a **deterministic grader** — no LLM judge (a local model
judging itself is unreliable). Graders are **lenient by design** because
LM-Studio-class models emit messy output:

- `extract_json` strips ` ```json ` fences and pulls JSON out of surrounding prose.
- `label_match` / `contains_any` do normalized, synonym-tolerant matching — never
  exact string equality.
- `has_number` tolerates `$` and thousands-commas.
- Tool scenarios add an **informational** check ("model emitted a tool call") so a
  model that simply doesn't support the tools API is *diagnostically distinct*
  from one that called the wrong tool with the wrong arguments.

A scenario passes when all its **required** checks pass; informational checks are
reported for diagnosis but don't decide pass/fail.

## Why there's a `selftest`

The hard part of an eval suite is grader **calibration** — a grader that can't
fail is worthless, and one that's too strict fails good output. Every scenario
ships `sample_good` (canned correct output) and `sample_bad` (a believable
mistake). `python run_evals.py selftest` replays both through the **real runner**
(with a `FakeClient`) and asserts each grader **passes good and fails bad** — a
full calibration check that needs no model and runs in milliseconds. Run it after
editing any grader.

## Calibration status

The graders are calibrated against a **live local model**, not just the offline
mocks — an eval that only ever sees canned output is almost always miscalibrated.
On the DGX box (`deepseek-v4-flash`, a reasoning model), all six scenarios pass
(`6/6`). Treat that as *calibrated and currently green*, not deterministically
stable: reasoning models decide **nondeterministically** whether to dump
chain-of-thought into the response (even at temperature 0), which observably
flipped a JSON scenario pass↔fail across runs. That variance is *mitigated* (see
below: generous `max_tokens` + `extract_json` recovering JSON from prose), not
eliminated — a heavy-reasoning run can still surprise you. Re-run before trusting
a number.

Several graders were tuned during the live runs — exactly what calibration is for: 

- `call_summarization` — `max_tokens` was raised from 800 → 2000. At 800 the
  model was truncated mid-recap and never reached its Action Items section, so it
  "lost" the planted needles to the token cap rather than to poor recall. With
  room it captures all of them.
- `sentiment_intent` — the `primary_intent` check was broadened. The model
  correctly tagged `negative` / `high` churn but labeled intent
  `"reliability complaint"` (a valid reading of the message); the churn dimension
  is already pinned by the separate `churn_risk` check, so the intent check now
  accepts the whole valid space while still rejecting genuinely-wrong labels
  (billing question, feature request, upsell, …). `max_tokens` was also raised
  (256 → 1536): the reasoning model truncated mid-thought before emitting any JSON.
- `objection_handling` — the CTA check was a **substring false-positive**: it
  matched `"connect"` inside the deal context's "third-party **connect**or", so a
  truncated reply with no actual next step passed spuriously. Replaced with a
  word-boundary regex, and `max_tokens` raised 512 → 1000 so the reply can reach
  its closing CTA.
- `harness.extract_json` — now recovers the JSON object even when reasoning prose
  (with stray braces) precedes it, by preferring the *last* balanced object.

Re-run `selftest` after any grader change, and re-run live against your target
model — local models differ, and a grader tuned for one may be miscalibrated for
another.

## Architecture

```
run_evals.py              CLI entry point
sales_eval/
  client.py               LM Studio AsyncOpenAI client + .local mDNS resolution
  harness.py              Scenario contract, RunOutcome, lenient grading helpers,
                          and the tool-call runner (run_scenario)
  fakeclient.py           scripted stand-in for AsyncOpenAI (offline self-test)
  selftest.py             offline calibration gate
  report.py               rich/plaintext result tables + JSON export
  scenarios/
    __init__.py           auto-discovers every module in this dir (pkgutil)
    *.py                  one self-contained scenario each (fixtures + grader)
```

## Adding a scenario

Drop a new `sales_eval/scenarios/<name>.py` that:

1. defines inline fixtures and a `grade(out: RunOutcome) -> GradeResult`,
2. calls `register(Scenario(... grade=grade, sample_good=..., sample_bad=...))`,
3. uses the lenient helpers from `sales_eval.harness` (never `==` on model text).

Auto-discovery picks it up — no registry to edit. Run `python run_evals.py
selftest` and confirm your scenario shows `[ok] ... good passes, bad fails`.
