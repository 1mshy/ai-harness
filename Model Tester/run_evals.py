#!/usr/bin/env python3
"""Run the sales-agent capability evals against an LM Studio endpoint.

Companion to stress_test.py. Where that one measures throughput, this one
measures whether the model can actually do sales-agent work: call CRM tools,
answer from a price sheet, read customer sentiment, summarize a long call,
qualify a lead, handle an objection.

    python run_evals.py list                 # show registered scenarios
    python run_evals.py selftest             # offline grader gate (no model)
    python run_evals.py run                   # run all scenarios live
    python run_evals.py --only sentiment_intent call_summarization run
    python run_evals.py --category tool_use run
    python run_evals.py --model nemotron-3-super --json results.json run
    python run_evals.py --url http://10.150.0.154:1234/v1 run

Global flags: --url, --model, --max-tokens, --timeout, --verbose.
Run without --model on a terminal to pick the model interactively from /v1/models.
Heavy-reasoning models (qwen3, deepseek-r1-class) burn the token budget on
reasoning before answering — give them room, e.g. `run --max-tokens 4000`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sales_eval import all_scenarios, run_scenario
import sales_eval.scenarios  # noqa: F401  (import side effect: registers scenarios)
from sales_eval.client import DEFAULT_URL, list_models, open_client
from sales_eval.report import render_results, save_json


def _select(only, category):
    scenarios = all_scenarios()
    if only:
        names = set(only)
        scenarios = [s for s in scenarios if s.name in names]
        missing = names - {s.name for s in scenarios}
        if missing:
            print(f"Unknown scenario(s): {', '.join(sorted(missing))}")
            print(f"Available: {', '.join(s.name for s in all_scenarios())}")
            sys.exit(2)
    if category:
        scenarios = [s for s in scenarios if s.category == category]
    return scenarios


def cmd_list(args) -> None:
    print(f"{'name':30} {'category':16} description")
    print("-" * 80)
    for s in sorted(all_scenarios(), key=lambda x: (x.category, x.name)):
        print(f"{s.name:30} {s.category:16} {s.description}")


def cmd_selftest(args) -> None:
    from sales_eval.selftest import _main
    sys.exit(asyncio.run(_main()))


async def resolve_model(args) -> str | None:
    """Decide which model to run.

    --model wins outright (scriptable, no prompt). Otherwise fetch /v1/models and,
    on an interactive TTY, let the user pick from the list. With no TTY (CI /
    background) fall back to the first non-embedding model so unattended runs work.
    """
    if args.model:
        return args.model
    try:
        models = await list_models(args.url)
    except Exception as e:
        print(f"Could not list models at {args.url}: {type(e).__name__}: {e}\n"
              f"Is LM Studio up? Pass --url / --model, or try `selftest` (offline).")
        return None
    if not models:
        print(f"No models loaded at {args.url}.")
        return None

    non_embed = [m for m in models if "embed" not in m.lower()]
    default = non_embed[0] if non_embed else models[0]

    if not sys.stdin.isatty():  # unattended -> don't block on input()
        return default

    print("\nAvailable models (from /v1/models):")
    for i, m in enumerate(models):
        tags = []
        if m == default:
            tags.append("default")
        if "embed" in m.lower():
            tags.append("embedding")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  {i}) {m}{suffix}")
    raw = input(f"\nPick a model number (enter for default '{default}'): ").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) < len(models):
        return models[int(raw)]
    print(f"Invalid selection {raw!r}; using default '{default}'.")
    return default


async def _run(args) -> None:
    scenarios = _select(args.only, args.category)
    if not scenarios:
        print("No scenarios selected.")
        return

    model = await resolve_model(args)
    if not model:
        sys.exit(1)
    print(f"Endpoint: {args.url}\nModel:    {model}\n"
          f"Running {len(scenarios)} scenario(s)...\n")

    results = []
    async with open_client(args.url, timeout=args.timeout) as client:
        for s in scenarios:
            print(f"  -> {s.name} [{s.category}] ...", flush=True)
            out, grade = await run_scenario(client, model, s,
                                            timeout=args.timeout, verbose=args.verbose,
                                            max_tokens_override=args.max_tokens)
            results.append((out, grade))

    render_results(results)
    if args.json_out:
        save_json(args.json_out, results, model)

    failed = [g for _, g in results if not g.passed]
    sys.exit(1 if failed else 0)


def cmd_run(args) -> None:
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sales-agent capability evals for LM Studio")
    p.add_argument("--url", default=DEFAULT_URL, help=f"endpoint (default {DEFAULT_URL})")
    p.add_argument("--model", default=None,
                   help="model id; if omitted, prompts to pick from /v1/models on a TTY "
                        "(falls back to the first non-embedding model when unattended)")
    p.add_argument("--timeout", type=float, default=180.0, help="per-request timeout (s)")
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                   help="override every scenario's token cap; raise it (e.g. 4000+) for "
                        "reasoning models that spend most of the budget on reasoning_content")
    p.add_argument("--verbose", action="store_true", help="print tool calls as they happen")

    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list registered scenarios").set_defaults(func=cmd_list)
    sub.add_parser("selftest", help="offline grader gate (no model needed)").set_defaults(func=cmd_selftest)

    run = sub.add_parser("run", help="run scenarios against the model")
    run.add_argument("--only", nargs="+", help="run only these scenario names")
    run.add_argument("--category", help="run only scenarios in this category")
    run.add_argument("--json", dest="json_out", help="write results JSON to FILE")
    run.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not getattr(args, "cmd", None):
        cmd_list(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
