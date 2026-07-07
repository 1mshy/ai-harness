"""Rendering for eval results — rich table if available, plaintext otherwise."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

from .harness import GradeResult, RunOutcome

try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    HAS_RICH = True
except Exception:  # rich is optional, same as stress_test.py
    _console = None
    HAS_RICH = False


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render_results(results: list[tuple[RunOutcome, GradeResult]]) -> None:
    """Print a per-scenario summary plus per-check detail for failures."""
    passed = sum(1 for _, g in results if g.passed)
    total = len(results)

    if HAS_RICH:
        table = Table(title=f"Sales-agent eval — {passed}/{total} scenarios passed")
        table.add_column("Scenario")
        table.add_column("Category")
        table.add_column("Result")
        table.add_column("Score", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Turns", justify="right")
        for out, g in results:
            color = "green" if g.passed else "red"
            table.add_row(
                out.scenario,
                _category(out),
                f"[{color}]{_mark(g.passed)}[/{color}]",
                f"{g.score:.0%}",
                f"{out.latency_s:.1f}s",
                str(out.turns),
            )
        _console.print(table)
    else:
        print(f"\nSales-agent eval — {passed}/{total} scenarios passed\n")
        print(f"{'scenario':28} {'result':6} {'score':>6} {'lat':>7} {'turns':>5}")
        for out, g in results:
            print(f"{out.scenario:28} {_mark(g.passed):6} {g.score:>5.0%} "
                  f"{out.latency_s:>6.1f}s {out.turns:>5}")

    # Detail: show the checks for anything that failed.
    for out, g in results:
        if g.passed:
            continue
        print(f"\n  {out.scenario}:")
        if out.error:
            print(f"    error: {out.error}")
        for c in g.checks:
            tag = "ok " if c.passed else "XX "
            req = "" if c.required else " (info)"
            print(f"    [{tag}] {c.name}{req}" + (f" — {c.detail}" if c.detail else ""))


def _category(out: RunOutcome) -> str:
    # category isn't on RunOutcome; resolve from registry lazily for display.
    from .harness import _REGISTRY
    s = _REGISTRY.get(out.scenario)
    return s.category if s else ""


def save_json(path: str, results: list[tuple[RunOutcome, GradeResult]], model: str) -> None:
    payload = {
        "model": model,
        "summary": {
            "passed": sum(1 for _, g in results if g.passed),
            "total": len(results),
        },
        "scenarios": [
            {
                "name": out.scenario,
                "passed": g.passed,
                "score": g.score,
                "latency_s": out.latency_s,
                "turns": out.turns,
                "tools_offered": out.tools_offered,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments} for tc in out.tool_calls
                ],
                "final_text": out.final_text,
                "error": out.error,
                "checks": [asdict(c) for c in g.checks],
                "notes": g.notes,
            }
            for out, g in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {path}")
