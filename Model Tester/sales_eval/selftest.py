"""Offline self-test — proves the harness and every grader work without the DGX.

Two gates, neither needs a live model:

  1. smoke_harness()      drive a trivial tool-use scenario through the REAL
                          runner with a FakeClient: confirms the tool-call loop,
                          RunOutcome assembly, and grading path are sound.

  2. validate_scenarios() for each registered scenario, replay its `sample_good`
                          and `sample_bad` through the runner. A grader is only
                          trustworthy if it PASSES good output and FAILS bad
                          output — a grader that can't fail is worthless.

Run:  python -m sales_eval.selftest   (from the "Model Tester" dir)
Exit code is non-zero if any gate fails, so it works in CI / pre-fan-out checks.
"""

from __future__ import annotations

import asyncio
import sys

from .fakeclient import FakeClient
from .harness import GradeResult, RunOutcome, Scenario, all_scenarios, has_number, run_scenario


# --------------------------------------------------------------------------- #
# Gate 1: smoke the runner end-to-end with a trivial inline scenario.
# --------------------------------------------------------------------------- #
def _smoke_scenario() -> Scenario:
    def router(name: str, args: dict):
        if name == "get_balance":
            return {"account": args.get("account"), "balance_usd": 4200}
        return {"error": "unknown tool"}

    def grade(out: RunOutcome) -> GradeResult:
        g = GradeResult()
        g.add("called get_balance", "get_balance" in out.tool_names(),
              detail=f"tools called: {out.tool_names()}")
        tc = out.first_call("get_balance")
        g.add("passed the account arg", bool(tc and tc.arguments.get("account") == "ACME-1"),
              detail=str(tc.arguments if tc else None))
        g.add("answer grounded on tool result (4200)", has_number(out.final_text, 4200),
              detail=out.final_text[:120])
        return g

    return Scenario(
        name="_smoke",
        category="tool_use",
        description="internal smoke test",
        system="You are a sales assistant. Use tools to answer.",
        user_messages=[{"role": "user", "content": "What's the balance on account ACME-1?"}],
        grade=grade,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_balance",
                "description": "Get the USD balance for an account.",
                "parameters": {
                    "type": "object",
                    "properties": {"account": {"type": "string"}},
                    "required": ["account"],
                },
            },
        }],
        tool_router=router,
        sample_good={
            "tool_calls": [{"name": "get_balance", "arguments": {"account": "ACME-1"}}],
            "final_text": "The balance on account ACME-1 is $4,200.",
        },
        sample_bad={"final_text": "I think the balance is about a billion dollars."},
    )


async def smoke_harness() -> bool:
    sc = _smoke_scenario()
    out, grade = await run_scenario(FakeClient.from_sample(sc.sample_good), "fake", sc)
    ok = grade.passed and "get_balance" in out.tool_names()
    print(f"[{'ok' if ok else 'XX'}] smoke_harness: runner+toolloop+grading "
          f"(score {grade.score:.0%}, tools={out.tool_names()})")
    if not ok:
        for c in grade.checks:
            print(f"      - {c.name}: {'ok' if c.passed else 'FAIL'} {c.detail}")
    return ok


# --------------------------------------------------------------------------- #
# Gate 2: every grader must pass its good sample and fail its bad sample.
# --------------------------------------------------------------------------- #
async def _grade_sample(sc: Scenario, sample: dict) -> GradeResult:
    _, grade = await run_scenario(FakeClient.from_sample(sample), "fake", sc)
    return grade


async def validate_scenarios() -> bool:
    scenarios = all_scenarios()
    all_ok = True
    print(f"\nValidating {len(scenarios)} registered scenario(s):")
    for sc in scenarios:
        problems = []
        if sc.sample_good is None or sc.sample_bad is None:
            print(f"[XX] {sc.name}: missing sample_good/sample_bad")
            all_ok = False
            continue
        good = await _grade_sample(sc, sc.sample_good)
        bad = await _grade_sample(sc, sc.sample_bad)
        if not good.passed:
            problems.append(f"sample_good did NOT pass (score {good.score:.0%})")
        if bad.passed:
            problems.append("sample_bad PASSED (grader can't fail — too lenient)")
        if problems:
            all_ok = False
            print(f"[XX] {sc.name} [{sc.category}]: " + "; ".join(problems))
            for label, gr in (("good", good), ("bad", bad)):
                for c in gr.checks:
                    if c.required and ((label == "good" and not c.passed) or (label == "bad")):
                        print(f"      {label}: [{'ok' if c.passed else 'XX'}] {c.name} — {c.detail}")
        else:
            print(f"[ok] {sc.name} [{sc.category}]: good passes, bad fails "
                  f"({len(good.required_checks)} checks)")
    return all_ok


async def _main() -> int:
    g1 = await smoke_harness()
    g2 = await validate_scenarios()
    ok = g1 and g2
    print(f"\n{'ALL GATES PASSED' if ok else 'SELF-TEST FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
