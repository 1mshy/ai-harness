"""Core contract for the sales-agent evaluation suite.

A *capability* test harness (as opposed to ``stress_test.py``, which is a load
tester). Each scenario describes a realistic sales-agent task — call a CRM tool,
answer from a price sheet, classify a customer's sentiment, summarize a long
phone call — and ships a deterministic grader that scores the model's behavior.

The pieces a scenario author touches:

    Scenario        the task: system prompt, opening messages, optional tools,
                    optional tool_router, and a `grade` callable.
    RunOutcome      everything the runner observed (final text, tool calls made,
                    parsed JSON, full transcript). The grader's only input.
    Check / GradeResult   how a grader reports per-criterion pass/fail.
    register(...)   add a scenario to the registry (called at import time).

Graders MUST be lenient. Local LM-Studio-class models emit messy output:
markdown-fenced JSON, prose around the answer, paraphrased labels. Use the
provided helpers (`extract_json`, `norm`, `contains_any`, `label_match`) rather
than exact string equality, and distinguish "model didn't/can't call tools"
from "model called the wrong tool" so the suite stays diagnostic.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One tool/function call the model emitted, plus what the router returned."""
    name: str
    arguments: dict          # best-effort parse of raw_arguments
    raw_arguments: str
    result: Any = None       # whatever the scenario's tool_router returned
    call_id: str = ""


@dataclass
class RunOutcome:
    """Everything the runner observed for one scenario run. The grader's input."""
    scenario: str
    final_text: str                       # final assistant message content
    tool_calls: list[ToolCall]            # every tool call, in order, across turns
    messages: list[dict]                  # full transcript (system..final)
    parsed_json: Any | None               # final_text parsed as JSON, if possible
    turns: int                            # model turns taken
    latency_s: float
    tools_offered: bool                   # were tools given to the model?
    finished_ok: bool                     # did the run complete without error?
    error: Optional[str] = None

    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    def first_call(self, name: str) -> Optional[ToolCall]:
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None


@dataclass
class Check:
    """One grading criterion. Required checks decide pass/fail; informational
    checks (required=False) are reported for diagnostics but don't fail a run —
    e.g. 'model supports the tools API'."""
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


@dataclass
class GradeResult:
    checks: list[Check] = field(default_factory=list)
    notes: str = ""

    def add(self, name: str, passed: bool, detail: str = "", required: bool = True) -> "GradeResult":
        self.checks.append(Check(name, bool(passed), detail, required))
        return self

    @property
    def required_checks(self) -> list[Check]:
        return [c for c in self.checks if c.required]

    @property
    def passed(self) -> bool:
        req = self.required_checks
        return bool(req) and all(c.passed for c in req)

    @property
    def score(self) -> float:
        """Fraction of required checks passed (0.0–1.0)."""
        req = self.required_checks
        if not req:
            return 0.0
        return sum(1 for c in req if c.passed) / len(req)


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """A single sales-agent capability test.

    name           unique slug, e.g. "tool_calling_crm"
    category       grouping, e.g. "tool_use" | "grounding" | "sentiment" |
                   "summarization" | "extraction" | "reasoning"
    description    one line shown in reports
    system         system prompt the agent runs under
    user_messages  opening messages (usually a single user turn); list of
                   {"role": ..., "content": ...} dicts
    grade          Callable[[RunOutcome], GradeResult] — the deterministic grader
    tools          OpenAI tool schema list, or None
    tool_router    Callable[[name, args_dict], result] returning the tool result
                   (dict/str/JSON-serializable); required iff tools is set
    response_format passed through to the API (e.g. {"type": "json_object"});
                   graders must NOT rely on it — parse leniently
    sample_good / sample_bad
                   canned model output used by the OFFLINE self-test to prove the
                   grader passes good output and fails bad output. Schema:
                       {"tool_calls": [{"name": str, "arguments": dict}, ...],  # optional
                        "final_text": str}
    """
    name: str
    category: str
    description: str
    system: str
    user_messages: list[dict]
    grade: Callable[[RunOutcome], GradeResult]
    tools: Optional[list[dict]] = None
    tool_router: Optional[Callable[[str, dict], Any]] = None
    response_format: Optional[dict] = None
    max_tokens: int = 512
    temperature: float = 0.0
    max_tool_turns: int = 5
    sample_good: Optional[dict] = None
    sample_bad: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Registry (populated by scenarios/* at import time)
# --------------------------------------------------------------------------- #
_REGISTRY: "dict[str, Scenario]" = {}


def register(scenario: Scenario) -> Scenario:
    if scenario.name in _REGISTRY:
        raise ValueError(f"duplicate scenario name: {scenario.name!r}")
    if scenario.tools and scenario.tool_router is None:
        raise ValueError(f"scenario {scenario.name!r} offers tools but has no tool_router")
    _REGISTRY[scenario.name] = scenario
    return scenario


def all_scenarios() -> list[Scenario]:
    return list(_REGISTRY.values())


def get_scenario(name: str) -> Scenario:
    return _REGISTRY[name]


def clear_registry() -> None:  # for tests
    _REGISTRY.clear()


# --------------------------------------------------------------------------- #
# Lenient grading helpers — use these, not == / exact matching
# --------------------------------------------------------------------------- #
def norm(s: Any) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any | None:
    """Best-effort parse of JSON from messy model output.

    Handles: clean JSON, ```json fenced blocks, and JSON embedded in prose
    (grabs the first balanced {...} or [...] span). Returns None on failure.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    # Collect ALL top-level balanced {...} / [...] spans, preferring the LAST one:
    # reasoning models (deepseek/nemotron-class) often think out loud in prose and
    # emit the answer JSON last, so the final object is the real answer — and the
    # prose can itself contain stray braces we must skip past.
    for opener, closer in (("{", "}"), ("[", "]")):
        candidates.extend(reversed(_balanced_spans(text, opener, closer)))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def _balanced_spans(text: str, opener: str, closer: str) -> list[str]:
    """Every top-level balanced opener..closer span in text, in order, string-aware."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            if depth == 0:
                start = i
            depth += 1
        elif ch == closer and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                spans.append(text[start:i + 1])
                start = -1
    return spans


def contains_any(text: str, options) -> bool:
    """True if any option appears in text (both normalized)."""
    t = norm(text)
    return any(norm(o) in t for o in options)


def contains_all(text: str, options) -> bool:
    t = norm(text)
    return all(norm(o) in t for o in options)


def label_match(got: Any, expected: str, synonyms: dict | None = None) -> bool:
    """Tolerant label comparison for classification graders.

    Matches when the normalized expected label (or one of its synonyms) is a
    whole-word-ish substring of the normalized model output. `synonyms` maps a
    canonical label to a list of accepted variants.
    """
    if got is None:
        return False
    g = norm(got)
    accepted = {norm(expected)}
    if synonyms:
        for variant in synonyms.get(expected, []):
            accepted.add(norm(variant))
    return any(a and a in g for a in accepted)


def find_number(text: str) -> Optional[float]:
    """First number in text, tolerant of $ , and %."""
    m = re.search(r"-?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text))
    if not m:
        return None
    try:
        return float(re.sub(r"[,$\s]", "", m.group(0)))
    except ValueError:
        return None


def has_number(text: str, value: float, tol: float = 0.0) -> bool:
    """True if `value` appears as a number in text (commas/$ tolerated)."""
    for m in re.finditer(r"\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text)):
        try:
            n = float(re.sub(r"[,$\s]", "", m.group(0)))
        except ValueError:
            continue
        if abs(n - value) <= tol:
            return True
    return False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _parse_args(raw: str) -> dict:
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {"_value": v}
    except Exception:
        return {}


async def run_scenario(client, model: str, scenario: Scenario, *,
                       timeout: float = 120.0, verbose: bool = False,
                       max_tokens_override: Optional[int] = None
                       ) -> tuple[RunOutcome, GradeResult]:
    """Run one scenario against `client` (real AsyncOpenAI or a FakeClient).

    Drives a tool-call loop: if the model returns tool_calls and the scenario
    has a router, execute each call, feed results back, and continue until the
    model answers or `max_tool_turns` is hit. Always produces a RunOutcome —
    errors are captured, never raised — then grades it.

    max_tokens_override raises every scenario's token cap, e.g. for reasoning
    models that spend most of the budget on `reasoning_content` before answering.
    """
    messages: list[dict] = [{"role": "system", "content": scenario.system}]
    messages.extend(scenario.user_messages)
    max_tokens = max_tokens_override or scenario.max_tokens

    tool_calls: list[ToolCall] = []
    final_text = ""
    error: Optional[str] = None
    finished_ok = False
    turns = 0
    last_finish: Optional[str] = None   # finish_reason of the last completion
    reasoning_chars = 0                  # size of reasoning_content (reasoning models)
    last_completion_tokens: Optional[int] = None  # usage, to spot server-side clamping
    start = time.perf_counter()
    # Some LM Studio builds / models reject response_format={"type":"json_object"}
    # (they want "json_schema" or "text"). We never rely on JSON mode — graders
    # parse leniently with extract_json — so on that 400 we drop it and retry,
    # then skip it for the rest of the run.
    drop_response_format = False

    try:
        for _ in range(scenario.max_tool_turns + 1):
            turns += 1
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=scenario.temperature,
                timeout=timeout,
            )
            if scenario.tools:
                kwargs["tools"] = scenario.tools
                kwargs["tool_choice"] = "auto"
            if scenario.response_format and not drop_response_format:
                kwargs["response_format"] = scenario.response_format

            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception as e:
                if "response_format" in kwargs and "response_format" in str(e).lower():
                    drop_response_format = True
                    kwargs.pop("response_format", None)
                    resp = await client.chat.completions.create(**kwargs)
                else:
                    raise
            msg = resp.choices[0].message
            last_finish = getattr(resp.choices[0], "finish_reason", None)
            reasoning_chars = len(getattr(msg, "reasoning_content", None) or "")
            _usage = getattr(resp, "usage", None)
            if _usage is not None:
                last_completion_tokens = getattr(_usage, "completion_tokens", None)
            raw_calls = getattr(msg, "tool_calls", None) or []

            if raw_calls and scenario.tool_router:
                # Record the assistant turn that requested the tools.
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.function.name,
                                      "arguments": c.function.arguments}}
                        for c in raw_calls
                    ],
                })
                for c in raw_calls:
                    name = c.function.name
                    raw_args = c.function.arguments or "{}"
                    args = _parse_args(raw_args)
                    try:
                        result = scenario.tool_router(name, args)
                    except Exception as e:  # a bad tool name / args shouldn't crash the run
                        result = {"error": f"{type(e).__name__}: {e}"}
                    tc = ToolCall(name=name, arguments=args, raw_arguments=raw_args,
                                  result=result, call_id=c.id or "")
                    tool_calls.append(tc)
                    if verbose:
                        print(f"  tool: {name}({args}) -> {result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c.id or "",
                        "content": result if isinstance(result, str) else json.dumps(result),
                    })
                continue  # let the model use the tool results

            # No tool calls -> final answer.
            final_text = msg.content or ""
            messages.append({"role": "assistant", "content": final_text})
            finished_ok = True
            break
        else:
            error = f"exceeded max_tool_turns ({scenario.max_tool_turns})"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    outcome = RunOutcome(
        scenario=scenario.name,
        final_text=final_text,
        tool_calls=tool_calls,
        messages=messages,
        parsed_json=extract_json(final_text) if final_text else None,
        turns=turns,
        latency_s=time.perf_counter() - start,
        tools_offered=bool(scenario.tools),
        finished_ok=finished_ok,
        error=error,
    )

    if error and not finished_ok:
        grade = GradeResult().add("run completed", False, error)
    elif finished_ok and not final_text.strip() and not tool_calls:
        # The model returned no answer content. For reasoning models this usually
        # means the token budget was spent on reasoning before any answer was
        # emitted (finish_reason=length). Report that clearly instead of letting
        # the grader fail every check against an empty string.
        diag = "model returned empty answer content"
        if last_finish == "length":
            diag += f" (finish_reason=length; {last_completion_tokens} completion tokens, " \
                    f"{reasoning_chars} chars of reasoning)"
            # If the server returned far fewer tokens than we asked for, raising
            # --max-tokens won't help — it's clamping output server-side.
            clamped = (last_completion_tokens is not None
                       and last_completion_tokens < max_tokens * 0.6)
            if clamped:
                diag += (f" — server returned only ~{last_completion_tokens} of {max_tokens} "
                         f"requested tokens; it is capping output. Raise the model's max "
                         f"output tokens in LM Studio (client --max-tokens can't override it).")
            elif reasoning_chars:
                diag += f" — reasoning ate the budget; raise --max-tokens (now {max_tokens})"
        outcome.error = diag
        grade = GradeResult().add("model produced answer content", False, diag)
    else:
        try:
            grade = scenario.grade(outcome)
        except Exception as e:  # a grader bug shouldn't crash the suite
            grade = GradeResult().add("grader ran", False, f"{type(e).__name__}: {e}")
    return outcome, grade
