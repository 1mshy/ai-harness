"""A scripted stand-in for AsyncOpenAI, used by the offline self-test.

It replays canned model output so each scenario's grader can be exercised
*without* the DGX box: prove the grader PASSES the scenario's `sample_good`
and FAILS its `sample_bad`. This validates grader calibration (a grader that
can't fail is useless) and the tool-call loop, independent of any live model.

The objects returned mimic the shape the runner reads from the real SDK:
``resp.choices[0].message.{content, tool_calls}`` and, per tool call,
``.id`` and ``.function.{name, arguments}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Func:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Func
    type: str = "function"


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


def _script_from_sample(sample: dict) -> list[dict]:
    """Turn a scenario's sample_good/sample_bad into a sequence of canned turns.

    {"tool_calls": [...], "final_text": "..."} ->
        turn 1: emit the tool_calls; turn 2: emit final_text.
    {"final_text": "..."} -> single turn emitting final_text.
    """
    turns: list[dict] = []
    calls = sample.get("tool_calls") or []
    if calls:
        turns.append({"tool_calls": calls})
    turns.append({"content": sample.get("final_text", "")})
    return turns


class _Completions:
    def __init__(self, owner: "FakeClient"):
        self._owner = owner

    async def create(self, **kwargs) -> _Response:
        return self._owner._next_response()


class _Chat:
    def __init__(self, owner: "FakeClient"):
        self.completions = _Completions(owner)


class FakeClient:
    """Replays a fixed script of canned turns across successive create() calls.

    Build one from a scenario sample with ``FakeClient.from_sample(sample)``.
    """

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self._i = 0
        self.chat = _Chat(self)

    @classmethod
    def from_sample(cls, sample: dict) -> "FakeClient":
        return cls(_script_from_sample(sample))

    def _next_response(self) -> _Response:
        turn = self._script[self._i] if self._i < len(self._script) else {"content": ""}
        self._i += 1
        calls = turn.get("tool_calls")
        if calls:
            tool_calls = [
                _ToolCall(
                    id=f"call_{j}",
                    function=_Func(name=c["name"],
                                   arguments=json.dumps(c.get("arguments", {}))),
                )
                for j, c in enumerate(calls)
            ]
            return _Response([_Choice(_Message(content=None, tool_calls=tool_calls))])
        return _Response([_Choice(_Message(content=turn.get("content", "")))])
