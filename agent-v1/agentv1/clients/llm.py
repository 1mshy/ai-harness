"""Chat client for the DGX vLLM endpoint.

Two properties of this endpoint were measured on 2026-08-05 and both contradict
assumptions in AGENT_PLAN.md §10:

*Native tool calling works.* The server runs with
``--enable-auto-tool-choice --tool-call-parser gemma4``. Sending OpenAI-style
``tools=[...]`` returns ``finish_reason="tool_calls"`` with well-formed
arguments. The plan budgeted for hand-rolled free-form JSON tool calls; that is
not necessary.

*Constrained decoding via ``response_format={"type":"json_schema"}`` still
returns empty content.* That invariant is real and separate -- it is guided
decoding, not tool parsing. So structured *extraction* (the KB merge pass) uses
free-form JSON plus code-side validation and a repair retry, while *tool
calling* uses the native path.

Latency measured the same day: 1.2-1.6 s for a trivial completion while the
box was serving other work. The single 99.9 s figure quoted in earlier notes
did not reproduce across five consecutive calls.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .. import config


class LLMError(RuntimeError):
    pass


class LLMTransientError(LLMError):
    """Retrying may help: timeout, 5xx, connection reset."""


class LLMPermanentError(LLMError):
    """Retrying will not help: 400, unparseable request, model refusal."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Gemma-4 emits channel control markers into `content` once the conversation
# contains tool_calls and tool-role replies -- a plain single-turn completion
# never shows them, which is exactly why this is easy to ship without noticing.
# The visible answer follows the last closing marker. Strip at the client
# boundary so no downstream consumer has to know the model's template.
_CHANNEL_CLOSE = re.compile(r"<\s*channel\s*\|?\s*>")
_CHANNEL_ANY = re.compile(r"<\s*\|?\s*/?\s*(?:channel|start|end|im_start|im_end)[^>]*>")


def clean_content(text: str | None) -> str | None:
    if not text:
        return text
    matches = list(_CHANNEL_CLOSE.finditer(text))
    if matches:
        text = text[matches[-1].end() :]
    text = _CHANNEL_ANY.sub("", text)
    return text.strip()


def extract_json(text: str) -> Any:
    """Recover a JSON value from a free-form completion.

    Needed because guided decoding is unusable on this endpoint. Handles the
    three things the model actually does: bare JSON, fenced JSON, and prose
    wrapped around JSON.
    """
    if text is None:
        raise ValueError("no content")
    text = text.strip()
    if not text:
        raise ValueError("empty content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Balanced-brace scan from the first opener.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
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
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"no JSON found in completion: {text[:200]!r}")


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or config.LLM_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self._sem = threading.Semaphore(config.LLM_MAX_CONCURRENCY)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_API_KEY}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            if exc.code >= 500 or exc.code == 429:
                raise LLMTransientError(f"HTTP {exc.code}: {body}") from exc
            raise LLMPermanentError(f"HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMTransientError(str(exc)) from exc

    def chat(
        self,
        messages: Sequence[dict],
        *,
        tools: Sequence[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = tool_choice

        last: Exception | None = None
        for attempt in range(retries):
            started = time.monotonic()
            try:
                with self._sem:
                    body = self._post("/chat/completions", payload)
            except LLMPermanentError:
                raise
            except LLMTransientError as exc:
                last = exc
                # Exponential backoff. The endpoint is shared with a batch
                # pipeline and an internal staff app, so contention is normal
                # and a tight retry makes it worse.
                time.sleep(min(2**attempt, 8))
                continue

            choice = body["choices"][0]
            msg = choice.get("message", {})
            calls: list[ToolCall] = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        args = extract_json(raw)
                    except ValueError:
                        args = {}
                calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=args if isinstance(args, dict) else {},
                        raw_arguments=raw,
                    )
                )
            return ChatResult(
                content=clean_content(msg.get("content")),
                tool_calls=calls,
                finish_reason=choice.get("finish_reason", ""),
                usage=body.get("usage", {}) or {},
                latency_s=time.monotonic() - started,
            )
        raise LLMTransientError(f"exhausted {retries} attempts: {last}")

    def json_call(
        self,
        messages: Sequence[dict],
        *,
        validator: Callable[[Any], Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        repair_attempts: int = 2,
    ) -> Any:
        """Free-form JSON with code-side validation and a repair round-trip.

        Used instead of ``response_format={"type":"json_schema"}`` because that
        returns empty content on this model.
        """
        convo = list(messages)
        last_err: Exception | None = None
        for _ in range(repair_attempts + 1):
            result = self.chat(convo, temperature=temperature, max_tokens=max_tokens)
            try:
                value = extract_json(result.content or "")
                return validator(value) if validator else value
            except (ValueError, KeyError, TypeError) as exc:
                last_err = exc
                convo = convo + [
                    {"role": "assistant", "content": result.content or ""},
                    {
                        "role": "user",
                        "content": (
                            f"That response could not be parsed: {exc}. "
                            f"Reply with the JSON object only -- no prose, no code fence."
                        ),
                    },
                ]
        raise LLMPermanentError(f"could not obtain valid JSON: {last_err}")

    def map_concurrent(
        self, items: Sequence[Any], fn: Callable[[Any], Any], workers: int | None = None
    ) -> list[Any]:
        """Run ``fn`` over items with bounded concurrency, preserving order.

        Failures become ``None`` rather than killing the batch: a KB merge over
        thousands of clusters must not lose 3,499 successes to one bad cluster.
        """
        workers = workers or config.LLM_MAX_CONCURRENCY

        def safe(item):
            try:
                return fn(item)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(safe, items))


_client: LLMClient | None = None
_lock = threading.Lock()


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = LLMClient()
    return _client
