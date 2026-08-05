#!/usr/bin/env python3
"""Interactive load / stress tester for an OpenAI-compatible LM Studio endpoint.

Built for an LM Studio server running on a DGX Spark on the local network, but
works against any OpenAI-compatible /v1 endpoint.

Modes:
  single      one request, prints the response (smoke test)
  sequential  N requests, one at a time (latency under no contention)
  batch       N requests at concurrency C (throughput under contention)
  ramp        batches at increasing concurrency -> throughput-vs-concurrency
  soak        sustained load at concurrency C for D seconds

Run with no arguments for an interactive menu, or use a subcommand for scripting:
  python stress_test.py batch -n 32 -c 8
  python stress_test.py ramp --levels 1,2,4,8,16

The single most useful experiment for this box is `ramp`: if aggregate
output-tokens/sec stays flat as concurrency climbs, the server is serializing
requests (queueing); if it scales up, it is genuinely batching.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from openai import AsyncOpenAI

from hard_prompts import HARD_PROMPTS

try:
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    HAS_RICH = True
except Exception:  # rich is optional
    _console = None
    HAS_RICH = False


DEFAULT_URL = "http://10.150.0.30:1234/v1"

# Varied prompts so concurrent identical requests don't all hit a prompt cache
# and inflate the numbers. Used when no fixed --prompt is set.
PROMPTS = [
    "Explain quantum entanglement in exactly three sentences.",
    "Write a short haiku about distributed systems.",
    "List five practical uses for a local LLM server.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the difference between TCP and UDP? Be concise.",
    "Give me a one-paragraph description of the city of Kyoto.",
    "Describe how a hash map works",
    "Name three trade-offs of microservices versus a monolith.",
]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    url: str = DEFAULT_URL
    model: Optional[str] = None
    system: str = "You are a helpful assistant."
    prompt: Optional[str] = None  # None -> rotate through PROMPTS (or HARD_PROMPTS)
    hard: bool = False              # rotate through HARD_PROMPTS instead of PROMPTS
    max_tokens: int = 16000
    temperature: float = 0.7
    timeout: float = 120.0          # per-request timeout for measured runs
    warmup_timeout: float = 300.0   # generous, first request loads the model
    warmup: bool = True

    def messages(self, index: int = 0) -> list[dict]:
        if self.prompt:
            text = self.prompt
        elif self.hard:
            text = HARD_PROMPTS[index % len(HARD_PROMPTS)]
        else:
            text = PROMPTS[index % len(PROMPTS)]
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": text})
        return msgs


# --------------------------------------------------------------------------- #
# Per-request result + aggregate summary
# --------------------------------------------------------------------------- #
@dataclass
class RequestResult:
    ok: bool
    latency: float                       # total wall time, request -> last byte
    ttft: Optional[float] = None         # time to first token (streaming)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    estimated: bool = False              # token counts estimated, not reported
    content_chars: int = 0
    error: Optional[str] = None

    @property
    def gen_speed(self) -> Optional[float]:
        """Decode speed for this request: output tokens / (latency - ttft)."""
        if self.completion_tokens and self.ttft is not None:
            decode = self.latency - self.ttft
            if decode > 0:
                return self.completion_tokens / decode
        return None


@dataclass
class Summary:
    label: str
    concurrency: int
    total: int
    ok: int
    failed: int
    wall: float
    total_completion_tokens: int
    total_prompt_tokens: int
    ttft: dict = field(default_factory=dict)       # mean/p50/p90/p99
    latency: dict = field(default_factory=dict)
    gen_speed_mean: Optional[float] = None         # mean per-request decode tok/s
    errors: dict = field(default_factory=dict)     # error type -> count

    @property
    def throughput(self) -> float:
        """Aggregate output tokens/sec across the whole run (the headline)."""
        return self.total_completion_tokens / self.wall if self.wall > 0 else 0.0

    @property
    def req_per_sec(self) -> float:
        return self.ok / self.wall if self.wall > 0 else 0.0


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p99": None}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def summarize(label: str, concurrency: int, results: list[RequestResult], wall: float) -> Summary:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    speeds = [r.gen_speed for r in ok if r.gen_speed is not None]
    errors: dict = {}
    for r in failed:
        key = (r.error or "unknown").split("\n")[0][:80]
        errors[key] = errors.get(key, 0) + 1
    return Summary(
        label=label,
        concurrency=concurrency,
        total=len(results),
        ok=len(ok),
        failed=len(failed),
        wall=wall,
        total_completion_tokens=sum(r.completion_tokens or 0 for r in ok),
        total_prompt_tokens=sum(r.prompt_tokens or 0 for r in ok),
        ttft=_dist([r.ttft for r in ok if r.ttft is not None]),
        latency=_dist([r.latency for r in ok]),
        gen_speed_mean=statistics.fmean(speeds) if speeds else None,
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# Hostname resolution
# --------------------------------------------------------------------------- #
# Python's getaddrinfo can't resolve mDNS ".local" names on macOS (curl can, via
# Bonjour). So for ".local" hosts we resolve to an IP via the system resolver and
# rewrite the URL. Other hosts are left to httpx/the OS resolver as usual.
_resolve_cache: dict[str, Optional[str]] = {}


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


_IPV4 = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"


def _system_resolve(host: str) -> Optional[str]:
    """Resolve a hostname via OS tools, trying several mDNS-capable methods.

    dscacheutil only returns *cached* mDNS entries, so it's unreliable on its own;
    dns-sd and ping actively trigger the lookup. We try the cheapest first.
    """
    # 1. macOS directory cache (fast, but only if already cached).
    try:
        out = subprocess.run(["dscacheutil", "-q", "host", "-a", "name", host],
                             capture_output=True, text=True, timeout=8).stdout
        m = re.search(r"ip_address:\s*" + _IPV4, out)
        if m:
            return m.group(1)
    except Exception:
        pass
    # 2. macOS active mDNS query. dns-sd streams and never exits, so time-box it
    #    and parse whatever it printed (TimeoutExpired carries partial stdout).
    for getter in (lambda: subprocess.run(["dns-sd", "-G", "v4", host],
                                          capture_output=True, text=True, timeout=4).stdout,):
        try:
            out = getter()
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "ignore")
        except Exception:
            out = ""
        m = re.search(rf"{re.escape(host)}\.?\s+{_IPV4}", out or "")
        if m:
            return m.group(1)
    # 3. ping resolves via mDNS and prints the IP in parentheses.
    try:
        out = subprocess.run(["ping", "-c", "1", "-t", "2", host],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"\(" + _IPV4 + r"\)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    # 4. Linux fallback.
    try:
        out = subprocess.run(["getent", "hosts", host],
                             capture_output=True, text=True, timeout=8).stdout
        m = re.match(r"\s*" + _IPV4, out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def resolve_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host or _is_ip(host) or not host.endswith(".local"):
        return url
    ip = _resolve_cache.get(host)
    if not ip:
        ip = _system_resolve(host)
        if ip:
            _resolve_cache[host] = ip  # only cache successes, so we retry if it was down
            log(f"[dim]Resolved {host} -> {ip} (mDNS via system resolver)[/dim]")
        else:
            log(f"[yellow]Could not resolve {host} (mDNS). If this persists, pass the IP "
                f"directly, e.g. --url http://10.x.x.x:1234/v1[/yellow]")
            return url
    netloc = f"{ip}:{parsed.port}" if parsed.port else ip
    return urlunparse(parsed._replace(netloc=netloc))


# --------------------------------------------------------------------------- #
# API layer
# --------------------------------------------------------------------------- #
@contextlib.asynccontextmanager
async def open_client(url: str, pool: int, timeout: float):
    """AsyncOpenAI client with retries off and a connection pool sized to the run.

    max_retries=0    -> a stress test must measure failures, not silently retry.
    pool >= concurrency -> otherwise excess requests queue at the client (httpx
                           defaults to ~100 connections) and we'd be measuring
                           the laptop, not the DGX.
    """
    limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
    http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(timeout))
    client = AsyncOpenAI(base_url=resolve_url(url), api_key="lm-studio",
                         max_retries=0, http_client=http_client)
    try:
        yield client
    finally:
        await http_client.aclose()


def _delta_text(delta) -> str:
    """Text from a streaming delta, covering reasoning models (reasoning_content)."""
    text = getattr(delta, "content", None) or ""
    extra = getattr(delta, "reasoning_content", None)
    if extra is None:
        me = getattr(delta, "model_extra", None)
        if me:
            extra = me.get("reasoning_content")
    return text + (extra or "")


async def run_one(client: AsyncOpenAI, cfg: Config, index: int, timeout: float,
                  collect_text: bool = False) -> tuple[RequestResult, str]:
    start = time.perf_counter()
    ttft: Optional[float] = None
    prompt_tokens = completion_tokens = None
    content_chars = 0
    text_parts: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=cfg.model,
            messages=cfg.messages(index),
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout,
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                completion_tokens = chunk.usage.completion_tokens
                prompt_tokens = chunk.usage.prompt_tokens
            if chunk.choices:
                piece = _delta_text(chunk.choices[0].delta)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    content_chars += len(piece)
                    if collect_text:
                        text_parts.append(piece)
        end = time.perf_counter()
        estimated = False
        if completion_tokens is None:  # server didn't report usage -> estimate
            completion_tokens = max(1, content_chars // 4)
            estimated = True
        return RequestResult(
            ok=True, latency=end - start, ttft=ttft,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            estimated=estimated, content_chars=content_chars,
        ), "".join(text_parts)
    except Exception as e:  # timeouts, resets, 5xx — record and keep going
        end = time.perf_counter()
        return RequestResult(
            ok=False, latency=end - start, ttft=ttft,
            error=f"{type(e).__name__}: {e}",
        ), "".join(text_parts)


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
async def list_models(url: str) -> list[str]:
    async with open_client(url, pool=4, timeout=30) as client:
        resp = await client.models.list()
        return [m.id for m in resp.data]


async def warmup(client: AsyncOpenAI, cfg: Config) -> RequestResult:
    log(f"[dim]Warming up {cfg.model} (timeout {cfg.warmup_timeout:.0f}s, first request loads the model)...[/dim]")
    res, _ = await run_one(client, cfg, index=0, timeout=cfg.warmup_timeout)
    if res.ok:
        log(f"[green]Warm-up OK[/green]  {res.latency:.1f}s, ttft {fmt(res.ttft, 's')}, "
            f"{res.completion_tokens} tok")
    else:
        log(f"[red]Warm-up FAILED[/red]  {res.error}")
    return res


async def run_batch(client: AsyncOpenAI, cfg: Config, n: int, concurrency: int,
                    progress: Optional[Callable[[], None]] = None) -> tuple[list[RequestResult], float]:
    sem = asyncio.Semaphore(concurrency)

    async def task(i: int) -> RequestResult:
        async with sem:
            res, _ = await run_one(client, cfg, index=i, timeout=cfg.timeout)
            if progress:
                progress()
            return res

    start = time.perf_counter()
    results = await asyncio.gather(*(task(i) for i in range(n)), return_exceptions=True)
    wall = time.perf_counter() - start
    norm = [r if isinstance(r, RequestResult)
            else RequestResult(ok=False, latency=0.0, error=repr(r)) for r in results]
    return norm, wall


async def run_soak(client: AsyncOpenAI, cfg: Config, concurrency: int, duration: float,
                   progress: Optional[Callable[[], None]] = None) -> tuple[list[RequestResult], float]:
    results: list[RequestResult] = []
    start = time.perf_counter()
    deadline = start + duration
    counter = {"i": 0}

    async def worker():
        while time.perf_counter() < deadline:
            i = counter["i"]
            counter["i"] += 1
            res, _ = await run_one(client, cfg, index=i, timeout=cfg.timeout)
            results.append(res)
            if progress:
                progress()

    await asyncio.gather(*(worker() for _ in range(concurrency)), return_exceptions=True)
    wall = time.perf_counter() - start
    return results, wall


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    if HAS_RICH:
        _console.print(msg)
    else:
        # strip the simplest rich markup tags for the plain fallback
        import re
        print(re.sub(r"\[/?[a-z #]+\]", "", msg))


def fmt(x: Optional[float], suffix: str = "", nd: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}{suffix}"


def progress_printer(total: Optional[int], label: str) -> Callable[[], None]:
    state = {"done": 0}

    def cb() -> None:
        state["done"] += 1
        tail = f"/{total}" if total else ""
        print(f"\r  {label}: {state['done']}{tail} done ", end="", flush=True)

    return cb


def render_summary(s: Summary) -> None:
    print()
    if HAS_RICH:
        t = Table(title=f"{s.label}  (concurrency {s.concurrency})", title_style="bold cyan")
        t.add_column("metric", style="dim")
        t.add_column("value", justify="right")
        rows = [
            ("requests", f"{s.ok} ok / {s.failed} failed of {s.total}"),
            ("wall time", fmt(s.wall, "s")),
            ("throughput (aggregate)", f"{fmt(s.throughput, ' tok/s')}"),
            ("requests/sec", fmt(s.req_per_sec, "", 2)),
            ("per-req decode (mean)", fmt(s.gen_speed_mean, " tok/s")),
            ("TTFT  mean / p50 / p90 / p99",
             f"{fmt(s.ttft['mean'],'s')} / {fmt(s.ttft['p50'],'s')} / {fmt(s.ttft['p90'],'s')} / {fmt(s.ttft['p99'],'s')}"),
            ("latency  mean / p50 / p90 / p99",
             f"{fmt(s.latency['mean'],'s')} / {fmt(s.latency['p50'],'s')} / {fmt(s.latency['p90'],'s')} / {fmt(s.latency['p99'],'s')}"),
            ("output tokens (total)", str(s.total_completion_tokens)),
        ]
        for k, v in rows:
            t.add_row(k, v)
        _console.print(t)
    else:
        print(f"=== {s.label}  (concurrency {s.concurrency}) ===")
        print(f"  requests        : {s.ok} ok / {s.failed} failed of {s.total}")
        print(f"  wall time       : {fmt(s.wall,'s')}")
        print(f"  throughput      : {fmt(s.throughput,' tok/s')}  (aggregate output)")
        print(f"  requests/sec    : {fmt(s.req_per_sec,'',2)}")
        print(f"  decode (mean)   : {fmt(s.gen_speed_mean,' tok/s')}  per request")
        print(f"  TTFT  m/50/90/99: {fmt(s.ttft['mean'],'s')} / {fmt(s.ttft['p50'],'s')} / {fmt(s.ttft['p90'],'s')} / {fmt(s.ttft['p99'],'s')}")
        print(f"  lat   m/50/90/99: {fmt(s.latency['mean'],'s')} / {fmt(s.latency['p50'],'s')} / {fmt(s.latency['p90'],'s')} / {fmt(s.latency['p99'],'s')}")
        print(f"  output tokens   : {s.total_completion_tokens}")
    if s.errors:
        log("[red]  errors:[/red]")
        for k, v in s.errors.items():
            print(f"    {v}x  {k}")


def render_ramp(summaries: list[Summary]) -> None:
    print()
    if HAS_RICH:
        t = Table(title="Ramp: throughput vs concurrency", title_style="bold cyan")
        for col in ["concurrency", "ok/fail", "agg tok/s", "req/s", "decode tok/s", "ttft p50", "lat p50", "lat p90"]:
            t.add_column(col, justify="right")
        for s in summaries:
            t.add_row(
                str(s.concurrency), f"{s.ok}/{s.failed}",
                fmt(s.throughput), fmt(s.req_per_sec, "", 2), fmt(s.gen_speed_mean),
                fmt(s.ttft["p50"], "s"), fmt(s.latency["p50"], "s"), fmt(s.latency["p90"], "s"),
            )
        _console.print(t)
    else:
        print("=== Ramp: throughput vs concurrency ===")
        hdr = f"{'conc':>5} {'ok/fail':>9} {'agg tok/s':>10} {'req/s':>7} {'decode':>8} {'ttftp50':>8} {'latp50':>8} {'latp90':>8}"
        print(hdr)
        for s in summaries:
            print(f"{s.concurrency:>5} {f'{s.ok}/{s.failed}':>9} {s.throughput:>10.1f} "
                  f"{s.req_per_sec:>7.2f} {fmt(s.gen_speed_mean):>8} "
                  f"{fmt(s.ttft['p50'],'s'):>8} {fmt(s.latency['p50'],'s'):>8} {fmt(s.latency['p90'],'s'):>8}")
    # interpretation hint
    oks = [s for s in summaries if s.ok > 0]
    if len(oks) >= 2:
        first, last = oks[0].throughput, oks[-1].throughput
        if first > 0:
            ratio = last / first
            if ratio >= 1.5:
                log(f"[green]Throughput scaled {ratio:.1f}x with concurrency -> the server is batching requests.[/green]")
            elif ratio <= 1.15:
                log(f"[yellow]Throughput flat ({ratio:.2f}x) -> the server is serializing/queueing concurrent requests.[/yellow]")
            else:
                log(f"[cyan]Throughput scaled {ratio:.2f}x -> partial batching.[/cyan]")


def save_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"[dim]Saved results to {path}[/dim]")


# --------------------------------------------------------------------------- #
# Mode entry points (async)
# --------------------------------------------------------------------------- #
async def ensure_model(cfg: Config) -> bool:
    if cfg.model:
        return True
    models = await list_models(cfg.url)
    chat = [m for m in models if "embed" not in m.lower()]
    if not chat:
        log("[red]No chat models available on the server.[/red]")
        return False
    cfg.model = chat[0]
    log(f"[dim]No model set; using {cfg.model}[/dim]")
    return True


async def do_single(cfg: Config) -> None:
    if not await ensure_model(cfg):
        return
    async with open_client(cfg.url, pool=4, timeout=cfg.warmup_timeout) as client:
        log(f"[bold]Single request[/bold] -> {cfg.model}")
        res, text = await run_one(client, cfg, index=0, timeout=cfg.warmup_timeout, collect_text=True)
        if res.ok:
            log(f"[green]OK[/green]  latency {fmt(res.latency,'s')}, ttft {fmt(res.ttft,'s')}, "
                f"{res.completion_tokens} tok, decode {fmt(res.gen_speed,' tok/s')}"
                + ("  [estimated]" if res.estimated else ""))
            print("\n" + (text.strip() or "(no text content)") + "\n")
        else:
            log(f"[red]FAILED[/red]  {res.error}")


async def do_batch(cfg: Config, n: int, concurrency: int, label: str = "Batch",
                   json_out: Optional[str] = None) -> Optional[Summary]:
    if not await ensure_model(cfg):
        return None
    pool = max(concurrency + 4, 8)
    async with open_client(cfg.url, pool=pool, timeout=cfg.timeout) as client:
        if cfg.warmup:
            await warmup(client, cfg)
        log(f"[bold]{label}[/bold]  {n} requests @ concurrency {concurrency} -> {cfg.model}")
        prog = progress_printer(n, label.lower())
        results, wall = await run_batch(client, cfg, n, concurrency, prog)
        print()
        summary = summarize(label, concurrency, results, wall)
        render_summary(summary)
        if json_out:
            save_json(json_out, {"config": asdict(cfg), "summary": asdict(summary),
                                 "results": [asdict(r) for r in results]})
        return summary


async def do_ramp(cfg: Config, levels: list[int], n: int, json_out: Optional[str] = None) -> None:
    if not await ensure_model(cfg):
        return
    pool = max(max(levels) + 4, 8)
    summaries: list[Summary] = []
    async with open_client(cfg.url, pool=pool, timeout=cfg.timeout) as client:
        if cfg.warmup:
            await warmup(client, cfg)
        for c in levels:
            # Several waves per level so we measure steady-state throughput, not a
            # single burst whose wall time is set by the one slowest straggler.
            count = max(n, c * 4)
            log(f"\n[bold]Ramp level[/bold] concurrency={c}  ({count} requests)")
            prog = progress_printer(count, f"c={c}")
            results, wall = await run_batch(client, cfg, count, c, prog)
            print()
            s = summarize(f"ramp c={c}", c, results, wall)
            summaries.append(s)
            log(f"  -> {fmt(s.throughput,' tok/s')} aggregate, {fmt(s.latency['p50'],'s')} p50 latency")
    render_ramp(summaries)
    if json_out:
        save_json(json_out, {"config": asdict(cfg), "levels": [asdict(s) for s in summaries]})


async def do_soak(cfg: Config, concurrency: int, duration: float, json_out: Optional[str] = None) -> None:
    if not await ensure_model(cfg):
        return
    pool = max(concurrency + 4, 8)
    async with open_client(cfg.url, pool=pool, timeout=cfg.timeout) as client:
        if cfg.warmup:
            await warmup(client, cfg)
        log(f"[bold]Soak[/bold]  {duration:.0f}s @ concurrency {concurrency} -> {cfg.model}")
        prog = progress_printer(None, "soak")
        results, wall = await run_soak(client, cfg, concurrency, duration, prog)
        print()
        summary = summarize("Soak", concurrency, results, wall)
        render_summary(summary)
        if json_out:
            save_json(json_out, {"config": asdict(cfg), "summary": asdict(summary)})


# --------------------------------------------------------------------------- #
# Interactive menu
# --------------------------------------------------------------------------- #
def ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def ask_int(prompt: str, default: int) -> int:
    while True:
        try:
            return int(ask(prompt, str(default)))
        except ValueError:
            print("  enter a number")


def ask_float(prompt: str, default: float) -> float:
    while True:
        try:
            return float(ask(prompt, str(default)))
        except ValueError:
            print("  enter a number")


def choose_model(cfg: Config) -> None:
    try:
        models = asyncio.run(list_models(cfg.url))
    except Exception as e:
        log(f"[red]Could not list models: {e}[/red]")
        return
    if not models:
        log("[red]No models returned.[/red]")
        return
    print("\nAvailable models:")
    for i, m in enumerate(models):
        marker = " *" if m == cfg.model else ""
        print(f"  {i}) {m}{marker}")
    raw = input("Pick a model number (enter to keep current): ").strip()
    if raw.isdigit() and int(raw) < len(models):
        cfg.model = models[int(raw)]
        log(f"[green]Model set to {cfg.model}[/green]")


def edit_settings(cfg: Config) -> None:
    cfg.url = ask("URL", cfg.url)
    cfg.max_tokens = ask_int("max_tokens", cfg.max_tokens)
    cfg.temperature = ask_float("temperature", cfg.temperature)
    cfg.timeout = ask_float("per-request timeout (s)", cfg.timeout)
    cfg.warmup_timeout = ask_float("warm-up timeout (s)", cfg.warmup_timeout)
    p = ask("fixed prompt (blank = rotate varied prompts)", cfg.prompt or "")
    cfg.prompt = p or None
    h = ask("use hard prompts (transcripts/extraction) instead of short ones? (y/n)",
            "y" if cfg.hard else "n")
    cfg.hard = h.lower().startswith("y")
    w = ask("warm up before runs? (y/n)", "y" if cfg.warmup else "n")
    cfg.warmup = w.lower().startswith("y")


MENU = """
==================== LM Studio Stress Tester ====================
  URL    : {url}
  Model  : {model}
  Prompt : {prompt}
  Tokens : max_tokens={max_tokens}  temp={temperature}  warmup={warmup}  hard={hard}
-----------------------------------------------------------------
  1) Single request (smoke test)
  2) Sequential run   (N requests, one at a time)
  3) Concurrent batch (N requests @ concurrency C)
  4) Ramp test        (throughput vs concurrency)   <- headline
  5) Soak test        (sustained load for D seconds)
  6) Settings
  7) Pick model
  8) Warm up model
  q) Quit
================================================================="""


def interactive(cfg: Config) -> None:
    if not cfg.model:
        try:
            models = asyncio.run(list_models(cfg.url))
            chat = [m for m in models if "embed" not in m.lower()]
            if chat:
                cfg.model = chat[0]
        except Exception as e:
            log(f"[yellow]Could not reach {cfg.url}: {e}[/yellow]")

    while True:
        print(MENU.format(
            url=cfg.url, model=cfg.model or "(none)",
            prompt=(cfg.prompt or ("hard (rotating)" if cfg.hard else "varied (rotating)")),
            max_tokens=cfg.max_tokens, temperature=cfg.temperature, warmup=cfg.warmup,
            hard=cfg.hard,
        ))
        choice = input("> ").strip().lower()
        try:
            if choice == "1":
                asyncio.run(do_single(cfg))
            elif choice == "2":
                n = ask_int("number of requests", 10)
                asyncio.run(do_batch(cfg, n, concurrency=1, label="Sequential"))
            elif choice == "3":
                n = ask_int("number of requests", 20)
                c = ask_int("concurrency", 8)
                asyncio.run(do_batch(cfg, n, concurrency=c, label="Batch"))
            elif choice == "4":
                raw = ask("concurrency levels (comma-separated)", "1,2,4,8,16")
                levels = [int(x) for x in raw.split(",") if x.strip()]
                n = ask_int("requests per level (min)", max(levels))
                asyncio.run(do_ramp(cfg, levels, n))
            elif choice == "5":
                d = ask_float("duration (s)", 30)
                c = ask_int("concurrency", 8)
                asyncio.run(do_soak(cfg, c, d))
            elif choice == "6":
                edit_settings(cfg)
            elif choice == "7":
                choose_model(cfg)
            elif choice == "8":
                if not cfg.model:
                    asyncio.run(ensure_model(cfg))
                asyncio.run(_warmup_only(cfg))
            elif choice in ("q", "quit", "exit"):
                return
            else:
                print("  unknown option")
        except KeyboardInterrupt:
            print("\n  (interrupted, back to menu)")
        except Exception as e:
            log(f"[red]Error: {type(e).__name__}: {e}[/red]")


async def _warmup_only(cfg: Config) -> None:
    if not await ensure_model(cfg):
        return
    async with open_client(cfg.url, pool=4, timeout=cfg.warmup_timeout) as client:
        await warmup(client, cfg)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stress test an OpenAI-compatible LM Studio endpoint.")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=None, help="model id (default: first non-embedding model)")
    p.add_argument("--prompt", default=None, help="fixed prompt (default: rotate varied prompts)")
    p.add_argument("--hard", action="store_true",
                   help="rotate through hard prompts (transcripts, extraction, JSON output) instead of short ones")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout (s)")
    p.add_argument("--warmup-timeout", type=float, default=300.0)
    p.add_argument("--no-warmup", action="store_true", help="skip the warm-up request")
    p.add_argument("--json", dest="json_out", default=None, help="write results to this JSON file")

    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("models", help="list available models")
    sub.add_parser("single", help="one request (smoke test)")

    seq = sub.add_parser("sequential", help="N requests, one at a time")
    seq.add_argument("-n", type=int, default=10)

    bat = sub.add_parser("batch", help="N requests at concurrency C")
    bat.add_argument("-n", type=int, default=20)
    bat.add_argument("-c", "--concurrency", type=int, default=8)

    ramp = sub.add_parser("ramp", help="throughput vs concurrency")
    ramp.add_argument("--levels", default="1,2,4,8,16")
    ramp.add_argument("-n", type=int, default=0, help="min requests per level (default: = concurrency)")

    soak = sub.add_parser("soak", help="sustained load for D seconds")
    soak.add_argument("-d", "--duration", type=float, default=30.0)
    soak.add_argument("-c", "--concurrency", type=int, default=8)

    return p


def cfg_from_args(args) -> Config:
    return Config(
        url=args.url, model=args.model, system=args.system, prompt=args.prompt,
        hard=args.hard,
        max_tokens=args.max_tokens, temperature=args.temperature,
        timeout=args.timeout, warmup_timeout=args.warmup_timeout,
        warmup=not args.no_warmup,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = cfg_from_args(args)

    if not args.cmd:
        interactive(cfg)
        return

    try:
        if args.cmd == "models":
            for m in asyncio.run(list_models(cfg.url)):
                print(m)
        elif args.cmd == "single":
            asyncio.run(do_single(cfg))
        elif args.cmd == "sequential":
            asyncio.run(do_batch(cfg, args.n, concurrency=1, label="Sequential", json_out=args.json_out))
        elif args.cmd == "batch":
            asyncio.run(do_batch(cfg, args.n, concurrency=args.concurrency, label="Batch", json_out=args.json_out))
        elif args.cmd == "ramp":
            levels = [int(x) for x in args.levels.split(",") if x.strip()]
            asyncio.run(do_ramp(cfg, levels, args.n, json_out=args.json_out))
        elif args.cmd == "soak":
            asyncio.run(do_soak(cfg, args.concurrency, args.duration, json_out=args.json_out))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
