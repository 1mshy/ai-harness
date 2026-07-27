"""LM Studio connection helpers for the sales-agent eval suite.

Self-contained on purpose: mirrors the mDNS / `.local` resolution logic in
``stress_test.py`` so the eval suite can talk to the same DGX Spark box without
importing the stress tester. Python's resolver can't see `.local` mDNS names
that ``curl`` can, so we shell out to the system resolver and patch the URL.
"""

from __future__ import annotations

import contextlib
import ipaddress
import re
import subprocess
from urllib.parse import urlparse, urlunparse

import httpx
from openai import AsyncOpenAI

DEFAULT_URL = "http://10.150.0.30:1234/v1"

_IPV4 = r"(\d{1,3}(?:\.\d{1,3}){3})"
_resolve_cache: dict[str, str] = {}


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _system_resolve(host: str) -> str | None:
    """Resolve a hostname via OS tools, trying several mDNS-capable methods.

    dscacheutil only returns *cached* mDNS entries, so it's unreliable on its
    own; dns-sd and ping actively trigger the lookup. Cheapest method first.
    """
    # 1. macOS directory cache (fast, only if already cached).
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
    try:
        out = subprocess.run(["dns-sd", "-G", "v4", host],
                             capture_output=True, text=True, timeout=4).stdout
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
    """Patch a `.local` URL to an IP if Python can't resolve it but the OS can."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host or _is_ip(host) or not host.endswith(".local"):
        return url
    ip = _resolve_cache.get(host)
    if not ip:
        ip = _system_resolve(host)
        if ip:
            _resolve_cache[host] = ip  # cache successes only, so we retry if it was down
        else:
            return url
    netloc = f"{ip}:{parsed.port}" if parsed.port else ip
    return urlunparse(parsed._replace(netloc=netloc))


@contextlib.asynccontextmanager
async def open_client(url: str = DEFAULT_URL, timeout: float = 120.0):
    """AsyncOpenAI client pointed at LM Studio, retries off.

    max_retries=0 so a flaky tool call surfaces as a failure instead of being
    silently retried (the DGX box is fragile under load).
    """
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    client = AsyncOpenAI(base_url=resolve_url(url), api_key="lm-studio",
                         max_retries=0, http_client=http_client)
    try:
        yield client
    finally:
        await http_client.aclose()


async def list_models(url: str = DEFAULT_URL) -> list[str]:
    async with open_client(url) as client:
        resp = await client.models.list()
        return [m.id for m in resp.data]


async def pick_model(url: str = DEFAULT_URL, prefer: str | None = None) -> str | None:
    """Return `prefer` if given, else the first non-embedding model on the box."""
    if prefer:
        return prefer
    try:
        models = await list_models(url)
    except Exception:
        return None
    for m in models:
        if "embed" not in m.lower():
            return m
    return models[0] if models else None
