"""Read-only access to the local Unitronic MariaDB (the `unidb` docker
container's CONTENT_MGMT_SYS database).

Only parameterized SELECTs defined in tools.py ever run — the model is never
handed raw SQL, so the catalog can't be mutated from a conversation.
"""

import json
import os
import re

import pymysql

DB_CONFIG = {
    "host": os.environ.get("UNIDB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("UNIDB_PORT", "3306")),
    "user": os.environ.get("UNIDB_USER", "root"),
    "password": os.environ.get("UNIDB_PASSWORD", ""),
    "database": os.environ.get("UNIDB_NAME", "CONTENT_MGMT_SYS"),
}


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run one SELECT on a fresh connection and return rows as dicts.

    A connection per call is cheap against a local container and sidesteps
    stale-connection handling in a long-lived chat session.
    """
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())
    finally:
        conn.close()


_TAG_RE = re.compile(r"<[^>]+>")


def rich_text(raw: str | None, limit: int = 600) -> str:
    """Flatten the CMS's Editor.js fields (a JSON list of JSON-encoded block
    strings) into plain text. Falls back to tag-stripping on anything that
    isn't valid Editor.js."""
    if not raw:
        return ""
    parts: list[str] = []
    try:
        for block_str in json.loads(raw):
            block = json.loads(block_str) if isinstance(block_str, str) else block_str
            data = block.get("data", {})
            if text := data.get("text"):
                parts.append(text)
            for item in data.get("items", []):
                # list items are strings or {content: ...} depending on editor version
                parts.append(item if isinstance(item, str) else item.get("content", ""))
    except (json.JSONDecodeError, AttributeError, TypeError):
        parts = [raw]
    text = _TAG_RE.sub(" ", " ".join(parts))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "…" if len(text) > limit else text
