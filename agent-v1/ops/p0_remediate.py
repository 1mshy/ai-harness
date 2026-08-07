#!/usr/bin/env python3
"""Reversible remediation for the AGENT_PLAN.md §3.1 PII collections.

§3.1 says "delete ``unitronic_call_transcriptions_0_6b`` outright". That is the
right call and it is also irreversible against a collection nobody can re-embed
on demand, so this tool refuses to make it irreversible: quarantine is
*snapshot, verify the snapshot, then delete*, and it aborts before the delete if
any verification step fails. A snapshot that Qdrant reported as written but that
never landed on this filesystem is the failure mode worth engineering against,
so the local file is checked for existence, for size against the server's own
figure, for its SHA-256 against the server's checksum, and for being a readable
tar archive -- Qdrant snapshots are tars, and a truncated download is still a
plausible-looking file.

Nothing mutates without ``--yes``. The default for every subcommand is a dry run
that prints the exact collection, its live point count, and the snapshot path
that would be written, then stops.

    python ops/p0_remediate.py list
    python ops/p0_remediate.py snapshot unitronic_faq_0_6b --yes
    python ops/p0_remediate.py verify ops/snapshots/<file>.snapshot
    python ops/p0_remediate.py quarantine unitronic_call_transcriptions_0_6b --yes
    python ops/p0_remediate.py restore ops/snapshots/<file>.snapshot --yes

Exit codes: 0 success (including a completed dry run), 1 refused or failed,
2 bad arguments / Qdrant unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402  (a qdrant-client dependency; already present)

from agentv1 import config  # noqa: E402
from agentv1.clients.qdrant import get_client  # noqa: E402

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
# A Qdrant collection snapshot is a tar of segment directories. Even an
# empty collection produced ~74 MB in testing, so anything in the kilobyte
# range means the transfer failed, not that the collection was small.
MIN_SNAPSHOT_BYTES = 4096
DOWNLOAD_CHUNK = 1 << 20


class Refused(RuntimeError):
    """Raised when a precondition fails. Nothing has been mutated."""


@dataclass
class SnapshotResult:
    collection: str
    snapshot_name: str
    local_path: Path
    manifest_path: Path
    size_bytes: int
    sha256: str
    points_at_snapshot: int


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _base_url() -> str:
    return config.QDRANT_URL.rstrip("/")


def _headers() -> dict:
    return {"api-key": config.QDRANT_API_KEY} if config.QDRANT_API_KEY else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(DOWNLOAD_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024.0
    return f"{n} B"


def _require_collection(client, name: str) -> int:
    if not client.collection_exists(name):
        raise Refused(f"collection {name!r} does not exist")
    return client.get_collection(name).points_count or 0


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------


def take_snapshot(client, collection: str, *, out_dir: Path = SNAPSHOT_DIR) -> SnapshotResult:
    """Create a server-side snapshot, download it, and verify the local copy.

    Raises ``Refused`` on any verification failure. The server-side snapshot is
    left in place on failure so an operator can retry the download by hand.
    """
    points = _require_collection(client, collection)
    desc = client.create_snapshot(collection_name=collection, wait=True)
    if desc is None:
        raise Refused(f"Qdrant returned no snapshot descriptor for {collection!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / desc.name
    url = f"{_base_url()}/collections/{collection}/snapshots/{desc.name}"

    # Stream to a .part file so a crashed download can never be mistaken for a
    # complete one by the size check below.
    part = local.with_suffix(local.suffix + ".part")
    written = 0
    with httpx.stream("GET", url, headers=_headers(), timeout=None) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_bytes(DOWNLOAD_CHUNK):
                fh.write(chunk)
                written += len(chunk)
    part.replace(local)

    problems = []
    if not local.exists():
        problems.append("downloaded file is not on disk")
    elif local.stat().st_size < MIN_SNAPSHOT_BYTES:
        problems.append(f"file is {local.stat().st_size} B, below the {MIN_SNAPSHOT_BYTES} B floor")
    elif desc.size and local.stat().st_size != desc.size:
        problems.append(f"size {local.stat().st_size} != server-reported {desc.size}")

    digest = ""
    if not problems:
        digest = _sha256(local)
        if desc.checksum and digest != desc.checksum:
            problems.append(f"sha256 {digest[:16]}... != server checksum {desc.checksum[:16]}...")
        elif not tarfile.is_tarfile(local):
            problems.append("file is not a readable tar archive")

    if problems:
        raise Refused(
            f"snapshot of {collection!r} did not verify:\n    - " + "\n    - ".join(problems)
        )

    manifest = out_dir / (desc.name + ".manifest.json")
    manifest.write_text(json.dumps({
        "collection": collection,
        "snapshot_name": desc.name,
        "qdrant_url": _base_url(),
        "points_at_snapshot": points,
        "size_bytes": local.stat().st_size,
        "sha256": digest,
        "server_checksum": desc.checksum,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qwen_size": config.QWEN_SIZE,
    }, indent=2) + "\n")

    return SnapshotResult(
        collection=collection,
        snapshot_name=desc.name,
        local_path=local,
        manifest_path=manifest,
        size_bytes=local.stat().st_size,
        sha256=digest,
        points_at_snapshot=points,
    )


def verify_snapshot_file(path: Path) -> dict:
    """Re-check a snapshot on disk. Read-only; safe to run at any time."""
    if not path.exists():
        raise Refused(f"{path} does not exist")
    size = path.stat().st_size
    problems = []
    if size < MIN_SNAPSHOT_BYTES:
        problems.append(f"size {size} B is below the {MIN_SNAPSHOT_BYTES} B floor")
    if not tarfile.is_tarfile(path):
        problems.append("not a readable tar archive")
    digest = _sha256(path)

    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("sha256") and manifest["sha256"] != digest:
            problems.append("sha256 does not match the manifest recorded at snapshot time")
        if manifest.get("size_bytes") and manifest["size_bytes"] != size:
            problems.append("size does not match the manifest recorded at snapshot time")
    members = 0
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            for _ in tar:
                members += 1
                if members >= 64:
                    break

    if problems:
        raise Refused(f"{path.name} failed verification:\n    - " + "\n    - ".join(problems))
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest,
        "tar_members_seen": members,
        "manifest": manifest,
    }


def restore_snapshot(client, path: Path, collection: str) -> int:
    """Upload a snapshot back into Qdrant, creating the collection if needed.

    ``recover_snapshot`` on the client takes a *URL the server fetches*, which
    is useless here: Qdrant runs in Docker and cannot see this filesystem. The
    multipart upload endpoint pushes the bytes over the same HTTP connection
    the rest of this tool uses, so it works regardless of where Qdrant lives.
    """
    url = f"{_base_url()}/collections/{collection}/snapshots/upload?priority=snapshot"
    with path.open("rb") as fh:
        resp = httpx.post(
            url,
            headers=_headers(),
            files={"snapshot": (path.name, fh, "application/octet-stream")},
            timeout=None,
        )
    if resp.status_code >= 400:
        raise Refused(f"upload rejected ({resp.status_code}): {resp.text[:400]}")
    body = resp.json()
    if body.get("result") is not True:
        raise Refused(f"Qdrant did not confirm recovery: {json.dumps(body)[:400]}")
    if not client.collection_exists(collection):
        raise Refused(f"upload reported success but {collection!r} still does not exist")
    return client.get_collection(collection).points_count or 0


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_list(client, args) -> int:
    names = sorted(c.name for c in client.get_collections().collections)
    quarantine_targets = set(config.QUARANTINED_COLLECTIONS)
    print(f"{len(names)} live collections at {_base_url()}")
    for name in names:
        count = client.get_collection(name).points_count or 0
        tag = "  <- QUARANTINE TARGET (AGENT_PLAN.md 3.1)" if name in quarantine_targets else ""
        print(f"  {count:>8,}  {name}{tag}")
    print()
    local = sorted(SNAPSHOT_DIR.glob("*.snapshot")) if SNAPSHOT_DIR.exists() else []
    print(f"{len(local)} local snapshots in {SNAPSHOT_DIR}")
    for p in local:
        print(f"  {_human(p.stat().st_size):>12}  {p.name}")
    return 0


def _prune_server_copy(client, res: SnapshotResult) -> None:
    """Drop Qdrant's own copy once the verified bytes are on this filesystem.

    Qdrant keeps every snapshot it creates under its data volume. Quarantining
    a 24,760-point collection and leaving the server copy behind reclaims no
    disk and keeps a second copy of the exact payloads §3.1 wants out of
    circulation -- so this is opt-in, but it is the right default for an
    operator who has the local file.
    """
    client.delete_snapshot(collection_name=res.collection, snapshot_name=res.snapshot_name, wait=True)
    print(f"      pruned Qdrant's server-side copy of {res.snapshot_name}")


def cmd_snapshot(client, args) -> int:
    points = _require_collection(client, args.collection)
    print(f"PLAN: snapshot {args.collection!r} ({points:,} points)")
    print(f"      snapshot directory: {SNAPSHOT_DIR}")
    print("      nothing is deleted by this subcommand"
          + (" except Qdrant's own copy, after the local file verifies"
             if args.prune_server_snapshot else ""))
    if not args.yes:
        print("DRY RUN -- nothing written. Re-run with --yes.")
        return 0
    res = take_snapshot(client, args.collection)
    print(f"OK    wrote {res.local_path}")
    print(f"      {_human(res.size_bytes)}  sha256 {res.sha256[:32]}...")
    print(f"      manifest {res.manifest_path.name}")
    if args.prune_server_snapshot:
        _prune_server_copy(client, res)
    return 0


def cmd_verify(client, args) -> int:
    info = verify_snapshot_file(Path(args.path))
    print(f"OK    {info['path']}")
    print(f"      {_human(info['size_bytes'])}  sha256 {info['sha256'][:32]}...")
    print(f"      readable tar, {info['tar_members_seen']}+ members")
    if info["manifest"]:
        m = info["manifest"]
        print(f"      manifest: collection={m['collection']} points={m['points_at_snapshot']:,} "
              f"created={m['created_at']}")
    else:
        print("      no manifest alongside this file -- restore needs --collection")
    return 0


def cmd_quarantine(client, args) -> int:
    points = _require_collection(client, args.collection)
    planned = SNAPSHOT_DIR / f"{args.collection}-<timestamp>.snapshot"
    print(f"PLAN: DESTROY collection {args.collection!r}")
    print(f"      it currently holds {points:,} points")
    print(f"      a verified snapshot is written to {planned} FIRST")
    print("      the delete is skipped entirely if the snapshot does not verify")
    print(f"      reverse with: python ops/p0_remediate.py restore {SNAPSHOT_DIR}/<file> --yes")
    if not args.yes:
        print("DRY RUN -- nothing snapshotted, nothing deleted. Re-run with --yes.")
        return 0

    res = take_snapshot(client, args.collection)
    print(f"OK    snapshot verified: {res.local_path} ({_human(res.size_bytes)})")
    print(f"      sha256 {res.sha256[:32]}...")

    # Re-read from disk rather than trusting the in-memory result: the point of
    # the gate is that the bytes are recoverable now, not that a function
    # returned without raising a minute ago.
    verify_snapshot_file(res.local_path)
    print("      re-verified from disk immediately before delete")

    client.delete_collection(args.collection)
    if client.collection_exists(args.collection):
        raise Refused(f"delete of {args.collection!r} did not take effect")
    print(f"OK    deleted {args.collection!r} ({points:,} points removed from service)")
    if args.prune_server_snapshot:
        # After the delete, never before: if the delete had failed we would
        # still be holding two copies rather than one.
        _prune_server_copy(client, res)
    print(f"      recovery: python ops/p0_remediate.py restore {res.local_path} --yes")
    return 0


def cmd_restore(client, args) -> int:
    path = Path(args.path)
    info = verify_snapshot_file(path)
    collection = args.collection or (info["manifest"] or {}).get("collection")
    if not collection:
        raise Refused("no --collection given and no manifest to read it from")
    exists = client.collection_exists(collection)
    print(f"PLAN: restore {path.name} into collection {collection!r}")
    print(f"      {_human(info['size_bytes'])}  sha256 {info['sha256'][:32]}...")
    if info["manifest"]:
        print(f"      snapshot held {info['manifest']['points_at_snapshot']:,} points")
    if exists:
        live = client.get_collection(collection).points_count or 0
        print(f"      WARNING: {collection!r} already exists with {live:,} points and will be OVERWRITTEN")
    else:
        print(f"      {collection!r} does not exist and will be created")
    if not args.yes:
        print("DRY RUN -- nothing uploaded. Re-run with --yes.")
        return 0
    restored = restore_snapshot(client, path, collection)
    print(f"OK    {collection!r} restored with {restored:,} points")
    return 0


def cmd_selftest(client, args) -> int:
    """Offline checks of the pure logic. No collection is touched."""
    failures = []
    tmp = SNAPSHOT_DIR / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)

    tiny = tmp / "tiny.snapshot"
    tiny.write_bytes(b"nope")
    try:
        verify_snapshot_file(tiny)
        failures.append("verify accepted a 4-byte file")
    except Refused:
        pass

    missing = tmp / "does-not-exist.snapshot"
    try:
        verify_snapshot_file(missing)
        failures.append("verify accepted a missing file")
    except Refused:
        pass

    # A real tar, large enough to clear the floor, must pass; and mutating one
    # byte must make the recorded sha256 disagree.
    good = tmp / "good.snapshot"
    payload = tmp / "payload.bin"
    payload.write_bytes(b"0" * (MIN_SNAPSHOT_BYTES * 2))
    with tarfile.open(good, "w") as tar:
        tar.add(payload, arcname="segments/payload.bin")
    manifest = good.with_name(good.name + ".manifest.json")
    manifest.write_text(json.dumps({
        "collection": "selftest_collection", "snapshot_name": good.name,
        "points_at_snapshot": 0, "size_bytes": good.stat().st_size,
        "sha256": _sha256(good), "created_at": "1970-01-01T00:00:00+00:00",
    }))
    try:
        info = verify_snapshot_file(good)
        if info["tar_members_seen"] < 1:
            failures.append("verify did not read any tar members")
    except Refused as exc:
        failures.append(f"verify rejected a good snapshot: {exc}")

    with good.open("r+b") as fh:
        fh.seek(good.stat().st_size // 2)
        fh.write(b"\xff")
    try:
        verify_snapshot_file(good)
        failures.append("verify accepted a file whose sha256 no longer matches its manifest")
    except Refused:
        pass

    for p in (tiny, good, payload, manifest):
        p.unlink(missing_ok=True)
    tmp.rmdir()

    if _human(1536) != "1.5 KiB":
        failures.append(f"_human is wrong: {_human(1536)}")

    for f in failures:
        print(f"FAIL  {f}")
    print(f"{6 - len(failures)}/6 remediation checks passed")
    return 1 if failures else 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # --yes is defined on every subparser and deliberately NOT on the top-level
    # parser. argparse lets a subparser shadow a parent's flag and reset it to
    # its own default, so `p0_remediate.py --yes quarantine X` would parse
    # cleanly and then silently dry-run. Leaving it off the top level turns that
    # typo into an "unrecognized arguments" error instead of an operator who
    # believes a collection was quarantined when it was not.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--yes", action="store_true",
                        help="actually do it; without this every subcommand is a dry run")
    common.add_argument("--prune-server-snapshot", action="store_true",
                        help="delete Qdrant's own copy once the local file has verified")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", parents=[common],
                   help="live collections, point counts and local snapshots").set_defaults(fn=cmd_list)
    sub.add_parser("selftest", parents=[common],
                   help="offline checks of the verification logic").set_defaults(fn=cmd_selftest)

    p = sub.add_parser("snapshot", parents=[common],
                       help="snapshot a collection to ops/snapshots/ (never deletes)")
    p.add_argument("collection")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("verify", parents=[common], help="re-verify a snapshot file on disk")
    p.add_argument("path")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("quarantine", parents=[common],
                       help="snapshot, verify, then DELETE the collection")
    p.add_argument("collection")
    p.set_defaults(fn=cmd_quarantine)

    p = sub.add_parser("restore", parents=[common], help="upload a snapshot back into Qdrant")
    p.add_argument("path")
    p.add_argument("--collection", help="target name; defaults to the manifest's collection")
    p.set_defaults(fn=cmd_restore)

    args = ap.parse_args(argv)

    # selftest exercises pure local logic and is documented as needing no
    # Qdrant. Dispatching it before the connection check keeps that true --
    # otherwise the one command an operator runs to decide whether to trust
    # this tool is the one that fails when the server is down.
    if args.fn is cmd_selftest:
        return cmd_selftest(None, args)

    try:
        client = get_client()
        client.get_collections()
    except Exception as exc:
        print(f"could not reach Qdrant at {_base_url()}: {exc}", file=sys.stderr)
        return 2

    try:
        return args.fn(client, args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"HTTP error, nothing was deleted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
