"""Remap the supersession relation off PII and onto opaque unit ids.

``case_knowledge_units[].supersedes_calls[]`` is a list of 3CX recording
filenames::

    [Coles, Zoe]_124-8012307610_20250603202223(152).wav
     ^staff name    ^caller's phone number

So the supersession relation -- which is the entire point of case units -- is
expressed in PII today. Every one of the 3,716 case units carries it. Nothing
may be embedded until this remap has run, because the relation has to travel
into the payload for the retrieval-time filter to work at all.

The join resolves completely: measured 2026-08-06, 1,757 of 1,757 sampled
filenames match a ``calls_analysis.file_name`` exactly. So this is a lookup,
not a fuzzy match, and a miss is a real error worth reporting rather than
absorbing.

Resolution runs in three hops, and the last one is why this module exists at
all rather than being three lines inside the extractor::

    filename -> calls_analysis._id -> raw unit_ids -> CANONICAL unit_ids

Merging changes unit identity, so supersession can only be resolved after the
merge decides which canonical unit each raw unit ended up in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..clients.mongo import source_db


@dataclass
class SupersessionStats:
    case_units_with_refs: int = 0
    filenames_seen: int = 0
    filenames_resolved: int = 0
    filenames_unresolved: int = 0
    edges_emitted: int = 0
    unresolved_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "case_units_with_refs": self.case_units_with_refs,
            "filenames_seen": self.filenames_seen,
            "filenames_resolved": self.filenames_resolved,
            "filenames_unresolved": self.filenames_unresolved,
            "edges_emitted": self.edges_emitted,
            "unresolved_examples": self.unresolved_examples[:5],
        }


def build_filename_index(filenames: set[str]) -> dict[str, str]:
    """``file_name`` -> ``calls_analysis._id``, fetched in bounded batches."""
    db = source_db()
    out: dict[str, str] = {}
    batch: list[str] = []

    def flush() -> None:
        if not batch:
            return
        for doc in db.calls_analysis.find(
            {"file_name": {"$in": batch}}, {"file_name": 1}
        ):
            out[doc["file_name"]] = str(doc["_id"])
        batch.clear()

    for name in filenames:
        batch.append(name)
        if len(batch) >= 500:
            flush()
    flush()
    return out


def resolve(
    raw_units: list,
    canonical_of: dict[str, str],
) -> tuple[dict[str, list[str]], SupersessionStats]:
    """Return ``canonical_unit_id -> [superseded canonical unit ids]``.

    ``canonical_of`` maps a raw ``unit_id`` to the canonical ``unit_id`` it was
    merged into. Produced by the merge stage.
    """
    stats = SupersessionStats()

    wanted: set[str] = set()
    for u in raw_units:
        if u.source_kind == "case" and u.supersedes_calls:
            stats.case_units_with_refs += 1
            wanted.update(u.supersedes_calls)
    stats.filenames_seen = len(wanted)
    if not wanted:
        return {}, stats

    fname_to_source = build_filename_index(wanted)
    stats.filenames_resolved = len(fname_to_source)
    stats.filenames_unresolved = len(wanted) - len(fname_to_source)
    stats.unresolved_examples = sorted(wanted - set(fname_to_source))[:5]

    # Which raw units came from which source document.
    units_by_source: dict[str, list[str]] = defaultdict(list)
    for u in raw_units:
        if u.source_kind == "call":
            units_by_source[u.source_id].append(u.unit_id)

    edges: dict[str, set[str]] = defaultdict(set)
    for u in raw_units:
        if u.source_kind != "case" or not u.supersedes_calls:
            continue
        target = canonical_of.get(u.unit_id)
        if not target:
            continue
        for fname in u.supersedes_calls:
            source_id = fname_to_source.get(fname)
            if not source_id:
                continue
            for raw_id in units_by_source.get(source_id, ()):
                canonical = canonical_of.get(raw_id)
                # A unit never supersedes itself, and a case unit that merged
                # into the same canonical unit as the call unit it supersedes
                # has already absorbed it -- emitting the edge would make the
                # retrieval filter drop the very unit it just selected.
                if canonical and canonical != target:
                    edges[target].add(canonical)

    result = {k: sorted(v) for k, v in edges.items()}
    stats.edges_emitted = sum(len(v) for v in result.values())
    return result, stats
