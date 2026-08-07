"""Three-stage near-duplicate clustering.

Deduplication is a correctness requirement here, not an optimisation. Measured
over the live corpus three ways: exact normalised-title match collapses ~15%,
title token-set Jaccard collapses 25-45% depending on threshold, and ~41% of
units share a verbatim hypothetical-question token set with another unit. The
union is over half the corpus.

Indexed raw, top-k is routinely five restatements of the same fact. The model
reads repetition as corroboration, and the tens of thousands of *singletons* --
the actual reason to have this corpus at all -- never surface.

The three stages are union-find over the same node set, cheapest first:

1. exact match on the normalised title
2. token-set Jaccard over titles, blocked by a shared rare token so this stays
   near-linear instead of O(n^2) over ~65k units
3. verbatim hypothetical-question token-set overlap, which catches units whose
   titles were phrased differently but which answer the identical question

Stage 2 is the one that needs blocking. A naive all-pairs comparison of 65k
titles is 2.1 billion pairs; blocking on the rarest token in each title cuts it
to something that finishes in seconds, because a duplicate pair essentially
always shares its rarest term.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..text.normalize import normalize_text

TITLE_JACCARD_THRESHOLD = 0.72
HQ_JACCARD_THRESHOLD = 0.80
MAX_BLOCK = 400  # a token this common is not evidence of anything

_STOP = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is", "are",
    "with", "how", "what", "do", "does", "can", "i", "my", "it", "this", "that",
    "process", "question", "info", "information", "about",
}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def title_tokens(title: str) -> frozenset[str]:
    return frozenset(t for t in normalize_text(title).split() if t not in _STOP and len(t) > 1)


def hq_tokens(questions: Sequence[str]) -> frozenset[str]:
    out: set[str] = set()
    for q in questions:
        out.update(t for t in normalize_text(q).split() if t not in _STOP and len(t) > 2)
    return frozenset(out)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


# A cluster larger than this cannot be shown to the merge model in one prompt,
# and past ~12 units the marginal unit adds nothing the model has not already
# seen. Head topics really are this large: "UniConnect Plus Cable Reset Process"
# alone reaches four figures.
MAX_MERGE_CLUSTER = 12
ANSWER_JACCARD_THRESHOLD = 0.55


def answer_tokens(unit) -> frozenset[str]:
    text = f"{unit.answer} {unit.conditions}"
    return frozenset(t for t in normalize_text(text).split() if t not in _STOP and len(t) > 2)


def refine_cluster(members: list[int], units: Sequence) -> list[list[int]]:
    """Split an over-merged cluster on answer text.

    Union-find is transitive but similarity is not: A~B and B~C does not make
    A~C, so single-link chaining over 60k titles produces clusters far larger
    than any real topic. Stages 1-3 all key on the *question* side (title,
    hypothetical questions), which is exactly where that chaining happens --
    a hundred differently-worded ways to ask about a cable reset.

    So refine on the *answer* side, which is what the merge actually has to
    reconcile. Units that agree on the answer are genuinely the same fact;
    units that merely sound like the same question are not. This is also the
    mechanism that keeps a $150 license-transfer unit from being averaged with
    a $300 cable-reset unit that happens to share a title.
    """
    if len(members) <= MAX_MERGE_CLUSTER:
        return [members]

    atoks = {i: answer_tokens(units[i]) for i in members}
    local = {m: pos for pos, m in enumerate(members)}
    uf = UnionFind(len(members))

    df: dict[str, int] = defaultdict(int)
    for toks in atoks.values():
        for t in toks:
            df[t] += 1
    blocks: dict[str, list[int]] = defaultdict(list)
    for i in members:
        if atoks[i]:
            blocks[min(atoks[i], key=lambda t: (df[t], t))].append(i)

    for block in blocks.values():
        if len(block) < 2 or len(block) > MAX_BLOCK:
            continue
        for pos_a in range(len(block)):
            i = block[pos_a]
            for j in block[pos_a + 1 :]:
                if uf.find(local[i]) == uf.find(local[j]):
                    continue
                if jaccard(atoks[i], atoks[j]) >= ANSWER_JACCARD_THRESHOLD:
                    uf.union(local[i], local[j])

    sub: dict[int, list[int]] = defaultdict(list)
    for m in members:
        sub[uf.find(local[m])].append(m)

    out: list[list[int]] = []
    for group in sub.values():
        if len(group) <= MAX_MERGE_CLUSTER:
            out.append(group)
            continue
        # Still oversized after refining on answers: this is a genuine head
        # topic with many near-identical restatements. Keep the most
        # informative representatives for the merge prompt and carry the rest
        # as absorbed duplicates -- the cluster's size survives as
        # `occurrences`, which is a bounded boost and never a sort key.
        ranked = sorted(
            group,
            key=lambda i: (
                {"high": 0, "medium": 1, "low": 2}.get(units[i].confidence, 1),
                -len(units[i].answer or ""),
            ),
        )
        out.append(ranked)
    return out


@dataclass
class ClusterStats:
    n_units: int = 0
    n_clusters: int = 0
    singletons: int = 0
    pairs: int = 0
    triple_plus: int = 0
    stage1_merges: int = 0
    stage2_merges: int = 0
    stage3_merges: int = 0
    largest: int = 0
    largest_title: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def cluster_units(units: Sequence) -> tuple[list[list[int]], ClusterStats]:
    """Return clusters as lists of indices into ``units``, plus statistics.

    ``units`` items need ``.title``, ``.hypothetical_questions`` and ``.kind``.
    """
    n = len(units)
    uf = UnionFind(n)
    stats = ClusterStats(n_units=n)

    norm_titles = [normalize_text(u.title) for u in units]
    ttokens = [title_tokens(u.title) for u in units]
    hqtokens = [hq_tokens(u.hypothetical_questions) for u in units]
    kinds = [(u.kind or "") for u in units]

    # -- stage 1: exact normalised title ------------------------------------
    by_title: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, title in enumerate(norm_titles):
        if title:
            by_title[(kinds[i], title)].append(i)
    for group in by_title.values():
        for j in group[1:]:
            uf.union(group[0], j)
            stats.stage1_merges += 1

    # -- stage 2: blocked title token-set Jaccard ---------------------------
    # Block on the rarest token in each title: a near-duplicate pair shares its
    # rarest term with near-certainty, and this avoids 2.1e9 comparisons.
    doc_freq: dict[str, int] = defaultdict(int)
    for toks in ttokens:
        for t in toks:
            doc_freq[t] += 1

    blocks: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(ttokens):
        if not toks:
            continue
        rarest = min(toks, key=lambda t: (doc_freq[t], t))
        blocks[rarest].append(i)

    for members in blocks.values():
        if len(members) < 2 or len(members) > MAX_BLOCK:
            continue
        for a_pos in range(len(members)):
            i = members[a_pos]
            for j in members[a_pos + 1 :]:
                if uf.find(i) == uf.find(j):
                    continue
                if kinds[i] != kinds[j]:
                    continue
                if jaccard(ttokens[i], ttokens[j]) >= TITLE_JACCARD_THRESHOLD:
                    uf.union(i, j)
                    stats.stage2_merges += 1

    # -- stage 3: hypothetical-question token-set ---------------------------
    hq_blocks: dict[str, list[int]] = defaultdict(list)
    hq_df: dict[str, int] = defaultdict(int)
    for toks in hqtokens:
        for t in toks:
            hq_df[t] += 1
    for i, toks in enumerate(hqtokens):
        if len(toks) < 4:
            continue
        rarest = min(toks, key=lambda t: (hq_df[t], t))
        hq_blocks[rarest].append(i)

    for members in hq_blocks.values():
        if len(members) < 2 or len(members) > MAX_BLOCK:
            continue
        for a_pos in range(len(members)):
            i = members[a_pos]
            for j in members[a_pos + 1 :]:
                if uf.find(i) == uf.find(j):
                    continue
                if kinds[i] != kinds[j]:
                    continue
                if jaccard(hqtokens[i], hqtokens[j]) >= HQ_JACCARD_THRESHOLD:
                    uf.union(i, j)
                    stats.stage3_merges += 1

    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        grouped[uf.find(i)].append(i)

    refined: list[list[int]] = []
    for members in grouped.values():
        refined.extend(refine_cluster(members, units))

    clusters = sorted(refined, key=lambda g: (-len(g), g[0]))
    stats.n_clusters = len(clusters)
    for c in clusters:
        size = len(c)
        if size == 1:
            stats.singletons += 1
        elif size == 2:
            stats.pairs += 1
        else:
            stats.triple_plus += 1
    if clusters:
        stats.largest = len(clusters[0])
        stats.largest_title = units[clusters[0][0]].title
    return clusters, stats
