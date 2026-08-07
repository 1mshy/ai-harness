"""Deterministic supersession. Code, not model judgement.

This is the property that makes case units worth building. A per-call unit
says *"disable Windows Driver Signature Enforcement"*. A case unit -- built
from a `calls_cases` chronology where that attempt is recorded with
``result: failed`` -- says that **fails when Secure Boot is enabled in BIOS**,
and names the call unit in ``superseded_unit_ids``. Retrieval that cannot
express supersession recommends the thing that did not work, confidently,
because the thing that did not work is stated more plainly and appears more
often.

It has to be code. Measured with the live reranker on exactly that pair
(``rerank.py`` self-check output): against the query *"my UniCONNECT+ cable is
not detected by the Windows flashing software"* the naive fix scores +1.147 and
the case unit that corrects it scores **-10.047**. The cross-encoder is doing
its job -- the case unit is phrased as a negation and matches the query surface
badly -- and it would rank the wrong answer first every single time. No amount
of reranker quality fixes this, and asking an LLM to arbitrate re-introduces
the failure it was meant to remove. The relation is recorded in the data; the
filter reads it.

Hard-drop, before context assembly, never "demote". A superseded unit that
survives into the prompt at rank 5 is a unit the generator can quote.

Cycles are treated as a data defect and resolved deterministically rather than
crashing or dropping both sides: see :func:`apply_supersession`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

SUPERSEDES_FIELD = "superseded_unit_ids"


@dataclass(frozen=True)
class Drop:
    """One unit removed, and what removed it. Surfaced in the search trace so
    "why did that answer disappear" is answerable without a rerun."""

    unit_id: str
    superseded_by: str
    reason: str = "superseded"


@dataclass
class SupersessionResult:
    kept: list[Any]
    dropped: list[Drop] = field(default_factory=list)
    # superseder -> [superseded ids that were actually present in the pool]
    edges: dict[str, list[str]] = field(default_factory=dict)

    @property
    def dropped_ids(self) -> set[str]:
        return {d.unit_id for d in self.dropped}


def _default_unit_id(item: Any) -> str:
    uid = getattr(item, "unit_id", None)
    if uid:
        return str(uid)
    payload = _default_payload(item)
    uid = payload.get("unit_id")
    if uid:
        return str(uid)
    # Legacy collections carry no unit_id; the point id is the identity there.
    return str(getattr(item, "id", "") or id(item))


def _default_payload(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    return dict(getattr(item, "payload", None) or {})


def superseded_ids(payload: dict) -> list[str]:
    """Normalised read of ``superseded_unit_ids``.

    Tolerates the single-string form because the field is remapped from
    ``case_knowledge_units[].supersedes_calls[]`` and a one-element list that
    lost its brackets somewhere in the pipeline should not silently disable the
    safety property.
    """
    raw = payload.get(SUPERSEDES_FIELD)
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(x) for x in dict.fromkeys(raw) if x]


def build_edges(
    items: Sequence[Any],
    *,
    unit_id_of: Callable[[Any], str] = _default_unit_id,
    payload_of: Callable[[Any], dict] = _default_payload,
) -> dict[str, list[str]]:
    """superseder -> superseded, restricted to units present in ``items``.

    Restricting to the present set matters: a case unit routinely supersedes
    call units that this particular query never retrieved, and carrying those
    ids forward would make the edge set look far larger than the decision it
    supports.
    """
    present = {unit_id_of(i) for i in items}
    edges: dict[str, list[str]] = {}
    for item in items:
        uid = unit_id_of(item)
        targets = [t for t in superseded_ids(payload_of(item)) if t in present and t != uid]
        if targets:
            edges.setdefault(uid, [])
            for t in targets:
                if t not in edges[uid]:
                    edges[uid].append(t)
    return edges


def _tarjan_scc(nodes: Sequence[str], edges: dict[str, list[str]]) -> list[list[str]]:
    """Strongly connected components, iterative, reverse-topological order.

    Iterative rather than recursive because the candidate pool is caller-sized,
    not fixed, and a deep supersession chain must not be able to blow the
    Python stack inside a request.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        # (node, iterator over successors) frames
        work: list[tuple[str, int]] = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, i = work[-1]
            succs = edges.get(node, ())
            if i < len(succs):
                work[-1] = (node, i + 1)
                nxt = succs[i]
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    comp: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == node:
                            break
                    result.append(comp)
    return result


def _cycle_winner_key(payload: dict, rank: int, unit_id: str):
    """Deterministic ordering used only to break a supersession cycle.

    A cycle means two units each claim to supersede the other, which is a data
    defect -- but dropping both, or crashing, turns a data defect into a
    retrieval outage. Prefer case-derived evidence (the whole reason the
    relation exists), then the explicit ``outranks_call_units`` flag, then
    cluster size, then retrieval rank, and finally the unit id so the outcome
    is stable across processes and reruns.
    """
    return (
        payload.get("evidence") == "multi_call_case",
        bool(payload.get("outranks_call_units")),
        int(payload.get("occurrences") or 0),
        -rank,
        unit_id,
    )


def apply_supersession(
    items: Sequence[Any],
    *,
    unit_id_of: Callable[[Any], str] = _default_unit_id,
    payload_of: Callable[[Any], dict] = _default_payload,
) -> SupersessionResult:
    """Drop every unit superseded by a *surviving* unit.

    "Surviving" is load-bearing. If A supersedes B and B supersedes C and all
    three are retrieved, C must survive: B is dead, so B's opinion about C
    carries no weight. A single pass in arbitrary order gets this wrong, so the
    graph is condensed (cycles collapsed, winner chosen by
    :func:`_cycle_winner_key`) and then walked in topological order.

    Idempotent: running it on its own output is a no-op, which is why the
    pipeline can afford to run it both before and after reranking.
    """
    if not items:
        return SupersessionResult(kept=[])

    order = {unit_id_of(item): i for i, item in enumerate(items)}
    payloads = {unit_id_of(item): payload_of(item) for item in items}
    nodes = list(order)
    edges = build_edges(items, unit_id_of=unit_id_of, payload_of=payload_of)

    dropped: dict[str, Drop] = {}

    # 1. Collapse cycles: one winner per component, the rest die immediately.
    comp_of: dict[str, int] = {}
    comps = _tarjan_scc(nodes, edges)
    for ci, comp in enumerate(comps):
        for n in comp:
            comp_of[n] = ci
        if len(comp) > 1:
            winner = max(
                comp, key=lambda n: _cycle_winner_key(payloads[n], order[n], n)
            )
            for n in comp:
                if n != winner:
                    dropped[n] = Drop(n, winner, "superseded_cycle")

    # 2. Walk the condensation in topological order. Tarjan emits components in
    #    reverse topological order, so reversing gives predecessors first.
    topo: list[str] = []
    for comp in reversed(comps):
        topo.extend(sorted(comp, key=lambda n: order[n]))

    killers: dict[str, list[str]] = {}
    for node in topo:
        if node in dropped:
            continue  # a dead unit's supersessions do not fire
        for target in edges.get(node, ()):
            if target == node or comp_of.get(target) == comp_of.get(node):
                continue  # intra-component edges were settled in step 1
            killers.setdefault(target, []).append(node)
            dropped.setdefault(target, Drop(target, node, "superseded"))
    # When several survivors supersede the same unit, attribute the drop to the
    # highest-ranked one. Any of them is correct; naming a stable one makes the
    # trace reproducible across runs.
    for target, names in killers.items():
        best = min(names, key=lambda n: (order[n], n))
        dropped[target] = Drop(target, best, dropped[target].reason)

    kept = [item for item in items if unit_id_of(item) not in dropped]
    drops = sorted(dropped.values(), key=lambda d: order.get(d.unit_id, 0))
    return SupersessionResult(kept=kept, dropped=drops, edges=edges)


if __name__ == "__main__":  # self-check
    def unit(uid, supersedes=(), **extra):
        return {"unit_id": uid, SUPERSEDES_FIELD: list(supersedes), **extra}

    # 1. The real case: a case unit hard-drops the call unit it corrects.
    call_unit = unit("u_call_dse", evidence="single_call", occurrences=12)
    case_unit = unit(
        "u_case_secureboot",
        supersedes=["u_call_dse"],
        evidence="multi_call_case",
        outranks_call_units=True,
        occurrences=1,
    )
    res = apply_supersession([call_unit, case_unit])
    assert [u["unit_id"] for u in res.kept] == ["u_case_secureboot"], res.kept
    assert res.dropped == [Drop("u_call_dse", "u_case_secureboot", "superseded")]
    print("1 direct:", [u["unit_id"] for u in res.kept], "dropped", res.dropped)

    # 2. A dead unit's opinion does not fire: A>B>C keeps C.
    chain = [unit("A", ["B"]), unit("B", ["C"]), unit("C")]
    res = apply_supersession(chain)
    assert [u["unit_id"] for u in res.kept] == ["A", "C"], res.kept
    print("2 chain  :", [u["unit_id"] for u in res.kept])

    # 3. Targets absent from the pool are not edges.
    res = apply_supersession([unit("A", ["ZZZ"])])
    assert res.edges == {} and len(res.kept) == 1
    print("3 absent :", res.edges)

    # 4. A cycle resolves to the case unit, never to a crash or a wipe-out.
    cyc = [
        unit("X", ["Y"], evidence="single_call", occurrences=99),
        unit("Y", ["X"], evidence="multi_call_case", occurrences=1),
    ]
    res = apply_supersession(cyc)
    assert [u["unit_id"] for u in res.kept] == ["Y"], res.kept
    assert res.dropped[0].reason == "superseded_cycle"
    print("4 cycle  :", [u["unit_id"] for u in res.kept], res.dropped)

    # 5. Idempotent, so running it twice around the reranker is free.
    once = apply_supersession(chain)
    twice = apply_supersession(once.kept)
    assert [u["unit_id"] for u in twice.kept] == [u["unit_id"] for u in once.kept]
    assert twice.dropped == []
    print("5 idempot: ok")

    # 6. Diamond: two survivors both dropping the same target, one Drop record.
    dia = [unit("P", ["T"]), unit("Q", ["T"]), unit("T")]
    res = apply_supersession(dia)
    assert [u["unit_id"] for u in res.kept] == ["P", "Q"]
    assert len(res.dropped) == 1
    print("6 diamond:", [u["unit_id"] for u in res.kept], res.dropped)

    # 7. Deep chain: iterative Tarjan, no recursion limit.
    deep = [unit(f"n{i}", [f"n{i+1}"]) for i in range(5000)] + [unit("n5000")]
    res = apply_supersession(deep)
    assert [u["unit_id"] for u in res.kept] == [
        f"n{i}" for i in range(0, 5001, 2)
    ], res.kept[:5]
    print("7 deep   :", len(res.kept), "kept of", len(deep))

    # 8. Single-string form still disables the wrong answer.
    res = apply_supersession([{"unit_id": "A", SUPERSEDES_FIELD: "B"}, {"unit_id": "B"}])
    assert [u["unit_id"] for u in res.kept] == ["A"]
    print("8 scalar :", [u["unit_id"] for u in res.kept])

    print("supersede.py self-check OK")
