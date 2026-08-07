"""Qdrant access with the two footguns from AGENT_PLAN.md §3.3 designed out.

*Collections are never created on a read path.* The stack this replaces created
them lazily inside ``get_vector_store``, so pointing the reader at a collection
that did not exist produced an empty collection and zero results instead of an
error. Here, creation is an explicit call in the builder and nowhere else.

*Hybrid is decided at birth.* Adding sparse vectors to an existing Qdrant
collection means deleting and recreating it. New collections are created with
both named vectors from the start, so ``enable_hybrid_search`` is not a runtime
flag that can be flipped over live data.

Builds write to a physical collection named ``<alias>_<size>__<generation>``
and the alias is moved onto it at the end. Rolling back a bad build is an alias
flip measured in seconds, not a re-embed measured in hours.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .. import config


class CollectionMissing(RuntimeError):
    pass


class CollectionEmpty(RuntimeError):
    pass


_client: QdrantClient | None = None
_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = QdrantClient(
                    url=config.QDRANT_URL,
                    api_key=config.QDRANT_API_KEY,
                    timeout=config.QDRANT_TIMEOUT,
                )
    return _client


def collection_exists(name: str) -> bool:
    return get_client().collection_exists(name)


def resolve_alias(alias: str) -> str | None:
    """Physical collection an alias currently points at, if any."""
    client = get_client()
    for entry in client.get_aliases().aliases:
        if entry.alias_name == alias:
            return entry.collection_name
    return None


def open_collection(name_or_alias: str, *, create: bool = False) -> str:
    """Resolve a name for reading. ``create`` defaults to False deliberately."""
    client = get_client()
    if client.collection_exists(name_or_alias):
        return name_or_alias
    resolved = resolve_alias(name_or_alias)
    if resolved:
        return resolved
    if create:
        raise CollectionMissing(
            f"{name_or_alias!r} does not exist. Create it explicitly with "
            f"create_hybrid_collection() from the builder -- read paths do not create."
        )
    raise CollectionMissing(f"collection or alias {name_or_alias!r} does not exist")


def create_hybrid_collection(name: str, *, dim: int | None = None, recreate: bool = False) -> None:
    """Create a collection carrying both a dense and a sparse named vector."""
    client = get_client()
    dim = dim or config.EMBED_DIM
    if client.collection_exists(name):
        if not recreate:
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            config.DENSE_VECTOR: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
        },
        sparse_vectors_config={
            config.SPARSE_VECTOR: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False)
            )
        },
        optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
    )


# Payload fields that must be indexed for a *filter* to be cheap. training_safe
# in particular is AND-ed into every single query as a server-side floor, so an
# unindexed scan on it would tax the whole system.
_PAYLOAD_INDEXES: list[tuple[str, Any]] = [
    ("training_safe", qm.PayloadSchemaType.BOOL),
    ("unit_id", qm.PayloadSchemaType.KEYWORD),
    ("point_role", qm.PayloadSchemaType.KEYWORD),
    ("kind", qm.PayloadSchemaType.KEYWORD),
    ("department", qm.PayloadSchemaType.KEYWORD),
    ("language", qm.PayloadSchemaType.KEYWORD),
    ("evidence", qm.PayloadSchemaType.KEYWORD),
    ("emissions_risk", qm.PayloadSchemaType.BOOL),
    ("safety_gated", qm.PayloadSchemaType.BOOL),
    ("dealer_pricing", qm.PayloadSchemaType.BOOL),
    ("internal_only", qm.PayloadSchemaType.BOOL),
    ("time_sensitive", qm.PayloadSchemaType.BOOL),
    ("platform_id", qm.PayloadSchemaType.INTEGER),
    ("source_ids", qm.PayloadSchemaType.KEYWORD),
]


def ensure_payload_indexes(name: str) -> list[str]:
    client = get_client()
    made = []
    for field, schema in _PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema, wait=True
            )
            made.append(field)
        except Exception:
            # Already present, or the field does not occur in this collection.
            continue
    return made


def swap_alias(alias: str, new_collection: str, *, drop_old: bool = True) -> str | None:
    """Point ``alias`` at ``new_collection``; optionally drop what it left.

    Order matters and matches the pipeline convention this repo follows
    elsewhere: write the new generation, move the alias, *then* delete the old
    one. A crash after the move leaves an orphan collection wasting disk, which
    is recoverable. A crash after a delete-first would leave the alias dangling
    and retrieval dead.
    """
    client = get_client()
    old = resolve_alias(alias)
    # `qm.AliasOperations` is a typing.Union in qdrant-client, not a model --
    # instantiating it raises "Cannot instantiate typing.Union". The concrete
    # operation classes are what the API actually takes.
    actions: list[Any] = []
    if old:
        actions.append(
            qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=alias))
        )
    actions.append(
        qm.CreateAliasOperation(
            create_alias=qm.CreateAlias(collection_name=new_collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=actions)
    if old and drop_old and old != new_collection:
        client.delete_collection(old)
    return old


def upsert(name: str, points: Sequence[qm.PointStruct], *, wait: bool = False) -> None:
    get_client().upsert(collection_name=name, points=list(points), wait=wait)


def points_count(name_or_alias: str) -> int:
    client = get_client()
    target = open_collection(name_or_alias)
    return client.get_collection(target).points_count or 0


def assert_routed_collections_populated(names: Iterable[str]) -> None:
    """Refuse to serve when a routed collection is missing or empty.

    This is the assertion AGENT_PLAN.md §3.3 asks for, and it is meant to fail:
    ``unitronic_faq_0_6b`` holds 0 points, is a live member of the legacy
    default collection set, and owns the *troubleshooting* summary in the
    router -- so the highest-volume intent in the corpus routes to an empty
    collection. Answering nothing is not better than refusing to start.
    """
    problems = []
    for name in names:
        try:
            count = points_count(name)
        except CollectionMissing as exc:
            problems.append(f"{name}: {exc}")
            continue
        if count == 0:
            problems.append(f"{name}: exists but holds 0 points")
    if problems:
        raise CollectionEmpty(
            "routed collections are unusable:\n  " + "\n  ".join(problems)
        )


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


def _to_hits(rows) -> list[Hit]:
    return [Hit(id=str(r.id), score=float(r.score or 0.0), payload=dict(r.payload or {})) for r in rows]


def hybrid_search(
    name_or_alias: str,
    *,
    dense: Sequence[float],
    sparse_indices: Sequence[int],
    sparse_values: Sequence[float],
    limit: int,
    query_filter: qm.Filter | None = None,
    prefetch_multiplier: int = 3,
) -> list[Hit]:
    """Dense + sparse prefetch fused server-side by RRF.

    Both halves run against the same filter, so the ``training_safe`` floor
    cannot be satisfied by one branch and bypassed by the other.
    """
    client = get_client()
    target = open_collection(name_or_alias)
    prefetch = [
        qm.Prefetch(
            query=list(dense),
            using=config.DENSE_VECTOR,
            limit=limit * prefetch_multiplier,
            filter=query_filter,
        )
    ]
    if sparse_indices:
        prefetch.append(
            qm.Prefetch(
                query=qm.SparseVector(
                    indices=list(sparse_indices), values=list(sparse_values)
                ),
                using=config.SPARSE_VECTOR,
                limit=limit * prefetch_multiplier,
                filter=query_filter,
            )
        )
    res = client.query_points(
        collection_name=target,
        prefetch=prefetch,
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
    return _to_hits(res.points)


def dense_search(
    name_or_alias: str,
    *,
    dense: Sequence[float],
    limit: int,
    query_filter: qm.Filter | None = None,
    vector_name: str | None = None,
) -> list[Hit]:
    """Dense-only search, for the legacy collections that have no sparse half."""
    client = get_client()
    target = open_collection(name_or_alias)
    kwargs: dict[str, Any] = {}
    if vector_name:
        kwargs["using"] = vector_name
    res = client.query_points(
        collection_name=target,
        query=list(dense),
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
        **kwargs,
    )
    return _to_hits(res.points)


def has_named_vectors(name_or_alias: str) -> bool:
    """Legacy collections use a single unnamed vector; new ones use names."""
    client = get_client()
    target = open_collection(name_or_alias)
    params = client.get_collection(target).config.params
    return isinstance(params.vectors, dict)


def delete_by_unit_ids(name_or_alias: str, unit_ids: Sequence[str]) -> int:
    """Targeted delete backing the ``training_safe`` revocation path.

    Because ``kb_units`` owns canonical identity in Mongo, un-publishing a unit
    is a filtered delete rather than a full collection rebuild. The same
    mechanism serves a Law 25 / PIPEDA deletion request, which is unservable
    against a vector store whose payloads carry no stable source id.
    """
    if not unit_ids:
        return 0
    client = get_client()
    target = open_collection(name_or_alias)
    client.delete(
        collection_name=target,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key="unit_id", match=qm.MatchAny(any=list(unit_ids)))]
            )
        ),
        wait=True,
    )
    return len(unit_ids)
