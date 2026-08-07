"""Product catalogue lookup over the pre-existing Qdrant collections.

These are legacy collections built by the website scraper, not by this project:
``unitronic_products_0_6b`` (11,781 points) and
``unitronic_comprehensive_0_6b`` (28,472). Two consequences.

*Vector naming is not assumable.* Both happen to carry named ``dense`` +
``sparse`` vectors, but the other legacy collections in the same Qdrant do not
-- ``unitronic_call_transcriptions_0_6b`` and the classification/training sets
use a single unnamed vector. ``has_named_vectors()`` is checked at call time
and the vector name is passed only when it exists, because a ``using=`` on an
unnamed collection is a 400 and an omitted ``using=`` on a named one is also a
400. The check is cached per collection since the answer cannot change without
the collection being recreated.

*The payload is not ours and must be filtered on the way out.* It carries
``_node_content``, a llama-index blob that re-serialises every metadata field a
second time. Nothing here returns it: the projection is an allowlist of
``sku / title / url / primary_price / context / page_type / brand``, which is
also why a payload key added upstream cannot leak through this tool.

**``primary_price`` is frequently the empty string** and is *not* a fee. It is a
scraped list price whose freshness nobody owns. It is returned so the model can
link the customer to the page, and the description says plainly that it is not
a quotable price. Quotable pricing comes from ``get_fee_schedule`` only.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from qdrant_client import models as qm

from .. import config
from ..clients.qdrant import (
    CollectionMissing,
    dense_search,
    get_client,
    has_named_vectors,
    open_collection,
)
from .base import Tool, ToolDependencyError, ToolInputError, obj_schema

# Search hits the narrow product collection first; the comprehensive one is the
# same scraper output plus support/blog pages, so it is the fallback rather
# than the default -- a product question answered from a blog post reads like a
# hallucination even when it is faithful.
PRODUCT_COLLECTION = f"unitronic_products_{config.QWEN_SIZE}"
COMPREHENSIVE_COLLECTION = f"unitronic_comprehensive_{config.QWEN_SIZE}"

_PAYLOAD_FIELDS = ("sku", "title", "url", "primary_price", "context", "page_type", "brand")

MAX_RESULTS = 8

# UH010FLA -> UH010-FLA. Letters, digits, then a trailing alphanumeric group.
_SKU_JOINED_RE = re.compile(r"^([A-Z]{2,3}\d{2,5})([A-Z][A-Z0-9]{1,4})$")


@lru_cache(maxsize=8)
def _vector_name(collection: str) -> str | None:
    """Cached because it cannot change without the collection being recreated."""
    return config.DENSE_VECTOR if has_named_vectors(collection) else None


def _view(payload: dict, score: float | None = None) -> dict:
    out: dict[str, Any] = {}
    for key in _PAYLOAD_FIELDS:
        val = payload.get(key)
        if key == "context" and isinstance(val, str):
            val = val.strip()[:500]
        if key == "primary_price" and not val:
            val = None
        out[key] = val
    if score is not None:
        out["score"] = round(score, 4)
    return out


def _dedupe(rows: list[dict]) -> list[dict]:
    """One row per SKU. The scraper emits a point per page section, so a single
    part legitimately appears five times with different vehicle-fitment text."""
    # Keyed on title, not SKU: the scraper emits a point per page section and
    # only some of them carry the sku field, so a SKU key leaves the same part
    # in the list twice -- once identified and once anonymous. Whichever copy
    # arrives first wins, and a later copy donates the fields it is missing.
    index: dict[str, dict] = {}
    out: list[dict] = []
    for row in rows:
        key = (row.get("title") or "").strip().lower() or (
            (row.get("sku") or "").strip().upper() or str(row.get("url"))
        )
        kept = index.get(key)
        if kept is None:
            index[key] = row
            out.append(row)
            continue
        for field in ("sku", "primary_price", "url", "context"):
            if not kept.get(field) and row.get(field):
                kept[field] = row[field]
    return out


def search_products(query: str) -> dict:
    if not (query or "").strip():
        raise ToolInputError("search_products needs a query")

    from ..clients.embeddings import get_embedder

    vec = [float(x) for x in get_embedder().embed_query(query)]

    collections_tried: list[str] = []
    rows: list[dict] = []
    for collection in (PRODUCT_COLLECTION, COMPREHENSIVE_COLLECTION):
        try:
            target = open_collection(collection)
        except CollectionMissing:
            continue
        collections_tried.append(collection)
        hits = dense_search(
            target,
            dense=vec,
            limit=MAX_RESULTS * 3,
            vector_name=_vector_name(target),
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="page_type", match=qm.MatchValue(value="product"))]
            ),
        )
        rows.extend(_view(h.payload, h.score) for h in hits)
        if len(_dedupe(rows)) >= MAX_RESULTS:
            break

    if not collections_tried:
        raise ToolDependencyError(
            f"neither {PRODUCT_COLLECTION!r} nor {COMPREHENSIVE_COLLECTION!r} exists"
        )

    results = _dedupe(rows)[:MAX_RESULTS]
    return {
        "query": query,
        "collections": collections_tried,
        "result_count": len(results),
        "results": results,
        "price_note": (
            "primary_price is a scraped list price and is NOT quotable. Quote only "
            "figures returned by get_fee_schedule."
        ),
    }


def lookup_product_by_sku(sku: str) -> dict:
    """Exact SKU read. A filtered scroll, not a nearest-neighbour query --
    a part number is an identifier and embedding it would return neighbours."""
    if not (sku or "").strip():
        raise ToolInputError("lookup_product_by_sku needs a sku")
    wanted = sku.strip().upper()
    candidates = {wanted, wanted.replace(" ", ""), wanted.replace("-", "")}
    # The catalogue stores UH010-FLA; customers read UH010FLA off an invoice
    # where the dash did not survive. Re-insert it at the letter/digit boundary
    # rather than fuzzy-matching, because a part number is an exact key.
    m = _SKU_JOINED_RE.match(wanted.replace("-", ""))
    if m:
        candidates.add(f"{m.group(1)}-{m.group(2)}")

    client = get_client()
    rows: list[dict] = []
    for collection in (PRODUCT_COLLECTION, COMPREHENSIVE_COLLECTION):
        try:
            target = open_collection(collection)
        except CollectionMissing:
            continue
        for candidate in candidates:
            points, _ = client.scroll(
                collection_name=target,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="sku", match=qm.MatchValue(value=candidate))]
                ),
                limit=12,
                with_payload=True,
                with_vectors=False,
            )
            rows.extend(_view(p.payload) for p in points)
        if rows:
            break

    results = _dedupe(rows)
    if not results:
        return {
            "sku": sku,
            "found": False,
            "note": (
                "No catalogue entry for that SKU. Do not guess what the part is -- "
                "try search_products with a description, or escalate."
            ),
        }
    merged = results[0]
    # Fitment lives in the per-section text, so keep every distinct context
    # rather than only the first point's.
    merged["fitment_notes"] = [r["context"] for r in results[:5] if r.get("context")]
    return {"sku": sku, "found": True, "product": merged, "matches": len(results)}


TOOLS = [
    Tool(
        name="search_products",
        description=(
            "Search the Unitronic product catalogue by description, e.g. 'intake for "
            "MK7 GTI', 'UniCONNECT cable', 'downpipe 2.0 TSI'. Returns SKU, title and "
            "product page URL. The primary_price field is a scraped list price and must "
            "never be quoted to a customer as the price."
        ),
        parameters=obj_schema(
            {"query": {"type": "string", "description": "What the customer is looking for"}},
            ["query"],
        ),
        handler=search_products,
        dependency="qdrant",
    ),
    Tool(
        name="lookup_product_by_sku",
        description=(
            "Look up one product by its exact part number / SKU, e.g. 'UH010-FLA'. Use "
            "when the customer reads a part number off an invoice or a box."
        ),
        parameters=obj_schema(
            {"sku": {"type": "string", "description": "Part number, e.g. UH010-FLA"}},
            ["sku"],
        ),
        handler=lookup_product_by_sku,
        dependency="qdrant",
    ),
]


def self_check() -> None:
    import json

    print("named vectors:", PRODUCT_COLLECTION, has_named_vectors(PRODUCT_COLLECTION))
    print("--- search_products('UniCONNECT+ flashing cable') ---")
    res = search_products("UniCONNECT+ flashing cable")
    print(json.dumps(res, indent=1)[:1800])
    assert res["result_count"] > 0
    blob = json.dumps(res)
    assert "_node_content" not in blob and "original_text" not in blob

    print("--- search_products('cold air intake for MK7 GTI') ---")
    res2 = search_products("cold air intake for MK7 GTI")
    for r in res2["results"][:5]:
        print(f"  {r['sku']!s:14s} {r['score']:.3f}  {r['title']}")

    print("--- lookup_product_by_sku('UH010-FLA') ---")
    one = lookup_product_by_sku("UH010-FLA")
    print(json.dumps(one, indent=1)[:1200])
    assert one["found"]

    print("--- lookup_product_by_sku('UH010FLA') dash-stripped ---")
    print(lookup_product_by_sku("UH010FLA")["found"])

    print("--- lookup_product_by_sku('NOT-A-SKU') ---")
    print(json.dumps(lookup_product_by_sku("NOT-A-SKU"), indent=1))
    print("products.py self-check OK")


if __name__ == "__main__":
    self_check()
