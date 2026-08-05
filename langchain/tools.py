"""Catalog tools for the Unitronic sales-rep agent.

Every tool returns compact plain text (not JSON): the local models on the DGX
box ground better on short readable tables, and it keeps tool-result tokens
down. IDs are included wherever a follow-up tool call needs them.
"""

from langchain_core.tools import tool

from db import query, rich_text

DEFAULT_REGION = "United States (USD)"


def _usd(value) -> str:
    return f"${value:,.2f}" if value is not None else "n/a"


def _years(start, end) -> str:
    if not start and not end:
        return ""
    return f" [{start or '?'}-{end or '?'}]"


@tool
def find_vehicle(query_text: str, year: int | None = None) -> str:
    """Resolve a customer's vehicle to the engine variants Unitronic supports.
    Search by model ("Golf R"), engine ("2.5 TFSI"), or make ("Audi").
    Always call this first when the customer names their car — the returned
    engine_variant_id is what software_for_vehicle needs.
    """
    like = f"%{query_text}%"
    sql = """
        SELECT DISTINCT man.name AS manufacturer, m.name AS model,
               ev.name AS engine, ev.year_start, ev.year_end,
               ev.id AS engine_variant_id
        FROM EngineVariant ev
        JOIN EngineVariant_model evm ON evm.enginevariant_id = ev.id
        JOIN Model m ON m.id = evm.model_id
        LEFT JOIN Model_manufacturers mm ON mm.model_id = m.id
        LEFT JOIN Manufacturer man ON man.id = mm.manufacturer_id
        WHERE (m.name LIKE %s OR ev.name LIKE %s OR man.name LIKE %s
               OR CONCAT(man.name, ' ', m.name) LIKE %s)
    """
    params: list = [like, like, like, like]
    if year is not None:
        sql += " AND ev.year_start <= %s AND ev.year_end >= %s"
        params += [year, year]
    sql += " ORDER BY man.name, m.name, ev.name LIMIT 40"
    rows = query(sql, tuple(params))
    if not rows:
        hint = f" in {year}" if year else ""
        return f"No supported vehicle matches '{query_text}'{hint}. Try just the model name, or list_supported_vehicles."
    lines = [
        f"{r['manufacturer'] or '?'} {r['model']} — {r['engine']}"
        f"{_years(r['year_start'], r['year_end'])} (engine_variant_id: {r['engine_variant_id']})"
        for r in rows
    ]
    return "\n".join(lines)


@tool
def list_supported_vehicles() -> str:
    """List every make and model Unitronic has products for, with year ranges.
    Use when the customer asks what cars are supported or their car didn't match."""
    rows = query(
        """
        SELECT man.name AS manufacturer, m.name AS model, m.year_start, m.year_end
        FROM Model m
        LEFT JOIN Model_manufacturers mm ON mm.model_id = m.id
        LEFT JOIN Manufacturer man ON man.id = mm.manufacturer_id
        ORDER BY man.name, m.name
        """
    )
    by_make: dict[str, list[str]] = {}
    for r in rows:
        by_make.setdefault(r["manufacturer"] or "Other", []).append(
            f"{r['model']}{_years(r['year_start'], r['year_end'])}"
        )
    return "\n".join(f"{make}: {', '.join(models)}" for make, models in by_make.items())


def _stage_hardware(stage_id: str, table: str) -> list[str]:
    rows = query(
        f"""
        SELECT hp.productName, hp.usdMsrp, hp.availability
        FROM {table} sh
        JOIN HardwareProducts hp ON hp.productId = sh.hardware_product_id
        WHERE sh.stage_id = %s
        """,
        (stage_id,),
    )
    return [f"{r['productName']} ({_usd(r['usdMsrp'])}, {r['availability']})" for r in rows]


@tool
def software_for_vehicle(engine_variant: str, region: str = DEFAULT_REGION) -> str:
    """List Unitronic software stages (performance tunes) for an engine variant,
    with customer pricing and any required/recommended hardware per stage.
    Pass the engine_variant_id from find_vehicle (an engine name works too).
    Region defaults to 'United States (USD)'; other regions like
    'Canada (CAD)', 'United Kingdom (GBP)', 'Spain (EUR)' are available.
    """
    variants = query(
        """
        SELECT DISTINCT ev.id, ev.name, ev.year_start, ev.year_end
        FROM EngineVariant ev
        WHERE ev.id = %s OR ev.name LIKE %s
        LIMIT 4
        """,
        (engine_variant, f"%{engine_variant}%"),
    )
    if not variants:
        return f"No engine variant matches '{engine_variant}'. Use find_vehicle to resolve the car first."

    out: list[str] = []
    for v in variants:
        stages = query(
            """
            SELECT s.id, s.stage_name, s.special_note,
                   p.suggested_price, p.currency
            FROM Stage s
            LEFT JOIN Price p ON p.stage_id = s.id AND p.status = 'Active'
                 AND p.continent LIKE %s
            WHERE s.engine_variant_id = %s AND s.status = 'Active'
            ORDER BY s.`order`
            """,
            (f"%{region}%", v["id"]),
        )
        out.append(f"== {v['name']}{_years(v['year_start'], v['year_end'])} ==")
        if not stages:
            out.append("  (no active software stages)")
            continue
        for s in stages:
            price = (
                f"{_usd(s['suggested_price'])} {s['currency']}"
                if s["suggested_price"] is not None
                else "price not listed for this region"
            )
            out.append(f"- {s['stage_name']}: {price}")
            if note := rich_text(s["special_note"], 200):
                out.append(f"    note: {note}")
            if req := _stage_hardware(s["id"], "Stage_required_hardware"):
                out.append(f"    required hardware: {'; '.join(req)}")
            if rec := _stage_hardware(s["id"], "Stage_recommended_hardware"):
                out.append(f"    recommended hardware: {'; '.join(rec)}")
    return "\n".join(out)


@tool
def hardware_for_vehicle(model: str, year: int | None = None, category: str | None = None) -> str:
    """List Unitronic hardware that fits a vehicle. Match by model name
    ("Golf R") or engine ("2.5 TFSI"), optionally narrowed by year and by
    category (Intakes, Exhausts, Fueling, Accessories, ...)."""
    like = f"%{model}%"
    sql = """
        SELECT DISTINCT hp.productId, hp.productName, hc.name AS category,
               hp.usdMsrp, hp.availability
        FROM HardwareFitment hf
        JOIN HardwareProducts hp ON hp.productId = hf.product_id
        JOIN Model m ON m.id = hf.model_id
        LEFT JOIN EngineVariant ev ON ev.id = hf.engine_variant_id
        LEFT JOIN HardwareCategories hc ON hc.id = hp.category_id
        WHERE hp.live_on_website = 1 AND hf.exclude = 0
          AND (m.name LIKE %s OR ev.name LIKE %s)
    """
    params: list = [like, like]
    if year is not None:
        sql += """ AND (hf.year_start IS NULL OR hf.year_start <= %s)
                   AND (hf.year_end IS NULL OR hf.year_end >= %s)"""
        params += [year, year]
    if category:
        sql += " AND hc.name LIKE %s"
        params.append(f"%{category}%")
    sql += " ORDER BY hc.name, hp.productName LIMIT 40"
    rows = query(sql, tuple(params))
    if not rows:
        return f"No live hardware found for '{model}'" + (f" ({year})" if year else "") + "."
    lines = [
        f"[{r['category'] or 'Uncategorized'}] {r['productName']} — "
        f"{_usd(r['usdMsrp'])}, {r['availability']} (productId: {r['productId']})"
        for r in rows
    ]
    return "\n".join(lines)


@tool
def search_hardware(query_text: str, category: str | None = None) -> str:
    """Search the hardware catalog by product name or keyword (e.g. "downpipe",
    "intake", "DSG"). Optionally filter by category. Returns live products with
    US pricing and stock status."""
    like = f"%{query_text}%"
    sql = """
        SELECT hp.productId, hp.productName, hc.name AS category,
               hp.usdMsrp, hp.availability
        FROM HardwareProducts hp
        LEFT JOIN HardwareCategories hc ON hc.id = hp.category_id
        WHERE hp.live_on_website = 1
          AND (hp.productName LIKE %s OR hp.menuDisplayName LIKE %s OR hp.keywords LIKE %s)
    """
    params: list = [like, like, like]
    if category:
        sql += " AND hc.name LIKE %s"
        params.append(f"%{category}%")
    sql += " ORDER BY hp.productName LIMIT 25"
    rows = query(sql, tuple(params))
    if not rows:
        return f"No live hardware products match '{query_text}'."
    lines = [
        f"[{r['category'] or 'Uncategorized'}] {r['productName']} — "
        f"{_usd(r['usdMsrp'])}, {r['availability']} (productId: {r['productId']})"
        for r in rows
    ]
    return "\n".join(lines)


def _related(product_id: str, table: str) -> list[str]:
    rows = query(
        f"""
        SELECT hp.productName FROM {table} rel
        JOIN HardwareProducts hp ON hp.productId = rel.to_hardware_product_id
        WHERE rel.from_hardware_product_id = %s AND hp.live_on_website = 1
        """,
        (product_id,),
    )
    return [r["productName"] for r in rows]


@tool
def hardware_details(product: str) -> str:
    """Full detail on one hardware product: pricing (incl. any active sale),
    stock, description, features, vehicle fitment, and required/recommended
    add-on hardware. Pass a productId from a search, or a product name."""
    rows = query(
        """
        SELECT hp.*, hc.name AS category,
               (hp.scheduleUsdSalePrice IS NOT NULL
                AND NOW() BETWEEN hp.scheduleUsdStartDate AND hp.scheduleUsdEndDate) AS sale_active
        FROM HardwareProducts hp
        LEFT JOIN HardwareCategories hc ON hc.id = hp.category_id
        WHERE hp.productId = %s OR hp.productName LIKE %s
        LIMIT 5
        """,
        (product, f"%{product}%"),
    )
    if not rows:
        return f"No hardware product matches '{product}'."
    if len(rows) > 1 and rows[0]["productId"] != product:
        names = "; ".join(f"{r['productName']} (productId: {r['productId']})" for r in rows)
        return f"Multiple matches — ask the customer or pick one by productId: {names}"
    p = rows[0]

    price = f"MSRP {_usd(p['usdMsrp'])} USD"
    if p["sale_active"]:
        price += f" — ON SALE: {_usd(p['scheduleUsdSalePrice'])} until {p['scheduleUsdEndDate']:%Y-%m-%d}"
    stock = p["availability"] or "unknown"
    if p["quantity"]:  # qty 0 alongside "in-stock" just means untracked
        stock += f" (qty {p['quantity']})"

    fitment = query(
        """
        SELECT DISTINCT m.name, hf.year_start, hf.year_end
        FROM HardwareFitment hf JOIN Model m ON m.id = hf.model_id
        WHERE hf.product_id = %s AND hf.exclude = 0
        ORDER BY m.name LIMIT 30
        """,
        (p["productId"],),
    )

    out = [
        f"{p['productName']} (productId: {p['productId']})",
        f"category: {p['category'] or 'Uncategorized'} | part #: {p['Product_Number'] or 'n/a'}",
        f"price: {price}",
        f"stock: {stock}",
    ]
    if p["installationDifficulty"]:
        out.append(f"install difficulty: {p['installationDifficulty']}/5")
    if desc := rich_text(p["productDescription"], 600):
        out.append(f"description: {desc}")
    if feats := rich_text(p["features"], 400):
        out.append(f"features: {feats}")
    if fitment:
        out.append("fits: " + "; ".join(f"{f['name']}{_years(f['year_start'], f['year_end'])}" for f in fitment))
    for label, table in [
        ("required with this product", "HardwareProducts_requiredHardwareProduct"),
        ("recommended add-ons", "HardwareProducts_recommendedHardwareProduct"),
        ("related products", "HardwareProducts_relatedHardwareProduct"),
    ]:
        if names := _related(p["productId"], table):
            out.append(f"{label}: " + "; ".join(names[:8]))
    return "\n".join(out)


SALES_TOOLS = [
    find_vehicle,
    list_supported_vehicles,
    software_for_vehicle,
    hardware_for_vehicle,
    search_hardware,
    hardware_details,
]
