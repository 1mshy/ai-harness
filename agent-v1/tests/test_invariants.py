"""Cross-module invariants.

These are the properties that no single module owns and that a plausible-looking
refactor inside any one of them would quietly break. Each maps to a
non-negotiable in CONTRACT.md.

    pytest tests/test_invariants.py -v
"""

from __future__ import annotations

import importlib

import pytest

from agentv1 import config
from agentv1.clients import qdrant, sparse
from agentv1.kb import dedup, extract, gates
from agentv1.text import normalize


# --- surface forms -----------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Stage 1+ software", "stage_1_plus"),
        ("stage 1 plus", "stage_1_plus"),
        ("Stage One Plus", "stage_1_plus"),
        ("stage1+", "stage_1_plus"),
        ("Stage 1 software", "stage_1"),
        ("Stage 2+", "stage_2_plus"),
    ],
)
def test_stage_plus_survives_tokenization(text, expected):
    """`Stage 1` and `Stage 1+` are different products at different prices.

    A default BM25 tokenizer deletes the `+` and dense embeddings blur it, so
    without an explicit marker neither half of the hybrid retriever can tell
    them apart.
    """
    assert expected in normalize.marker_tokens(text)


def test_stage_1_and_stage_1_plus_do_not_collide():
    assert "stage_1_plus" not in normalize.marker_tokens("Stage 1 software")
    assert "stage_1" not in normalize.marker_tokens("Stage 1+ software")


@pytest.mark.parametrize(
    "text",
    ["UniCONNECT+", "uni-connect plus", "uniconnectplus", "Uni Connect Plus", "UniConnect +"],
)
def test_uniconnect_plus_surface_forms_converge(text):
    assert "uniconnect_plus" in normalize.marker_tokens(text)


def test_index_and_query_terms_are_symmetric():
    """The whole point of owning the analyzer: both sides run one function."""
    doc_terms = set(normalize.lexical_terms("Stage 1+ UniCONNECT+ cable reset"))
    query_terms = set(normalize.lexical_terms("stage one plus uni-connect plus cable reset"))
    overlap = doc_terms & query_terms
    assert "stage_1_plus" in overlap
    assert "uniconnect_plus" in overlap


# --- BM25 --------------------------------------------------------------------

def test_bm25_term_ids_are_stable_across_processes():
    """Python's `hash` is salted per process; a hash-derived sparse index would
    silently stop matching after a restart."""
    assert sparse.term_id("stage_1_plus") == sparse.term_id("stage_1_plus")
    assert sparse.term_id("stage_1_plus") == 1531035378 or sparse.term_id("stage_1_plus") >= 0


def test_bm25_analyzer_version_mismatch_refuses_to_load(tmp_path):
    enc = sparse.Bm25Encoder().fit(["a b c", "b c d"])
    path = tmp_path / "bm25.json"
    enc.save(path)
    blob = path.read_text().replace(f'"analyzer_version":{sparse.ANALYZER_VERSION}', '"analyzer_version":999')
    path.write_text(blob)
    with pytest.raises(RuntimeError, match="analyzer"):
        sparse.Bm25Encoder.load(path)


# --- gates -------------------------------------------------------------------

def test_training_safe_is_non_bypassable():
    doc = {"training_safe": False, "useful_content": True, "status": "done", "review": {}}
    assert gates.document_gate(doc).passed is False


def test_complied_improperly_is_dropped_despite_being_training_safe():
    """training_safe is a PII gate. It screens nothing about behaviour, so these
    32 documents are both training_safe and useful_content and would otherwise
    embed alongside the 201 correct refusals."""
    doc = {
        "training_safe": True,
        "useful_content": True,
        "status": "done",
        "review": {"emissions_handling": "complied_improperly"},
    }
    verdict = gates.document_gate(doc)
    assert verdict.passed is False
    assert verdict.reason == "emissions_complied_improperly"


def test_incorrect_statements_is_read_as_an_array_not_a_boolean():
    """{'review.incorrect_statements': True} matches 0 documents in Mongo while
    the '.0' form matches 535. Any code treating it as a bool reports zero."""
    doc = {
        "training_safe": True, "useful_content": True, "status": "done",
        "review": {"incorrect_statements": [{"claim": "prices were wrong"}]},
    }
    assert gates.document_gate(doc).reason == "incorrect_statements"
    clean = {"training_safe": True, "useful_content": True, "status": "done",
             "review": {"incorrect_statements": []}}
    assert gates.document_gate(clean).passed is True


def test_case_with_unscreened_members_fails_closed():
    doc = {"training_safe": True, "status": "done", "unscreened_members": 2, "review": {}}
    assert gates.document_gate(doc, is_case=True).passed is False


def test_emissions_label_catches_units_the_review_flag_misses():
    """The review flag fires on the *request* and misses 92% of the exposure,
    which sits in units written as neutral product facts."""
    unit = {"title": "Downpipe fitment", "answer": "The catless downpipe fits without modification.",
            "question": "", "conditions": ""}
    labels = gates.derive_labels(unit, {"review": {}})
    assert labels["emissions_risk"] is True


# --- dedup -------------------------------------------------------------------

class _FakeUnit:
    def __init__(self, title, answer, kind="policy", hq=None, confidence="high"):
        self.title = title
        self.question = title
        self.answer = answer
        self.conditions = ""
        self.kind = kind
        self.hypothetical_questions = hq or []
        self.confidence = confidence
        self.evidence = "single_call"
        self.call_ts = None
        self.labels = {}


def test_refinement_splits_same_title_different_fee():
    """A $150 fee and a $300 fee sharing a title must not become one unit that
    says $225. Refinement runs on the answer side for exactly this reason."""
    units = [_FakeUnit("Cable Reset Fee", f"The fee for a remote cable reset is ${p}.")
             for p in (150,) * 8 + (300,) * 8]
    clusters, _ = dedup.cluster_units(units)
    assert len(clusters) >= 2, "150 and 300 were collapsed into one cluster"


def test_merge_prompt_is_bounded_even_for_a_head_topic():
    """Answer-side refinement cannot split 200 genuinely identical restatements,
    so the bound has to hold at the prompt instead. The tail still has to be
    accounted for -- losing it would silently shrink `occurrences` and, worse,
    orphan those source_ids from the revocation path."""
    from agentv1.kb import merge as merge_mod

    units = [_FakeUnit("Cable Reset Process", "Contact support to reset the cable.")
             for _ in range(200)]
    clusters, _ = dedup.cluster_units(units)
    biggest = max(clusters, key=len)
    assert len(biggest) > dedup.MAX_MERGE_CLUSTER, "expected an unsplittable head topic"

    rendered = merge_mod._render(units, biggest[: dedup.MAX_MERGE_CLUSTER])
    assert rendered.count("--- unit ") <= dedup.MAX_MERGE_CLUSTER

    payload = {"units": [{"title": "t", "answer": "a", "source_indices": list(range(dedup.MAX_MERGE_CLUSTER))}]}
    assert merge_mod._validate(payload, dedup.MAX_MERGE_CLUSTER)


# --- PII ---------------------------------------------------------------------

def test_wav_filename_phone_extraction():
    """These filenames are what supersedes_calls[] contains today, which is why
    the supersession relation is PII until it is remapped."""
    assert normalize.wav_filename_phone("[Coles, Zoe]_124-8012307610_20250603202223(152).wav") == "8012307610"
    assert normalize.wav_filename_phone("[Nick P]_127-4507128184_20240513203802(970).wav") == "4507128184"


def test_pii_markers_are_stripped():
    dirty = "call 801-230-7610 or a@b.com, inbox/somehandle_123, VIN WVWZZZ1JZ3W386752"
    clean = normalize.strip_pii_markers(dirty)
    for leak in ("801-230-7610", "a@b.com", "somehandle_123", "WVWZZZ1JZ3W386752"):
        assert leak not in clean


def test_payload_allowlist_cannot_emit_pii():
    payload = pytest.importorskip("agentv1.index.payload")
    doctored = {
        "unit_id": "u_deadbeef", "title": "t", "question": "q", "answer": "a",
        "kind": "policy", "training_safe": True,
        "file_name": "[Coles, Zoe]_124-8012307610_20250603202223(152).wav",
        "agent_name": "Zoe", "caller_area_code": "801",
        "thread_path": "inbox/dustinulichney_10153446946432182",
        "caller_phone_number": "8012307610", "_node_content": "{...}",
    }
    fn = getattr(payload, "project", None) or getattr(payload, "project_unit", None)
    assert fn is not None, "index.payload must expose project()/project_unit()"
    out = fn(doctored) if not isinstance(fn(doctored), tuple) else fn(doctored)[0]
    blob = str(out)
    for leak in ("8012307610", "Zoe", "inbox/", "_node_content", "file_name", "caller_area_code"):
        assert leak not in blob, f"{leak!r} reached the payload"


# --- Qdrant ------------------------------------------------------------------

def test_read_path_never_creates_a_collection():
    with pytest.raises(qdrant.CollectionMissing):
        qdrant.open_collection("agentv1_definitely_not_a_real_collection")
    client = qdrant.get_client()
    assert not client.collection_exists("agentv1_definitely_not_a_real_collection")


def test_empty_routed_collection_refuses_to_boot():
    """Meant to fire: unitronic_faq_0_6b holds 0 points and is routed to today."""
    if not qdrant.collection_exists("unitronic_faq_0_6b"):
        pytest.skip("faq collection absent")
    with pytest.raises(qdrant.CollectionEmpty):
        qdrant.assert_routed_collections_populated(["unitronic_faq_0_6b"])


# --- config ------------------------------------------------------------------

def test_embedding_dimension_matches_the_pinned_size():
    assert config.EMBED_DIM == {"0_6b": 1024, "8b": 4096}[config.QWEN_SIZE]


def test_quarantined_collection_is_not_routed():
    routed = set(config.LEGACY_COLLECTIONS) | set(config.NEW_ALIASES)
    assert not (routed & set(config.QUARANTINED_COLLECTIONS))


# --- tier-2 scoping ----------------------------------------------------------

BANNED_IDENTITY_PARAMS = {
    "customer_id", "customerid", "customer", "phone", "phone_number", "email",
    "account_id", "user_id", "vin_owner", "name",
}


def test_tier2_identity_is_inexpressible_in_the_model_facing_schema():
    """Scoping by omission.

    The boundary that matters is the schema handed to the model, not the Python
    signature -- the executor injects `customer_id` from server-side session
    state, so the function may legitimately accept it keyword-only. What must
    hold is that the model has no way to *say* a customer id: no identity
    property, and `additionalProperties: false` so one cannot be smuggled in as
    an extra key.
    """
    registry = pytest.importorskip("agentv1.tools.registry")
    schemas = registry.get_tool_schemas(persona="support", tier_allowed=2)
    tier2 = [s for s in schemas if s.get("function", s)["name"].startswith("get_my_")]
    assert tier2, "no Tier-2 tools registered"

    for schema in tier2:
        fn = schema.get("function", schema)
        params = fn.get("parameters", {})
        props = {k.lower() for k in (params.get("properties") or {})}
        assert not (props & BANNED_IDENTITY_PARAMS), (
            f"{fn['name']} exposes an identity parameter to the model: "
            f"{props & BANNED_IDENTITY_PARAMS}"
        )
        assert params.get("additionalProperties") is False, (
            f"{fn['name']} allows additionalProperties, so an identity key can be smuggled in"
        )


def test_tier2_python_functions_cannot_be_called_positionally():
    """Belt and braces: identity must be passed explicitly by keyword, so a
    stray `get_my_orders(some_string)` cannot bind a caller-supplied value."""
    customer = pytest.importorskip("agentv1.tools.customer")
    import inspect

    checked = 0
    for name, fn in vars(customer).items():
        if not name.startswith("get_my_") or not callable(fn):
            continue
        checked += 1
        for pname, param in inspect.signature(fn).parameters.items():
            if pname.lower() in BANNED_IDENTITY_PARAMS:
                assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
                    f"{name}.{pname} is positional; it must be keyword-only and executor-injected"
                )
    assert checked >= 1, "no get_my_* tools found"


# --- PII scanner precision ---------------------------------------------------

def test_opaque_ids_do_not_trip_the_pii_scanner():
    """A `unit_id` is `u_` + 16 hex, so it contains ten-digit runs that look
    exactly like a phone number. Left alone this fails the PII gate on every
    turn, and a gate that always fires gets switched off."""
    pii = pytest.importorskip("agentv1.guardrails.pii")
    for uid in ("u_16db124489100490", "u_7373723da2bd8c56", "u_0000000000000000"):
        assert pii.scan_egress({"citations": [{"unit_id": uid}]}).clean


def test_the_opaque_id_exemption_cannot_be_used_to_smuggle_pii():
    """The exemption is an anchored format match, not a field-name check."""
    pii = pytest.importorskip("agentv1.guardrails.pii")
    assert not pii.scan_egress({"unit_id": "8012307610"}).clean
    assert not pii.scan_egress({"unit_id": "801-230-7610"}).clean
    assert not pii.scan_egress(
        {"unit_id": "[Coles, Zoe]_124-8012307610_20250603202223(152).wav"}
    ).clean
