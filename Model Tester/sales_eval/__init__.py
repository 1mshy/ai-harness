"""Sales-agent capability evaluation suite for an LM Studio endpoint.

Companion to ``stress_test.py``: that one measures *how fast* the box serves
tokens; this one measures *whether the model can actually do the job* of a
sales agent — call CRM tools, answer from a price sheet, read customer
sentiment, summarize a long phone call, qualify a lead, handle an objection.

Run it with ``python run_evals.py`` (see that file / the README).
"""

from __future__ import annotations

from .harness import (
    Check,
    GradeResult,
    RunOutcome,
    Scenario,
    ToolCall,
    all_scenarios,
    contains_all,
    contains_any,
    extract_json,
    find_number,
    get_scenario,
    has_number,
    label_match,
    norm,
    register,
    run_scenario,
)

__all__ = [
    "Check", "GradeResult", "RunOutcome", "Scenario", "ToolCall",
    "all_scenarios", "contains_all", "contains_any", "extract_json",
    "find_number", "get_scenario", "has_number", "label_match", "norm",
    "register", "run_scenario",
]
