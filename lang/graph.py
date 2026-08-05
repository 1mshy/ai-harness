"""A small LangGraph workflow: plan -> write -> review, looping back to
write until the reviewer approves or MAX_REVISIONS is hit.

Talks to any OpenAI-compatible server (vLLM / LM Studio). The model id is
discovered from /v1/models at startup because the lineup on the box changes
often; override with MODEL_ID if needed.
"""

import json
import os
from typing import TypedDict

import httpx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

BASE_URL = os.environ.get("MODEL_BASE_URL", "http://10.150.0.30:1234/v1")
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", "2"))


def discover_model() -> str:
    if override := os.environ.get("MODEL_ID"):
        return override
    resp = httpx.get(f"{BASE_URL}/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError(f"No models loaded at {BASE_URL}")
    return models[0]["id"]


def extract_last_json(text: str) -> dict | None:
    """Grab the LAST balanced {...} in the text. Some local models (e.g.
    deepseek-v4) dump chain-of-thought prose before the JSON, so the first
    brace is not trustworthy."""
    depth, start, last = 0, None, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                last = text[start : i + 1]
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    critique: str
    approved: bool
    revision: int


def build_graph():
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key="not-needed",
        model=discover_model(),
        temperature=0.7,
        max_tokens=1024,
        timeout=120,
    )

    def plan(state: State) -> dict:
        msg = llm.invoke(
            [
                ("system", "You are a writing planner. Produce a short bullet outline (3-5 bullets) for a ~200 word answer. Outline only, no prose."),
                ("user", f"Topic: {state['topic']}"),
            ]
        )
        return {"outline": msg.content}

    def write(state: State) -> dict:
        user = f"Topic: {state['topic']}\n\nOutline:\n{state['outline']}"
        if state.get("critique"):
            user += (
                f"\n\nYour previous draft:\n{state['draft']}"
                f"\n\nReviewer feedback to address:\n{state['critique']}"
            )
        msg = llm.invoke(
            [
                ("system", "You are a concise technical writer. Write a ~200 word answer following the outline. If reviewer feedback is given, revise the previous draft to address it."),
                ("user", user),
            ]
        )
        return {"draft": msg.content, "revision": state["revision"] + 1}

    def review(state: State) -> dict:
        msg = llm.invoke(
            [
                ("system", 'You are a strict reviewer. Judge the draft for accuracy, clarity and completeness. Reply with a single JSON object, nothing else: {"approved": true/false, "critique": "one or two sentences"}'),
                ("user", f"Topic: {state['topic']}\n\nDraft:\n{state['draft']}"),
            ]
        )
        verdict = extract_last_json(msg.content)
        if verdict is None:
            # Unparseable review — accept the draft rather than loop blindly.
            return {"approved": True, "critique": "(reviewer output unparseable)"}
        return {
            "approved": bool(verdict.get("approved", False)),
            "critique": str(verdict.get("critique", "")),
        }

    def after_review(state: State) -> str:
        if state["approved"] or state["revision"] > MAX_REVISIONS:
            return END
        return "write"

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("write", write)
    g.add_node("review", review)
    g.add_edge(START, "plan")
    g.add_edge("plan", "write")
    g.add_edge("write", "review")
    g.add_conditional_edges("review", after_review, {"write": "write", END: END})
    return g.compile()
