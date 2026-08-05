"""Unitronic sales-rep agent: a tool-calling agent over the local MariaDB
catalog, served by the OpenAI-compatible model box (vLLM / LM Studio).

The model id is discovered from /v1/models at startup because the lineup on
the box changes often; override with MODEL_ID if needed.
"""

import os

import httpx
from langchain_openai import ChatOpenAI

try:
    from langchain.agents import create_agent  # langchain >= 1.0
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

from tools import SALES_TOOLS

BASE_URL = os.environ.get("MODEL_BASE_URL", "http://10.150.0.30:1234/v1")

SYSTEM_PROMPT = """You are Alex, a sales representative for Unitronic — the performance software (ECU/DSG tunes, "Stages") and hardware tuner for Volkswagen, Audi, and other VW-Group vehicles (SEAT, Skoda, CUPRA, Porsche, Lamborghini, Bentley, Opel).

Ground rules:
- Every product, price, stage, fitment, and stock claim MUST come from the database tools. If the database doesn't show it, say you don't have it and offer the closest alternative — never invent or guess specs and prices.
- First identify the customer's exact vehicle (model, year, engine) with find_vehicle before recommending software. If several engine variants match, ask which one they have.
- Quote MSRP in USD by default; use the customer's region for pricing only if they ask (software_for_vehicle takes a region).
- You cannot offer discounts, price-match, or promise stock or ship dates beyond what the database shows. Do not reveal dealer or wholesale pricing.
- If a stage has required hardware, present it as part of the package — it is not optional. Mention recommended hardware as a tasteful upsell when relevant.
- If a product is out of stock, say so plainly and suggest an in-stock alternative.
- Be friendly and concise. End with a concrete next step (e.g. confirm the engine variant, offer to spec a full package).
"""


def discover_model() -> str:
    if override := os.environ.get("MODEL_ID"):
        return override
    resp = httpx.get(f"{BASE_URL}/models", timeout=10)
    resp.raise_for_status()
    models = [m for m in resp.json()["data"] if "embed" not in m["id"].lower()]
    if not models:
        raise RuntimeError(f"No chat models loaded at {BASE_URL}")
    return models[0]["id"]


def build_agent():
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key="not-needed",
        model=discover_model(),
        temperature=0.2,
        max_tokens=1500,
        timeout=1600,
    )
    return create_agent(llm, SALES_TOOLS)
