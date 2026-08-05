import json
import os

import requests
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are the metrics analyser for a running vllm instance."""

VLLM_HOST = "http://10.150.0.30:1234"

# Flip to True to watch tool-call arguments assemble token by token.
# Requires vLLM to be launched with --enable-auto-tool-choice and a
# --tool-call-parser that emits streaming deltas. If it stays silent,
# leave this off; the "updates" stream still shows the call before it runs.
SHOW_LIVE_TOOL_ARGS = True


@tool
def curl_metrics() -> str:
    """Pulls the metrics from the vllm server"""
    req = requests.get(f"{VLLM_HOST}/metrics", timeout=10)
    if req.status_code < 300:
        return req.text
    return "Sorry there are no metrics available at this time."


def chunk_text(chunk) -> str:
    """AIMessageChunk.content is a str for most providers, a list for some."""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def main() -> None:
    # llm = ChatOpenAI(
    #     model="nvidia/Gemma-4-31B-IT-NVFP4",
    #     base_url=f"{VLLM_HOST}/v1",
    #     api_key="fd",
    #     streaming=True,  # without this the model node may buffer the full reply
    # )

    llm = ChatOpenAI(
        model="z-ai/glm-5.2",
        base_url="https://openrouter.ai/api/v1",
        extra_body={"provider": {"only": ["novita/fp8"], "allow_fallbacks": False, "zdr": True}},
        api_key=str(os.getenv("OPENROUTER_API_KEY")),
        streaming=True,  # without this the model node may buffer the full reply
    )

    agent = create_agent(
        model=llm,
        tools=[curl_metrics],
        system_prompt=SYSTEM_PROMPT,
    )

    # Keeps the conversation across turns; your original loop was stateless.
    history: list = []

    while True:
        try:
            question = input("\nAsk a question about metrics: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        history.append({"role": "user", "content": question})

        for mode, payload in agent.stream(
            {"messages": history},
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                if not isinstance(chunk, AIMessageChunk):
                    continue

                if SHOW_LIVE_TOOL_ARGS:
                    for tcc in chunk.tool_call_chunks or []:
                        if tcc.get("name"):
                            print(f"\n  → {tcc['name']}(", end="", flush=True)
                        if tcc.get("args"):
                            print(tcc["args"], end="", flush=True)
                    
                text = chunk_text(chunk)
                if text:
                    print(text, end="", flush=True)

            elif mode == "updates":
                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages") or []:
                        history.append(msg)

                        if msg.type == "ai" and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(
                                    f"\n  [call] {tc['name']}({tc['args']})",
                                    flush=True,
                                )
                        elif msg.type == "tool":
                            body = str(msg.content)
                            preview = body[:100].replace("\n", " ")
                            print(
                                f"  [result] {msg.name} — "
                                f"{len(body)} chars: {preview}...",
                                flush=True,
                            )

        print()

        with open("data.json", "w") as f:
            json.dump(
                [
                    m.model_dump() if hasattr(m, "model_dump") else m
                    for m in history
                ],
                f,
                indent=2,
                default=str,
            )


if __name__ == "__main__":
    main()
