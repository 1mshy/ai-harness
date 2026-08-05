"""Interactive chat with the Unitronic sales rep.

    uv run main.py            # chat REPL
    uv run main.py --verbose  # also print tool results, not just tool calls
    uv run main.py "one-shot question"
"""

import re
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import BASE_URL, SYSTEM_PROMPT, build_agent, discover_model

# Gemma on the vLLM box leaks reasoning-channel markers ("<|channel>thought",
# "<channel|>") into message content; strip them before display.
_CHANNEL_RE = re.compile(r"<[^>]*channel[^>]*>")


def clean(text: str) -> str:
    text = _CHANNEL_RE.sub("", text)
    return re.sub(r"^\s*thought\s*\n", "", text).strip()


def render(msg, verbose: bool) -> None:
    if isinstance(msg, AIMessage):
        for call in msg.tool_calls:
            print(f"  -> {call['name']}({call['args']})")
        if msg.content and not msg.tool_calls:
            print(f"\nrep: {clean(str(msg.content))}\n")
    elif isinstance(msg, ToolMessage) and verbose:
        text = str(msg.content)
        preview = text if len(text) < 500 else text[:500] + "…"
        print(f"  <- {preview}")


def chat_turn(agent, messages: list, verbose: bool) -> list:
    """Run one agent turn, printing activity as it streams; returns the full
    updated message history (incl. tool traffic) to carry into the next turn."""
    seen = len(messages)
    state = {"messages": messages}
    for state in agent.stream({"messages": messages}, stream_mode="values"):
        for msg in state["messages"][seen:]:
            render(msg, verbose)
        seen = len(state["messages"])
    return state["messages"]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    verbose = "--verbose" in sys.argv[1:]

    print(f"server: {BASE_URL}")
    print(f"model:  {discover_model()}")
    agent = build_agent()
    messages: list = [SystemMessage(SYSTEM_PROMPT)]

    if args:  # one-shot mode
        messages.append(HumanMessage(" ".join(args)))
        chat_turn(agent, messages, verbose)
        return

    print("Unitronic sales rep — describe your car or ask about products. Ctrl-D to quit.\n")
    while True:
        try:
            user = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"quit", "exit"}:
            break
        messages.append(HumanMessage(user))
        try:
            messages = chat_turn(agent, messages, verbose)
        except Exception as e:  # keep the session alive on model-box hiccups
            print(f"[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
