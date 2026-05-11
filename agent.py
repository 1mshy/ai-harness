import json
import os
import subprocess
import sys

from openai import OpenAI


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries in a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def list_dir(path: str) -> str:
    return "\n".join(sorted(os.listdir(path)))


def run_bash(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    return f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


HANDLERS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "run_bash": run_bash,
}


SYSTEM_PROMPT = """You are an autonomous agent with access to tools.
Use the tools to investigate, gather information, and complete the user's task.
When you have enough information, respond with a final answer and stop calling tools."""


def run_agent(prompt: str, max_iters: int = 15) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for step in range(max_iters):
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            print(f"\n[step {step}] {name}({args})")

            handler = HANDLERS.get(name)
            if handler is None:
                result = f"Error: unknown tool {name}"
            else:
                try:
                    result = handler(**args)
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"

            preview = result if len(result) < 300 else result[:300] + "..."
            print(f"[result] {preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return "Reached max iterations without a final answer."


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "List the files in the current directory and summarize what this project does."
    print("\n=== FINAL ANSWER ===")
    print(run_agent(task))
