import os
import sys

from openai import OpenAI


def main(prompt: str) -> None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in one sentence."
    main(prompt)
