"""Thin provider abstraction so generation.py and eval can swap openai <-> anthropic via config only."""
import os
from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


def chat(provider: str, model: str, system: str, user: str, temperature: float = 0.0) -> ChatResult:
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return ChatResult(
            text=resp.choices[0].message.content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            default_headers={"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]},)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return ChatResult(
            text=resp.content[0].text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            model=model,
        )

    raise ValueError(f"unknown LLM provider: {provider}")
