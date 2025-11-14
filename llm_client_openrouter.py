# ~/Projects/Assistant/llm_stream_openrouter.py
from __future__ import annotations
import os
from typing import Iterable, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def _client() -> OpenAI:
    # Default headers per OpenRouter attribution guidance
    headers = {}
    ttl = os.getenv("OPENROUTER_TITLE")

    if ttl:
        headers["X-Title"] = ttl

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers=headers or None,
    )

def stream_chat(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    usage: bool = False,           # include final usage chunk if desired
) -> Iterable[str]:
    """
    Yields text deltas as they stream. Final chunk may contain usage info if enabled.
    """
    client = _client()
    req_kwargs: Dict[str, Any] = {
        "model": os.getenv("OPENROUTER_MODEL"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if usage:
        # OpenRouter returns usage in the last SSE event when enabled
        req_kwargs["extra_body"] = {"usage": {"include": True}}  # optional
    stream = client.chat.completions.create(**req_kwargs)

    for chunk in stream:
        # Comments like ": OPENROUTER PROCESSING" may appear; SDK filters them.
        # Normal content:
        delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
        if delta:
            yield delta
        # Optional: handle usage on the last event
        if usage and hasattr(chunk, "usage") and chunk.usage:
            # Example: expose token/cost stats to logs/metrics
            pass
