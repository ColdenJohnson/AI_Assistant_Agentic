from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_qwq import ChatQwen

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not _DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY is not set; check your environment or .env file.")

_llm = ChatQwen(
    model="qwen-flash",
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


@tool
def secret_number() -> int:
    """Return the secret number 31."""
    return 31


_llm_with_tools = _llm.bind_tools([secret_number])

_HISTORY: List[Any] | None = None


def _ensure_history_initialized() -> None:
    global _HISTORY
    if _HISTORY is None:
        _HISTORY = [
            SystemMessage(content="You are a home assistant. Be concise."),
        ]


def _truncate_history(max_messages: int = 20) -> None:
    """Keep simple short-term memory: system + last (max_messages-1) messages."""
    global _HISTORY
    if _HISTORY is None:
        return
    if len(_HISTORY) <= max_messages:
        return
    system_msg = _HISTORY[0]
    rest = _HISTORY[1:]
    _HISTORY = [system_msg] + rest[-(max_messages - 1) :]


def _run_qwen_with_tools(user_text: str) -> str:
    """Invoke Qwen with LangChain tools and return final assistant text."""
    global _HISTORY
    _ensure_history_initialized()
    assert _HISTORY is not None

    _HISTORY.append(HumanMessage(content=user_text))

    ai_msg = _llm_with_tools.invoke(_HISTORY)
    _HISTORY.append(ai_msg)

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    for tool_call in tool_calls:
        name = tool_call.get("name")
        call_id = tool_call.get("id")
        args = tool_call.get("args") or {}

        if name == "secret_number":
            result = secret_number.invoke(args)
            _HISTORY.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id,
                )
            )

    final_msg = _llm_with_tools.invoke(_HISTORY)
    _HISTORY.append(final_msg)

    _truncate_history()

    content = final_msg.content
    if isinstance(content, str):
        return content
    return str(content)


def stream_chat(messages: List[Dict[str, Any]], usage: bool = False) -> Iterator[str]:
    """
    Minimal drop-in replacement for existing stream_chat:
    - Takes a list of OpenAI-style messages.
    - Uses only the latest user message as input.
    - Returns an iterator over text chunks (here: characters).
    """
    _ensure_history_initialized()

    last_user_content: str | None = None
    for msg in messages:
        if msg.get("role") == "user":
            last_user_content = msg.get("content")
    if not last_user_content:
        return iter(())

    final_text = _run_qwen_with_tools(last_user_content)

    for ch in final_text:
        yield ch

    # `usage` flag is ignored but kept for compatibility
