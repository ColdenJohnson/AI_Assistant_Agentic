from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import message_chunk_to_message
from langchain_qwq import ChatQwen

# Mirror llm_client_openrouter.py environment loading to pick up Qwen settings.
load_dotenv()

_QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-flash")
_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
_QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

if not _DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY is not set; check your environment variables.")

_llm_kwargs: Dict[str, Any] = {
    "model": _QWEN_MODEL,
    "max_tokens": 1024,
    "timeout": None,
    "max_retries": 2,
    "api_key": _DASHSCOPE_API_KEY,
}
if _QWEN_BASE_URL:
    _llm_kwargs["base_url"] = _QWEN_BASE_URL # This is critical, as otherwise it defaults to something non China (gives eror Incorrect API key provided)

_llm = ChatQwen(**_llm_kwargs)


@tool
def secret_number() -> int:
    """Return the secret number."""
    return 31

@tool
def secret_phrase(stra) -> str:
    """get a secret phrase after passing in a particular string"""
    return "fuck shit up " + stra


_llm_with_tools = _llm.bind_tools([secret_number, secret_phrase])

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


def _stream_qwen_with_tools(user_text: str) -> Iterator[str]:
    """Invoke Qwen with LangChain tools and stream assistant text."""
    global _HISTORY
    _ensure_history_initialized()
    assert _HISTORY is not None

    # Step 1: add the new human message to history
    _HISTORY.append(HumanMessage(content=user_text))

    # Step 2: first pass – let the model decide on any tool calls
    ai_msg = _llm_with_tools.invoke(_HISTORY)
    print(f"invoking _llm_with_tools got ai_msg: {ai_msg}")
    _HISTORY.append(ai_msg)

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    for tool_call in tool_calls:
        name = tool_call.get("name")
        call_id = tool_call.get("id")
        args = tool_call.get("args") or {}
        print(tool_call)
        print(f"tool_call name: {name}, call_id: {call_id}, args: {args}", flush = True)

        if name == "secret_number":
            result = secret_number.invoke(args)
            _HISTORY.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id,
                )
            )
        if name == "secret_phrase":
            result = secret_phrase.invoke(args)
            _HISTORY.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id,
                )
            )


    # Step 3: second pass – stream the final answer
    full_chunk = None
    for chunk in _llm_with_tools.stream(_HISTORY):
        # Each chunk is an AIMessageChunk; stream incremental text
        text = getattr(chunk, "text", None)
        if text:
            yield text
        full_chunk = chunk if full_chunk is None else full_chunk + chunk

    # Step 4: fold the aggregated chunk back into history as a full message
    if full_chunk is not None:
        final_msg = message_chunk_to_message(full_chunk)
        _HISTORY.append(final_msg)

    _truncate_history()


def stream_chat(messages: List[Dict[str, Any]], usage: bool = False) -> Iterator[str]:
    """
    Minimal drop-in replacement for existing stream_chat:
    - Takes a list of OpenAI-style messages.
    - Uses only the latest user message as input.
    - Returns an iterator over text chunks from Qwen's streaming API.
    """
    _ensure_history_initialized()

    last_user_content: str | None = None
    for msg in messages:
        if msg.get("role") == "user":
            last_user_content = msg.get("content")
    if not last_user_content:
        return iter(())

    # Delegate to real Qwen streaming (token/text chunks), not per-character slicing
    for token in _stream_qwen_with_tools(last_user_content):
        yield token

    # `usage` flag is ignored but kept for compatibility
