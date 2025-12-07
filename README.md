# AI_Assistant_Agentic

Qwen (DashScope) toggle:
- Set `USE_QWEN=true` to route chat calls to DashScope's OpenAI-compatible endpoint (defaults to OpenRouter when unset/false).
- Provide `DASHSCOPE_API_KEY` (or `QWEN_API_KEY`), optional `QWEN_MODEL` (default `qwen-flash`), and optional `QWEN_BASE_URL` (default `https://dashscope.aliyuncs.com/compatible-mode/v1`).

Qwen realtime ASR (optional):
- Set `USE_REMOTE_STT=true` to send each locally VAD-delimited utterance to DashScope realtime ASR instead of local Whisper.
- Optional `QWEN_ASR_MODEL` (default `qwen3-asr-flash-realtime`) and `QWEN_ASR_BASE_URL` (default `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`); reuses `DASHSCOPE_API_KEY`.
- Install `websocket-client` when using the remote ASR toggle.

# TODO: Whisper-streaming library exists -- can simply stream whisper with it

# TODO: sentences are still being flushed in chunks: should be flushed smaller than taht? I believe the queue can handle just appending words to it dierctly as they become available (verify)