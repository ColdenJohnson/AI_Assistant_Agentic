# Project Structure

Root-level items and responsibilities:

```
README.md  # Project overview, architecture, setup, and file roles
llm_client_langchain.py  # LLM client using LangChain + Qwen (tool calling, streaming)
llm_client_openrouter.py  # OpenRouter/OpenAI-compatible streaming LLM client
orchestrator.py  # Main pipeline wiring wake word -> VAD -> STT -> LLM -> TTS
phase_timer.py  # Timing/latency helper used across the pipeline
stt_faster_whisper.py  # Local STT using faster-whisper
stt_qwen_dashscope.py  # Cloud STT using Qwen ASR (DashScope)
tts_piper.py  # Local TTS using Piper
tts_qwen_dashscope.py  # Cloud/streaming TTS using Qwen (DashScope)
wake_listener.py  # Wake-word listener + VAD-based speech chunking
```
