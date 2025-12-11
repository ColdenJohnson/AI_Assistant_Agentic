# AI_Assistant_Agentic

# Raspberry Pi AI Voice Assistant

Always-on, wake-word voice assistant running on Raspberry Pi, with switchable **local vs. cloud** Speech-to-Text (STT), Text-to-Speech (TTS), and LLM backend. Designed to be usable even with unreliable networks and behind the Great Firewall (China-compatible), even without VPN. Highly customizable tooling capabilities, to give your personal assistant access to anything you want!

---

## Table of Contents

1. [Project Description](#project-description)  
2. [Architecture Diagram](#architecture-diagram)  
3. [System Overview](#system-overview)  
4. [Setup](#setup)  
5. [Latency](#latency)
6. [Tool Integrations](#tool-integrations)  
7. [Glossary](#glossary)

---

## Project Description

The goal of this project was to build a an AI assistant that:

- Runs on a **Raspberry Pi** with a USB microphone and speaker/DAC.
- Listens passively for a **wake word**, then records your speech.
- Converts speech to text (**STT**), sends it to a large language model (**LLM**), and reads the reply out loud (**TTS**).
- Can switch between **local** and **cloud** models depending on latency and network conditions.
- Is structured enough that custom tools (home automation, APIs, MCP Servers etc.) can be easily integrated.

All of this is implemented and working end-to-end: you can say the wake word, talk to the assistant, and get streaming spoken replies.

---

## Architecture Diagram

### Diagram
![AI Assistant Architecture](https://raw.githubusercontent.com/ColdenJohnson/Drawio_Diagrams/main/AIAssistantArchitecture.png)

### End-to-End Flow

1. Custom **Wake word** (“Paxton”) is detected by Picovoice **Porcupine**. This can be swapped out by downloading your own picovoice file and replacing the local one (pax-ton_en_raspberry-pi_v3_0_0.ppn).
2. Once awake, **Cobra VAD** (Voice Activity Detection) decides when you are speaking and when you’ve stopped.
3. Audio during speech is:
   - Sent to **local Whisper (faster-whisper)** on the Pi, **or**
   - Streamed to **Qwen ASR** in the cloud via DashScope.
4. The resulting text is sent to **Qwen3** (via LangChain / Dashscope client) as the LLM.
5. The LLM streams text tokens back.
6. Text is converted to audio using:
   - **Local Piper TTS** on the Pi, **or**
   - **Qwen Streaming TTS** in the cloud.
7. Audio is played out through the DAC + speaker while the rest of the response is still being generated (true streaming)

---

## System Overview

### Hardware

| Component | Example Used | Role |
|----------|--------------|------|
| Single-board computer | Raspberry Pi (tested on Pi 5) | Runs wake-word, STT, LLM client, and TTS pipeline. |
| Microphone | USB mic | Captures user speech for Porcupine / VAD / STT. |
| DAC + Speaker | USB / HAT DAC + powered speaker | Plays assistant responses with low noise and stable volume. |
| Storage | microSD / SSD | Holds OS, Python environment, models (Whisper, Piper, etc.). |
| Network | Ethernet or Wi-Fi | Used for cloud STT/TTS/LLM (Qwen, OpenRouter) when enabled. |

### Software Pipeline

| Stage | Local Option | Cloud Option | Main Files |
|-------|--------------|-------------|-----------|
| Wake Word + VAD | Picovoice Porcupine + Cobra | N/A (always local) | `wake_listener.py`, Porcupine `.ppn` model |
| STT | `faster-whisper` | Qwen ASR (DashScope realtime) | `stt_faster_whisper.py`, `stt_qwen_dashscope.py` |
| LLM | Qwen3 via DashScope / OpenRouter | Same, but can route via OpenRouter | `llm_client_langchain.py`, `llm_client_openrouter.py` |
| TTS | Piper | Qwen TTS (DashScope realtime) | `tts_piper.py`, `tts_qwen_dashscope.py` |
| Orchestration | — | — | `orchestrator.py`, `wake_listener.py`, `phase_timer.py` |

The central brain of the system is `orchestrator.py`. It wires everything together:

- Chooses **local vs cloud STT/TTS** based on environment variables.
- Streams LLM tokens into TTS so speech starts quickly.
- Logs timing checkpoints using `PhaseTimer` for latency debugging.

### File Roles

| File | Purpose |
|------|---------|
| `orchestrator.py` | Entry point. Connects wake-word → VAD → STT → LLM → TTS; manages streaming TTS via `StreamingSpeaker`. |
| `wake_listener.py` | Listens for the wake word using Porcupine and segments speech using Cobra VAD; yields full utterances. |
| `stt_faster_whisper.py` | Local STT using `faster-whisper` (tiny model, quantized) to transcribe VAD-delimited chunks. |
| `stt_qwen_dashscope.py` | Cloud STT via Qwen realtime ASR with server-side VAD; feeds transcripts back to `orchestrator`. |
| `tts_piper.py` | Local TTS using Piper; plays PCM via `sox` + `aplay`. |
| `tts_qwen_dashscope.py` | Streaming Qwen TTS client; exposes `QwenStreamingTtsSession` used by `orchestrator`. |
| `llm_client_langchain.py` | Qwen LLM client using LangChain. Supports streaming tokens and basic tool calling. |
| `llm_client_openrouter.py` | OpenRouter/OpenAI-compatible streaming client; can also hit DashScope’s OpenAI-style endpoint when `USE_QWEN=true`. |
| `phase_timer.py` | Tiny helper to log checkpoints and measure latency through the pipeline. |
| `test.py` | Offline harness: routes a WAV file through the same wake-word + STT path for debugging. |
| `pvrecorderavailabledevices.py` | Lists available audio input devices (helpful when configuring the Pi). |
| `see_open_router_models.py` | Quick hack script to inspect available models on OpenRouter. Contains API key → treat with caution. |
| `backlog_note.txt` | Notes on latency issues, STT chunking, and tuning ideas. |
| `*.onnx`, `*.ppn`, `*.wav` | Model assets (Piper voice, Porcupine wake-word), and example audio files. |

---

## Setup

To install, you should be familiar with basic Raspberry Pi setup.

Place access keys in all relevant .env variables. Then, run the following commands on the cloned project directory.

```bash
sudo apt-get update
sudo apt-get install -y sox alsa-utils

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

source .venv/bin/activate
python orchestrator.py
```

---

## Latency

| Layer | Local Option                                         | Cloud Option         | Pros (Local)                                                              | Pros (Cloud)                                                              |
| ----- | ---------------------------------------------------- | -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| STT   | faster-whisper on Pi                               | Qwen ASR (DashScope) | No wifi requirements, free, predictively stable. | Faster response time, better transcription in noisy environment. |
| LLM   | Qwen or other model via cloud. | Same                 | N/A – always remote (could be configured for a VM/local computer.                                            | Stronger reasoning and language abilities than you can fit on the Pi.       |
| TTS   | Piper on Pi                                          | Qwen TTS (DashScope) | No wifi requirements, free, stable.         | Better and more customizable voices, lower latency.        |

For latency and firewall reasons in China, cloud options have been switched from Openrouter (Claude, etc.) to instead run with the Qwen family of models (Beijing data center). This has made for much faster response times and better overall performance. The local options of STT and TTS are at times useful, but it is faster to do cloud processing for these steps given relatively low ping to cloud models. `Phase_timer.py` is accessible throughout to measure this accurately.

---

## Tool Integrations

This project currently already has basic LangChain support, and is structurally ready for any custom Python-designable or MCP tool that needs to be integrated. This allows for file access, data retrieval, and arbitrary Python functions to all be exposed to the LLM.

To create a new tool, simply do the following:

- Write a small Python function (ex: `def foo(): ...)`
-Add an `@tool` decorator describing the function (name and input type)
-Add it to the tool list passed to the LLM client

The AI assistant will immediately have access to this new tool, and will dynamically call it when logically useful to solve problems.

---

## Glossary

| Term | Meaning |
|------|---------|
| **STT (Speech-to-Text)** | Converting spoken audio into written text. Implemented with **Whisper (faster-whisper)** locally and **Qwen ASR** in the cloud. |
| **TTS (Text-to-Speech)** | Converting written text into spoken audio. Implemented with **Piper** locally and **Qwen TTS** in the cloud. |
| **LLM (Large Language Model)** | The “brain” that understands your text and generates replies (here: **Qwen3** via DashScope / OpenRouter). |
| **Wake Word** | A specific phrase that “wakes up” the assistant (e.g., “Pax-ton”), handled by **Porcupine**. |
| **VAD (Voice Activity Detection)** | Algorithm that detects when the user is actually speaking vs. background silence; here: **Cobra VAD**. |
| **Latency** | The delay between you finishing a sentence and hearing the assistant start speaking back. |
| **Pi** | Short for **Raspberry Pi**, the small, low-power computer running all of this. |




