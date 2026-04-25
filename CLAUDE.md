# Qwen3-ASR Transcription Server

Speech-to-text service using local Qwen3-ASR-1.7B model via vLLM.

## Start

```bash
python server.py                             # ASR + chat API, binds 0.0.0.0:9002
python server.py --port 9000                 # custom port
python server.py --qwenvl                    # + Qwen3-VL-2B-Instruct on VL_PORT (default 9004)
python server.py --qwenvl Qwen/Model-Name    # custom VL model
python server.py --qwenvl --vl-device 1     # VL on GPU 1 (2nd GPU), ASR on GPU 0
python server.py --asr-device 0 --qwenvl --vl-device 1  # explicit GPU assignment
python web_server.py                         # Web UI, binds 0.0.0.0:8001
```

Server loads models in the background; poll `GET /health` until `"status": "ready"`.

## Core Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server — `/transcribe`, `/transcribe-streaming` (WebSocket), `/chat` (SSE), `/vl/health`, `/vl/proxy/{path}` |
| `web_server.py` | Web UI server (port 8001) — serves `web/index.html`, `/api/chat`, `/api/translate`, `/api/extract-context`, `/api/session/*`, `/api/models`, `/api/config` |
| `web/index.html` | Instructor UI — sessions, AI chat (with image input), live mic transcription, auto-translation |
| `web/viewer.html` | Viewer UI — live transcription + translations via SSE, AI chat |
| `client_file.py` | **Primary client** — vocal extraction → resample → VAD → streaming ASR (outputs TXT format) |
| `client_mic.py` | Live microphone streaming client with VAD-based utterance detection |
| `process_video.py` | Extract audio from video, start server, transcribe, save JSON |
| `Qwen3-ASR-1.7B/` | ASR model weights |
| `Qwen3-ForcedAligner-0.6B/` | Forced-aligner model weights (word-level timestamps) |

## Web UI (`web/index.html`)

Three-panel layout served from `web_server.py` at `http://localhost:8001`:

- **Left**: Session list — auto-saved to `localStorage`, double-click to rename, ✕ to delete
- **Middle**: AI chat — Claude / Gemini / Mistral / Local VL; image attachment (🖼) visible only when `Local VL` selected; image thumbnails shown in history, clickable to enlarge (lightbox)
- **Right**: Live mic transcription — VAD-based, language selector, **auto-translate** target selector (shown next to source language when VL available), PDF/MD/TXT context upload (📎), export (⇩)

**Panel divider**: Draggable 4px divider between chat and transcription panels; width saved to `localStorage`.

**Auto-translation**: each new segment is auto-translated if target language ≠ source language and VL is available. Result stored in `entry.translated`, broadcast to viewers via `pushToServer()`. Manual `⇄ Translate` / `✕ Delete` buttons appear at bottom-right of each entry on hover.

**Chat backend**: `POST /api/chat` on `web_server.py`. Server URL configurable via ⚙ settings button (persisted to `localStorage` and `POST /api/config`).

**Microphone**: requires a **secure context** — access via `http://localhost:8001`, not an IP over HTTP.

**Audio source selector**: 🎙 Mic (echo/noise cancellation on) vs 🔊 Speaker (all processing off for line-in). Changes VAD thresholds: Mic uses `silenceTrigger=20`, `energyThreshold=0.018`, `maxUttFrames=400` (~12 s force-flush); Speaker uses `silenceTrigger=30`, `energyThreshold=0.006`, `maxUttFrames=400` (~12 s force-flush).

**Streaming**: WebSocket opened at VAD speech-start; partial results shown as the model decodes. Partials broadcast to viewers via `pushPartial`.

**VL proxy**: all VL requests go through `GET|POST /vl/proxy/{path}` on the main server — no separate tunnel needed for `VL_PORT`.

**Pop-out button** (⧉): opens `/viewer?popout=1` in a 400×620 px window — live transcription feed that can be pinned on top via OS window manager.

**Mermaid rendering**: `scheduleRenderMermaid()` renders ` ```mermaid ``` ` fenced blocks. `normalizeMermaid()` (called inside `renderMd`) auto-wraps bare mermaid diagrams (lines starting with `graph TD/LR`, `flowchart`, `sequenceDiagram`, etc.) that VL models output as plain text.

## Viewer Page (`web/viewer.html`)

Served at `http://localhost:8001/viewer`.

- **Left**: AI chat (instructor's API keys — students need no keys; viewer stores its own keys in `localStorage` via ⚙ settings panel)
- **Right**: Live transcription via SSE — segments, partials, and translations all displayed; `entry.translated` patched into existing DOM entries when broadcast arrives after initial render; ⇄ Trans toggle shows/hides translations
- Export button (⇩) downloads TXT; auto-reconnects on SSE drop
- When opened as pop-out (`?popout=1`), shows transcription panel only

## Viewer Broadcast System

Session state held in-memory in `web_server.py`:

- `POST /api/session/push` — instructor page posts full session (name + segments + seq)
- `POST /api/session/partial` — live partial text pushed per frame
- `GET /api/session/stream` — SSE stream consumed by all viewer tabs; heartbeat every 20 s
- Sequence numbers deduplicate updates; restarting `web_server.py` clears state; viewers reconnect automatically

## VL Model (`--qwenvl`)

Started as a separate vLLM OpenAI-compatible subprocess on `VL_PORT` (default 9004).

- GPU memory auto-sized from actual free GPU at startup:
  - **Shared GPU** (no `--vl-device`): capped at `_VL_ESTIMATED_GB_4K = 8 GB` → leaves ~1.7 GB KV cache → `max_model_len = 2048`
  - **Dedicated GPU** (`--vl-device`): capped at `_VL_MAX_GB = 20 GB` → leaves ~2.6 GB KV cache → `max_model_len = 4096` on ≥10 GB GPU
- Minimum GPU free memory at startup: ~20 GB on a single shared GPU; ~10 GB each on separate GPUs (tested on 2× 11 GB)
- Subprocess env strips ASR CPU vars (`VLLM_TARGET_DEVICE=cpu`, etc.) so VL runs on GPU
- Accessed via `/vl/proxy/...` on main server — `web_server.py` never connects to `VL_PORT` directly

## client_file.py Pipeline

```
input audio
  → demucs (htdemucs --two-stems vocals)   # isolate human voice
  → resample to 16kHz mono                  # scipy.signal.resample_poly
  → WebRTC VAD aggressiveness=2             # split into speech segments
  → stream each segment over WebSocket      # with optional vocabulary context
  → <stem>.txt                             # one text line per segment with timestamp
```

```bash
python client_file.py openclaw.mp3 --context foo.md --language en
```

## client_mic.py

VAD-based real-time microphone transcription. Each detected utterance is sent as a separate WebSocket session.

```bash
python client_mic.py                   # English, localhost:9002
python client_mic.py -l zh             # Chinese
python client_mic.py -v                # verbose VAD debug output
python client_mic.py -e ws://host:port/transcribe-streaming
```

**VAD settings** (tunable at top of file):
- `VAD_AGGRESSIVENESS = 3` — level 3 filters background noise best
- `SILENCE_END_FRAMES = 33` — ~1 s of silence ends an utterance
- `ENERGY_THRESHOLD = 0.018` — RMS threshold; raise if background noise triggers false starts

## server.py `/chat` Endpoint

Uses the already-loaded ASR model for text chat:

```python
asr = models["asr"]           # Qwen3ASRModel instance
asr.model                     # underlying vllm.LLM — has .generate()
asr.processor.tokenizer       # has .apply_chat_template()
```

Quality is limited — Qwen3-ASR-1.7B is trained for audio→text, not chat.

## Server Env Vars

| Variable | Default | Notes |
|---|---|---|
| `ASR_MODEL_NAME` | `Qwen3-ASR-1.7B` | local dir or HF model id |
| `ALIGNER_MODEL_NAME` | `Qwen3-ForcedAligner-0.6B` | |
| `GPU_MEMORY_UTILIZATION` | auto | vLLM GPU fraction for ASR; auto targets ~6 GB |
| `VL_GPU_MEMORY_UTILIZATION` | auto | vLLM GPU fraction for VL; 8 GB cap on shared GPU, 20 GB cap on dedicated GPU |
| `VL_MAX_MODEL_LEN` | auto | VL context length; 2048 on shared GPU, 4096 on dedicated GPU ≥10 GB |
| `VL_PORT` | `9004` | Internal port for VL subprocess |
| `ASR_DEVICE` | `""` | GPU index for ASR model (overridden by `--asr-device`) |
| `VL_DEVICE` | `""` | GPU index for VL subprocess (overridden by `--vl-device`); empty = share GPU with ASR |
| `MAX_NEW_TOKENS` | `8192` | |
| `ENABLE_ASR_MODEL` | `true` | set `false` to skip |
| `ENABLE_ALIGNER_MODEL` | `false` | set `true` to enable word-level timestamps |
| `ENABLE_PREFIX_CACHING` | `true` | vLLM APC — caches KV blocks for shared prefix |
| `ASR_PORT` | `9002` | default port; overridden by `--port` CLI arg |

## Web Server CLI Args

| Arg | Default | Description |
|---|---|---|
| `--asr-host` | `localhost` | ASR server host |
| `--asr-port` | `9002` | ASR server port |
| `--port` | `8001` | Web server port |

## Web Server API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve `web/index.html` (no-cache) |
| `/viewer` | GET | Serve `web/viewer.html` (no-cache) |
| `/api/chat` | POST | SSE chat stream — routes to Claude/Gemini/Mistral/local-vl |
| `/api/translate` | POST | Translate a text segment via VL model |
| `/api/extract-context` | POST | Parse PDF/MD/TXT; returns first 4000 chars |
| `/api/models` | GET | List available models; `?all=true` includes disabled ones |
| `/api/config` | POST | Save ASR host/port to persistent JSON |
| `/api/session/push` | POST | Instructor pushes full session state to broadcast |
| `/api/session/partial` | POST | Instructor pushes live partial text |
| `/api/session/stream` | GET | SSE stream for viewers |

## Key Notes

- **Audio format for streaming**: PCM 16-bit signed little-endian, 16kHz mono. Send `{"type":"start","format":"pcm_s16le","sample_rate_hz":16000}` before audio bytes, then `{"type":"stop"}`.
- **Forced alignment**: pass `?forced_alignment=true` to `/transcribe` for word-level timestamps.
- **Event recordings**: always run demucs vocal extraction first — background music causes hallucination.
- **`Qwen3ASRModel` does NOT have `.generate()`** — use `models["asr"].model.generate()`.
- **Server restart required** after any change to `server.py` endpoints.
- **Prefix caching (APC)**: context + system prompt tokens cached after first utterance. Disable with `ENABLE_PREFIX_CACHING=false` if unsupported.
- **Viewer broadcast relay**: session state held in-memory in `web_server.py`; restarting clears it. Viewers reconnect automatically via SSE.
- **VL subprocess env**: `VLLM_TARGET_DEVICE=cpu` (set for ASR CPU backend) is stripped before launching VL subprocess, so VL always runs on GPU.
- **Translation history**: `entry.translated` stored in session alongside `entry.text`; persisted to `localStorage` and broadcast via SSE so viewers receive translations.
- **HTML caching**: `web/index.html` and `web/viewer.html` served with `Cache-Control: no-cache` headers.
- **Context stripping** (`strip_prompt()` in `server.py`): removes instruction prompts from ASR output when a vocabulary context is provided. Uses sentinel tags, instruction markers, exact-line matching, and prefix detection as cascading fallbacks.
- **API keys**: server reads from env vars; clients can override per-request via `api_keys` field in `/api/chat`. Viewer stores its own keys in `localStorage` only.
