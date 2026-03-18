# Qwen3-ASR Transcription Server

Speech-to-text service using local Qwen3-ASR-1.7B model via vLLM.

## Start

```bash
python server.py                             # ASR + chat API, binds 0.0.0.0:9002
python server.py --port 9000                 # custom port
python server.py --qwenvl                    # + Qwen3-VL-2B-Instruct on VL_PORT (default 9004)
python server.py --qwenvl Qwen/Model-Name    # custom VL model
python web_server.py                         # Web UI, binds 0.0.0.0:8001
```

Server loads models in the background; poll `GET /health` until `"status": "ready"`.

## Core Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server — `/transcribe`, `/transcribe-streaming` (WebSocket), `/chat` (SSE), `/vl/health`, `/vl/proxy/{path}` |
| `web_server.py` | Web UI server (port 8001) — serves `web/index.html`, `/api/chat`, `/api/translate`, `/api/extract-context` |
| `web/index.html` | Instructor UI — sessions, AI chat (with image input), live mic transcription, auto-translation |
| `web/viewer.html` | Viewer UI — live transcription + translations via SSE, AI chat |
| `client_file.py` | **Primary client** — vocal extraction → resample → VAD → streaming ASR (outputs TXT format) |
| `client_mic.py` | Live microphone streaming client with VAD-based utterance detection |
| `process_video.py` | Extract audio from video, start server, transcribe, save JSON |
| `Qwen3-ASR-1.7B/` | ASR model weights |
| `Qwen3-ForcedAligner-0.6B/` | Forced-aligner model weights (word-level timestamps) |

## Web UI (`web/index.html`)

Three-panel layout served from `web_server.py` at `http://localhost:8001`:

- **Left**: Session list — auto-saved to `localStorage`, double-click to rename
- **Middle**: AI chat — Claude / Gemini / Mistral / Local VL; image attachment (🖼) visible only when `Local VL` selected; image thumbnails shown in history, clickable to enlarge (lightbox)
- **Right**: Live mic transcription — VAD-based, language selector, **auto-translate** target selector (shown next to source language when VL available), PDF/MD context upload, export (↓)

**Auto-translation**: each new segment is auto-translated if target language ≠ source language and VL is available. Result stored in `entry.translated`, broadcast to viewers via `pushToServer()`. Manual `⇄ Translate` / `✕ Delete` buttons appear at bottom-right of each entry on hover.

**Chat backend**: `POST /api/chat` on `web_server.py`. Server URL configurable via ⚙ settings button.

**Microphone**: requires a **secure context** — access via `http://localhost:8001`, not an IP over HTTP.

**Streaming**: WebSocket opened at VAD speech-start; partial results shown as the model decodes. Partials broadcast to viewers via `pushPartial`.

**VL proxy**: all VL requests go through `GET|POST /vl/proxy/{path}` on the main server — no separate tunnel needed for `VL_PORT`.

## Viewer Page (`web/viewer.html`)

Served at `http://localhost:8001/viewer`.

- **Left**: AI chat (instructor's API keys — students need no keys)
- **Right**: Live transcription via SSE — segments, partials, and translations all displayed; `entry.translated` patched into existing DOM entries when broadcast arrives after initial render
- Export button downloads TXT; auto-reconnects on SSE drop

## VL Model (`--qwenvl`)

Started as a separate vLLM OpenAI-compatible subprocess on `VL_PORT` (default 9004).

- GPU memory auto-sized from actual free GPU at startup, capped at `_VL_MAX_GB = 20 GB`
- `max_model_len` auto-sized from free GPU (16384 if ≥20 GB free, else 8192/4096/2048)
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
| `VL_GPU_MEMORY_UTILIZATION` | auto | vLLM GPU fraction for VL; auto = free GPU − 2 GB buffer, capped at 20 GB |
| `VL_MAX_MODEL_LEN` | auto | VL context length; auto from free GPU, max 16384 |
| `VL_PORT` | `9004` | Internal port for VL subprocess |
| `MAX_NEW_TOKENS` | `8192` | |
| `ENABLE_ASR_MODEL` | `true` | set `false` to skip |
| `ENABLE_ALIGNER_MODEL` | `true` | set `false` to skip |
| `ENABLE_PREFIX_CACHING` | `true` | vLLM APC — caches KV blocks for shared prefix |
| `ASR_PORT` | `9002` | default port; overridden by `--port` CLI arg |

## Web Server CLI Args

| Arg | Default | Description |
|---|---|---|
| `--asr-host` | `localhost` | ASR server host |
| `--asr-port` | `9002` | ASR server port |
| `--port` | `8001` | Web server port |

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
