# Qwen3-ASR Transcription Server

Speech-to-text service using local Qwen3-ASR-1.7B model via vLLM.

## Start

```bash
python server.py          # ASR + chat API, binds 0.0.0.0:8000
python web_server.py      # Web UI, binds 0.0.0.0:8001 (optional, for PDF context extraction)
```

Server loads models in the background; poll `GET /health` until `"status": "ready"`.

## Core Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server — `/transcribe`, `/transcribe-streaming` (WebSocket), `/chat` (SSE) |
| `web_server.py` | Web UI server (port 8001) — serves `web/index.html`, `/api/chat` (Anthropic), `/api/extract-context` |
| `web/index.html` | Single-page browser UI — sessions, AI chat, live mic transcription |
| `client_file.py` | **Primary client** — vocal extraction → resample → VAD → streaming ASR |
| `client_mic.py` | Live microphone streaming client with VAD-based utterance detection |
| `process_video.py` | Extract audio from video, start server, transcribe, save JSON |
| `Qwen3-ASR-1.7B/` | ASR model weights |
| `Qwen3-ForcedAligner-0.6B/` | Forced-aligner model weights (word-level timestamps) |
| `qwen_asr_inference/

## Web UI (`web/index.html`)

Three-panel layout served from `web_server.py` at `http://localhost:8001`:

- **Left**: Session list — auto-saved to `localStorage`, double-click to rename
- **Middle**: AI chat — asks questions about the current session's transcription
- **Right**: Live mic transcription — VAD-based utterance detection, language selector, PDF/MD context upload, resizable via drag handle

**Chat backend**: calls `http://localhost:8000/chat` (ASR server) by default. Server URL is configurable via the ⚙ settings button (stored in `localStorage`). Requires server restart after adding `/chat` endpoint.

**Microphone**: requires a **secure context** — access via `http://localhost:8001`, not an IP address over HTTP. For remote access, use HTTPS (self-signed cert with `openssl req -x509 ...`).

**PDF context extraction**: requires `web_server.py` running (`pip install pypdf`). MD/TXT files are parsed directly in the browser.

## client_file.py Pipeline

```
input audio
  → demucs (htdemucs --two-stems vocals)   # isolate human voice
  → resample to 16kHz mono                  # scipy.signal.resample_poly
  → WebRTC VAD aggressiveness=2             # split into speech segments
  → stream each segment over WebSocket      # with optional vocabulary context
  → <stem>.jsonl                            # one JSON line per segment
```

```bash
python client_file.py openclaw.mp3 --context foo.md --language en
```

- `--context` accepts `.pdf` or `.md`; text is prepended to the model prompt as vocabulary hint.

## client_mic.py

VAD-based real-time microphone transcription. Each detected utterance is sent as a separate WebSocket session (not one continuous stream).

```bash
python client_mic.py                   # English, localhost:8000
python client_mic.py -l zh             # Chinese
python client_mic.py -v                # verbose VAD debug output (shows silence/speech counters)
python client_mic.py -e ws://host:port/transcribe-streaming
```

**VAD settings** (tunable at top of file):
- `VAD_AGGRESSIVENESS = 3` — level 3 filters background noise best; lower if speech is clipped
- `SILENCE_END_FRAMES = 33` — ~1 s of silence ends an utterance
- `ENERGY_THRESHOLD = 0.018` — RMS threshold; raise if background noise triggers false starts
- If stuck on `[Recording...]`: background noise is being detected as speech; increase aggressiveness or `ENERGY_THRESHOLD`

## server.py `/chat` Endpoint

Added `POST /chat` for text-chat using the already-loaded ASR model:

```python
# Internal implementation — key facts:
asr = models["asr"]                  # Qwen3ASRModel instance
asr.model                            # underlying vllm.LLM — has .generate()
asr.processor.tokenizer              # Qwen3ASRProcessor tokenizer — has .apply_chat_template()
asr.backend                          # "vllm" or "transformers"
```

The chat endpoint uses `asr.processor.tokenizer.apply_chat_template()` to format messages, then calls `asr.model.generate()` (text-only, no audio `multi_modal_data`). Quality is limited — Qwen3-ASR-1.7B is trained for audio→text, not chat.

Request body: `{"messages": [{"role": "user", "content": "..."}], "transcription": "...", "context": "..."}`
Response: SSE stream (`data: {"text": "..."}` lines, ending with `data: [DONE]`).

## Server Env Vars

| Variable | Default | Notes |
|---|---|---|
| `ASR_MODEL_NAME` | `Qwen3-ASR-1.7B` | local dir or HF model id |
| `ALIGNER_MODEL_NAME` | `Qwen3-ForcedAligner-0.6B` | |
| `GPU_MEMORY_UTILIZATION` | `0.75` | vLLM GPU fraction |
| `MAX_NEW_TOKENS` | `8192` | |
| `ENABLE_ASR_MODEL` | `true` | set `false` to skip |
| `ENABLE_ALIGNER_MODEL` | `true` | set `false` to skip |
| `VLLM_TARGET_DEVICE` | `cpu` | override to `cuda` for GPU |

## Key Notes

- **Audio format for streaming**: PCM 16-bit signed little-endian, 16kHz mono. Send WebSocket JSON `{"type":"start","format":"pcm_s16le","sample_rate_hz":16000}` before audio bytes, then `{"type":"stop"}` when done.
- **Forced alignment**: pass `?forced_alignment=true` to `/transcribe` for word-level timestamps (requires aligner model).
- **Event recordings**: always run demucs vocal extraction first — background music causes LLM hallucination in the ASR model (repetitive generic phrases).
- **`Qwen3ASRModel` does NOT have `.generate()`** — use `models["asr"].model.generate()` for the underlying vLLM engine.
- **Server restart required** after any change to `server.py` endpoints.
- The `transcription.json` (non-streaming) is produced by `process_video.py` via the `/transcribe` HTTP endpoint.
