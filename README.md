# Qwen3-ASR Transcription Server

Local speech-to-text service powered by [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), with a FastAPI backend and clients for files, microphone, and video.

## Features

- **File transcription** — MP3/WAV/M4A/video audio → timestamped JSONL
- **Vocal extraction** — isolates human voice from background music before ASR (via [demucs](https://github.com/facebookresearch/demucs))
- **VAD segmentation** — WebRTC VAD splits audio into speech segments
- **Vocabulary context** — feed a PDF or Markdown document to improve domain-specific terminology
- **Streaming** — real-time transcription over WebSocket (partial + final results)
- **Forced alignment** — word-level timestamps via Qwen3-ForcedAligner
- **Microphone input** — live transcription from system mic
- **Web UI** — browser-based chat interface with session management and live mic transcription

## Models

| Model | Size | Purpose |
|---|---|---|
| `Qwen3-ASR-1.7B` | ~3.5 GB | Speech recognition |
| `Qwen3-ForcedAligner-0.6B` | ~1.2 GB | Word-level timestamps |

Both are loaded from local directories at startup. ASR inference is handled by [`qwen_asr_inference`](https://github.com/QwenLM/Qwen3-ASR).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the ASR server

```bash
CUDA_VISIBLE_DEVICES=0 python server.py
```

Poll `GET /health` until `"status": "ready"` before sending requests.

**Key env vars:**

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL_NAME` | `Qwen3-ASR-1.7B` | Local path or HF model ID |
| `ALIGNER_MODEL_NAME` | `Qwen3-ForcedAligner-0.6B` | |
| `GPU_MEMORY_UTILIZATION` | `0.75` | vLLM GPU fraction |
| `ENABLE_ASR_MODEL` | `true` | |
| `ENABLE_ALIGNER_MODEL` | `true` | |

### 2. Transcribe a file

```bash
python client_file.py audio.mp3 --language English
```

For event recordings with background music, add `--vocal-extraction` to run demucs first:

```bash
python client_file.py event_recording.mp3 --vocal-extraction --context slides.md --language English
```

Output: `transcription_streaming.jsonl` — one line per speech segment:

```json
{"timestamp": "[0:01:23 - 0:01:45]", "text": "So clustering is an unsupervised learning task."}
```

**How it works internally:**

```
input audio
  → [demucs htdemucs]        only with --vocal-extraction
  → resample to 16kHz mono
  → WebRTC VAD (level 2)     split into speech segments
  → stream over WebSocket    with optional vocabulary context
  → transcription_streaming.jsonl
```

> **When to use `--vocal-extraction`:** event recordings (conferences, meetups) with background music and PA noise. Without it the ASR model hallucinates repetitive generic phrases when fed music. For clean lecture/interview audio it is unnecessary overhead (adds several minutes of CPU time). Separated tracks are cached in `separated/` next to the audio file and reused on subsequent runs.

### 3. Live microphone transcription

```bash
python client_mic.py                       # English, localhost:8000
python client_mic.py -l zh                 # Chinese
python client_mic.py -l English            # full name also works
python client_mic.py -v                    # verbose VAD debug output
python client_mic.py -e ws://host:8000/transcribe-streaming  # remote server
```

Speak into the mic; each detected utterance is transcribed and printed with a timestamp. Press `Ctrl+C` to stop.

**Tunable VAD constants** (top of `client_mic.py`):

| Constant | Default | Effect |
|---|---|---|
| `VAD_AGGRESSIVENESS` | `3` | 0–3; higher = stricter speech detection |
| `SILENCE_END_FRAMES` | `~33` | frames of silence to end an utterance (~1 s) |
| `ENERGY_THRESHOLD` | `0.018` | RMS floor; raise to suppress background noise |

If stuck on `[Recording...]`, background noise is triggering speech detection — increase `VAD_AGGRESSIVENESS` or `ENERGY_THRESHOLD`.

### 4. Transcribe a video file

```bash
python process_video.py lecture.mp4 --text-out transcription.json
```

Extracts audio via ffmpeg, starts the server, transcribes, saves JSON.

### 5. Web UI

Start the web UI server alongside the ASR server:

```bash
# Set at least one AI provider API key
ANTHROPIC_API_KEY=sk-...  python web_server.py   # Claude
GOOGLE_API_KEY=...         python web_server.py   # Gemini
MISTRAL_API_KEY=...        python web_server.py   # Mistral
```

Then open `http://localhost:8001` in a browser.

Three-panel layout:
- **Left** — session list, auto-saved to `localStorage`; double-click to rename
- **Middle** — AI chat about the current session's transcription (Claude / Gemini / Mistral)
- **Right** — live mic transcription with VAD, language selector, and PDF/MD context upload

> **Microphone** requires a secure context. Access via `http://localhost:8001`, not an IP address over HTTP. For remote access, use HTTPS (self-signed cert with `openssl req -x509 ...`).

## HTTP API

### `POST /transcribe`

Upload one or more audio files for batch transcription.

```bash
curl -F "files=@audio.wav" "http://localhost:8000/transcribe?language=English"

# With word-level timestamps
curl -F "files=@audio.wav" "http://localhost:8000/transcribe?language=English&forced_alignment=true"
```

Response:
```json
[{"text": "Hello world.", "language": "English"}]
```

### `WS /transcribe-streaming`

WebSocket streaming endpoint. Send a `start` control message, then raw PCM frames, then `stop`.

```
→ {"type": "start", "format": "pcm_s16le", "sample_rate_hz": 16000, "context": "...optional..."}
→ <binary PCM int16 mono 16kHz frames>
→ {"type": "stop"}
← {"type": "partial", "text": "...", "language": "English"}
← {"type": "final",   "text": "...", "language": "English"}
```

## Supported Languages

English, Chinese, Cantonese, Japanese, Korean, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Russian, Thai, Vietnamese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Persian, Greek, Romanian, Hungarian, Macedonian.

Pass the **full name** (e.g. `--language Chinese`, not `--language zh`) for `client_file.py`. Both short codes and full names work for `client_mic.py`.
