# Qwen3-ASR Transcription Server

Speech-to-text service using the local Qwen3-ASR-1.7B model via llama.cpp.

## Start

```bash
python server.py                             # ASR, binds 0.0.0.0:9002
python server.py --port 9000                 # custom port
python server.py --qwenvl                    # + Qwen3-VL-4B-Instruct on VL_PORT (default 9004)
python server.py --qwenvl path/to/model.gguf # custom VL model
python server.py --qwenvl --vl-device 1      # VL on GPU 1 (2nd GPU), ASR on GPU 0
python server.py --asr-device 0 --qwenvl --vl-device 1  # explicit GPU assignment
python web_server.py                         # Web UI, binds 0.0.0.0:8001
```

Server loads models in the background; poll `GET /health` until `"status": "ready"`.

**Requires `llama-server` (llama.cpp ≥ b9173) on `PATH`.** Earlier builds load the model and
encode audio but transcribe everything as empty output.

**Model files are optional.** If the local GGUF pair is missing, the server falls back to
`llama-server -hf <repo>`, which downloads the decoder *and* its mmproj into the shared HF
cache (`~/.cache/huggingface/hub`) and reuses them on later runs:

| | default repo | override |
|---|---|---|
| ASR | `ggml-org/Qwen3-ASR-1.7B-GGUF` | `--asr-hf` / `ASR_HF_REPO` |
| VL | `unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M` | `--vl-hf` / `VL_HF_REPO` |

Local files always win, so an existing setup downloads nothing. Set the repo to `""` to
require local files and fail with an explicit error instead. First run pulls ~2.5 GB (ASR) or
~3.3 GB (VL), hence `STARTUP_TIMEOUT` defaults to 1800 s.

## Core Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI server — `/transcribe`, `/transcribe-streaming` (WebSocket), `/vl/health`, `/vl/proxy/{path}`; spawns `llama-server` subprocesses |
| `web_server.py` | Web UI server (port 8001) — serves `web/index.html`, `/api/chat`, `/api/translate`, `/api/extract-context`, `/api/session/*`, `/api/models`, `/api/config` |
| `web/index.html` | Instructor UI — sessions, AI chat (with image input), live mic transcription, auto-translation |
| `web/viewer.html` | Viewer UI — live transcription + translations via SSE, AI chat |
| `client_file.py` | **Primary client** — accepts a local audio path or URL (Bilibili, YouTube, etc.); URL inputs are downloaded via yt-dlp, then vocal extraction → resample → VAD → streaming ASR (outputs TXT format) |
| `client_mic.py` | Live microphone streaming client with VAD-based utterance detection |
| `process_video.py` | Extract audio from video, start server, transcribe, save JSON |
| `Qwen/*.gguf` | GGUF weights — each model needs a decoder **and** a matching `mmproj-*` encoder |

## Web UI (`web/index.html`)

Three-panel layout served from `web_server.py` at `http://localhost:8001`:

- **Left**: Session list — auto-saved to `localStorage`, double-click to rename, ✕ to delete
- **Middle**: AI chat — Claude / Gemini / Mistral / Local VL; image attachment (🖼) visible only when `Local VL` selected; image thumbnails shown in history, clickable to enlarge (lightbox)
- **Right**: Live mic transcription — VAD-based, language selector, **auto-translate** target selector (shown next to source language when VL available), PDF/MD/TXT context upload (📎), export (⇩)

**Panel divider**: Draggable 4px divider between chat and transcription panels; width saved to `localStorage`.

**Auto-translation**: each new segment is auto-translated if target language ≠ source language and VL is available. Result stored in `entry.translated`, broadcast to viewers via `pushToServer()`. Manual `⇄ Translate` / `✕ Delete` buttons appear at bottom-right of each entry on hover.

**Auto-answer + manual Ask-AI**: when a transcription segment ends with `?` (matched by `isQuestion()` — trailing `?` only, so self-answered segments like *"How much do you remember? Not much."* are skipped), `scheduleAutoAnswer()` arms a 10 s timer. The timer is cancelled if (a) a new VAD utterance starts at the mic level (`processVADFrame` → `cancelAutoAnswer()`), (b) the next final segment isn't itself a question, or (c) the user manually sends a chat message. On fire, it calls `askAIAboutSegment(-1, null, questionText)` which streams a response into the chat. Every segment also shows a **💬 Answer** button (questions) or **💬 Explain** button (statements) on hover; clicking routes through the same `askAIAboutSegment(idx, btn)` and bypasses the 10 s wait. Silence is measured from the mic VAD, not from ASR `final` arrivals, so ASR decoding lag does not eat into the 10 s window.

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

Started as a separate `llama-server` subprocess on `VL_PORT` (default 9004), serving an
OpenAI-compatible API.

- Needs a decoder GGUF **and** its `mmproj-*` vision encoder (`--qwenvl` / `--vl-mmproj`).
  Without the mmproj the server starts fine but reports `"vision": false` and rejects images —
  startup fails loudly rather than serving a text-only model.
- Uses ~3.5 GB VRAM for `Qwen3-VL-4B-Instruct-Q4_K_M` (vs 8-20 GB budgeted by the old vLLM
  path), so it comfortably shares a GPU with ASR; `--vl-device` pins it to a second GPU.
- Accessed via `/vl/proxy/...` on the main server — `web_server.py` never connects to
  `VL_PORT` directly.

## client_file.py Pipeline

```
input (local path OR URL)
  → yt-dlp download to downloads/<id>.<ext>   # only if input is a URL; cached by video id, skipped on re-run
  → demucs (htdemucs --two-stems vocals)      # optional, with --vocal-extraction
  → resample to 16kHz mono                    # scipy.signal.resample_poly
  → WebRTC VAD aggressiveness=2               # split into speech segments
  → stream each segment over WebSocket        # with optional vocabulary context
  → <stem>.txt                               # one text line per segment with timestamp
```

```bash
python client_file.py openclaw.mp3 --context foo.md --language en
python client_file.py 'https://www.bilibili.com/video/BV1jEnuz9ESc/' --language Chinese
```

**URL inputs** (`is_url()` → `download_url()` in `client_file.py`):
- Detected by `http://` / `https://` prefix; routed through `yt-dlp -f bestaudio/best --no-playlist -o 'downloads/%(id)s.%(ext)s'`
- `yt-dlp --get-id` resolves the video id first; if `downloads/<id>.*` already exists, the download step is skipped (re-runs are effectively instant)
- `detect_browser_for_cookies()` scans common profile dirs (Firefox, Chrome, Chromium, Edge, Brave) and auto-attaches `--cookies-from-browser <name>`; override via `--cookies-from-browser`
- Override the cache directory with `--download-dir`
- Requires `yt-dlp` (already in environment); not in `requirements.txt`

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

## Server Internals (`server.py`)

Rather than loading models in-process, the server spawns `llama-server` subprocesses and proxies
to them (ASR on `ASR_INTERNAL_PORT`, default 9003; VL on `VL_PORT`).

**Requires llama.cpp ≥ b9173.** Earlier builds load the model and encode audio but transcribe
everything as empty output (`language None<asr_text>`) — see llama.cpp issue #22357.

**GGUF models ship as two files**: a decoder plus an `mmproj-*` encoder. This differs from the
HF safetensors layout that vLLM uses, where the vision/audio encoder lives inside the same file.
Passing only the decoder starts the server successfully but *silently* disables audio/image
input, so `load_models()` verifies `/props` reports the expected modality and fails startup if
not.

**Language forcing** uses an assistant-prefill message (`_asr_request_body`):

```python
{"role": "assistant", "content": f"language {language}<asr_text>"}
```

llama-server's `--prefill-assistant` (on by default) continues this message rather than starting
a new turn — the same mechanism as vLLM's `_build_text_prompt`. Without it the model
auto-detects language per segment and drifts to Chinese/Thai/Spanish on short filler utterances
(24 drifted lines over a 57-min lecture; 0 with prefill). Setting the language via the `system`
field does **not** work — that slot is the vocabulary context.

**Streaming is re-transcription, not incremental.** llama.cpp has no equivalent of
`init_streaming_state`/`streaming_transcribe`, so partials come from re-transcribing a growing
window of the utterance every `STREAM_PARTIAL_EVERY_S`. Partials run as cancellable background
tasks so audio intake never blocks; the `final` is one clean pass over the whole utterance and
matches `/transcribe` exactly.

**Server-side VAD re-segmentation.** Both `/transcribe` and the streaming `final` re-segment
audio via `vad_split()` before decoding: whole-file uploads and long utterances otherwise
overflow the model context (llama-server returns 400, and quality collapses past ~120 s well
before that). Segments are merged toward `STREAM_VAD_TARGET_S` and boundaries are placed
**only in the gaps between speech regions**, never inside one — the rule Qwen3-ASR-Toolkit
uses to avoid mid-sentence cuts. Defaults to Silero VAD (`STREAM_VAD_BACKEND=silero`), falling
back to `webrtcvad` if `silero-vad` is not installed. The browser's own VAD in
`web/index.html` is untouched: its ~12 s force-flush (`maxUttFrames`, lowered in commit
`d2e5300`) exists for caption latency and drives the auto-answer timer.

**Context leakage guard.** With a vocabulary context set, a segment containing no speech makes
the highest-probability continuation a copy of the system prompt — the model emits the context
verbatim as if it were transcribed speech. This is *not* fixable by prompt wording:
Qwen3-ASR is an audio→text model, not instruction-following, so "Reference only — do NOT
transcribe this" and the `[ASR_CONTEXT_START]`/`[ASR_CONTEXT_END]` sentinels are echoed along
with the terms (`server.py`'s `strip_prompt()` removes them after the fact instead). Three
layers *prevent* the generation, verified 6/6 leaks blocked with 5/5 real speech preserved:

1. `speech_ratio()` — webrtcvad frame ratio. Non-speech measures ≤0.02, real speech ≥0.30, so
   a low ratio skips the request entirely (free).
2. Language probe — re-asks *without* the language prefill and drops the segment if the model
   answers `language None`. The prefill that forces language suppresses this signal, which is
   why it must be probed separately. Only runs for borderline audio (one extra request).
3. `context_overlap()` — drops output whose 4-grams overlap the context past
   `CONTEXT_GUARD_OVERLAP`. Catches the residue, e.g. long stretches of digital silence.

Set `CONTEXT_GUARD=false` to disable.

Measured against a human-checked transcript (`data/08_19_1_10min.txt`, 1279 words):
**12.4% WER** through `/transcribe`. For reference, on the same audio the browser's live RMS
VAD scores 25.6% and `client_file.py`'s webrtcvad scores 26.0%.

Measured on `data/08_14_3.wav` (57 min) vs the vLLM reference transcript: 4.6% word divergence,
0 language-drift lines, ~84 s wall (vs ~597 s for vLLM). Q8_0 and bf16 differ by only 0.9% from
each other, while bf16 is 2.2× slower and 1.9× the disk — Q8_0 is the better default.

### Server Env Vars

| Variable | Default | Notes |
|---|---|---|
| `ASR_MODEL_PATH` | `Qwen/Qwen3-ASR-1.7B-Q8_0.gguf` | decoder GGUF (`--asr-model`) |
| `ASR_MMPROJ_PATH` | `Qwen/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` | audio encoder (`--asr-mmproj`) |
| `VL_MODEL_PATH` | `Qwen/Qwen3-VL-4B-Instruct-Q4_K_M.gguf` | VL decoder (`--qwenvl MODEL`) |
| `VL_MMPROJ_PATH` | `Qwen/mmproj-Qwen3-VL-4B-Instruct-F16.gguf` | vision encoder (`--vl-mmproj`) |
| `LLAMA_SERVER_BIN` | `llama-server` | full path if not on `PATH` |
| `ASR_HF_REPO` | `ggml-org/Qwen3-ASR-1.7B-GGUF` | HF GGUF repo used when local ASR files are absent (`--asr-hf`) |
| `VL_HF_REPO` | `unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M` | HF GGUF repo used when local VL files are absent (`--vl-hf`) |
| `ASR_INTERNAL_PORT` | `9003` | internal ASR llama-server port |
| `ASR_CTX_SIZE` / `VL_CTX_SIZE` | `8192` | `-c` context size |
| `ASR_NGL` / `VL_NGL` | `99` | layers to offload to GPU |
| `ASR_PARALLEL` | `2` | llama-server `--parallel` slots |
| `STREAM_VAD_BACKEND` | `silero` | `silero` or `webrtc` for server-side re-segmentation |
| `STREAM_VAD_TARGET_S` | `8.0` | merge speech regions toward this length |
| `STREAM_VAD_MAX_S` | `15.0` | hard cap per segment |
| `STREAM_VAD_MIN_S` | `8.0` | utterances shorter than this are sent unsegmented |
| `CONTEXT_GUARD` | `true` | prevent context regurgitation on non-speech |
| `CONTEXT_GUARD_VAD_RATIO` | `0.05` | below this speech ratio, skip the request |
| `CONTEXT_GUARD_PROBE_RATIO` | `0.15` | below this, run the language probe |
| `CONTEXT_GUARD_OVERLAP` | `0.5` | drop output with this 4-gram overlap vs context |
| `STREAM_PARTIAL_EVERY_S` | `1.5` | seconds between streaming partials |
| `STREAM_PARTIAL_MIN_S` | `1.0` | min audio before first partial |
| `STREAM_MAX_UTTERANCE_S` | `30.0` | cap on re-transcribed window |
| `STARTUP_TIMEOUT` | `1800` | seconds to wait for llama-server ready (first run may download GBs) |

`MAX_NEW_TOKENS` (default `512` here), `ASR_PORT`, `VL_PORT`, `ASR_DEVICE`, `VL_DEVICE`, and
`ENABLE_ASR_MODEL` behave as in `server.py`.

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
- **Event recordings**: always run demucs vocal extraction first — background music causes hallucination.
- **GGUF needs an mmproj**: a decoder-only GGUF starts fine but silently reports `"audio": false` / `"vision": false` and rejects media. Always pass `--mmproj`.
- **llama.cpp build matters**: < b9173 transcribes audio as empty output. Check with `llama-server --version`.
- **Server restart required** after any change to `server.py` endpoints.
- **Viewer broadcast relay**: session state held in-memory in `web_server.py`; restarting clears it. Viewers reconnect automatically via SSE.
- **Translation history**: `entry.translated` stored in session alongside `entry.text`; persisted to `localStorage` and broadcast via SSE so viewers receive translations.
- **HTML caching**: `web/index.html` and `web/viewer.html` served with `Cache-Control: no-cache` headers.
- **Context stripping** (`strip_prompt()` in `server.py`): removes instruction prompts from ASR output when a vocabulary context is provided. Uses sentinel tags, instruction markers, exact-line matching, and prefix detection as cascading fallbacks.
- **API keys**: server reads from env vars; clients can override per-request via `api_keys` field in `/api/chat`. Viewer stores its own keys in `localStorage` only.
