"""llama.cpp backend for the Qwen3-ASR transcription server.

Drop-in replacement for server.py: same endpoints, same request/response
shapes, same default port (9002), so web_server.py and the web UI need no
changes. Instead of loading models in-process via vLLM, this spawns
llama-server subprocesses and proxies to them.

    python server_llama_cpp.py
    python server_llama_cpp.py --qwenvl                 # + VL model
    python server_llama_cpp.py --qwenvl --vl-device 1   # VL on GPU 1

Requires llama.cpp >= b9173 (earlier builds transcribe audio as empty output)
and a *pair* of GGUF files per model: the decoder plus its mmproj audio/vision
encoder. See ASR_MODEL_PATH / ASR_MMPROJ_PATH below.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

# ── CLI args ──────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--qwenvl",
    nargs="?",
    const=os.getenv("VL_MODEL_PATH", "Qwen/Qwen3-VL-4B-Instruct-Q4_K_M.gguf"),
    metavar="MODEL",
    help="Enable Qwen-VL model (path to decoder GGUF)",
)
_parser.add_argument("--port", type=int, default=int(os.getenv("ASR_PORT", "9002")), help="Port to listen on (default: 9002)")
_parser.add_argument("--asr-device", default=os.getenv("ASR_DEVICE", ""), metavar="N", help="GPU index for ASR llama-server, e.g. 0")
_parser.add_argument("--vl-device", default=os.getenv("VL_DEVICE", ""), metavar="N", help="GPU index for VL llama-server, e.g. 1")
_parser.add_argument("--asr-model", default=os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B-Q8_0.gguf"), help="Path to ASR decoder GGUF")
_parser.add_argument("--asr-mmproj", default=os.getenv("ASR_MMPROJ_PATH", "Qwen/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"), help="Path to ASR mmproj GGUF")
_parser.add_argument("--vl-mmproj", default=os.getenv("VL_MMPROJ_PATH", "Qwen/mmproj-Qwen3-VL-4B-Instruct-F16.gguf"), help="Path to VL mmproj GGUF")
_cli, _ = _parser.parse_known_args()

VL_MODEL_NAME = _cli.qwenvl or os.getenv("VL_MODEL_NAME", "")
VL_PORT = int(os.getenv("VL_PORT", "9004"))
ASR_DEVICE = _cli.asr_device
VL_DEVICE = _cli.vl_device

import numpy as np
import psutil
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
def get_env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")


MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "4"))
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "1"))
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 5)))

STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "1600"))
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "120"))
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))

# llama-server process settings
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "llama-server")
ASR_INTERNAL_PORT = int(os.getenv("ASR_INTERNAL_PORT", "9003"))
ASR_CTX_SIZE = int(os.getenv("ASR_CTX_SIZE", "8192"))
VL_CTX_SIZE = int(os.getenv("VL_CTX_SIZE", "8192"))
ASR_NGL = os.getenv("ASR_NGL", "99")
VL_NGL = os.getenv("VL_NGL", "99")
ASR_PARALLEL = int(os.getenv("ASR_PARALLEL", "2"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "300"))

# Streaming: how much trailing audio to re-decode for each partial. llama.cpp
# has no incremental/streaming ASR API, so partials are produced by
# re-transcribing a growing window of the utterance.
STREAM_PARTIAL_MIN_S = float(os.getenv("STREAM_PARTIAL_MIN_S", "1.0"))
STREAM_PARTIAL_EVERY_S = float(os.getenv("STREAM_PARTIAL_EVERY_S", "1.5"))
STREAM_MAX_UTTERANCE_S = float(os.getenv("STREAM_MAX_UTTERANCE_S", "30.0"))

# Server-side VAD re-segmentation for the `final` result.
#
# The browser's VAD (web/index.html) exists for *latency*: it force-flushes every
# ~12 s (maxUttFrames=400, lowered from 2000 in commit d2e5300) so captions appear
# promptly, and its utterance boundaries drive the auto-answer timer. That force-flush
# necessarily cuts mid-sentence — 76% of utterances on a measured lecture — and its
# RMS gate (energyThreshold) misses ~52% of genuine speech frames.
#
# Rather than change those client-side tradeoffs, the server re-runs a proper VAD over
# the buffered utterance before producing `final`. Partials still stream at browser
# cadence, so perceived latency is unchanged; only the committed text improves.
# Set STREAM_SERVER_VAD=false to send the utterance as a single request instead.
STREAM_SERVER_VAD = get_env_bool("STREAM_SERVER_VAD", "true")
# Silero (neural VAD) measured 19.2% WER against a human-checked transcript vs
# 26.0% for webrtcvad on the same audio — it recovers quiet speech, notably
# student answers, that an energy/webrtc gate drops. Falls back to webrtcvad if
# silero-vad is not installed.
STREAM_VAD_BACKEND = os.getenv("STREAM_VAD_BACKEND", "silero")  # silero | webrtc
STREAM_VAD_TARGET_S = float(os.getenv("STREAM_VAD_TARGET_S", "8.0"))
STREAM_VAD_MAX_S = float(os.getenv("STREAM_VAD_MAX_S", "15.0"))
STREAM_VAD_MIN_S = float(os.getenv("STREAM_VAD_MIN_S", "8.0"))

CONTEXT_PREFIX = "Reference only — do NOT transcribe this. Vocabulary hint: "
CONTEXT_TAG_START = "[ASR_CONTEXT_START]"
CONTEXT_TAG_END = "[ASR_CONTEXT_END]"

# ── Context leakage ───────────────────────────────────────────
# When a vocabulary context is supplied and a segment contains no speech, the
# highest-probability continuation is to copy the system prompt — so the model
# emits the context verbatim as if it were transcribed speech.
#
# This is not fixable by prompt wording: Qwen3-ASR is an audio→text model, not an
# instruction-following one. Measured on silence, "Reference only — do NOT
# transcribe this" and [ASR_CONTEXT_START]/[ASR_CONTEXT_END] sentinels are simply
# echoed along with the terms. server.py's strip_prompt() removes them after the
# fact; the checks below prevent the generation instead.
#
# Layer 1 (free): webrtcvad speech ratio. Non-speech measures <= 0.02, real speech
#   >= 0.30 — a wide margin, so a low ratio skips the request entirely.
# Layer 2 (one extra request, only for borderline audio): ask without a language
#   prefill and let the model self-report. It answers "language None" for
#   non-speech. Note the prefill that forces language (and prevents drift)
#   suppresses this signal, which is why it must be probed separately.
# Layer 3 (free): reject output whose 4-grams overlap the context — catches the
#   residual cases where layers 1-2 both pass (e.g. long digital silence).
CONTEXT_GUARD = get_env_bool("CONTEXT_GUARD", "true")
CONTEXT_GUARD_VAD_RATIO = float(os.getenv("CONTEXT_GUARD_VAD_RATIO", "0.05"))
CONTEXT_GUARD_PROBE_RATIO = float(os.getenv("CONTEXT_GUARD_PROBE_RATIO", "0.15"))
CONTEXT_GUARD_OVERLAP = float(os.getenv("CONTEXT_GUARD_OVERLAP", "0.5"))

# -----------------------------
# App state
# -----------------------------
models = {}
model_status = "starting"
model_ready_event = asyncio.Event()

decode_sem = asyncio.Semaphore(MAX_CONCURRENT_DECODE)
infer_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)


# -----------------------------
# Helpers
# -----------------------------
async def to_thread_limited(sem: asyncio.Semaphore, fn, *args, **kwargs):
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)


def map_language(lang_code: Optional[str]) -> Optional[str]:
    """Map ISO code to Qwen full name."""
    if lang_code is None:
        return None
    mapping = {
        "en": "English", "de": "German", "fr": "French", "es": "Spanish",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
        "ru": "Russian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
        "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese", "hi": "Hindi",
        "ar": "Arabic",
    }
    return mapping.get(lang_code.lower(), lang_code)


def read_audio_file(file_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """Sync decode; call via asyncio.to_thread. soundfile first, ffmpeg fallback."""
    import tempfile

    try:
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        pass

    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(file_bytes)
        tmp_path = tf.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-f", "wav", "-ac", "1", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[:300]}")
        audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    finally:
        os.unlink(tmp_path)


def resample_16k(audio_f32: np.ndarray, sr: int) -> np.ndarray:
    """Resample float32 mono audio to 16 kHz."""
    if sr == 16000:
        return audio_f32
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(int(sr), 16000)
    return resample_poly(audio_f32, 16000 // g, int(sr) // g).astype(np.float32)


def to_wav_b64(audio_f32: np.ndarray, sr: int = 16000) -> str:
    """Encode float32 mono audio as base64 16-bit PCM WAV for llama-server."""
    pcm = np.clip(audio_f32, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


_ASR_MARKER = "<asr_text>"


def split_asr_output(raw: str) -> Tuple[str, Optional[str]]:
    """Split llama.cpp's 'language X<asr_text>TEXT' into (text, language).

    vLLM strips this header itself; llama.cpp returns it verbatim.
    See https://github.com/ggml-org/llama.cpp/issues/26749
    """
    if not raw:
        return "", None
    i = raw.find(_ASR_MARKER)
    if i == -1:
        return raw.strip(), None
    head, text = raw[:i], raw[i + len(_ASR_MARKER):]
    lang = head.strip()
    if lang.lower().startswith("language"):
        lang = lang[len("language"):].strip()
    if lang.lower() in ("none", ""):
        lang = None
    return text.strip(), lang


def speech_ratio(audio_f32: np.ndarray) -> float:
    """Fraction of 30 ms frames webrtcvad classifies as speech.

    Measured separation is wide: digital silence / hum 0.00, room tone 0.02,
    real lecture speech 0.31-0.40. Returns 1.0 if webrtcvad is unavailable so
    callers fail open (transcribe) rather than silently dropping audio.
    """
    try:
        import webrtcvad
    except ImportError:
        return 1.0
    pcm = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    vad = webrtcvad.Vad(2)
    frame = int(STREAM_EXPECT_SR * 0.03)
    total = speech = 0
    for i in range(0, len(pcm) - frame, frame):
        total += 1
        try:
            if vad.is_speech(pcm[i:i + frame].tobytes(), STREAM_EXPECT_SR):
                speech += 1
        except Exception:
            return 1.0
    return speech / total if total else 0.0


def _ngrams(text: str, n: int = 4) -> set:
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def context_overlap(text: str, context: str, n: int = 4) -> float:
    """Fraction of the output's n-grams that also appear in the context."""
    if not text or not context:
        return 0.0
    t = _ngrams(text, n)
    if not t:
        return 0.0
    return len(t & _ngrams(context, n)) / len(t)


async def _probe_language(audio_f32: np.ndarray, context: str) -> Optional[str]:
    """Ask without a language prefill so the model can answer 'language None'."""
    import httpx

    body = {
        "messages": [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": to_wav_b64(audio_f32), "format": "wav"}}]},
        ],
        "max_tokens": 24, "temperature": 0,
    }
    url = f"http://localhost:{ASR_INTERNAL_PORT}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None  # fail open
    _, lang = split_asr_output(raw)
    return lang


def _asr_request_body(audio_f32: np.ndarray, language: Optional[str], context: str = "") -> dict:
    """Build the llama-server chat request for one audio segment.

    The trailing assistant message is a *prefill*: llama-server's
    --prefill-assistant (on by default) continues it rather than starting a new
    turn, which is how vLLM's _build_text_prompt forces the output language.
    Without it the model auto-detects per segment and drifts to another
    language on short filler utterances.
    """
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": to_wav_b64(audio_f32), "format": "wav"}}
        ]},
    ]
    if language:
        msgs.append({"role": "assistant", "content": f"language {language}{_ASR_MARKER}"})
    return {"messages": msgs, "max_tokens": MAX_NEW_TOKENS, "temperature": 0}


_silero_model = None
_silero_failed = False


def _silero_regions(audio_f32: np.ndarray):
    """Speech regions via Silero VAD, or None if unavailable (caller falls back)."""
    global _silero_model, _silero_failed
    if _silero_failed:
        return None
    try:
        if _silero_model is None:
            from silero_vad import load_silero_vad
            _silero_model = load_silero_vad()
        from silero_vad import get_speech_timestamps
        import torch

        ts = get_speech_timestamps(
            torch.from_numpy(audio_f32.astype(np.float32)), _silero_model,
            sampling_rate=STREAM_EXPECT_SR, return_seconds=False,
            min_speech_duration_ms=250, min_silence_duration_ms=500,
        )
    except Exception as e:
        logger.warning("silero VAD unavailable (%s); falling back to webrtcvad", e)
        _silero_failed = True
        return None
    return [[t["start"], t["end"]] for t in ts] or None


def _merge_regions(regions, n: int) -> List[Tuple[int, int]]:
    """Merge speech regions toward STREAM_VAD_TARGET_S, never cutting inside one.

    Boundaries land only in the gaps between regions — the rule Qwen3-ASR-Toolkit
    uses — so sentences stay intact. The hard cap is a last resort for a single
    uninterrupted region longer than the model's usable window.
    """
    pad = int(0.2 * STREAM_EXPECT_SR)
    target = STREAM_VAD_TARGET_S * STREAM_EXPECT_SR
    merged: List[List[int]] = []
    for r in regions:
        if merged and (r[1] - merged[-1][0]) <= target:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))

    cap = STREAM_VAD_MAX_S * STREAM_EXPECT_SR
    out: List[Tuple[int, int]] = []
    for s, e in merged:
        s = max(0, s - pad)
        e = min(n, e + pad)
        if (e - s) <= cap:
            out.append((s, e))
            continue
        parts = int(np.ceil((e - s) / cap))
        step = (e - s) // parts
        for k in range(parts):
            out.append((s + k * step, e if k == parts - 1 else s + (k + 1) * step))
    return out


def vad_split(audio_f32: np.ndarray) -> List[Tuple[int, int]]:
    """Split a buffered utterance at natural speech boundaries.

    Uses webrtcvad (already a project dependency, same settings as client_file.py's
    apply_vad) to find speech regions, then merges them toward STREAM_VAD_TARGET_S
    *without ever cutting inside a speech region* — the approach Qwen3-ASR-Toolkit
    uses. Splitting only at speech onsets is what keeps sentences intact.

    Returns [(start, end)] sample offsets, or [(0, len)] if VAD is unavailable or
    finds nothing (callers then send the buffer unchanged).
    """
    n = len(audio_f32)
    if n == 0:
        return []
    if n / STREAM_EXPECT_SR <= STREAM_VAD_MIN_S:
        return [(0, n)]  # short enough to send whole; VAD would only add risk

    if STREAM_VAD_BACKEND == "silero":
        regions = _silero_regions(audio_f32)
        if regions is not None:
            return _merge_regions(regions, n)

    try:
        import webrtcvad
    except ImportError:
        logger.warning("webrtcvad not installed; sending utterance unsegmented")
        return [(0, n)]

    pcm = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    vad = webrtcvad.Vad(2)  # matches client_file.apply_vad
    frame = int(STREAM_EXPECT_SR * 0.03)  # 30 ms
    pad = frame * 5                       # lead-in/out, matches client_file.py

    # Collect speech regions, closing a region after ~0.5 s of silence.
    regions: List[List[int]] = []
    start = -1
    silence = 0
    max_silence = int(0.5 / 0.03)
    for i in range(0, n - frame, frame):
        try:
            speech = vad.is_speech(pcm[i:i + frame].tobytes(), STREAM_EXPECT_SR)
        except Exception:
            return [(0, n)]
        if speech:
            silence = 0
            if start == -1:
                start = max(0, i - pad)
        elif start != -1:
            silence += 1
            if silence > max_silence:
                regions.append([start, min(n, i + pad)])
                start = -1
                silence = 0
    if start != -1:
        regions.append([start, n])
    if not regions:
        return [(0, n)]

    return _merge_regions(regions, n)


async def transcribe_utterance(audio_f32: np.ndarray, language: Optional[str], context: str = "") -> Tuple[str, Optional[str]]:
    """Transcribe a full utterance, re-segmenting server-side when it is long enough."""
    if not STREAM_SERVER_VAD:
        return await llama_transcribe(audio_f32, language, context)

    spans = vad_split(audio_f32)
    if len(spans) <= 1:
        return await llama_transcribe(audio_f32, language, context)

    results = await asyncio.gather(
        *(llama_transcribe(audio_f32[s:e], language, context) for s, e in spans)
    )
    texts = [t for t, _ in results if t]
    lang = next((l for _, l in results if l), language)
    return " ".join(texts).strip(), lang


async def llama_transcribe(audio_f32: np.ndarray, language: Optional[str], context: str = "") -> Tuple[str, Optional[str]]:
    """Transcribe one audio array via the ASR llama-server."""
    import httpx

    # Context-leak guard: only relevant when a vocabulary context is in play.
    if CONTEXT_GUARD and context:
        ratio = await asyncio.to_thread(speech_ratio, audio_f32)
        if ratio < CONTEXT_GUARD_VAD_RATIO:
            logger.debug("context guard: no speech (vad ratio %.3f)", ratio)
            return "", language
        if ratio < CONTEXT_GUARD_PROBE_RATIO:
            # Borderline: let the model itself decide before forcing a language.
            if await _probe_language(audio_f32, context) is None:
                logger.debug("context guard: model reported no speech")
                return "", language

    body = _asr_request_body(audio_f32, language, context)
    url = f"http://localhost:{ASR_INTERNAL_PORT}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    text, detected = split_asr_output(raw)

    # Final layer: the model may still regurgitate the context even when the audio
    # passed the gates above (observed on long stretches of digital silence).
    if CONTEXT_GUARD and context and text:
        ov = context_overlap(text, context)
        if ov >= CONTEXT_GUARD_OVERLAP:
            logger.debug("context guard: dropped output, %.0f%% n-gram overlap with context", ov * 100)
            return "", language

    # With a prefill the model echoes the language we forced; report that.
    return text, (language or detected)


# -----------------------------
# llama-server subprocess management
# -----------------------------
def _resolve(path: str) -> str:
    """Resolve a model path relative to this file's directory if not absolute."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return here if os.path.exists(here) else path


def _check_llama_server() -> None:
    if shutil.which(LLAMA_SERVER_BIN) is None:
        raise RuntimeError(
            f"'{LLAMA_SERVER_BIN}' not found on PATH. Install llama.cpp (>= b9173) "
            f"or set LLAMA_SERVER_BIN to its full path."
        )


def _start_llama_server(model: str, mmproj: str, port: int, device: str, ctx: int, ngl: str, parallel: int, tag: str) -> subprocess.Popen:
    model, mmproj = _resolve(model), _resolve(mmproj)
    for p, what in ((model, "model"), (mmproj, "mmproj")):
        if not os.path.exists(p):
            raise RuntimeError(
                f"{tag} {what} GGUF not found: {p}\n"
                f"Note: GGUF multimodal models ship as two files — the decoder and its "
                f"mmproj encoder. Without the mmproj, audio/image input is silently disabled."
            )
    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model,
        "--mmproj", mmproj,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", str(ngl),
        "-c", str(ctx),
        "--parallel", str(parallel),
    ]
    env = os.environ.copy()
    # These are set for the vLLM CPU backend in server.py; they break llama.cpp GPU offload.
    for k in ("VLLM_TARGET_DEVICE", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_CPU_KVCACHE_SPACE", "VLLM_LIMIT_MM_PER_PROMPT"):
        env.pop(k, None)
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
        logger.info(f"{tag} llama-server pinned to GPU {device}")
    logger.info(f"Starting {tag} llama-server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def _wait_ready(port: int, proc: subprocess.Popen, tag: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.error(f"{tag} llama-server exited early with code {proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(2)
    logger.error(f"{tag} llama-server did not become ready within {timeout}s")
    return False


def _check_modality(port: int, key: str, tag: str) -> bool:
    """Verify the mmproj actually loaded — llama-server starts fine without it."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/props", timeout=5) as r:
            mods = json.loads(r.read()).get("modalities", {})
        if not mods.get(key):
            logger.error(f"{tag} server reports {key}=false — mmproj missing or incompatible.")
            return False
        return True
    except Exception as e:
        logger.error(f"{tag} /props check failed: {e}")
        return False


def load_models():
    """Start llama-server subprocesses and wait for them to become ready."""
    global model_status

    if get_env_bool("ENABLE_ASR_MODEL", "true"):
        try:
            _check_llama_server()
            proc = _start_llama_server(
                _cli.asr_model, _cli.asr_mmproj, ASR_INTERNAL_PORT,
                ASR_DEVICE, ASR_CTX_SIZE, ASR_NGL, ASR_PARALLEL, "ASR",
            )
            models["_asr_proc"] = proc
            if not _wait_ready(ASR_INTERNAL_PORT, proc, "ASR"):
                model_status = "failed"
                return
            if not _check_modality(ASR_INTERNAL_PORT, "audio", "ASR"):
                model_status = "failed"
                return
            models["asr"] = True
            logger.info(f"ASR llama-server ready on port {ASR_INTERNAL_PORT}")
        except Exception as e:
            logger.exception(f"Failed to start ASR llama-server: {e}")
            model_status = "failed"
            return

    if VL_MODEL_NAME:
        try:
            _check_llama_server()
            proc = _start_llama_server(
                VL_MODEL_NAME, _cli.vl_mmproj, VL_PORT,
                VL_DEVICE, VL_CTX_SIZE, VL_NGL, 1, "VL",
            )
            models["_vl_proc"] = proc
            if _wait_ready(VL_PORT, proc, "VL") and _check_modality(VL_PORT, "vision", "VL"):
                logger.info(f"VL llama-server ready on port {VL_PORT}")
            else:
                logger.error("VL llama-server failed to start; continuing without it.")
                models.pop("_vl_proc", None)
                proc.terminate()
        except Exception as e:
            logger.exception(f"Failed to start VL llama-server: {e}")
            models.pop("_vl_proc", None)

    model_status = "ready"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Qwen3-ASR Server (llama.cpp backend)...")

    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    await asyncio.to_thread(load_models)
    model_ready_event.set()
    try:
        yield
    finally:
        for key, tag in (("_vl_proc", "VL"), ("_asr_proc", "ASR")):
            proc = models.pop(key, None)
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                logger.info(f"{tag} llama-server stopped.")
        models.clear()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Shutdown complete.")


# -----------------------------
# App
# -----------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    return {
        "status": model_status,
        "backend": "llama.cpp",
        "limits": {
            "max_concurrent_decode": MAX_CONCURRENT_DECODE,
            "max_concurrent_infer": MAX_CONCURRENT_INFER,
            "threadpool_workers": THREADPOOL_WORKERS,
        },
        "memory": {
            "ram_total_mb": mem.total // (1024 * 1024),
            "ram_available_mb": mem.available // (1024 * 1024),
            "ram_percent": mem.percent,
        },
    }


@app.get("/vl/health")
async def vl_health():
    proc = models.get("_vl_proc")
    running = proc is not None and proc.poll() is None
    return {"enabled": running, "model": VL_MODEL_NAME or None, "port": VL_PORT if running else None}


@app.api_route("/vl/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def vl_proxy(path: str, request: Request):
    """Proxy VL requests through the main server so clients need no direct tunnel to VL_PORT."""
    import httpx
    from fastapi.responses import Response as _R
    from fastapi.responses import StreamingResponse as _SR

    url = f"http://localhost:{VL_PORT}/{path}"
    body = await request.body()
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    try:
        req_body_json = json.loads(body) if body else {}
    except Exception:
        req_body_json = {}

    if req_body_json.get("stream"):

        async def _iter():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(request.method, url, content=body, headers=fwd_headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return _SR(_iter(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.request(request.method, url, content=body, headers=fwd_headers)
    return _R(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))


@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g. en, de, fr). None for auto-detect."),
    forced_alignment: bool = Query(False, description="Not supported by the llama.cpp backend"),
):
    await model_ready_event.wait()

    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")
    if "asr" not in models:
        raise HTTPException(status_code=503, detail="ASR model is not enabled or failed to load.")
    if forced_alignment:
        raise HTTPException(
            status_code=501,
            detail="forced_alignment requires the Qwen3-ForcedAligner model, which has no GGUF build. Use server.py for word-level timestamps.",
        )

    full_lang = map_language(language)

    async def decode_one(f: UploadFile):
        content = await f.read()
        audio, sr = await to_thread_limited(decode_sem, read_audio_file, content, f.filename or "")
        return await asyncio.to_thread(resample_16k, audio, sr)

    try:
        audio_batch = await asyncio.gather(*(decode_one(f) for f in files))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    try:
        async def one(a):
            async with infer_sem:
                # Re-segment long uploads: a whole file sent as one request overflows
                # the model's context (llama-server 400s, and quality collapses well
                # before that). transcribe_utterance() splits at speech boundaries.
                return await transcribe_utterance(a, full_lang)

        results = await asyncio.gather(*(one(a) for a in audio_batch))
        return [{"text": t, "language": lang} for t, lang in results]
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/transcribe-streaming")
async def websocket_endpoint(
    ws: WebSocket,
    language: Optional[str] = Query(None),
    forced_alignment: bool = Query(False),  # kept for API symmetry
):
    """Streaming transcription.

    llama.cpp exposes no incremental ASR state, so partials are produced by
    re-transcribing a growing window of the utterance every
    STREAM_PARTIAL_EVERY_S. The `final` sent on stop is a single clean
    transcription of the whole utterance, so quality matches /transcribe.
    """
    await ws.accept()
    await model_ready_event.wait()

    if model_status != "ready" or "asr" not in models:
        await ws.close(code=1011, reason=f"Server not ready: {model_status}")
        return

    full_lang = map_language(language)
    started = False
    context = ""

    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    buf_parts: List[np.ndarray] = []
    buf_n = 0
    last_partial_mono = 0.0
    partial_task: Optional[asyncio.Task] = None

    def _collected(truncate: bool = True) -> np.ndarray:
        """Buffered audio for this utterance.

        Partials pass truncate=True: only the trailing STREAM_MAX_UTTERANCE_S is
        re-decoded, bounding the cost of the growing-window approach. The final
        pass uses truncate=False so no speech is dropped — vad_split() re-segments
        it properly instead.
        """
        if not buf_parts:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(buf_parts) if len(buf_parts) > 1 else buf_parts[0]
        if not truncate:
            return audio
        max_n = int(STREAM_MAX_UTTERANCE_S * STREAM_EXPECT_SR)
        return audio[-max_n:] if audio.size > max_n else audio

    async def send_partial():
        """Re-transcribe the utterance so far and emit a partial."""
        audio = _collected()
        if audio.size < int(STREAM_PARTIAL_MIN_S * STREAM_EXPECT_SR):
            return
        try:
            async with infer_sem:
                text, lang = await llama_transcribe(audio, full_lang, context)
            if text:
                await ws.send_json({"type": "partial", "text": text, "language": lang})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"partial transcription failed: {e}")

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] != "websocket.receive":
                continue

            # Control messages
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    data = None

                if isinstance(data, dict):
                    t = data.get("type")

                    if t == "start":
                        started = True
                        client_sr = int(data.get("sample_rate_hz", 0)) if data.get("sample_rate_hz") else None
                        fmt = data.get("format")
                        context = data.get("context", "") or ""

                        if client_sr != STREAM_EXPECT_SR or fmt not in (None, "pcm_s16le"):
                            await ws.send_json({"type": "error", "message": f"Only pcm_s16le @ {STREAM_EXPECT_SR}Hz supported"})
                            await ws.close(code=1003)
                            return

                        buf_parts, buf_n = [], 0
                        last_partial_mono = time.monotonic()
                        if full_lang is not None:
                            await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                        continue

                    if t == "stop":
                        if partial_task and not partial_task.done():
                            partial_task.cancel()
                        audio = _collected(truncate=False)
                        text, lang = "", full_lang
                        if audio.size > 0:
                            try:
                                async with infer_sem:
                                    text, lang = await transcribe_utterance(audio, full_lang, context)
                            except Exception as e:
                                logger.exception(f"final transcription failed: {e}")
                                await ws.send_json({"type": "error", "message": str(e)})
                        await ws.send_json({"type": "final", "text": text, "language": lang})
                        await ws.close(code=1000)
                        return

            # Audio frames
            if msg.get("bytes"):
                if not started:
                    await ws.send_json({"type": "error", "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first"})
                    await ws.close(code=1002)
                    return

                audio_int16 = np.frombuffer(msg["bytes"], dtype=np.int16)
                if audio_int16.size == 0:
                    continue

                buf_parts.append(audio_int16.astype(np.float32) / 32768.0)
                buf_n += audio_int16.size

                now = time.monotonic()
                if buf_n >= STREAM_MIN_SAMPLES and (now - last_partial_mono) >= STREAM_PARTIAL_EVERY_S:
                    last_partial_mono = now
                    # Fire-and-forget so audio intake is never blocked by decoding.
                    if partial_task is None or partial_task.done():
                        partial_task = asyncio.create_task(send_partial())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass
    finally:
        if partial_task and not partial_task.done():
            partial_task.cancel()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=_cli.port)
