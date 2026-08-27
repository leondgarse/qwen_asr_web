"""Qwen3-ASR transcription server (llama.cpp backend).

Serves /transcribe, /transcribe-streaming, /vl/health and /vl/proxy on port 9002.
Rather than loading models in-process, it spawns llama-server subprocesses and
proxies to them: the ASR model on ASR_INTERNAL_PORT and, with --qwenvl, a VL
model on VL_PORT.

    python server.py
    python server.py --qwenvl                 # + VL model
    python server.py --qwenvl --vl-device 1   # VL on GPU 1

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
import re
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
_parser.add_argument("--asr-model", default=os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B-Q8_0.gguf"), help="Path to ASR decoder GGUF; falls back to --asr-hf if missing")
_parser.add_argument("--asr-mmproj", default=os.getenv("ASR_MMPROJ_PATH", "Qwen/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"), help="Path to ASR mmproj GGUF")
_parser.add_argument("--vl-mmproj", default=os.getenv("VL_MMPROJ_PATH", "Qwen/mmproj-Qwen3-VL-4B-Instruct-F16.gguf"), help="Path to VL mmproj GGUF")
_parser.add_argument("--asr-hf", default=os.getenv("ASR_HF_REPO", "ggml-org/Qwen3-ASR-1.7B-GGUF"), metavar="REPO[:QUANT]",
                     help="HF GGUF repo used when the local ASR files are absent (llama-server -hf)")
_parser.add_argument("--vl-hf", default=os.getenv("VL_HF_REPO", "unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M"), metavar="REPO[:QUANT]",
                     help="HF GGUF repo used when the local VL files are absent (llama-server -hf)")
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
# Generous by default: a first run with no local GGUF downloads ~2.5 GB (ASR) or
# ~3.3 GB (VL) before the server binds its port.
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "1800"))

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
# echoed along with the terms, so stripping them after generation (as the previous
# vLLM server did) is a losing game; the checks below prevent the generation instead.
#
# Layer 1 (free): webrtcvad speech ratio. Non-speech measures <= 0.02, real speech
#   >= 0.30 — a wide margin, so a low ratio skips the request entirely.
# Layer 2 (one extra request, only for borderline audio): ask without a language
#   prefill and let the model self-report. It answers "language None" for
#   non-speech. Note the prefill that forces language (and prevents drift)
#   suppresses this signal, which is why it must be probed separately.
# Layer 3 (free): reject output whose 4-grams overlap the context — catches the
#   residual cases where layers 1-2 both pass (e.g. long digital silence).
# ── Audio capture (for building test fixtures) ───────────────────────────────
# Off by default: this writes every utterance received over the WebSocket to
# disk, which is a privacy-relevant side effect and grows without bound. Enable
# deliberately when collecting real in-class audio, which is otherwise the
# hardest thing to get for testing the live mic path.
#
# Each session writes <dir>/<session>/NNN_<ms>.wav plus a transcript.jsonl with
# the text, language and offset per utterance, so a capture replays directly as
# a scoring fixture. Audio is saved *before* auto-gain so the recording reflects
# what the browser actually sent.
CAPTURE_AUDIO = get_env_bool("CAPTURE_AUDIO", "false")
CAPTURE_DIR = os.getenv("CAPTURE_DIR", "captures")
CAPTURE_MAX_MB = float(os.getenv("CAPTURE_MAX_MB", "2048"))

# Auto-gain. Quiet room recordings are the dominant cause of both mis-decoding and
# context regurgitation; normalizing before inference fixes far more than any
# post-hoc filter. Set AUDIO_TARGET_RMS=0 to disable.
AUDIO_TARGET_RMS = float(os.getenv("AUDIO_TARGET_RMS", "0.08"))
AUDIO_MAX_GAIN = float(os.getenv("AUDIO_MAX_GAIN", "20.0"))

CONTEXT_GUARD = get_env_bool("CONTEXT_GUARD", "true")
CONTEXT_GUARD_VAD_RATIO = float(os.getenv("CONTEXT_GUARD_VAD_RATIO", "0.05"))
CONTEXT_GUARD_PROBE_RATIO = float(os.getenv("CONTEXT_GUARD_PROBE_RATIO", "0.15"))
# Measured on real leaks vs real speech: regurgitated lines score 1.00, while
# genuine speech quoting a slide term tops out around 0.50. 0.75 separates them.
CONTEXT_GUARD_OVERLAP = float(os.getenv("CONTEXT_GUARD_OVERLAP", "0.75"))

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


def _capture_dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def save_capture_pcm(session: str, pcm: np.ndarray, meta: dict) -> None:
    """Write already-int16 PCM verbatim (used for the raw session stream)."""
    _write_capture(session, 0, pcm, meta)


def save_capture(session: str, index: int, audio_f32: np.ndarray, meta: dict) -> None:
    """Persist one utterance as 16-bit WAV plus a JSONL metadata line.

    Best-effort: capture must never break a live lecture, so every failure is
    logged and swallowed. Called with the raw (pre-gain) audio.
    """
    if audio_f32.size == 0:
        return
    _write_capture(session, index, (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16), meta)


def _write_capture(session: str, index: int, pcm: np.ndarray, meta: dict) -> None:
    """Write one WAV + JSONL row. Best-effort: capture must never break a live
    lecture, so every failure is logged and swallowed."""
    if not CAPTURE_AUDIO or pcm.size == 0:
        return
    try:
        base = os.path.join(CAPTURE_DIR, session)
        os.makedirs(base, exist_ok=True)
        if _capture_dir_size_mb(CAPTURE_DIR) > CAPTURE_MAX_MB:
            logger.warning("capture: %s over CAPTURE_MAX_MB (%.0f MB), skipping",
                           CAPTURE_DIR, CAPTURE_MAX_MB)
            return
        # index 0 is the continuous raw stream for the whole recording; 1+ are
        # the individual utterances the VAD produced.
        name = (f"{int(time.time() * 1000)}_RAW" if index == 0
                else f"{int(time.time() * 1000)}_{index:03d}")
        with wave.open(os.path.join(base, name + ".wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(STREAM_EXPECT_SR)
            w.writeframes(pcm.tobytes())
        rec = {"file": name + ".wav", "seconds": round(audio_f32.size / STREAM_EXPECT_SR, 2), **meta}
        with open(os.path.join(base, "transcript.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("capture failed: %s", e)


def normalize_gain(audio_f32: np.ndarray, target_rms: float = 0.0) -> np.ndarray:
    """Scale audio to a target RMS.

    Lecture-hall recordings arrive far quieter than a close mic: measured RMS
    0.0085-0.014 versus ~0.05 for a good capture. At those levels the model
    mis-decodes ("consent" -> "science") and, when a vocabulary context is set,
    falls back to regurgitating it. Normalizing is the single highest-impact fix
    measured — it took mic WER from 96.4% to 31.6% on a quiet lecture.

    Peak-limited so a loud transient cannot clip the whole utterance.
    """
    target_rms = target_rms or AUDIO_TARGET_RMS
    if audio_f32.size == 0 or target_rms <= 0:
        return audio_f32
    rms = float(np.sqrt(np.mean(audio_f32.astype(np.float64) ** 2)))
    if rms < 1e-6:
        return audio_f32          # digital silence: nothing to amplify
    gain = target_rms / rms
    peak = float(np.max(np.abs(audio_f32))) or 1e-6
    gain = min(gain, 0.97 / peak, AUDIO_MAX_GAIN)
    if gain <= 1.0:
        return audio_f32          # already loud enough; never attenuate
    return (audio_f32 * gain).astype(np.float32)


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


def _words(text: str) -> List[str]:
    """Lowercase alphanumeric tokens. Punctuation is dropped, not kept attached:
    the model writes "Consent Obligation." where a slide says "Consent
    Obligation", so any comparison that keeps punctuation misses the match."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _ngrams(tokens: List[str], n: int) -> set:
    return {" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def _bare(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def collapse_repeats(text: str, max_rep: int = 1) -> str:
    """Collapse a phrase repeated back-to-back ("X. X. X." -> "X.").

    Repetition loops are the classic ASR hallucination on unclear audio, and are
    how leaked context usually arrives. Comparison ignores punctuation because
    the model varies it between repeats.
    """
    w = text.split()
    for plen in range(1, 8):
        out: List[str] = []
        i = 0
        while i < len(w):
            head = [_bare(x) for x in w[i:i + plen]]
            run = 1
            while head and head == [_bare(x) for x in w[i + run * plen:i + (run + 1) * plen]]:
                run += 1
            out += w[i:i + plen * min(run, max_rep)] if run > 1 else w[i:i + plen]
            i += plen * run
        w = out
    return " ".join(w)


def context_fraction(text: str, context: str, n: int = 3) -> float:
    """Share of the output's n-grams that occur verbatim in the context."""
    t = _words(text)
    if not t:
        return 0.0
    if len(t) < n:
        return 1.0 if " ".join(t) in " ".join(_words(context)) else 0.0
    g = _ngrams(t, n)
    return len(g & _ngrams(_words(context), n)) / len(g)


def scrub_context(text: str, context: str, n: int = 3) -> str:
    """Remove regurgitated context from a decoded line.

    Three steps, in order:
      1. collapse repetition loops — "X. X. X." shares its n-gram profile with a
         single "X", so a loop would otherwise hide from the ratio test;
      2. drop the line outright if what remains is mostly context;
      3. trim context spans from the *edges* only. Leaks attach at a boundary,
         while a term appearing mid-sentence is usually the lecturer genuinely
         saying it. Leading trims are gated on a loop having been present,
         because a line may legitimately open with a slide term.
    """
    if not context or not text:
        return text
    collapsed = collapse_repeats(text)
    looped = collapsed != text
    text = collapsed

    if context_fraction(text, context, n) >= CONTEXT_GUARD_OVERLAP:
        return ""

    words = text.split()
    toks = [_bare(w) for w in words]
    cg = _ngrams(_words(context), n)

    end = len(words)
    while end >= n and " ".join(toks[end - n:end]) in cg:
        end -= 1
    if end < len(words) and (len(words) - end) >= n - 1:
        words, toks = words[:end], toks[:end]

    if looped:
        start = 0
        while start + n <= len(toks) and " ".join(toks[start:start + n]) in cg:
            start += 1
        if start > 0 and start >= n - 1:
            words = words[start + n - 1:] if start + n - 1 <= len(words) else []

    return " ".join(words).strip()


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

    audio_f32 = normalize_gain(audio_f32)

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

    # Final layer: scrub context regurgitation out of the decoded text.
    # This must handle leaks *appended to real speech* — the common form in
    # practice — not just whole-line leaks, and must be punctuation-insensitive:
    # the model writes "Consent Obligation." where the slide says "Consent
    # Obligation", and a whitespace-only n-gram comparison misses that entirely.
    if CONTEXT_GUARD and context and text:
        cleaned = scrub_context(text, context)
        if cleaned != text:
            logger.debug("context guard: scrubbed leaked context from output")
        text = cleaned

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


def _start_llama_server(model: str, mmproj: str, port: int, device: str, ctx: int, ngl: str,
                        parallel: int, tag: str, hf_repo: str = "") -> subprocess.Popen:
    """Start a llama-server for one model.

    Prefers local GGUF files. If either half of the pair is missing and hf_repo is
    set, falls back to `llama-server -hf <repo>`, which downloads the decoder *and*
    its mmproj into the shared HF cache (~/.cache/huggingface/hub) and reuses them
    on later runs. That pairing matters: a decoder without its mmproj starts fine
    but silently disables audio/image input.
    """
    model, mmproj = _resolve(model), _resolve(mmproj)
    missing = [what for p, what in ((model, "model"), (mmproj, "mmproj")) if not os.path.exists(p)]

    if missing and not hf_repo:
        raise RuntimeError(
            f"{tag} {' and '.join(missing)} GGUF not found: {model} / {mmproj}\n"
            f"Note: GGUF multimodal models ship as two files — the decoder and its "
            f"mmproj encoder. Without the mmproj, audio/image input is silently disabled.\n"
            f"Either place both files locally or set a HF repo (--{tag.lower()}-hf)."
        )

    source = ["--model", model, "--mmproj", mmproj]
    if missing:
        logger.info("%s GGUF not found locally (%s); downloading from HF repo %s",
                    tag, ", ".join(missing), hf_repo)
        # -hf pulls the matching mmproj automatically (--mmproj-auto, on by default).
        source = ["-hf", hf_repo]
    models[f"_{tag.lower()}_source"] = hf_repo if missing else model

    cmd = [
        LLAMA_SERVER_BIN,
        *source,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", str(ngl),
        "-c", str(ctx),
        "--parallel", str(parallel),
    ]
    env = os.environ.copy()
    # Left over from the previous vLLM CPU backend; they break llama.cpp GPU offload.
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
                hf_repo=_cli.asr_hf,
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
                hf_repo=_cli.vl_hf,
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
        "capture_audio": CAPTURE_AUDIO,
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


@app.post("/capture/raw")
async def capture_raw(request: Request, session: str = Query("session")):
    """Store the browser's continuous pre-VAD stream for one recording.

    The per-utterance WAVs written during a session only contain what the VAD
    chose to send — on a sample session that discarded 21% of wall time, with
    overlapping utterances, so the original cannot be reconstructed from them.
    This keeps the unbroken stream alongside them.
    """
    if not CAPTURE_AUDIO:
        raise HTTPException(status_code=403, detail="capture disabled (set CAPTURE_AUDIO=true)")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    pcm = np.frombuffer(body, dtype=np.int16)
    name = re.sub(r"[^\w.-]", "_", session)[:64].strip("_") or "session"
    # Written straight through as int16: a float round-trip costs a bit of
    # precision for no benefit, and this file is meant to be a faithful fixture.
    await asyncio.to_thread(save_capture_pcm, name, pcm,
                            {"kind": "raw_session", "text": "", "language": None})
    logger.info("capture: stored raw session %s (%.1f s)", name, pcm.size / STREAM_EXPECT_SR)
    return {"saved": True, "seconds": round(pcm.size / STREAM_EXPECT_SR, 1)}


@app.get("/vl/health")
async def vl_health():
    proc = models.get("_vl_proc")
    running = proc is not None and proc.poll() is None
    model = models.get("_vl_source") or VL_MODEL_NAME
    return {"enabled": running, "model": model or None, "port": VL_PORT if running else None}


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
):
    await model_ready_event.wait()

    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")
    if "asr" not in models:
        raise HTTPException(status_code=503, detail="ASR model is not enabled or failed to load.")

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
    capture_session = time.strftime("%Y%m%d_%H%M%S") if CAPTURE_AUDIO else ""
    capture_index = 0

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
                        if CAPTURE_AUDIO:
                            # Group a lecture's utterances (each its own socket)
                            # under one directory, per the client's session name.
                            client_session = re.sub(r"[^\w.-]", "_", str(data.get("session", ""))[:64]).strip("_")
                            if client_session:
                                capture_session = client_session

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
                        if CAPTURE_AUDIO and audio.size > 0:
                            capture_index += 1
                            await asyncio.to_thread(
                                save_capture, capture_session, capture_index, audio,
                                {"text": text, "language": lang, "has_context": bool(context)},
                            )
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
