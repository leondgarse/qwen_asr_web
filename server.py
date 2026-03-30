import argparse
import os
import io
import json
import asyncio
import logging
import subprocess
import sys
import tempfile
import urllib.request
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import time

# ── CLI args ──────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--qwenvl",
    nargs="?",
    const="Qwen/Qwen3-VL-2B-Instruct",
    metavar="MODEL",
    help="Enable Qwen-VL model (optional model name, default: Qwen2.5-VL-3B-Instruct)",
)
_parser.add_argument("--port", type=int, default=int(os.getenv("ASR_PORT", "9002")), help="Port to listen on (default: 9002)")
_parser.add_argument("--asr-device", default=os.getenv("ASR_DEVICE", ""), metavar="N", help="GPU index for ASR model, e.g. 0 (default: unset)")
_parser.add_argument("--vl-device", default=os.getenv("VL_DEVICE", ""), metavar="N", help="GPU index for VL subprocess, e.g. 1 (default: unset)")
_cli, _ = _parser.parse_known_args()
VL_MODEL_NAME = _cli.qwenvl or os.getenv("VL_MODEL_NAME", "")
VL_PORT = int(os.getenv("VL_PORT", "9004"))
ASR_DEVICE = _cli.asr_device  # GPU index for ASR model, e.g. "0"
VL_DEVICE = _cli.vl_device    # GPU index for VL subprocess, e.g. "1"

import uvicorn
import numpy as np
import soundfile as sf
import torch
import psutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel
from qwen_asr_inference import Qwen3ASRModel, Qwen3ForcedAligner


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
def get_env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")


MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "4"))
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "1"))  # GPU: usually 1
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 5)))

# Streaming buffering/throttling
STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "1600"))  # 100ms @ 16kHz
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "120"))  # throttle partials
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))

CONTEXT_PREFIX = "Reference only — do NOT transcribe this. Vocabulary hint: "
CONTEXT_TAG_START = "[ASR_CONTEXT_START]"
CONTEXT_TAG_END = "[ASR_CONTEXT_END]"

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
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "ru": "Russian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "tr": "Turkish",
        "sv": "Swedish",
        "id": "Indonesian",
        "vi": "Vietnamese",
        "hi": "Hindi",
        "ar": "Arabic",
    }
    return mapping.get(lang_code.lower(), lang_code)


def read_audio_file(file_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """
    Sync decode. Must be called via asyncio.to_thread (or threadpool).
    soundfile first; fallback to ffmpeg via temp file for mp3/m4a/etc.
    Temp file is used (instead of pipe) so container formats like m4a that
    require seeking (moov atom) are handled correctly.
    """
    try:
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    except Exception:
        pass

    suffix = os.path.splitext(filename)[1] if filename else ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        process = subprocess.Popen(
            ["ffmpeg", "-y", "-i", tmp_path, "-f", "wav", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate()
        if process.returncode != 0:
            raise ValueError(f"FFmpeg decoding failed: {err.decode(errors='ignore')}")
        with io.BytesIO(out) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    finally:
        os.unlink(tmp_path)


# -----------------------------
# GPU memory auto-sizing
# -----------------------------
# Estimated GB needed for each component (model weights + KV cache)
# Updated based on actual measurement: Qwen3-ASR-1.7B weights ~3.87 GB + KV cache buffer
_ASR_ESTIMATED_GB = 5.7   # 3.87 GB weights + ~1.8 GB KV cache + overhead (max_model_len=2048, measured on 2080 Ti)
_ASR_ESTIMATED_GB_4K = 6.0  # weights + ~2 GB KV cache (max_model_len=4096, large GPUs)
_VL_ESTIMATED_GB = 6.7    # free GPU needed for VL: profiling peak ~6.17 GB + min KV cache 0.44 GB
_ALIGNER_GB = 1.5  # rough footprint of the 0.6B aligner
_VL_BUFFER_GB = 0.5  # safety margin; util = (free - buffer) / total, vLLM requires util*total <= free
_VL_MAX_GB = 20.0  # cap VL server memory so KV cache doesn't balloon on large GPUs
_GPU_MAX_UTIL = 0.75       # maximum utilization for small GPUs (leave room for others)
_GPU_MAX_UTIL_SHARED = 0.70  # lower cap when ASR and VL share the same GPU; accounts for
                              # ~4 GB CUDA/NCCL overhead not counted in gpu_memory_utilization


def _asr_device_index() -> int:
    """Return the CUDA device index for ASR memory queries."""
    if ASR_DEVICE:
        try:
            return int(ASR_DEVICE)
        except ValueError:
            pass
    return 0


def _auto_asr_gpu_util(with_vl: bool = False) -> float:
    """Compute gpu_memory_utilization as a fraction of total GPU memory.
    When VL shares the same GPU, cap ASR to leave headroom for VL."""
    if not torch.cuda.is_available():
        return 0.15
    dev = _asr_device_index()
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    if with_vl and not VL_DEVICE:
        # VL shares this GPU — give VL _VL_ESTIMATED_GB + _VL_BUFFER_GB; ASR gets the rest.
        # Use _GPU_MAX_UTIL_SHARED (0.70) instead of _GPU_MAX_UTIL (0.75) to account for
        # ~4 GB CUDA/NCCL process overhead that is not captured by gpu_memory_utilization.
        asr_gb = max(total_gb - _VL_ESTIMATED_GB - _VL_BUFFER_GB, _ASR_ESTIMATED_GB)
        util = round(min(asr_gb / total_gb, _GPU_MAX_UTIL_SHARED), 3)
    else:
        asr_gb = _ASR_ESTIMATED_GB_4K if total_gb >= 16 else _ASR_ESTIMATED_GB
        util = round(min(asr_gb / total_gb, _GPU_MAX_UTIL), 3)
    logger.info(f"Auto ASR gpu_memory_utilization={util:.3f} (target {asr_gb:.1f} GB / {total_gb:.1f} GB total, with_vl={with_vl}, vl_device='{VL_DEVICE}')")
    return util


def _auto_asr_max_model_len() -> int:
    """Pick max_model_len for ASR based on total GPU memory."""
    if not torch.cuda.is_available():
        return 2048
    dev = _asr_device_index()
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    return 4096 if total_gb >= 16 else 2048


def _vl_device_index() -> int:
    """Return the CUDA device index to use for VL memory queries.
    If VL_DEVICE is set (e.g. '1'), use that; otherwise fall back to the ASR device
    (since VL shares the same GPU as ASR by default)."""
    if VL_DEVICE:
        try:
            return int(VL_DEVICE)
        except ValueError:
            pass
    return _asr_device_index()


def _auto_vl_gpu_util() -> float:
    """Compute gpu_memory_utilization for VL model using actual free GPU memory at call time."""
    if not torch.cuda.is_available():
        return 0.55
    dev = _vl_device_index()
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    free_gb = free_bytes / 1024**3
    if free_gb < _VL_ESTIMATED_GB + _VL_BUFFER_GB:
        raise RuntimeError(
            f"Not enough free GPU memory on device {dev} to start VL model: "
            f"{free_gb:.1f} GB free, need at least {_VL_ESTIMATED_GB + _VL_BUFFER_GB:.1f} GB. "
            f"Use --vl-device to select a different GPU."
        )
    usable_gb = min(max(0.0, free_gb - _VL_BUFFER_GB), _VL_MAX_GB)
    # vLLM requires gpu_memory_utilization * total <= free_at_startup.
    # util = usable / total satisfies this since usable = free - buffer < free.
    # available_kv = total * util - peak_memory, so buffer must be small enough
    # that util >= (peak + min_kv) / total.
    util = round(min(usable_gb / total_gb, 0.90), 3)
    logger.info(f"Auto VL gpu_memory_utilization={util:.3f} ({usable_gb:.1f} GB usable / {free_gb:.1f} GB free / {total_gb:.1f} GB total, device={dev})")
    return util


def _auto_vl_max_model_len() -> int:
    """Pick max_model_len for VL based on free GPU memory, capped at 16384."""
    if not torch.cuda.is_available():
        return 4096
    dev = _vl_device_index()
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    free_gb = free_bytes / 1024**3
    if free_gb >= 20:
        return 16384
    elif free_gb >= 12:
        return 8192
    elif free_gb >= 6:
        return 4096
    return 2048


# -----------------------------
# VL server helpers
# -----------------------------
def _start_vl_server(model_name: str, vl_util: float) -> subprocess.Popen:
    vl_max_len_env = os.getenv("VL_MAX_MODEL_LEN", "")
    vl_max_len = int(vl_max_len_env) if vl_max_len_env else _auto_vl_max_model_len()
    logger.info(f"VL max_model_len={vl_max_len}")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--port",
        str(VL_PORT),
        "--host",
        "0.0.0.0",
        "--trust-remote-code",
        "--max-model-len",
        str(vl_max_len),
        "--enable-prefix-caching",
        "--enforce-eager",  # skip CUDA graph compilation — faster startup, less memory spike
    ]
    if torch.cuda.is_available():
        cmd += ["--gpu-memory-utilization", str(vl_util)]
    vl_env = os.environ.copy()
    # ASR model sets VLLM_TARGET_DEVICE=cpu for its CPU backend; VL server must run on GPU
    vl_env.pop("VLLM_TARGET_DEVICE", None)
    vl_env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    vl_env.pop("VLLM_CPU_KVCACHE_SPACE", None)
    vl_env.pop("VLLM_LIMIT_MM_PER_PROMPT", None)
    if VL_DEVICE:
        # Pin VL subprocess to a specific GPU (e.g. VL_DEVICE=1 for second GPU)
        vl_env["CUDA_VISIBLE_DEVICES"] = VL_DEVICE
        logger.info(f"VL subprocess pinned to GPU {VL_DEVICE} via CUDA_VISIBLE_DEVICES")
    logger.info(f"Starting VL server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=vl_env)


def _wait_vl_ready(timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{VL_PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


# -----------------------------
# Model loading
# -----------------------------
def load_models():
    global model_status
    logger.info("Loading models...")
    model_status = "loading_models"
    asr_util = 0.0  # tracked for VL auto-sizing

    if get_env_bool("ENABLE_ASR_MODEL", "true"):
        if Qwen3ASRModel is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ASRModel missing).")
        _default_asr = next((m for m in ("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B") if os.path.isdir(m)), "Qwen/Qwen3-ASR-1.7B")
        model_name = os.getenv("ASR_MODEL_NAME", _default_asr)
        logger.info(f"Loading ASR Model: {model_name}...")
        if ASR_DEVICE:
            os.environ["CUDA_VISIBLE_DEVICES"] = ASR_DEVICE
            logger.info(f"ASR model pinned to GPU {ASR_DEVICE} via CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")
        os.environ.setdefault("VLLM_LIMIT_MM_PER_PROMPT", "audio=32768")
        asr_util_env = os.getenv("GPU_MEMORY_UTILIZATION", "")
        asr_util = float(asr_util_env) if asr_util_env else _auto_asr_gpu_util(with_vl=bool(VL_MODEL_NAME))  # noqa: F841 (used below)
        asr_max_len = int(os.getenv("MAX_MODEL_LEN", str(_auto_asr_max_model_len())))
        models["asr"] = Qwen3ASRModel.LLM(
            model=model_name,
            gpu_memory_utilization=asr_util,
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "8192")),
            max_model_len=asr_max_len,
            max_num_batched_tokens=asr_max_len,
            enforce_eager=True,
            trust_remote_code=True,
            enable_prefix_caching=get_env_bool("ENABLE_PREFIX_CACHING", "true"),
        )
        logger.info("ASR Model loaded successfully.")
    else:
        logger.info("ASR Model disabled via ENABLE_ASR_MODEL.")

    if get_env_bool("ENABLE_ALIGNER_MODEL", "false"):
        if Qwen3ForcedAligner is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ForcedAligner missing).")
        aligner_name = os.getenv("ALIGNER_MODEL_NAME", "Qwen/Qwen3-ForcedAligner-0.6B")
        logger.info(f"Loading Aligner Model: {aligner_name}...")
        models["aligner"] = Qwen3ForcedAligner.from_pretrained(
            aligner_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        logger.info("Aligner Model loaded successfully.")
    else:
        logger.info("Aligner Model disabled via ENABLE_ALIGNER_MODEL.")

    # Warmup (best-effort)
    if "asr" in models:
        logger.info("Warming up ASR model (best-effort)...")
        model_status = "warming_up"
        try:
            dummy_wav = np.zeros(16000, dtype=np.float32)
            dummy_sr = 16000
            models["asr"].transcribe(
                audio=[(dummy_wav, dummy_sr)],
                language=["English"],
                return_time_stamps=False,
            )
            state = models["asr"].init_streaming_state(
                unfixed_chunk_num=2,
                unfixed_token_num=5,
                chunk_size_sec=2.0,
            )
            for n in [320, 640, 1024, 3200] + [3200] * 25:
                models["asr"].streaming_transcribe(dummy_wav[:n], state)
            models["asr"].finish_streaming_transcribe(state)
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    if VL_MODEL_NAME:
        logger.info(f"Starting VL model server: {VL_MODEL_NAME} on port {VL_PORT} ...")
        vl_util_env = os.getenv("VL_GPU_MEMORY_UTILIZATION", "")
        vl_util = float(vl_util_env) if vl_util_env else _auto_vl_gpu_util()
        proc = _start_vl_server(VL_MODEL_NAME, vl_util)
        models["_vl_proc"] = proc
        if _wait_vl_ready():
            logger.info(f"VL server ready on port {VL_PORT}.")
        else:
            logger.warning("VL server did not become ready within timeout; continuing anyway.")

    model_status = "ready"
    model_ready_event.set()
    logger.info("Server is ready to accept requests.")


# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Qwen3-ASR Server...")

    # Bigger threadpool helps when decoding + websocket buffering + other to_thread calls happen together.
    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    load_models()
    try:
        yield
    finally:
        # Shutdown VL subprocess first
        proc = models.pop("_vl_proc", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            logger.info("VL server stopped.")
        models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    info = {
        "status": model_status,
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
    if torch.cuda.is_available():
        info["memory"]["gpu_allocated_mb"] = torch.cuda.memory_allocated() // (1024 * 1024)
        info["memory"]["gpu_reserved_mb"] = torch.cuda.memory_reserved() // (1024 * 1024)
    return info


@app.get("/vl/health")
async def vl_health():
    proc = models.get("_vl_proc")
    running = proc is not None and proc.poll() is None
    return {"enabled": running, "model": VL_MODEL_NAME or None, "port": VL_PORT if running else None}


@app.api_route("/vl/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def vl_proxy(path: str, request: Request):
    """Proxy VL requests through the main server so remote clients don't need a direct tunnel to VL_PORT."""
    import httpx
    from fastapi.responses import StreamingResponse as _SR

    url = f"http://localhost:{VL_PORT}/{path}"
    body = await request.body()
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    from fastapi.responses import Response as _R

    # Check if client wants streaming (SSE) or a plain response
    req_body_json = {}
    try:
        import json as _json

        req_body_json = _json.loads(body) if body else {}
    except Exception:
        pass

    if req_body_json.get("stream"):

        async def _iter():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(request.method, url, content=body, headers=fwd_headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return _SR(_iter(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.request(request.method, url, content=body, headers=fwd_headers)
        return _R(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))


@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g. en, de, fr). None for auto-detect."),
    forced_alignment: bool = Query(False, description="Enable forced alignment (timestamps)"),
):
    await model_ready_event.wait()

    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")
    if "asr" not in models:
        raise HTTPException(status_code=503, detail="ASR model is not enabled or failed to load.")

    full_lang = map_language(language)

    async def decode_one(f: UploadFile):
        content = await f.read()
        return await to_thread_limited(decode_sem, read_audio_file, content, f.filename or "")

    # Decode concurrently (limited)
    try:
        audio_batch = await asyncio.gather(*(decode_one(f) for f in files))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Inference (explicitly limited, because GPU concurrency is not free)
    try:
        async with infer_sem:
            results = await asyncio.to_thread(
                models["asr"].transcribe,
                audio=audio_batch,
                language=[full_lang] * len(audio_batch),
                return_time_stamps=False,
            )

        response_list = []

        if forced_alignment:
            if "aligner" not in models:
                raise HTTPException(status_code=503, detail="Aligner model is not enabled or failed to load.")

            texts = [r.text for r in results]

            async with infer_sem:
                alignment_results = await asyncio.to_thread(
                    models["aligner"].align,
                    audio=audio_batch,
                    text=texts,
                    language=[full_lang] * len(audio_batch),
                )

            for i, res in enumerate(results):
                response_list.append({"text": res.text, "language": res.language, "timestamps": alignment_results[i]})
        else:
            for res in results:
                response_list.append({"text": res.text, "language": res.language})

        return response_list

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/transcribe-streaming")
async def websocket_endpoint(
    ws: WebSocket,
    language: Optional[str] = Query(None),
    forced_alignment: bool = Query(False),  # kept for API symmetry; not yet used in streaming
):
    await ws.accept()

    # do wait until we know the outcome
    await model_ready_event.wait()

    if model_status != "ready" or "asr" not in models:
        await ws.close(code=1011, reason=f"Server not ready: {model_status}")
        return

    full_lang = map_language(language)
    client_sr = None
    started = False

    # Note: State initialization is moved down to after the `start` message is received,
    # so we can use the context provided by the client.
    state = None

    # Send ready
    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    buf_parts: List[np.ndarray] = []
    buf_n = 0
    last_partial_ts = 0.0

    async def flush_and_infer(send_partial: bool):
        nonlocal buf_parts, buf_n, last_partial_ts, state
        if state is None or buf_n <= 0:
            return
        chunk = np.concatenate(buf_parts, axis=0) if len(buf_parts) > 1 else buf_parts[0]
        buf_parts = []
        buf_n = 0

        async with infer_sem:
            await asyncio.to_thread(models["asr"].streaming_transcribe, chunk, state)

        if send_partial:
            now = time.monotonic()
            if (now - last_partial_ts) * 1000.0 >= PARTIAL_INTERVAL_MS:
                text = state.text or ""
                if context:
                    text = strip_prompt(text, context)
                await ws.send_json({"type": "partial", "text": text, "language": state.language})
                last_partial_ts = now

    def strip_prompt(text: str, ctx_raw: str) -> str:
        """Surgically strip context instructions and vocabulary content from response."""
        if not text:
            return ""

        # 1. Sentinel Tags are the highest priority and most reliable
        if CONTEXT_TAG_END in text:
            return text.split(CONTEXT_TAG_END, 1)[1].strip()

        # 2. If we see any "Instruction Zone" markers, we assume the model is echoing the prompt.
        # This allows us to safely strip the following lines if they match the context.
        instruction_markers = [
            CONTEXT_TAG_START,
            "Reference only",
            "Vocabulary hint:",
            "Vocabulary hint",
            "[Reference Only]",
            "[ASR_CONTEXT_START]"
        ]
        
        lines = text.splitlines()
        skip_idx = -1
        in_instruction_block = False
        
        # Prepare context lines for comparison
        ctx_lines = [l.strip() for l in ctx_raw.splitlines() if l.strip()]
        
        for i, line in enumerate(lines):
            l = line.strip()
            if not l:
                if in_instruction_block:
                    skip_idx = i
                continue
                
            # Check for instruction start or continuation
            is_instruction = any(m in l for m in instruction_markers)
            # Check if the line is exactly one of the vocabulary bits provided
            is_context_echo = any(l == cl for cl in ctx_lines) or (ctx_raw.strip() and l == ctx_raw.strip())
            
            if is_instruction:
                in_instruction_block = True
                skip_idx = i
            elif is_context_echo and (in_instruction_block or i == 0):
                # If it's a context echo and we've seen an instruction OR it's the very first line
                # (Model often jumps straight to echoing the context list)
                skip_idx = i
            else:
                # We hit a line that doesn't look like prompt instructions or context content.
                # Stop stripping here.
                break
                
        if skip_idx != -1:
            return "\n".join(lines[skip_idx+1:]).strip()

        # 3. Final fallback: Extreme safety check for raw context leakage WITHOUT instructions.
        # Only strip if it's a substantial exact match at the very beginning of the response.
        l_text = text.lstrip()
        l_ctx = ctx_raw.strip()
        if l_ctx and len(l_ctx) > 5 and l_text.startswith(l_ctx):
            remaining = l_text[len(l_ctx):].lstrip()
            # If what follows looks like a continuation of speech or a newline
            if not remaining or remaining[0] in ("\n", "\r", " ", ".", ",", ";"):
                return remaining.strip()
                
        return text.strip()

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
                        context = data.get("context", "")

                        if client_sr != STREAM_EXPECT_SR or fmt not in (None, "pcm_s16le"):
                            await ws.send_json({"type": "error", "message": f"Only pcm_s16le @ {STREAM_EXPECT_SR}Hz supported"})
                            await ws.close(code=1003)
                            return

                        # Init streaming state off event loop + limited concurrency (GPU touch)
                        try:
                            full_context = f"{CONTEXT_TAG_START}\n{CONTEXT_PREFIX}{context}\n{CONTEXT_TAG_END}" if context else ""
                            async with infer_sem:
                                state = await asyncio.to_thread(
                                    models["asr"].init_streaming_state,
                                    context=full_context,
                                    language=full_lang,
                                    unfixed_chunk_num=2,
                                    unfixed_token_num=5,
                                    chunk_size_sec=2.0,
                                )
                        except Exception as e:
                            logger.exception(f"Failed to init streaming state: {e}")
                            await ws.close(code=1011, reason="init_streaming_state failed")
                            return

                        # Optional: acknowledge language selection
                        if full_lang is not None:
                            await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                        continue

                    if t == "stop":
                        if state is not None:
                            # Flush remainder, finish, send final
                            await flush_and_infer(send_partial=False)
                            async with infer_sem:
                                await asyncio.to_thread(models["asr"].finish_streaming_transcribe, state)

                            text = state.text or ""
                            # Strip context prefix if the model echoed it back (happens on empty/silent audio)
                            if context:
                                text = strip_prompt(text, context)

                            await ws.send_json({"type": "final", "text": text, "language": state.language})
                        await ws.close(code=1000)
                        return

            # Audio frames
            if msg.get("bytes"):
                if not started:
                    # Require explicit start so we can validate format.
                    await ws.send_json({"type": "error", "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first"})
                    await ws.close(code=1002)
                    return

                chunk_bytes = msg["bytes"]
                # int16 mono little-endian -> float32 [-1, 1]
                audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                if audio_int16.size == 0:
                    continue

                audio_f32 = audio_int16.astype(np.float32) / 32768.0
                buf_parts.append(audio_f32)
                buf_n += audio_f32.size

                if buf_n >= STREAM_MIN_SAMPLES:
                    await flush_and_infer(send_partial=True)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass


if __name__ == "__main__":
    # NOTE: for GPU models, keep workers=1 unless you deliberately replicate the model per worker.
    uvicorn.run(app, host="0.0.0.0", port=_cli.port)
