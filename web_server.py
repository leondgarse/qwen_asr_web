"""
Web UI server for Qwen3-ASR Studio.
Serves the frontend and provides /api/chat (Claude / Gemini / Mistral) + /api/extract-context (PDF/MD).
The transcription WebSocket connects directly to the ASR server at localhost:8000.

Usage:
    # Set one or more API keys, then start:
    ANTHROPIC_API_KEY=sk-...  python web_server.py
    GOOGLE_API_KEY=...        python web_server.py
    MISTRAL_API_KEY=...       python web_server.py
    Then open http://localhost:8001
"""

import asyncio
import os
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-ASR Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Broadcast state (in-memory live session relay) ───────────
_broadcast: dict[str, Any] = {"name": "", "segments": [], "seq": 0}
_viewers: list[asyncio.Queue] = []
_broadcast_lock = asyncio.Lock()


async def _notify_viewers(payload: str) -> None:
    for q in _viewers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # slow viewer; they'll catch up on reconnect


# ── ASR / VL server config ────────────────────────────────────
ASR_HOST = os.getenv("ASR_HOST", "localhost")
ASR_PORT = int(os.getenv("ASR_PORT", "8000"))

_vl_checked = False  # True once we got a definitive "enabled" answer
_vl_available = False
_vl_info: dict = {}  # {"model": ..., "port": ...}


async def _check_vl() -> bool:
    """Lazy check whether the VL server on ASR host is up. Caches positive result."""
    global _vl_checked, _vl_available, _vl_info
    if _vl_checked and _vl_available:
        return True
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"http://{ASR_HOST}:{ASR_PORT}/vl/health")
            info = r.json()
            _vl_available = info.get("enabled", False)
            if _vl_available:
                _vl_info = info
                _vl_checked = True
            logger.info(f"VL health check -> {info} | vl_url={_vl_oai_url()}")
    except Exception as e:
        logger.warning(f"VL health check failed: {e}")
        _vl_available = False
    return _vl_available


def _vl_oai_url() -> str:
    # Route through the main server proxy so no extra tunnel is needed for VL_PORT
    return f"http://{ASR_HOST}:{ASR_PORT}/vl/proxy/v1/chat/completions"


_VL_MAX_HISTORY_CHARS = 12000  # rough guard: trim old turns if context gets too large


def _build_vl_messages(system: str, msgs: list, image: str, image_mime: str = "image/jpeg") -> list:
    # Trim history from the front (keep latest turns) to avoid exceeding context length.
    # Always keep the last user message; drop older pairs until total chars fit.
    trimmed = list(msgs)
    while len(trimmed) > 1:
        total = sum(len(str(m.get("content", ""))) for m in trimmed) + len(system)
        if total <= _VL_MAX_HISTORY_CHARS:
            break
        trimmed.pop(0)  # drop oldest message

    oai_msgs = []
    if system:
        oai_msgs.append({"role": "system", "content": system})
    for i, m in enumerate(trimmed):
        content = m["content"]
        if image and m["role"] == "user" and i == len(trimmed) - 1:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image}"}},
                {"type": "text", "text": content},
            ]
        oai_msgs.append({"role": m["role"], "content": content})
    return oai_msgs


# ── Model registry ───────────────────────────────────────────
MODELS = {
    "mistral": {
        "label": "Mistral",
        "env_key": "MISTRAL_API_KEY",
        "default_model": "mistral-small-latest",
    },
    "claude": {
        "label": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",
    },
    "gemini": {
        "label": "Gemini (Google)",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
}


def available_models() -> list[dict]:
    """Return models whose API key is set in the environment."""
    result = []
    for key, info in MODELS.items():
        if os.getenv(info["env_key"]):
            result.append({"id": key, "label": info["label"]})
    return result


# ── Request schemas ──────────────────────────────────────────
class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    messages: List[Msg]
    transcription: str = ""
    context: str = ""
    model: str = "auto"  # "auto" picks first available
    image: str = ""  # base64-encoded image; used only with local-vl model
    image_mime: str = "image/jpeg"


class TranslateReq(BaseModel):
    text: str
    target_language: str = "English"


# ── Routes ───────────────────────────────────────────────────
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}


@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "web" / "index.html", headers=_NO_CACHE)


@app.get("/viewer")
async def viewer_page():
    return FileResponse(Path(__file__).parent / "web" / "viewer.html", headers=_NO_CACHE)


# ── Broadcast endpoints ───────────────────────────────────────
class BroadcastPush(BaseModel):
    name: str = ""
    segments: list = []


class PartialPush(BaseModel):
    text: str = ""


@app.post("/api/session/partial")
async def push_partial(data: PartialPush):
    """Transcriber pushes live partial ASR text; forwarded to viewers immediately."""
    payload = json.dumps({"type": "partial", "text": data.text})
    await _notify_viewers(payload)
    return {"ok": True}


@app.post("/api/session/push")
async def push_session(data: BroadcastPush):
    """Transcriber posts current session state; all viewers are notified via SSE."""
    async with _broadcast_lock:
        _broadcast["name"] = data.name
        _broadcast["segments"] = data.segments
        _broadcast["seq"] += 1
        payload = json.dumps({"type": "update", "name": data.name, "segments": data.segments, "seq": _broadcast["seq"]})
    await _notify_viewers(payload)
    return {"ok": True, "viewers": len(_viewers)}


@app.get("/api/session/stream")
async def stream_session(request: Request):
    """SSE stream — sends full session state on connect, then incremental pushes."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _viewers.append(queue)

    async def event_gen():
        # Send current state immediately so viewer isn't blank
        async with _broadcast_lock:
            snapshot = json.dumps({"type": "update", "name": _broadcast["name"], "segments": _broadcast["segments"], "seq": _broadcast["seq"]})
        yield f"data: {snapshot}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # prevents proxy/browser timeout
        finally:
            if queue in _viewers:
                _viewers.remove(queue)

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/models")
async def list_models():
    result = available_models()
    if await _check_vl():
        label = (_vl_info.get("model") or "").split("/")[-1] or "Qwen-VL"
        result.insert(0, {"id": "local-vl", "label": f"Local VL ({label})"})
    return {"models": result}


@app.post("/api/chat")
async def chat(req: ChatReq):
    vl_ok = await _check_vl()
    avail = available_models()
    all_ids = [m["id"] for m in avail] + (["local-vl"] if vl_ok else [])
    if not all_ids:
        raise HTTPException(503, detail="No LLM available. Set an API key or start server.py with --qwenvl.")

    model_id = req.model if req.model != "auto" else all_ids[0]

    system_parts = ["You are a helpful assistant. Help the user understand and review a speech transcription."]
    if req.transcription:
        system_parts.append(f"Current transcription:\n{req.transcription}")
    if req.context:
        system_parts.append(f"Reference context document:\n{req.context}")
    system_text = "\n\n".join(system_parts)
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    if model_id == "local-vl":
        if not vl_ok:
            raise HTTPException(503, detail="VL model not available. Start server.py with --qwenvl.")
        return StreamingResponse(
            _stream_local_vl(system_text, msgs, req.image, req.image_mime), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    if model_id not in MODELS:
        raise HTTPException(400, detail=f"Unknown model: {model_id}")
    if not os.getenv(MODELS[model_id]["env_key"]):
        raise HTTPException(503, detail=f"API key not set: {MODELS[model_id]['env_key']}")

    if model_id == "claude":
        return StreamingResponse(_stream_claude(system_text, msgs), media_type="text/event-stream")
    elif model_id == "gemini":
        return StreamingResponse(_stream_gemini(system_text, msgs), media_type="text/event-stream")
    elif model_id == "mistral":
        return StreamingResponse(_stream_mistral(system_text, msgs), media_type="text/event-stream")


@app.post("/api/translate")
async def translate(req: TranslateReq):
    if not await _check_vl():
        raise HTTPException(503, detail="VL model not available. Start server.py with --qwenvl.")
    import httpx

    body = {
        "model": _vl_info.get("model", ""),
        "messages": [
            {"role": "system", "content": f"Translate the following text to {req.target_language}. Output only the translated text, nothing else."},
            {"role": "user", "content": req.text},
        ],
        "stream": False,
        "max_tokens": 512,
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(_vl_oai_url(), json=body)
            r.raise_for_status()
            translated = r.json()["choices"][0]["message"]["content"].strip()
            return {"translated": translated}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Local VL (direct OpenAI-compatible call to vLLM) ─────────
async def _stream_local_vl(system: str, msgs: list, image: str = "", image_mime: str = "image/jpeg"):
    import httpx

    url = _vl_oai_url()
    built_msgs = _build_vl_messages(system, msgs, image, image_mime)
    body = {
        "model": _vl_info.get("model", ""),
        "messages": built_msgs,
        "stream": True,
        "max_tokens": 1024,
    }
    logger.info(f"VL request -> {url} model={body['model']} msgs={len(built_msgs)}")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body) as resp:
                logger.info(f"VL response status: {resp.status_code}")
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    logger.error(f"VL error body: {err_body.decode()}")
                    yield f"data: {json.dumps({'error': f'VL server {resp.status_code}: {err_body.decode()}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        d = json.loads(payload)
                        delta = d["choices"][0]["delta"].get("content") or ""
                        if delta:
                            yield f"data: {json.dumps({'text': delta})}\n\n"
                    except Exception:
                        pass
    except Exception as e:
        logger.exception("Local VL stream error")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


# ── Claude ───────────────────────────────────────────────────
def _stream_claude(system: str, msgs: list):
    try:
        import anthropic
    except ImportError:
        yield f"data: {json.dumps({'error': 'Run: pip install anthropic'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        with client.messages.stream(
            model=MODELS["claude"]["default_model"],
            max_tokens=1024,
            system=system,
            messages=msgs,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
    except Exception as e:
        logger.exception("Claude error")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


# ── Gemini ───────────────────────────────────────────────────
def _stream_gemini(system: str, msgs: list):
    try:
        from google import genai
    except ImportError:
        yield f"data: {json.dumps({'error': 'Run: pip install google-genai'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # Build contents: system instruction + conversation history
        contents = []
        for m in msgs:
            role = "user" if m["role"] == "user" else "model"
            contents.append(genai.types.Content(role=role, parts=[genai.types.Part(text=m["content"])]))

        response = client.models.generate_content_stream(
            model=MODELS["gemini"]["default_model"],
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1024,
            ),
        )
        for chunk in response:
            if chunk.text:
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
    except Exception as e:
        logger.exception("Gemini error")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


# ── Mistral ──────────────────────────────────────────────────
def _stream_mistral(system: str, msgs: list):
    try:
        from mistralai import Mistral
    except ImportError:
        yield f"data: {json.dumps({'error': 'Run: pip install mistralai'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        # Mistral rejects assistant messages with None/empty content
        clean_msgs = [m for m in msgs if m.get("content") is not None]
        full_msgs = [{"role": "system", "content": system}] + clean_msgs

        response = client.chat.stream(
            model=MODELS["mistral"]["default_model"],
            max_tokens=1024,
            messages=full_msgs,
        )
        for event in response:
            delta = event.data.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'text': delta.content})}\n\n"
    except Exception as e:
        logger.exception("Mistral error")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


# ── Context extraction ───────────────────────────────────────
@app.post("/api/extract-context")
async def extract_context(file: UploadFile = File(...)):
    data = await file.read()
    ext = Path(file.filename or "").suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)[:4000]
        except ImportError:
            raise HTTPException(503, detail="Run: pip install pypdf")
    elif ext in (".md", ".markdown", ".txt"):
        text = data.decode("utf-8", errors="ignore")[:4000]
    else:
        raise HTTPException(400, detail=f"Unsupported file type: {ext}. Use .pdf, .md, or .txt")

    return {"text": text, "chars": len(text)}


if __name__ == "__main__":
    print("Qwen3-ASR Studio  →  http://localhost:8001")
    print("ASR server must be running on localhost:8000")
    avail = available_models()
    if avail:
        print(f"Chat models available: {', '.join(m['label'] for m in avail)}")
    else:
        print("⚠ No LLM API keys set. Chat will not work.")
        print("  Set one or more: ANTHROPIC_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY")
    uvicorn.run(app, host="0.0.0.0", port=8001)
