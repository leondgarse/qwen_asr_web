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

import os
import io
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
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

# ── Model registry ───────────────────────────────────────────
MODELS = {
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
    "mistral": {
        "label": "Mistral",
        "env_key": "MISTRAL_API_KEY",
        "default_model": "mistral-small-latest",
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


# ── Routes ───────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/api/models")
async def list_models():
    return {"models": available_models()}


@app.post("/api/chat")
async def chat(req: ChatReq):
    avail = available_models()
    if not avail:
        raise HTTPException(503, detail="No LLM API keys configured. Set ANTHROPIC_API_KEY, GOOGLE_API_KEY, or MISTRAL_API_KEY.")

    model_id = req.model
    if model_id == "auto":
        model_id = avail[0]["id"]
    if model_id not in MODELS:
        raise HTTPException(400, detail=f"Unknown model: {model_id}")
    if not os.getenv(MODELS[model_id]["env_key"]):
        raise HTTPException(503, detail=f"API key not set: {MODELS[model_id]['env_key']}")

    system_parts = ["You are a helpful assistant. Help the user understand and review a speech transcription."]
    if req.transcription:
        system_parts.append(f"Current transcription:\n{req.transcription}")
    if req.context:
        system_parts.append(f"Reference context document:\n{req.context}")
    system_text = "\n\n".join(system_parts)

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    if model_id == "claude":
        return StreamingResponse(_stream_claude(system_text, msgs), media_type="text/event-stream")
    elif model_id == "gemini":
        return StreamingResponse(_stream_gemini(system_text, msgs), media_type="text/event-stream")
    elif model_id == "mistral":
        return StreamingResponse(_stream_mistral(system_text, msgs), media_type="text/event-stream")


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
        full_msgs = [{"role": "system", "content": system}] + msgs

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
