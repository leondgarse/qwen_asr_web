import os
import io
import shutil
import subprocess
import asyncio
import json
import argparse
import sys
import websockets
import numpy as np
import soundfile as sf
import webrtcvad
from math import gcd
from scipy.signal import resample_poly
from datetime import datetime, timedelta
from pypdf import PdfReader
from tqdm import tqdm

# Audio settings
SAMPLING_RATE = 16000
CHANNELS = 1
MIN_SEGMENT_DURATION_S = 1.5  # skip VAD segments shorter than this (slide transitions, applause pops)
CONTAMINATION_WINDOW = 6      # consecutive words that must appear verbatim in context to flag contamination


def extract_context(file_path: str, max_chars: int = 1000) -> str:
    """Extract text from a PDF or Markdown file up to max_chars to use as vocabulary context."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            if len(text) > max_chars:
                text = text[:max_chars]
                break
    elif ext in (".md", ".markdown"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read(max_chars)
    else:
        raise ValueError(f"Unsupported context file type: {ext!r}. Expected .pdf or .md/.markdown.")

    instruction = "The following is background reference text to help with vocabulary and spelling. Do NOT read or transcribe this text. Only transcribe the audio. Reference: "
    final_context = instruction + text.replace('\n', ' ')
    print(f"Extracted {len(text)} chars from {file_path} as reference context.")
    return final_context


def extract_vocals(audio_path: str, device: str = "cuda", separated_dir: str | None = None) -> str:
    """
    Run demucs (htdemucs) to isolate the vocal stem.
    Tracks are saved persistently under separated_dir (default: ./separated/ next to the audio
    file) so subsequent runs reuse the existing output and skip separation.
    Defaults to CPU to avoid competing with the ASR server's GPU memory.
    Returns vocals_wav_path.
    """
    if separated_dir is None:
        separated_dir = os.path.join(os.path.dirname(os.path.abspath(audio_path)), "separated")

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(separated_dir, "htdemucs", stem, "vocals.wav")

    if os.path.exists(vocals_path):
        print(f"Reusing cached vocals: {vocals_path}")
        return vocals_path

    os.makedirs(separated_dir, exist_ok=True)
    print(f"Extracting vocals with demucs (htdemucs) on {device} — this may take a few minutes...")
    env = os.environ.copy()
    try:
        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demucs_runner.py")
        subprocess.run(
            [
                sys.executable, runner,
                "--two-stems", "vocals",
                "-d", device,
                "-o", separated_dir,
                audio_path,
            ],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs failed: {e}") from e

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(
            f"Demucs did not produce expected output at: {vocals_path}\n"
            f"Contents of {separated_dir}: {os.listdir(separated_dir)}"
        )

    print(f"Vocals saved to: {vocals_path}")
    return vocals_path


def resample_to_16k(audio_int16: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample int16 audio from src_sr to 16000 Hz using polyphase filtering."""
    if src_sr == SAMPLING_RATE:
        return audio_int16
    g = gcd(src_sr, SAMPLING_RATE)
    up = SAMPLING_RATE // g
    down = src_sr // g
    resampled = resample_poly(audio_int16.astype(np.float64), up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def is_context_contamination(text: str, context: str) -> bool:
    """
    Returns True if CONTAMINATION_WINDOW or more consecutive words from text appear
    verbatim in context, meaning the model generated from the context document rather
    than from actual audio.
    """
    if not context or not text:
        return False
    words = text.lower().split()
    if len(words) < CONTAMINATION_WINDOW:
        return False
    context_lower = context.lower()
    for i in range(len(words) - CONTAMINATION_WINDOW + 1):
        phrase = " ".join(words[i:i + CONTAMINATION_WINDOW])
        if phrase in context_lower:
            return True
    return False


def int16_to_bytes(audio_int16: np.ndarray) -> bytes:
    return audio_int16.tobytes()


async def stream_audio_segment(ws, audio_int16: np.ndarray, sample_rate: int, context: str = ""):
    # Handshake — include context if provided
    start_msg: dict = {
        "type": "start",
        "format": "pcm_s16le",
        "sample_rate_hz": sample_rate,
        "channels": CHANNELS,
    }
    if context:
        start_msg["context"] = context
    await ws.send(json.dumps(start_msg))

    # 100ms chunks to send over websocket
    chunk_size = int(sample_rate * 0.1)

    for i in range(0, len(audio_int16), chunk_size):
        chunk = audio_int16[i:i + chunk_size]
        await ws.send(int16_to_bytes(chunk))
        # Small sleep to yield control and simulate streaming / avoid overwhelming server buffers
        await asyncio.sleep(0.01)

    await ws.send(json.dumps({"type": "stop"}))


async def receiver(ws) -> str:
    """Receives websocket messages and returns the final transcribed text."""
    final_text = ""
    async for message in ws:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            evt = json.loads(message)
            msg_type = evt.get('type')
            text = evt.get('text', '')
            lang = evt.get('language', '')

            if msg_type == 'ready':
                pass
            elif msg_type == 'partial':
                sys.stdout.write(f"\r[Partial] ({lang}): {text}")
                sys.stdout.flush()
            elif msg_type == 'final':
                sys.stdout.write(f"\r[Final] ({lang}): {text}\n")
                sys.stdout.flush()
                final_text = text
            elif msg_type == 'error':
                print(f"\n[{timestamp}] [Error]: {evt.get('message')}")
            elif msg_type == 'info':
                pass

        except json.JSONDecodeError:
            print(f"\n[Raw]: {message}")

    return final_text


def apply_vad(audio_int16: np.ndarray, sample_rate: int, min_silence_duration_s: float = 1.0) -> list[tuple[int, int]]:
    """
    Applies WebRTC VAD to find non-silent segments separated by at least `min_silence_duration_s` of silence.
    Aggressiveness=2 rejects more non-speech (music, noise) than the previous value of 1.
    Returns a list of (start_idx, end_idx) tuples.
    """
    vad = webrtcvad.Vad(2)  # 0–3; 2 = balanced, filters music/noise better than 1

    frame_duration_ms = 30  # WebRTC VAD requires 10, 20, or 30 ms frames
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    min_silence_frames = int((min_silence_duration_s * 1000.0) / frame_duration_ms)

    segments = []
    current_start = -1
    silence_counter = 0

    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i:i + frame_size]
        if len(frame) < frame_size:
            break

        is_speech = vad.is_speech(frame.tobytes(), sample_rate)

        if is_speech:
            silence_counter = 0
            if current_start == -1:
                current_start = max(0, i - frame_size * 5)  # Include a bit of lead-in
        else:
            if current_start != -1:
                silence_counter += 1
                if silence_counter > min_silence_frames:
                    current_end = min(len(audio_int16), i + frame_size * 5)  # Include a bit of lead-out
                    segments.append((current_start, current_end))
                    current_start = -1
                    silence_counter = 0

    if current_start != -1:
        segments.append((current_start, len(audio_int16)))

    return segments


def read_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Read audio file, using ffmpeg for formats soundfile can't handle (e.g. mp3, m4a)."""
    try:
        audio, sr = sf.read(file_path, dtype='int16')
        return audio, sr
    except Exception:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-f", "wav", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {proc.stderr.decode(errors='ignore')}")
        audio, sr = sf.read(io.BytesIO(proc.stdout), dtype='int16')
        return audio, sr


async def process_file(file_path: str, endpoint: str, context: str = "", output: str | None = None, vocal_extraction: bool = False, demucs_device: str = "cuda", separated_dir: str | None = None):
    # ── Step 1: Vocal extraction (optional) ───────────────────────────────────
    audio_source = file_path
    if vocal_extraction:
        audio_source = extract_vocals(file_path, device=demucs_device, separated_dir=separated_dir)

    # ── Step 2: Load audio ────────────────────────────────────────────────
    if True:
        print(f"Loading audio: {audio_source}...")
        audio_data, sr = read_audio(audio_source)

        # Downmix to mono (demucs outputs stereo)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # ── Step 3: Resample to 16kHz ─────────────────────────────────────────
        if sr != SAMPLING_RATE:
            print(f"Resampling from {sr} Hz → {SAMPLING_RATE} Hz...")
            audio_data = resample_to_16k(audio_data, sr)
            sr = SAMPLING_RATE

        duration_s = len(audio_data) / sr
        print(f"Audio ready: {duration_s:.1f}s @ {sr}Hz mono")

        # ── Step 4: VAD segmentation ──────────────────────────────────────────
        print("Applying VAD to split into speech segments...")
        segments = apply_vad(audio_data, SAMPLING_RATE, min_silence_duration_s=0.8)

        if not segments:
            print("No speech detected after vocal extraction + VAD.")
            return

        print(f"Found {len(segments)} speech segments.")

        # ── Step 5: Transcribe each segment ──────────────────────────────────
        stem = os.path.splitext(file_path)[0]
        jsonl_output_file = output if output else f"{stem}.jsonl"
        print(f"Results will be written line by line to {jsonl_output_file}")

        with open(jsonl_output_file, "w", encoding="utf-8") as f:
            pass  # truncate

        pbar = tqdm(total=len(segments), desc="Transcribing")

        for start_idx, end_idx in segments:
            segment_duration = (end_idx - start_idx) / SAMPLING_RATE
            start_time = timedelta(seconds=start_idx / SAMPLING_RATE)
            end_time = timedelta(seconds=end_idx / SAMPLING_RATE)
            timestamp_str = f"[{start_time} - {end_time}]"

            if segment_duration < MIN_SEGMENT_DURATION_S:
                tqdm.write(f"Skipping {timestamp_str} (too short: {segment_duration:.1f}s)")
                pbar.update(1)
                continue

            tqdm.write(f"\nProcessing {timestamp_str}...")
            segment_audio = audio_data[start_idx:end_idx]

            try:
                async with websockets.connect(endpoint, max_size=None) as ws:
                    sender_task = asyncio.create_task(
                        stream_audio_segment(ws, segment_audio, SAMPLING_RATE, context=context)
                    )
                    receiver_task = asyncio.create_task(receiver(ws))

                    _, text = await asyncio.gather(sender_task, receiver_task)

                    text = text.strip()
                    if not text:
                        pass
                    elif is_context_contamination(text, context):
                        tqdm.write(f"Skipping {timestamp_str} (context contamination detected)")
                    else:
                        item = {"timestamp": timestamp_str, "text": text}
                        with open(jsonl_output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

            except Exception as e:
                tqdm.write(f"Failed to process segment {timestamp_str}: {e}")

            pbar.update(1)

        pbar.close()
        print(f"\n--- Final transcription written to {jsonl_output_file} ---")


async def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR File Streaming Transcriber with Vocal Extraction, VAD & Context")
    parser.add_argument("audio", help="Path to input audio file (.mp3, .wav, .m4a, ...)")
    parser.add_argument("-e", "--endpoint", default="ws://localhost:8000/transcribe-streaming", help="WebSocket Endpoint URL")
    parser.add_argument("-l", "--language", default="English", help="Forced language full name (e.g. English, Chinese, Japanese)")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL file path (default: <audio_stem>.jsonl)")
    parser.add_argument("--context", default=None, help="Path to PDF or Markdown reference document for vocabulary context")
    parser.add_argument("--vocal-extraction", action="store_true", help="Run demucs vocal extraction before ASR. Useful for event recordings with background music. Separated tracks are cached in --separated-dir for reuse.")
    parser.add_argument("--demucs-device", default="cuda", help="Device for demucs: cuda (default) or cpu / cuda:N.")
    parser.add_argument("--separated-dir", default=None, help="Where to store/reuse demucs output (default: separated/ next to the audio file).")
    args = parser.parse_args()

    endpoint = args.endpoint
    if args.language:
        sep = "&" if "?" in endpoint else "?"
        endpoint += f"{sep}language={args.language}"

    context = ""
    if args.context:
        context = extract_context(args.context)

    await process_file(args.audio, endpoint, context=context, output=args.output, vocal_extraction=args.vocal_extraction, demucs_device=args.demucs_device, separated_dir=args.separated_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
