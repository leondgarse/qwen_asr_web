import asyncio
import json
import argparse
import sys
import websockets
import numpy as np
import sounddevice as sd
import webrtcvad
from datetime import datetime
from collections import deque

# Audio settings
SAMPLING_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLING_RATE * FRAME_DURATION_MS / 1000)  # 480 samples @ 16kHz, 30ms

# VAD settings
VAD_AGGRESSIVENESS = 3                              # 0–3; 3 = most aggressive (best for noisy environments)
SPEECH_START_FRAMES = 5                             # consecutive speech frames to trigger recording
SILENCE_END_FRAMES = int(1000 / FRAME_DURATION_MS) # 1000ms silence → end of utterance (~33 frames)
MIN_SPEECH_FRAMES = int(400 / FRAME_DURATION_MS)   # ~400ms minimum speech to send
LEAD_IN_FRAMES = 5                                  # frames prepended before speech starts
MAX_UTTERANCE_FRAMES = int(60_000 / FRAME_DURATION_MS)  # force-flush after 60s regardless


async def transcribe_segment(audio_int16: np.ndarray, endpoint: str) -> str:
    """Send a PCM segment to the server over a fresh WebSocket and return the final text."""
    async with websockets.connect(endpoint, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": SAMPLING_RATE,
            "channels": CHANNELS,
        }))

        chunk_size = int(SAMPLING_RATE * 0.1)  # 100ms chunks
        for i in range(0, len(audio_int16), chunk_size):
            await ws.send(audio_int16[i:i + chunk_size].tobytes())
            await asyncio.sleep(0)  # yield control to event loop

        await ws.send(json.dumps({"type": "stop"}))

        async for message in ws:
            try:
                evt = json.loads(message)
                if evt.get("type") == "final":
                    return evt.get("text", "")
                elif evt.get("type") == "error":
                    print(f"\n  [Server error]: {evt.get('message')}", file=sys.stderr)
                    return ""
            except json.JSONDecodeError:
                pass
    return ""


async def vad_loop(audio_queue: asyncio.Queue, segment_queue: asyncio.Queue, verbose: bool = False):
    """
    VAD state machine: reads mic frames, detects utterance boundaries,
    and puts completed speech segments into segment_queue.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    ring_buffer: deque = deque(maxlen=LEAD_IN_FRAMES)
    speech_frames: list = []
    speaking = False
    speech_trigger = 0
    silence_count = 0
    total_frames = 0

    def log(msg: str):
        if verbose:
            print(f"\n  [VAD] {msg}", file=sys.stderr)

    sys.stdout.write("[Listening...]\n")
    sys.stdout.flush()

    while True:
        frame_bytes = await audio_queue.get()
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        if len(frame) != FRAME_SIZE:
            log(f"Unexpected frame size {len(frame)}, skipping")
            continue

        total_frames += 1
        is_speech = vad.is_speech(frame_bytes, SAMPLING_RATE)

        if not speaking:
            ring_buffer.append(frame)
            if is_speech:
                speech_trigger += 1
                log(f"Speech trigger {speech_trigger}/{SPEECH_START_FRAMES}")
                if speech_trigger >= SPEECH_START_FRAMES:
                    speaking = True
                    speech_frames = list(ring_buffer)
                    silence_count = 0
                    speech_trigger = 0
                    sys.stdout.write("\r[Recording...]    \n")
                    sys.stdout.flush()
                    log("Utterance started")
            else:
                if speech_trigger > 0:
                    log(f"Speech trigger reset (was {speech_trigger})")
                speech_trigger = 0
        else:
            speech_frames.append(frame)
            if not is_speech:
                silence_count += 1
                log(f"Silence {silence_count}/{SILENCE_END_FRAMES} ({silence_count * FRAME_DURATION_MS}ms)")
                if silence_count >= SILENCE_END_FRAMES:
                    # Drop trailing silence, keep a small tail for natural boundary
                    tail = SILENCE_END_FRAMES // 2
                    trim_end = len(speech_frames) - silence_count + tail
                    speech_only = speech_frames[:trim_end] if trim_end > 0 else speech_frames
                    log(f"Utterance ended: {len(speech_only)} frames ({len(speech_only) * FRAME_DURATION_MS}ms)")
                    if len(speech_only) >= MIN_SPEECH_FRAMES:
                        await segment_queue.put(np.concatenate(speech_only))
                        log("Segment queued for transcription")
                    else:
                        log(f"Segment too short ({len(speech_only)} < {MIN_SPEECH_FRAMES} frames), discarded")
                    # Reset state
                    speaking = False
                    speech_frames = []
                    silence_count = 0
                    ring_buffer.clear()
                    sys.stdout.write("[Listening...]\n")
                    sys.stdout.flush()
            else:
                if silence_count > 0:
                    log(f"Silence reset (was {silence_count})")
                silence_count = 0

            # Force-flush if utterance is too long
            if speaking and len(speech_frames) >= MAX_UTTERANCE_FRAMES:
                log(f"Max utterance duration reached ({MAX_UTTERANCE_FRAMES * FRAME_DURATION_MS / 1000:.0f}s), force-flushing")
                await segment_queue.put(np.concatenate(speech_frames))
                speaking = False
                speech_frames = []
                silence_count = 0
                ring_buffer.clear()
                sys.stdout.write("[Listening...]\n")
                sys.stdout.flush()


async def transcription_loop(segment_queue: asyncio.Queue, endpoint: str, verbose: bool = False):
    """
    Dequeues speech segments and transcribes them sequentially.
    Segments produced while transcribing are queued and processed in order.
    """
    while True:
        audio_data = await segment_queue.get()
        ts = datetime.now().strftime("%H:%M:%S")
        dur = len(audio_data) / SAMPLING_RATE
        sys.stdout.write(f"\r[Transcribing {dur:.1f}s segment...]    \n")
        sys.stdout.flush()
        try:
            text = await transcribe_segment(audio_data, endpoint)
            if text:
                print(f"[{ts}] {text}")
            elif verbose:
                print(f"  [ASR] Empty result for {dur:.1f}s segment", file=sys.stderr)
        except Exception as e:
            print(f"\n[{ts}] Transcription error: {e}", file=sys.stderr)


async def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Real-time Mic Client")
    parser.add_argument("-e", "--endpoint", default="ws://localhost:8000/transcribe-streaming",
                        help="WebSocket Endpoint URL")
    parser.add_argument("-l", "--language", default=None,
                        help="Language code or full name (e.g. en, English, zh, Chinese)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print VAD debug info (speech triggers, silence counters, etc.)")
    args = parser.parse_args()

    endpoint = args.endpoint
    if args.language:
        sep = "&" if "?" in endpoint else "?"
        endpoint += f"{sep}language={args.language}"

    audio_queue: asyncio.Queue = asyncio.Queue()
    segment_queue: asyncio.Queue = asyncio.Queue()

    loop = asyncio.get_running_loop()

    def mic_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy().tobytes())

    print(f"Connecting to {endpoint}")
    print(f"VAD aggressiveness={VAD_AGGRESSIVENESS}, silence_end={SILENCE_END_FRAMES * FRAME_DURATION_MS}ms")
    print("Speak into the mic. Press Ctrl+C to stop.\n")

    with sd.InputStream(samplerate=SAMPLING_RATE, channels=CHANNELS, dtype='int16',
                        blocksize=FRAME_SIZE, callback=mic_callback):
        vad_task = asyncio.create_task(vad_loop(audio_queue, segment_queue, verbose=args.verbose))
        asr_task = asyncio.create_task(transcription_loop(segment_queue, endpoint, verbose=args.verbose))
        try:
            await asyncio.gather(vad_task, asr_task)
        except asyncio.CancelledError:
            vad_task.cancel()
            asr_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
