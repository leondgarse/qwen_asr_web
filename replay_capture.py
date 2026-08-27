#!/usr/bin/env python3
"""Replay a captured session for offline testing.

`CAPTURE_AUDIO=true python server.py` writes each live utterance to
captures/<session>/. This stitches one back into a single WAV plus the
transcript the server produced at the time, so a real in-class recording
becomes a reusable fixture:

    python replay_capture.py captures/CS5432_week_3          # concatenate
    python replay_capture.py captures/CS5432_week_3 --rerun  # re-transcribe

--rerun sends each utterance back through /transcribe on a running server and
reports how the current code differs from what was produced live, which is the
point of keeping the audio.
"""
import argparse
import json
import os
import sys
import wave

import numpy as np


def load_session(path):
    meta_path = os.path.join(path, "transcript.jsonl")
    if not os.path.exists(meta_path):
        sys.exit(f"no transcript.jsonl in {path}")
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["file"])
    return rows


def read_wav(p):
    with wave.open(p, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise ValueError(f"{p}: expected 16 kHz mono")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="captures/<session> directory")
    ap.add_argument("-o", "--output", help="output WAV (default: <session>.wav)")
    ap.add_argument("--gap", type=float, default=0.5, help="silence inserted between utterances")
    ap.add_argument("--rerun", action="store_true", help="re-transcribe via a running server")
    ap.add_argument("--server", default="http://localhost:9002")
    a = ap.parse_args()

    rows = load_session(a.session)
    out_wav = a.output or a.session.rstrip("/") + ".wav"
    gap = np.zeros(int(a.gap * 16000), dtype=np.int16)

    parts, lines, t = [], [], 0.0
    for r in rows:
        p = os.path.join(a.session, r["file"])
        if not os.path.exists(p):
            print(f"  missing {r['file']}, skipping")
            continue
        pcm = read_wav(p)
        parts += [pcm, gap]
        h, rem = divmod(int(t), 3600)
        m, sec = divmod(rem, 60)
        lines.append(f"[{h:d}:{m:02d}:{sec:02d}] {r.get('text','')}")
        t += len(pcm) / 16000 + a.gap

    if not parts:
        sys.exit("no audio found")

    audio = np.concatenate(parts)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio.tobytes())

    txt = os.path.splitext(out_wav)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} utterances -> {out_wav} ({len(audio)/16000/60:.1f} min)")
    print(f"live transcript          -> {txt}")

    if a.rerun:
        import httpx

        changed = 0
        with httpx.Client(timeout=600) as c:
            for r in rows:
                p = os.path.join(a.session, r["file"])
                if not os.path.exists(p):
                    continue
                with open(p, "rb") as fh:
                    resp = c.post(f"{a.server}/transcribe?language=en",
                                  files={"files": (r["file"], fh.read(), "audio/wav")})
                new = resp.json()[0]["text"] if resp.status_code == 200 else f"[HTTP {resp.status_code}]"
                if new.strip() != (r.get("text") or "").strip():
                    changed += 1
                    print(f"\n  {r['file']}\n    live: {r.get('text','')[:90]}\n    now : {new[:90]}")
        print(f"\n{changed}/{len(rows)} utterances changed vs the live run")


if __name__ == "__main__":
    main()
