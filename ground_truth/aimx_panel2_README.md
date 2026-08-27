# AIMX panel 2 — "From Experimentation to Enterprise-Wide AI Deployment"

Paired sample. Laptop mic, `CAPTURE_AUDIO=true`, same room and time as
Snapsight's own live transcription.

- `aimx_panel2_snapsight.txt` — Snapsight's output for the overlapping stretch
- `../test_results/aimx_panel2_ours.txt` — ours, stitched from the per-utterance
  captures under `captures/20260827_14*/`

## Result

**WER 80.4%** against Snapsight over the best-aligned 271-word window.

## Why

Not decoding — segmentation. Over the 9.3-minute recording:

- audio kept by the browser VAD: **4.1 min**
- audio discarded between utterances: **5.9 min** (19 of 48 gaps exceed 3 s;
  the largest is 60 s)

So more speech was dropped than was transcribed, and the losses are whole
phrases rather than edge syllables. The transcript reads as a systematically
thinned version of the Snapsight text — same topics and sentence openings, with
words missing throughout.

This is the same `energyThreshold` failure seen on the 08_25 lectures, and it is
not addressed by the pre-roll fix (5160b60), which only recovers ~300 ms at an
utterance boundary. It needs the VAD to stop gating on a fixed energy level.

Caveat: Snapsight is a peer ASR system, not human ground truth.
