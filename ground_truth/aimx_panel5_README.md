# AIMX panel 5 — ROI / customer-service automation

Captured with `CONTINUOUS_CAPTURE` and the 8 s minimum from c436eef.

| | panel 2 | panel 3 | panel 4 | panel 5 |
|---|---|---|---|---|
| WER vs Snapsight | 80.4% | 52.3% | 44.1% | 59.4% |
| audio dropped | 56% | 20% | 0% | 0% |
| segments under 2 s | — | — | 30% | **0%** |

The 8 s minimum did what it was meant to: sub-2 s fragments went 27% → 0%.
But WER regressed, and the cause was the 1.2 s pause threshold shipped with it.

## Why the pause threshold was wrong

Silence runs *inside* continuous speech, measured on this capture:

| percentile | length |
|---|---|
| p50 | 0.09 s |
| p90 | 0.36 s |
| p95 | 0.54 s |
| p99 | 0.81 s |

**Nothing reaches 1.2 s.** The pause condition never fired, so 85% of segments
ran to the 20 s cap and ended mid-phrase — "...or maybe having.",
"...monitoring the stores." — which is the mid-sentence cutting visible in the
transcripts. Fixed in 42fccde by lowering it to 0.55 s (just above p95):
replaying the captured frames, cap-hits fall 92% → 20% and the median segment
goes 20.0 s → 10.7 s.

Re-decoding this audio confirms the direction: 61.9% at 20 s segments vs 57.4%
at ~10.7 s. Both are worse than panel 4's 44.1%, so this stretch of audio is
also intrinsically harder — the pause fix is an improvement, not a full
explanation of the gap.

Caveat: Snapsight is a peer ASR system, not human ground truth. Its own output
shows the same mid-sentence cuts, which is why they appear on both sides.
