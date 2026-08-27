# AIMX panel 3 — data/capability discussion

Second paired sample from the same event, laptop mic, after the pre-roll fix
(5160b60) shipped.

| | panel 2 (before) | panel 3 (after) |
|---|---|---|
| WER vs Snapsight | 80.4% | **52.3%** |
| audio dropped by VAD | 56% | 20% |
| largest gap between utterances | 60 s | 13 s |

Still far from Snapsight, and the residual cause is unchanged: the browser's
fixed `energyThreshold` decides what the model is allowed to see. That is what
`CONTINUOUS_CAPTURE` addresses — stream every frame and let the server segment
with Silero instead.

Caveat: Snapsight is a peer ASR system, not human ground truth.
