# AIMX panel 4 — build-vs-buy / orchestration

First capture with `CONTINUOUS_CAPTURE` enabled.

| | panel 2 | panel 3 | panel 4 |
|---|---|---|---|
| WER vs Snapsight | 80.4% | 52.3% | **44.1%** |
| audio dropped by VAD | 56% | 20% | **0%** |
| largest gap | 60 s | 13 s | 12.8 s |

Coverage is solved — no audio is discarded any more. The remaining error is
decode quality, and the next lever is segment length, not coverage.

## Segment-length sweep

Re-decoding this capture's audio at fixed segment lengths, scored against
Snapsight:

| segment | WER |
|---|---|
| 2.5 s | 76.3% |
| 5 s | 56.8% |
| **8 s** | **47.7%** |
| 12 s | 49.2% |
| 20 s | 51.1% |

Short segments starve the model of context; long ones invite repetition loops
("Is there? Is there? Is there?"). `CONTINUOUS_MIN_S` is set to 8 s on this
evidence. Before the change 30% of committed segments were under 2 s, many a
single "Oh." — a 0.6 s pause is a breath, not a sentence end.

Caveat: Snapsight is a peer ASR system, not human ground truth.
