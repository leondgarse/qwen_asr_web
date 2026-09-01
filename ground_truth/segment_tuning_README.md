# Segment-length tuning on real classroom audio

Swept `CONTINUOUS_MIN_S` and `CONTINUOUS_PAUSE_S` against 10 minutes of the
37.7-minute classroom capture (`captures/ACE5411_week3_08.27/*_RAW.wav`) — the
first fixture recorded in the room the system is actually used in.

## Pause distribution differs by room

Silence runs inside speech:

| percentile | AIMX panel | this classroom |
|---|---|---|
| p90 | 0.36 s | 0.54 s |
| p95 | 0.54 s | 0.81 s |
| p99 | 0.81 s | 1.57 s |

Classroom speech pauses noticeably more, which is why the same 0.55 s threshold
gave 85% cap-hits on the panel but 56% here.

## Result: the settings do not matter much

Scored against a 90-second-window reference (far more context than any live
setting can use, so it acts as a quality ceiling):

| min_s / pause_s | segments | divergence |
|---|---|---|
| 4 / 0.5 | 85 | 11.0% |
| 8 / 0.55 | 53 | 11.3% |
| 8 / 0.5 | 54 | 11.6% |
| 8 / 0.4 | 63 | 12.5% |
| 6 / 0.4 | 78 | 12.6% |
| 6 / 0.5 | 67 | 12.7% |

Spread is 1.7 pp across a 2x range of segment counts, and word counts are all
~1330 with ~5 redundant words. That is noise, not signal.

**Conclusion: segment tuning is exhausted.** The current defaults (8 s / 0.55 s)
are as good as anything tested. The earlier large wins came from *coverage* —
auto-gain, then continuous capture — not from segment length. Remaining error is
acoustic (domain words, accented speech, room noise), which segmentation cannot
fix.

Caveat: this is one 10-minute stretch of one recording, scored against another
ASR pass rather than human ground truth. A human-checked transcript of this
audio would be needed to measure the remaining error properly.
