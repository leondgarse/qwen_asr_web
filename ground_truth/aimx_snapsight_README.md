# AIMX / Snapsight reference transcript

`aimx_snapsight.txt` is Snapsight's transcript of the session "From Data to
Decisions: AI in the Next Generation of Supply Chains", copied from the
attendee live-channel page (the page renders client-side and its API requires a
key, so it cannot be fetched programmatically).

## Overlap with our captures

`captures/Session_1/` was recorded during part of the same panel, so a genuine
paired comparison is possible — but only over a **short overlap**:

- our capture ends mid-sentence at "when people get..." while the Snapsight
  text continues through the rapid-fire round;
- the `*_RAW.wav` files cover an *earlier* stretch than the pasted excerpt (1 of
  12 distinctive phrases matched), so only the live utterance transcripts line up.

On the ~35 words both cover, **WER vs Snapsight is 28.6%**. That is one
sentence — indicative only, far too small to draw conclusions from.

## To get a real number

Record a full session with `CAPTURE_AUDIO=true` running for its whole duration,
then export the complete Snapsight transcript for the same session. The
`*_RAW.wav` and that transcript would give a multi-thousand-word paired sample,
scoreable the way `*_plaud.txt` are.

Caveat: Snapsight is itself an ASR system, not human ground truth — the pasted
text contains its own errors (name drift: "Chunming"/"Chong Ming"/"Cho Ming",
"Pilot House"/"Pilot Pulse"; and it transcribes page UI labels such as
"Original / Live Text / Takeaways" as if they were speech). Treat it as a peer
system to compare against, not as truth.
