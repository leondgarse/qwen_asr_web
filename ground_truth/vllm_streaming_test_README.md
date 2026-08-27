# vLLM streaming test

Tested whether vLLM's incremental streaming decode (`init_streaming_state` /
`streaming_transcribe`) beats the chunked llama.cpp path.

## Version matters — the purged code did work

vLLM **0.23.0** (installed fresh for this test) dies on this GPU:

```
Cannot use FA version 2 ... FA2 is only supported on devices with compute capability >= 8
RuntimeError: BatchPrefillWithPagedKVCache failed with error invalid argument
```

Both Quadro RTX 8000s are compute capability 7.5, and XFORMERS / FLASH_ATTN /
TRITON_ATTN all fail identically.

vLLM **0.20.0** — still installed system-wide, the version the pre-purge code
ran against — loads and streams fine on the same hardware. So the earlier
conclusion that "streaming is unavailable on this GPU" was wrong: it was
unavailable on the *newer* vLLM only.

## But streaming loses on output quality

Both backends on the same 10 minutes of real lecture audio
(`captures/ACE5411_week3_08.27`):

| | words | redundant words | wall | RTF |
|---|---|---|---|---|
| llama.cpp chunked | 1398 | 5 | 29 s | 0.049 |
| vLLM 0.20.0 streaming | 6841 | 3611 | 133 s | 0.221 |

10 minutes of lecture speech at 130-150 wpm is ~1300-1500 words, so llama.cpp
is right and vLLM emits **4.6x** too much. Over half of vLLM's output is
duplicated text — the streaming state re-emits already-fixed tokens
("yeah as i mentioned the" x9). Per-phrase quality is comparable; the text is
just repeated.

De-duplicating that output is possible in principle, but it would be undoing a
defect rather than gaining accuracy, and llama.cpp is also 4.5x faster.

## Conclusion

Keep llama.cpp. The streaming path is reachable on this hardware with vLLM
0.20.0 but currently produces worse output, so the purge stands on quality
grounds rather than on the hardware limit claimed earlier.
