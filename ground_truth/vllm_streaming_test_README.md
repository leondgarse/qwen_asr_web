# vLLM streaming test — blocked by GPU hardware

Tested whether vLLM's incremental streaming decode (`init_streaming_state` /
`streaming_transcribe`) beats our chunked llama.cpp path on the captured AIMX
audio. It could not be run.

## What was set up

- Re-downloaded the safetensors weights removed in 0e993ca (4.4 GB,
  `~/workspace/models/Qwen3-ASR-1.7B-hf`)
- Built an isolated venv, because `~/workspace/Qwen3-ASR` pins
  `transformers==4.57.6` while this environment runs 5.7.0 — importing the
  model class under 5.x fails with
  `check_model_inputs() missing 1 required positional argument`
- Confirmed the streaming API is present and the model loads

## Why it fails

The engine dies during decode:

```
Cannot use FA version 2 ... FA2 is only supported on devices with compute capability >= 8
RuntimeError: BatchPrefillWithPagedKVCache failed with error invalid argument
```

Both Quadro RTX 8000s are **compute capability 7.5**. FlashAttention-2 and
FlashInfer's paged-KV prefill kernel both require ≥ 8.0 (Ampere). Forcing
`VLLM_ATTENTION_BACKEND` to XFORMERS, FLASH_ATTN and TRITON_ATTN all fail
identically, so this is not a configuration issue.

## Implication

The purge in 0e993ca is not reversible on this hardware for the streaming path
— current vLLM cannot serve this model here at all. That also revises the
earlier framing: we did not merely trade streaming for speed, the streaming
option is unavailable regardless.

Remaining routes to a streaming decoder:
- `qwen3-asr.cpp` (a separate runtime; the GGUF aligner exports on HF target it)
- an older vLLM whose kernels still support SM75
- Ampere-or-newer hardware

Until then, chunked decode with good segmentation is the ceiling, which makes
the VAD/segment tuning the productive line of work.
