---
type: Benchmark
title: Model loading costs (warm/cold)
description: What each loading phase costs and where load time actually goes; loading is NOT a bottleneck warm.
tags: [loading, safetensors, page-cache, prequantized]
timestamp: 2026-07-15T00:00:00Z
---

Measured on M3 Max 96 GB, mlx-swift 0.31.6 (klein-9b unless noted).

# Warm costs (file in page cache)

| Phase | Cost |
|-------|------|
| Transformer bf16, 17 GB, incl. QKV split + bf16→f16 conversion | **2.5-3.3 s** |
| Transformer + on-the-fly qint8 quantize | 1.3 s |
| Transformer + on-the-fly mxfp8 quantize | 3.0 s |
| Text encoder (Qwen3-8B-4bit, 4.3 GB) | 1.1-1.2 s |
| Pre-quantized checkpoint (9.6 GB, no mapping, no quantize) | 1.4-1.5 s |
| VAE (small decoder) | ~0 s |

`loadArrays` is lazy (header only); real IO happens at `eval` as parallel
32 MB preads straight into unified memory. On 96 GB the page cache makes
warm loads nearly free — **loading is not a bottleneck on large-RAM
machines**.

# Where it matters: small machines and cold starts

On 16-32 GB Macs the 17 GB bf16 file never stays in page cache, so **every**
load is a cold load (NVMe-bound, ~6-10 s of IO plus mapping). This is the
audience for [pre-quantized checkpoints](/docs/CLI.md) (`flux2
export-quantized`): ~half the bytes read, no transient bf16 materialization,
no quantize pass.

# Design facts

- The text encoder is reloaded on EVERY generation by default (memory-first
  design); `Flux2Pipeline.keepTextEncoderLoaded` opts out at ~+5 GB resident
  (PR #114).
- Debug weight statistics used to run on every transformer load even with
  logging off — gated since PR #114 (`Flux2Debug.isLoggable`).
- The MLX buffer cache retains ~3 GB after a generation; purged at end of
  generation since PR #114.
