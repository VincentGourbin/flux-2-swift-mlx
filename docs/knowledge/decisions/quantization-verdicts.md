---
type: Decision
title: Quantization verdicts for inference
description: Quantization buys memory, not step time; qint8 is quality-transparent on 9B; avoid mxfp8 for inference.
tags: [quantization, qint8, mxfp8, int4, benchmark]
timestamp: 2026-07-15T00:00:00Z
---

Benchmarked 2026-07-14 (klein-9b, 1 MP inpaint, 4 steps, equal seed, M3 Max;
mlx-swift 0.31.6 / core 0.31.1). Quality deltas measured on the inpainted
region against the bf16 output.

# Verdicts

| Mode | Speed vs bf16 | Quality vs bf16 | Verdict |
|------|--------------|-----------------|---------|
| qint8 (affine 8-bit, g64) | ~equal (14.9 vs 13.6 s/step) | near-identical on **9B** (mean RGB Δ 1.7/255, Δsat −0.002) | Use for **memory** (−47% active), not speed |
| mxfp8 (microscaling) | **~2x slower** (24-29 s/step) | **visibly degraded** (washed out, blurry; Δ 36/255) | **Avoid for inference** as of core 0.31.1 |
| int4 | not benchmarked for speed | aggressive | niche (32 GB Macs) |

# Why quantization does not speed up diffusion steps

LLM decode is GEMV/memory-bound, where 8-bit weights halve the traffic.
A diffusion denoising step is large-GEMM/SDPA-bound (compute-bound on
M-series): weight bandwidth is not the limiter, so `quantizedMM` wins
nothing. Measured, not theorized.

# Historical note

The qint8 color-drift issue (desaturation) is specific to **klein-4B**
(measured Δ −24 on blue vs 9B bf16 Δ −1); klein-9B qint8 does not exhibit it.
Do not generalize 4B quality findings to 9B or vice versa.

# Related

- On-the-fly quantize is cheap warm (1.3-3.0 s) — see
  [loading costs](/docs/knowledge/benchmarks/loading-costs.md).
- Pre-quantized checkpoints reload bit-identically to on-the-fly
  quantization (deterministic): mean RGB Δ 0.00 at equal seed.
