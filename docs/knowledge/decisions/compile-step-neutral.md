---
type: Decision
title: "MLX.compile on the denoising forward: measured neutral, kept opt-in"
description: Compiling the transformer step wins nothing here because the elementwise hot spots are already hand-fused; do not re-attempt without new evidence.
tags: [mlx-compile, performance, wontfix]
timestamp: 2026-07-15T00:00:00Z
---

Decision (2026-07-14, PR #113): `Flux2Pipeline.compileDenoisingStep` /
`--compile-step` exists but is **opt-in and never the default**. Measured
neutral — do not re-litigate without new evidence (e.g. a future mlx-swift
with materially different compile behavior).

# Measurements (klein-9b bf16, 1 MP, 4 steps, M3 Max)

| | Steady-state step | Output |
|---|---|---|
| baseline | 12.5-13.6 s | reference |
| compiled | 13.2 s | bit-identical (mean RGB Δ 0.01/255) |

One-time trace: ~1-2 s on the first step after each (re)trace.

# Why it is neutral

The 5-15% wins reported for `compile` elsewhere come from fusing elementwise
chains. This codebase already hand-optimized those: modulation precomputed
outside the block loop, fused RoPE Metal kernel, `MLXFast` rmsNorm/SDPA.
What remains of a step is GEMM/SDPA, which `compile` does not accelerate.

# Traps if ever revisited

- Intra-forward `eval()` (the `memoryOptimization` graph segmentation) is
  **illegal under compile tracing** — the flag forces `.disabled`, raising
  peak memory to the unsegmented level.
- The compiled closure **retains the transformer** (17 GB) — it must be
  invalidated in `unloadTransformer()` or unload frees nothing.
- Guidance arity changes the traced signature; shapes re-trace automatically.

# Related

- Same conclusion family as [quantization verdicts](/docs/knowledge/decisions/quantization-verdicts.md):
  the klein-9b step time is at the hardware floor on M3 Max; remaining gains
  are in loading/UX and in adopting `klein-9b-kv` for multi-reference I2I
  (~2.66x on steps, already in the framework).
- The asyncEval load/encode overlap idea (R1) was also dropped: ≤2-3 s once
  per session since the text encoder can stay resident (PR #114), against
  risky concurrency surgery in the pipeline.
