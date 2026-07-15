---
type: Benchmark
title: Klein 9B generation baselines (M3 Max)
description: Healthy per-step and wall-clock numbers; durations far above these indicate contention, not an engine regression.
tags: [klein-9b, klein-4b, baseline, performance]
timestamp: 2026-07-15T00:00:00Z
---

Measured on an M3 Max (14 cores, 96 GB), macOS 26.5, mlx-swift 0.31.6, on a
quiet machine. Configuration: `Flux2MaskedInpaintingChain` at app parity
(crop-and-stitch padding 48, 1 MP working canvas, 4 steps, guidance 1.0)
unless noted.

# Baselines

| Config | Per step | Wall (4 steps) |
|--------|----------|----------------|
| klein-9b bf16/bf16, ~1 MP canvas | **13-15 s** | **~60-66 s** |
| klein-4b qint8 ("Balanced"), 2048x2048 t2i | ~8 s | ~32 s |
| klein-9b bf16, 2048x1536 source, inpaint 1 MP crop | 13-15 s | ~70 s |

Overhead outside denoising (all loads + encodes + decode + composite):
**6-10 s warm**. Denoising is ~85% of a healthy run.

# Interpretation rule

A `generationDuration` far above these numbers (2x or more) on the same
config is **contention or host-app state, not the engine** — see the
[GPU contention playbook](/docs/knowledge/playbooks/gpu-contention-diagnosis.md)
and the [July 2026 investigation](/docs/knowledge/investigations/generation-slowness-2026-07.md)
that established this: identical engine commits produced 64 s and 577 s runs
depending solely on a competing GPU consumer.

# Notes

- First run in a process pays Metal kernel JIT — included in these numbers
  (the cold first run measured 63.8 s vs 60-66 s steady; the difference is
  noise-level). See [Metal full-JIT pitfall](/docs/knowledge/pitfalls/metal-full-jit.md).
- Step time is GEMM/SDPA-bound: neither quantization nor `MLX.compile`
  reduces it. See [quantization verdicts](/docs/knowledge/decisions/quantization-verdicts.md)
  and the [compile decision](/docs/knowledge/decisions/compile-step-neutral.md).
- Benchmarking method: per-phase profiler (`--profile`), repeated runs in one
  process (`--repeat-count`), and a GPU-quiet gate before each run —
  benchmarks on a machine running Xcode builds or an active UI are unusable.
