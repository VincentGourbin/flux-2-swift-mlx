---
type: Pitfall
title: MLX Metal kernels are full-JIT per process
description: There is no persistent MLX shader cache and no pre-warm API; first use of each kernel family pays compilation.
tags: [mlx-swift, metal, jit, cold-start]
timestamp: 2026-07-15T00:00:00Z
---

mlx-swift builds with JIT kernels only (no precompiled metallib ships, no
`MTLBinaryArchive` anywhere in the Metal backend as of core 0.31.1). Every
kernel family (GEMM, SDPA, quantized matmul, elementwise…) is compiled from
embedded source at **first use in each process**.

# Practical implications

- Cross-launch relief comes only from the **OS-level** Metal shader cache
  (source-hash → binary, system-wide): launch N+1 of the same binary is
  faster than the first launch after boot/OS-update/app-update. Not
  controllable from the app.
- There is **no pre-warm API**; the only pre-warm is running a tiny
  generation at startup.
- Measured impact here: small — the cold first run of klein-9b (63.8 s) was
  within noise of steady state (60-66 s), so kernel JIT is NOT a plausible
  explanation for multi-x slowdowns. See
  [the July 2026 investigation](/docs/knowledge/investigations/generation-slowness-2026-07.md),
  which explicitly disproved the "cold-start kernel compilation" hypothesis.
