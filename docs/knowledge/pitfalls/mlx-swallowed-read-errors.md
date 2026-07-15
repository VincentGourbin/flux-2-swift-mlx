---
type: Pitfall
title: MLX swallows deferred safetensors read errors
description: A truncated safetensors with a valid header loads silently uninitialized tensors — no crash, no exception.
resource: https://github.com/ml-explore/mlx
tags: [mlx, safetensors, corruption, silent-failure]
timestamp: 2026-07-15T00:00:00Z
---

Verified in mlx core 0.31.1 source (bundled by mlx-swift 0.31.6).

# The failure mode

`loadArrays` is lazy: tensor reads become `Load` primitives executed at
`eval`. `Load::eval_cpu` enqueues the pread on the IO thread pool via a
`std::packaged_task` and the scheduler only calls `fut.wait()` — **never
`get()`** — so a short-read exception is silently absorbed by the future's
shared state (`mlx/backend/common/load.cpp:53`). The pre-`malloc`ed output
buffer stays **uninitialized**; generation proceeds and produces garbage
with zero diagnostics.

A file with a *truncated header* fails loudly (header is read eagerly and
throws). The dangerous case is **valid header + truncated data**: every
header-derived check (metadata, key names, shapes) passes.

# The defense

Compare actual file size against `8 + header_len + max(data_offsets)` from
the header **before** trusting anything else. Implemented as
`Flux2PrequantizedCheckpoint.payloadIsComplete` (PR #116); apply the same
gate to any safetensors of uncertain provenance. Writing files atomically
(temp + rename) prevents creating such files in the first place.
