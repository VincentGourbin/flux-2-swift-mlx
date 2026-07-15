---
type: Pitfall
title: quantize() silently skips already-quantized layers
description: A fallback re-quantize on a partially mutated model is a no-op, not a repair.
tags: [mlx-swift, quantization, fallback]
timestamp: 2026-07-15T00:00:00Z
---

`quantizeSingle` in MLXNN returns nil for layers that are already
`Quantized` (mlx-swift Quantized.swift:34), so `quantize(model:)` on a model
whose structure was already (even partially) quantized silently does
nothing for those layers.

# Why it bites

Any "validate-after-mutate then fall back" pattern is broken by this: if a
loader quantizes a model's structure and THEN discovers the weights are
unusable, the standard path cannot repair the instance — its own quantize
call no-ops, and full-precision weights get written into packed
`QuantizedLinear.weight` slots (which update() accepts blindly — see
[no-verify pitfall](/docs/knowledge/pitfalls/module-update-no-verify.md)).

# The defense

Validate everything BEFORE mutating the model (build the expected manifest
on a throwaway structure clone — its lazy random weights are never
evaluated, so this is free), or recreate the model instance after a failed
mutating load. Implemented in `Flux2PrequantizedCheckpoint.load` (PR #116).
