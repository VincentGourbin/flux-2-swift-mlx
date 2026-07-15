---
type: Pitfall
title: Quanto-sourced models keep unquantized params in bfloat16
description: Production models are dtype-mixed depending on checkpoint source; never validate dtypes by strict equality.
tags: [quanto, dtype, bf16, float16, diffusers]
timestamp: 2026-07-15T00:00:00Z
---

The two weight-mapping paths in `Flux2WeightLoader` differ in dtype policy:

- **BFL path** (bf16 checkpoints, e.g. klein-9b): converts EVERY parameter
  bf16 → float16.
- **Diffusers/quanto path** (pre-quantized sources, e.g. klein-4b 8bit):
  converts only the *quantized* tensors to f16 during dequantization —
  unquantized params (norms, embedders) **stay bfloat16**.

So a production model is f16-uniform or f16/bf16-mixed depending on where
its checkpoint came from.

# Why it bites

Any dtype validation using strict equality against a single-dtype reference
rejects legitimate quanto-sourced models. This caused a silent
"always-fallback" bug in the pre-quantized checkpoint loader (caught by the
multi-model test matrix, fixed in PR #116): outputs stayed correct via the
fallback, so nothing looked wrong without reading the warnings.

# The defense

Compare dtypes by **category**: integer packing (uint32 weights, uint8
fp-mode scales) must match exactly; float params accept f16/bf16/f32
variants. Shapes stay strict.
