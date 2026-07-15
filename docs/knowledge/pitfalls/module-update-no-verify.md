---
type: Pitfall
title: Module.update(parameters:) does not verify shapes or dtypes
description: Parameter updates apply blindly by default; wrong-shaped tensors surface later as Metal errors or garbage output.
tags: [mlx-swift, module, update, validation]
timestamp: 2026-07-15T00:00:00Z
---

`Module.update(parameters:)` in mlx-swift defaults to `verify: .none`
(Module.swift:406): no shape or dtype check. Loading externally-produced
weights through it applies mismatched tensors silently; the failure surfaces
mid-first-forward as a Metal shape error, or not at all (garbage output).

# The defense

Validate key sets AND shapes/dtypes against an expected manifest before
calling `update` (see `Flux2PrequantizedCheckpoint.load`, PR #116). When
comparing dtypes, compare by **category** — see
[quanto sources keep bf16](/docs/knowledge/pitfalls/quanto-keeps-bf16.md)
for why strict dtype equality produces false rejections.
