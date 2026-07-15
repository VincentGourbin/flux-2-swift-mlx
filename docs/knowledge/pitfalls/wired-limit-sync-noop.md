---
type: Pitfall
title: Synchronous withWiredLimit is a silent no-op since mlx-swift 0.30.6
description: The old wired-memory API was deprecated into a no-op; wiring requires the WiredMemoryManager actor.
tags: [mlx-swift, wired-memory, api-migration]
timestamp: 2026-07-15T00:00:00Z
---

mlx-swift 0.30.6 (Feb 2026, PR #348) replaced the wired-memory API with a
coordinator (`WiredMemoryManager` actor, tickets, policies). The old
**synchronous** `Memory.withWiredLimit(_:_:)` / `GPU.withWiredLimit` still
compiles but is a **deprecated no-op** — code using it silently stopped
wiring memory. Only the async overload routes through the manager.

# Status in this framework

Neither API is used today (weights are not wired). If wiring is ever added
(the R4 follow-up: prevent macOS from evicting resident model weights in a
long-lived host app), use `WiredMemoryManager` tickets — and gate by
available RAM: wiring 17 GB on a 16 GB Mac would be catastrophic.
