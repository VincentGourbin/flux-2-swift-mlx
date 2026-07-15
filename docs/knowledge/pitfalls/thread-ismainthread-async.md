---
type: Pitfall
title: Thread.isMainThread is unavailable in Swift async contexts
description: Swift 6 bans the property where you most want it; use pthread_main_np() for thread assertions in async code.
tags: [swift6, concurrency, diagnostics]
timestamp: 2026-07-15T00:00:00Z
---

`Thread.isMainThread` is `NS_SWIFT_UNAVAILABLE_FROM_ASYNC` — the compiler
rejects it inside `async` functions ("Work intended for the main actor
should be marked with @MainActor"). Ironically this is exactly where you
want it when *proving* that supposedly off-main work really runs off-main
(e.g. validating the #103/#104/#105 nonisolated-loading work).

# The workaround

`pthread_main_np()` (C API, no async restriction): returns 1 on the main
thread, 0 otherwise. Used to verify issue #105: 6/6 text-encoder loads
during a real LoRA training run logged `pthread_main_np=0`.

# Related caveat

The off-main behavior of `nonisolated async` functions awaited from
`@MainActor` contexts depends on `NonisolatedNonsendingByDefault` (SE-0461)
NOT being enabled — this project's Swift 6.0 language mode does not enable
it. If that build setting ever changes, re-verify with the probe above.
