---
okf_version: "0.1"
---

# Engineering Knowledge Base

Durable, measured engineering knowledge about this framework: benchmarks,
decisions with their rationale, verified pitfalls, and diagnostic playbooks.
Structured as an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
knowledge bundle — plain markdown with YAML frontmatter, readable by humans
and agents alike. This is the **knowledge layer**: what is true and why. The
user docs in [`docs/`](../) explain how to use the framework; concepts here
record what was measured and decided so it is never re-derived or re-litigated.

# Benchmarks

* [Klein 9B generation baselines](benchmarks/klein9b-baselines.md) - healthy per-step and wall-clock numbers on M3 Max; anything far above them is contention, not the engine
* [Model loading costs](benchmarks/loading-costs.md) - what each loading phase costs warm/cold, and where the time actually goes

# Decisions

* [Quantization verdicts for inference](decisions/quantization-verdicts.md) - qint8/mxfp8/int4 speed and quality, measured; quantization buys memory, not step time
* [MLX.compile on the denoising forward: neutral, kept opt-in](decisions/compile-step-neutral.md) - why compiling the step wins nothing here, and the traps if it is ever revisited

# Pitfalls

* [MLX swallows deferred read errors](pitfalls/mlx-swallowed-read-errors.md) - a truncated safetensors with a valid header loads silently uninitialized tensors
* [Module.update() does not verify shapes](pitfalls/module-update-no-verify.md) - parameter updates apply blindly by default
* [quantize() skips already-quantized layers](pitfalls/quantize-skips-quantized.md) - a "fallback" re-quantize on a mutated model is a silent no-op
* [Quanto sources keep bf16 unquantized params](pitfalls/quanto-keeps-bf16.md) - production models are dtype-mixed depending on the checkpoint source
* [Synchronous withWiredLimit is a no-op](pitfalls/wired-limit-sync-noop.md) - silent behavior change since mlx-swift 0.30.6
* [Thread.isMainThread is banned in async contexts](pitfalls/thread-ismainthread-async.md) - use pthread_main_np() for thread assertions
* [Metal kernels are full-JIT per process](pitfalls/metal-full-jit.md) - there is no persistent MLX shader cache to pre-warm

# Investigations

* [Generation slowness campaign (July 2026)](investigations/generation-slowness-2026-07.md) - how a suspected engine regression turned out to be GPU contention from the host app's UI

# Playbooks

* [Diagnosing GPU contention](playbooks/gpu-contention-diagnosis.md) - the 5-minute protocol that attributes a slow generation to its real cause
