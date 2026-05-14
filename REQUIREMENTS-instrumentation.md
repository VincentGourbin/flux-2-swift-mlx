# flux-2-swift-mlx — Instrumentation Requirements

**Status:** Iteration 03 — minimal boundary instrumentation. No known anomalies; we are instrumenting choke points so that *when* something goes wrong we know *where*, not *why*. Per-kernel detail is deferred until a real failure points us at a region.
**Pattern source:** [Vinetas `docs/INSTRUMENTATION_PLAN.md`](https://github.com/intrusive-memory/Vinetas/blob/development/docs/INSTRUMENTATION_PLAN.md) + Produciesta `Docs/TELEMETRY_IMPL_PATTERN.md`
**Host:** Vinetas
**Depends on:** SwiftTuberia ≥ 0.7.0 (for the shared `TuberiaTensorStat` type — flux re-uses it instead of defining its own)
**Priority:** P0 — densest math surface in the Vinetas dep graph; the library that actually produces NaN/Inf when something goes wrong

## Design principle

Instrument **boundaries**, not internals. One event when each major phase starts, one when it ends, plus a single anomaly signal at each phase exit. If a phase reports an anomaly we add a follow-up iteration that drills into that phase with per-step stats. Until then, the cost of telemetry stays near zero.

Concretely, drop from iteration 02:

- Per-step `denoiseStepComplete` triplets (24 MLX reductions per step). Replaced by one anomaly check at loop exit.
- `dtypeHistogram` on weight load. Replaced by a single `weightLoadComplete` with paramCount + duration.
- `vaeBatchNormDenormalize` event. The denormalize math contributes to whatever `vaeDecodeComplete` observes; we don't need a separate event until we see muddy outputs.
- KV-cache hit/miss tracking. Add when we have a Klein9BKV bug to investigate.
- LoRA merge/unmerge detail, VLM interpret detail. Boundary events only.

## Anti-plan-bug checks (carried forward)

These came out of iteration 01 and still apply to anything we wire up:

- **P1** Verify counts match enumerations. If exit criteria say "exactly N emission sites," enumerate N items.
- **P2** Verify parallelism claims. Two sorties marked parallel must not touch the same file.
- **P3** Line numbers are advisory. Reference symbols (`loadTextEncoder`, `for stepIdx in`), not line numbers.
- **P4** Verify enum case existence before referencing it. grep the source.
- **P5** Verify unfamiliar API shape before using it. grep the dependency.

## CRITICAL CONSTRAINT — per-sortie compile + test gate

**Every code-touching sortie's exit criteria MUST include `make build` succeeding AND (for code that changes behavior) `make test` succeeding before commit.** This was the single biggest lesson from iteration 01; iteration 03 keeps the rule. Sub-agents run their own builds.

---

## 1. Why instrument flux-2-swift-mlx

`Flux2Pipeline` is where the diffusion math actually runs: text encoder forward, FlowMatchEuler scheduler init + per-step ODE update, transformer denoise step, and VAE decode. Every numerical anomaly visible to Vinetas — gray images, oversaturated outputs, NaN cascades, dtype-mismatch artifacts — originates here. SwiftTuberia's per-step events tell you *that* something went wrong; flux's events tell you *which phase produced it*.

The minimal surface must answer: "did each major phase complete, how long did it take, and did its output look numerically sane at the boundary?"

What it must NOT surface (this iteration):
- Per-step latent / noise-pred stats
- Per-transformer-block, per-attention-head, per-RoPE events
- Per-weight dtype histograms
- BatchNorm denormalize transitions
- KV-cache hit/miss
- LoRA merge layer counts
- VLM-internal call shape
- `Flux2Profiler` timings (already a separate seam)

If a future failure points us at one of these, we add a targeted iteration. Not before.

---

## 2. Coexistence with existing surfaces

| Surface | Status |
|---|---|
| `Flux2Debug.log` / `verbose` | Keep as-is. Telemetry fires alongside. |
| `Flux2Profiler.start/end(label)` | Keep as-is. Telemetry events carry their own `durationSeconds`. |
| `Flux2DownloadProgressCallback` | Keep as-is. |
| Per-step UI progress callback inside `generate*` | Keep as-is. |
| `Flux2Error` `LocalizedError` cases | Keep as-is. Every `throw` is preceded by an `errorThrown` emit. |

---

## 3. Public types to add

```
Sources/Flux2Core/Telemetry/
  Flux2TelemetryEvent.swift
  Flux2TelemetryReporter.swift
```

`TuberiaTensorStat` is imported from SwiftTuberia (`import Tuberia`).

### 3.1 `Flux2TelemetryEvent.swift`

```swift
@preconcurrency import MLX
import Foundation
import Tuberia  // for TuberiaTensorStat

public enum Flux2TelemetryEvent: Sendable {

    // --- Pipeline lifecycle ---
    // pipelineInit fires from Flux2Pipeline.init (sync), so hosts must call
    // setTelemetry() before the first generation to avoid losing this event.
    // pipelineDispose fires from an explicit `public func dispose() async`,
    // NOT deinit (deinit can't be async).
    case pipelineInit(model: String, quantization: String, vaeConfig: String)
    case pipelineDispose

    // --- Weight loading (one event per component, on success) ---
    case weightLoadComplete(component: WeightComponent, paramCount: Int, durationSeconds: Double)

    // --- Text encoding (boundary event with NaN/Inf check on the embedding) ---
    case textEncodeComplete(encoderName: String, finalPromptLength: Int, embeddingStat: TuberiaTensorStat, durationSeconds: Double)

    // --- Scheduler ---
    case schedulerConfigured(numInferenceSteps: Int, shift: Float, imageSeqLen: Int, mu: Float)

    // --- Denoise loop (start + end only; per-step events deferred) ---
    case denoiseLoopStart(variant: DenoiseVariant, totalSteps: Int, latentShape: [Int], latentDtype: String)
    case denoiseLoopEnd(variant: DenoiseVariant, totalSteps: Int, completedSteps: Int, finalLatentStat: TuberiaTensorStat, durationSeconds: Double)

    // --- VAE decode (boundary event with pixel-range check) ---
    case vaeDecodeComplete(pixelStat: TuberiaTensorStat, outputDims: [Int], durationSeconds: Double)

    // --- Anomaly side-channel (fires alongside any of the *Complete / *End events
    //     whose stat shows NaN, Inf, or out-of-range magnitude). One signal, not
    //     a per-step stream. ---
    case numericalAnomaly(phase: AnomalyPhase, kind: AnomalyKind, stat: TuberiaTensorStat)

    // --- Cancellation ---
    case generationCancelled(stepIndex: Int?)  // nil for pre-loop cancellation sites

    // --- Error side-channel — fires immediately before every Flux2Error throw ---
    case errorThrown(phase: ErrorPhase, errorDescription: String)

    public enum WeightComponent: String, Sendable {
        case textEncoderKlein     // Qwen3 (KleinTextEncoder)
        case textEncoderDev       // Mistral (DevTextEncoder / Flux2TextEncoder)
        case textEncoderTraining
        case transformer
        case vae
        case lora
    }

    public enum DenoiseVariant: String, Sendable {
        case textToImage
        case imageToImageKVExtractStep0      // single non-loop call; emits Start+End with totalSteps:1, completedSteps:1
        case imageToImageKVCached
        case imageToImageFullRecompute
    }

    public enum AnomalyPhase: String, Sendable {
        case textEncode
        case denoiseLoopEnd
        case vaeDecode
    }

    public enum AnomalyKind: String, Sendable {
        case nan
        case inf
        case outOfRange      // |x| > TuberiaTensorStat.defaultOutOfRangeThreshold
        case zeroLatent      // mean ≈ 0 && std ≈ 0
    }

    public enum ErrorPhase: String, Sendable {
        case modelNotLoaded
        case invalidConfiguration
        case insufficientMemory
        case modelNotDownloaded
        case generationCancelled
        case generationFailed
        case weightLoadFailed
        case vaeDecodeFailed
        case textEncoderFailed
        case vlmInterpretFailed
        case loraLoadFailed
        case imageProcessingFailed
        case other
    }
}
```

Notes:
- No `weightLoadStart`, `textEncoderForwardStart`, `vlmInterpretStart`, `loraLoadStart`, `vaeDecodeStart`, `vaeBatchNormDenormalize` events. Durations are carried on the `*Complete` side; we don't need before/after pairs at this iteration's resolution.
- Quantization is reported as a single string (e.g. `"klein4b-q4-g64"`), not a per-component manifest. Sufficient for "which model variant" answers.
- `embeddingStat`, `finalLatentStat`, `pixelStat` are the only `TuberiaTensorStat` samples we collect. Three per generation total (plus zero or one for each KV-extract path).

### 3.2 `Flux2TelemetryReporter.swift`

```swift
public protocol Flux2TelemetryReporter: Sendable {
    func capture(_ event: Flux2TelemetryEvent) async
}

public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter {
    public init() {}
    public func capture(_ event: Flux2TelemetryEvent) async {}
}
```

---

## 4. Injection points

### 4.1 The `@unchecked Sendable` problem

`Flux2Pipeline` is `public class Flux2Pipeline: @unchecked Sendable`. Same for `KleinTextEncoder`, `DevTextEncoder`, `FlowMatchEulerScheduler`, transformer classes. The reporter cannot be stored as a plain `var` without breaking `Sendable`.

**Pattern:** store telemetry behind an `OSAllocatedUnfairLock`:

```swift
import os.lock

public class Flux2Pipeline: @unchecked Sendable {
    private let _telemetryLock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)

    public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?) {
        _telemetryLock.withLock { $0 = reporter }
    }

    fileprivate func currentTelemetry() -> (any Flux2TelemetryReporter)? {
        _telemetryLock.withLock { $0 }
    }
}
```

Emission template:

```swift
if let telemetry = currentTelemetry() {
    let stat = TuberiaTensorStat.sample(latents)
    await telemetry.capture(.denoiseLoopEnd(/* … */))
}
```

Lock is uncontended in the steady state. `withLock` is ~10ns; cost is dominated by the reporter call. With per-step events removed, there is no hot-path concern.

### 4.2 Setters on each top-level type

| Type | Setter |
|---|---|
| `Flux2Pipeline` | `public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?)` |
| `KleinTextEncoder` | same |
| `DevTextEncoder` | same |
| `Flux2TextEncoder` (Mistral) | same |
| `FlowMatchEulerScheduler` | same |
| `Flux2WeightLoader` | same |
| `Flux2Transformer2DModel` | same |

The host only ever calls `Flux2Pipeline.setTelemetry`. The pipeline propagates the reporter to its owned subcomponents.

### 4.3 Why not migrate to `actor`?

The right long-term answer, but it is a breaking API change. The lock-based seam is non-breaking. Defer.

---

## 5. Per-event emission spec

| Event | Where it fires | Count per generation |
|---|---|---|
| `pipelineInit` | End of `Flux2Pipeline.init` | 1 per pipeline construction |
| `pipelineDispose` | New `dispose() async` method | 1 per pipeline tear-down (host-driven) |
| `weightLoadComplete` | After each `loadTextEncoder` / `loadTransformer` / `loadVAE` / LoRA load completes successfully | 3–4 per pipeline init (plus 0–1 LoRA per generate) |
| `textEncodeComplete` | After each `*Encoder.encode*(...)` call returns | 1–2 per generate |
| `schedulerConfigured` | After `scheduler.setTimesteps(...)` returns | 1 per generate |
| `denoiseLoopStart` | Just before each of the 3 `for stepIdx in ...` loops, plus once before the KV-extract one-shot. At most one path runs per generation. | 1 per generate |
| `denoiseLoopEnd` | Just after each loop exit (success or break-on-cancel), and immediately after the KV-extract one-shot. | 1 per generate |
| `vaeDecodeComplete` | After `postprocessVAEOutput` succeeds | 1 per generate |
| `numericalAnomaly` | Fires alongside `textEncodeComplete` / `denoiseLoopEnd` / `vaeDecodeComplete` whenever the carried `TuberiaTensorStat` has `hasNaN || hasInf || max.magnitude > TuberiaTensorStat.defaultOutOfRangeThreshold` or `(mean.magnitude < 1e-6 && std < 1e-6)` | 0 in the happy path; ≥1 only on failure |
| `generationCancelled` | Every cancellation check site. Pre-loop sites use `nil`; in-loop sites use the current step index. | 0–1 per generate |
| `errorThrown` | Immediately before every `throw Flux2Error.…` in the pipeline. | 0–1 per generate |

Total event count for a clean T2I generation: ~6 events (pipelineInit once-per-pipeline, then 3–4 weightLoadComplete once-per-pipeline, then per-generate: textEncodeComplete + schedulerConfigured + denoiseLoopStart + denoiseLoopEnd + vaeDecodeComplete = 5). Compared to iteration 02's per-step stream this is roughly 50–100× fewer events and roughly 100× fewer `TuberiaTensorStat.sample` calls.

---

## 6. Adapter mapping (Vinetas host side)

`Flux2TelemetryAdapter` at `Vinetas/Telemetry/Adapters/Flux2TelemetryAdapter.swift`. Exhaustive switch, no `default:`. Sink-phase strings mirror the event case names (`flux_pipeline_init`, `flux_weight_load_complete_<component>`, `flux_encode_complete_<encoderName>`, `flux_scheduler_configured`, `flux_denoise_loop_start`, `flux_denoise_loop_end`, `flux_vae_decode_complete`, `flux_anomaly_<kind>`, `flux_cancelled`, `flux_error_<phase>`).

Memory-snapshot routing (per Vinetas INSTRUMENTATION_PLAN §3.1):
- `weightLoadComplete`, `denoiseLoopStart`, `denoiseLoopEnd`, `vaeDecodeComplete` go through `captureWithMemorySnapshot`.
- Everything else uses plain `capture`.

---

## 7. Tests

Add to `Tests/Flux2CoreTests/`. Minimal set, matching the minimal surface:

| Test | Purpose |
|---|---|
| `Flux2TelemetryBoundaryEventsTests` | One T2I generate through a `MockReporter`. Assert the expected event sequence: pipelineInit → 3×weightLoadComplete → textEncodeComplete → schedulerConfigured → denoiseLoopStart → denoiseLoopEnd → vaeDecodeComplete. Assert per-event field shape. |
| `Flux2TelemetryNoopOverheadTests` | T2I run with `nil` reporter vs `NoopFlux2TelemetryReporter`. Wall-clock medians within ±2% over 20 iterations. Single most important test — proves the boundary emit path is cheap. |
| `Flux2TelemetryAnomalyTests` | Inject a transformer mock that returns a tensor containing NaN. Assert `denoiseLoopEnd` carries `finalLatentStat.hasNaN == true` and that a `numericalAnomaly(phase: .denoiseLoopEnd, kind: .nan, ...)` event fires alongside. |
| `Flux2TelemetryErrorPathTests` | Force a `Flux2Error.vaeDecodeFailed` and assert `errorThrown(phase: .vaeDecodeFailed, ...)` fires immediately before the throw. |
| `Flux2TelemetryLockContentionTests` | Concurrent `setTelemetry` toggles + a running denoise loop; assert no data races and emissions reflect the most-recently-set reporter. Plain XCTest (avoids the swift-testing + macOS 26.2 SDK issue noted in iteration 02). |

Per-step assertion, KV-cache assertion, BatchNorm-denormalize assertion, dtype-histogram assertion: all deferred to whatever iteration follows a real failure that demands them.

Test sortie discipline (from iteration 02 lessons):
- `import TestHelpers` in every test file that uses helpers; module dependency alone is insufficient.
- Verify unfamiliar API surface (e.g. `MLXArray.zeros(_:type:)`) against the dependency before writing the test.
- Under Swift 6 strict mode, private helpers must be non-static OR every call site must use `Self.` qualifier.

---

## 8. Out of scope (this iteration)

- Per-step latent / noise-pred stats. Re-add when a `numericalAnomaly` on `denoiseLoopEnd` points us at the denoise loop.
- Per-transformer-block / per-attention-head events.
- `dtypeHistogram` on weight load. Re-add if we see a "weights silently dequantized" symptom in the field.
- `vaeBatchNormDenormalize` event. Re-add if we see muddy-image reports.
- KV-cache hit/miss tracking. Re-add if Klein9BKV regresses.
- LoRA merge layer counts, VLM interpret detail.
- `Flux2Profiler` integration. Separate observability surface; both stay.
- Training instrumentation. Vinetas only runs inference.

---

## 9. Versioning

**Minor** version bump (additive). Pin floor: `3.2.0` post-release. Must ship AFTER SwiftTuberia ≥ 0.7.0.

---

## 10. Implementation checklist

- [ ] Add `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` per §3.1
- [ ] Add `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` per §3.2
- [ ] Add `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` and `setTelemetry`/`currentTelemetry` to `Flux2Pipeline`, `KleinTextEncoder`, `DevTextEncoder`, `Flux2TextEncoder`, `FlowMatchEulerScheduler`, `Flux2WeightLoader`, transformer classes
- [ ] In `Flux2Pipeline.setTelemetry`, propagate the reporter to all owned subcomponents
- [ ] Wire boundary emission sites per §5
- [ ] Add anomaly-check helper: given a `TuberiaTensorStat`, emit `numericalAnomaly` if any of {NaN, Inf, outOfRange, zeroLatent} hits. Call from the three boundary emit sites.
- [ ] Ensure every `throw Flux2Error.…` is preceded by `errorThrown` emit
- [ ] Add tests per §7
- [ ] Run baseline overhead test; commit results in PR description (focus: T2I, nil reporter, ±2% bound)
- [ ] Tag release with `MINOR` bump
