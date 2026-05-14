---
feature_name: OPERATION TWIN LIGHTHOUSE
iteration: 3
mission_branch: instrumentation/03
starting_point_commit: fba1583
---

# EXECUTION_PLAN.md — Boundary Telemetry for flux-2-swift-mlx + pixart-swift-mlx

**Source requirements:** [`REQUIREMENTS-instrumentation.md`](REQUIREMENTS-instrumentation.md) (this repo) + [`../pixart-swift-mlx/REQUIREMENTS-instrumentation.md`](../pixart-swift-mlx/REQUIREMENTS-instrumentation.md)
**Cross-library convention:** [`AGENTS.md §11`](AGENTS.md#11-telemetry-chokepoint-convention-cross-library)
**Generated:** 2026-05-13 (fresh breakdown — disregards all prior plans)

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

## Mission scope

Two sibling MLX-Swift image-generation libraries — `flux-2-swift-mlx` and `pixart-swift-mlx` — must expose **boundary-only** telemetry that initializes through the same `setTelemetry`-on-top-level-type seam and returns chokepoint events from the same catalog defined in [AGENTS.md §11](AGENTS.md#11-telemetry-chokepoint-convention-cross-library).

Asymmetric starting state:

| Library | Implementation | REQUIREMENTS doc |
|---|---|---|
| `pixart-swift-mlx` | **Live.** `PixArtTelemetryEvent`/`PixArtTelemetryReporter` ship; `OSAllocatedUnfairLock`-backed `setTelemetry` on `PixArtDiT`; emission sites in `PixArtDiT.swift`, `PixArtRecipe.swift`, `PixArtFP16Recipe.swift`; 4 telemetry test files; `MockReporter` helper. | **Stale** — 18 KB verbose draft predating the minimal-boundary decision. |
| `flux-2-swift-mlx` | **Zero code.** No telemetry directory, no setters, no dep on SwiftTuberia. | **Current** — slim iteration-03 spec. |

Therefore pixart is the **canonical pattern source**. Sortie agents implementing flux MUST read the matching pixart files for the exact lock-and-emit shape before writing flux code (per P5: verify API surface before use).

## Mandatory constraints (apply to every sortie)

- **Per-sortie compile + test gate.** Every code-touching sortie's exit criteria include `make build` succeeding AND `make test` succeeding (CI-safe suites) before commit. No sortie ships uncompiling code.
- **No `swift build` / `swift test`.** Use `make` targets locally; `xcodebuild` with `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES` in CI.
- **P1–P5 anti-plan-bug checks** ([REQUIREMENTS-instrumentation.md §Anti-plan-bug checks](REQUIREMENTS-instrumentation.md)): match counts to enumerations, verify parallelism claims, treat line numbers as advisory, grep-verify enum cases before reference, verify unfamiliar APIs against the dependency.
- **Naming convention.** All new event cases, enums, and adapter sink phases follow [AGENTS.md §11.2](AGENTS.md#112-naming-rules).
- **No new pixart event cases without first updating pixart's REQUIREMENTS doc and AGENTS.md §11 catalog.** Stability of the cross-library event surface is the point of this mission.

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| A. PixArt doc alignment | `../pixart-swift-mlx/` | 1 | 0 | none |
| B. Flux2 boundary telemetry | `Sources/Flux2Core/` + `Tests/Flux2CoreTests/` + `Tests/TestHelpers/` | 15 | 1 | none (Layer-0 A and Layer-1 B can run in parallel because they touch disjoint repos) |

Layer 0 work units gate Layer 1 only when dependencies are declared. Here, A and B touch disjoint repos and share no source files, so they may run in parallel — but the **cross-library consistency check** in Sortie B16 audits both at the end, so A must complete before B16.

---

## Work Unit A — PixArt doc alignment

Pixart's implementation is the canonical §11 pattern. The REQUIREMENTS-instrumentation.md on disk is stale and contradicts the live code (per-step events specified in the doc were never built; the live events are boundary-only). One sortie brings the doc into line with reality.

### Sortie A1: Rewrite pixart REQUIREMENTS-instrumentation.md to reflect live implementation

**Priority**: 4.5 — independent repo, gates only B16; can run any time during Work Unit B execution.

**Entry criteria**:
- [ ] First sortie — no prerequisites

**Tasks**:
1. Read the live event surface in `/Users/stovak/Projects/pixart-swift-mlx/Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` and `PixArtTelemetryReporter.swift`. These are the source of truth.
2. Read every actual emission site by grepping `setTelemetry|currentTelemetry|capture(.` across `/Users/stovak/Projects/pixart-swift-mlx/Sources/PixArtBackbone/*.swift`.
3. Read the existing telemetry tests in `/Users/stovak/Projects/pixart-swift-mlx/Tests/PixArtBackboneTests/PixArtTelemetry*.swift` and `MockReporter.swift` to confirm the contract the tests assert.
4. Overwrite `/Users/stovak/Projects/pixart-swift-mlx/REQUIREMENTS-instrumentation.md` with a slim minimal-boundary spec mirroring the structure of `/Users/stovak/Projects/flux-2-swift-mlx/REQUIREMENTS-instrumentation.md`. The new doc MUST:
   - State up-front that pixart is a *backbone*, not a pipeline (it does not own pipeline lifecycle, scheduler config, denoise loop, VAE decode, or cancellation — those live in SwiftTuberia's `DiffusionPipeline`).
   - Document the actual public event cases (`weightLoadComplete`, `weightUnloadComplete`, `recipeValidated`, `recipeValidationFailed`, `numericalAnomaly`, `errorThrown`) with their exact arg labels and nested enums.
   - Document the setTelemetry seam on `PixArtDiT` and the `telemetry:` parameter on `PixArtRecipe.validate(...)` / `PixArtFP16Recipe.validate(...)`.
   - Cross-link to flux's REQUIREMENTS-instrumentation.md and AGENTS.md §11.
   - Include the "Out of scope" section: per-block DiT events, per-attention-head events, internal kernel detail (the patchEmbedComplete / captionProjectionComplete / siluWorkaroundExecuted / etc. that were in the stale draft but never built).

**Exit criteria**:
- [ ] `/Users/stovak/Projects/pixart-swift-mlx/REQUIREMENTS-instrumentation.md` rewritten; `wc -c REQUIREMENTS-instrumentation.md` reports less than 11000 bytes (current is 21,678 bytes; target is ~6–9 KB).
- [ ] Every event case named in the new doc grep-matches a `case` declaration in `PixArtTelemetryEvent.swift`.
- [ ] Every emission site named in the new doc grep-matches an actual `await telemetry?.capture(.…)` or `Task { await telemetry.capture(…) }` call in pixart source.
- [ ] The doc contains a cross-reference to `flux-2-swift-mlx/AGENTS.md §11` and to flux's REQUIREMENTS doc.
- [ ] `cd /Users/stovak/Projects/pixart-swift-mlx && make build` succeeds (sanity — doc-only edit, but verify no accidental code damage).
- [ ] `cd /Users/stovak/Projects/pixart-swift-mlx && make test` succeeds.

---

## Work Unit B — Flux2 boundary telemetry

Implements the slim event surface in `REQUIREMENTS-instrumentation.md` for flux, using pixart as the canonical pattern source. Most sorties touch `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`, so they are inherently sequential within this work unit (refine-parallelism pass will confirm).

### Sortie B1: Add SwiftTuberia dependency

**Priority**: 48.5 — foundation; transitively blocks all 15 downstream B sorties. Must execute first within Work Unit B.

**Entry criteria**:
- [ ] First sortie in Work Unit B — no prerequisites.

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Package.swift` for the exact SwiftTuberia spec string (`url:` + version constraint).
2. Add the matching `.package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", .upToNextMajor(from: "0.7.0"))` entry to `Package.swift` in this repo, keeping the existing `sibling()`-pattern dep declarations intact.
3. Add `.product(name: "Tuberia", package: "SwiftTuberia")` to the `Flux2Core` target's `dependencies` array.
4. Confirm the version is available by checking SwiftTuberia's published releases (`gh release list -R intrusive-memory/SwiftTuberia --limit 5`).

**Exit criteria**:
- [ ] `grep "intrusive-memory/SwiftTuberia" Package.swift` returns exactly one match.
- [ ] `grep 'product(name: "Tuberia"' Package.swift` returns exactly one match inside the `Flux2Core` target block.
- [ ] `make build` succeeds (proves the resolved dep compiles in this project).
- [ ] `make test` succeeds (existing tests still green).

---

### Sortie B2: Define `Flux2TelemetryEvent` and `Flux2TelemetryReporter`

**Priority**: 46.75 — establishes the public event-surface contract used by every subsequent emission sortie.

**Entry criteria**:
- [ ] B1 complete: `Tuberia` product visible to `Flux2Core` target.

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` and `PixArtTelemetryReporter.swift` as the pattern reference.
2. Create directory `Sources/Flux2Core/Telemetry/`.
3. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` defining `public enum Flux2TelemetryEvent: Sendable` with exactly the cases enumerated in [REQUIREMENTS-instrumentation.md §3.1](REQUIREMENTS-instrumentation.md#31-flux2telemetryeventswift): `pipelineInit`, `pipelineDispose`, `weightLoadComplete`, `textEncodeComplete`, `schedulerConfigured`, `denoiseLoopStart`, `denoiseLoopEnd`, `vaeDecodeComplete`, `numericalAnomaly`, `generationCancelled`, `errorThrown`. Include the five nested enums: `WeightComponent` (6 cases), `DenoiseVariant` (4 cases), `AnomalyPhase` (3 cases), `AnomalyKind` (4 cases), `ErrorPhase` (13 cases).
4. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` defining `public protocol Flux2TelemetryReporter: Sendable` with `func capture(_ event: Flux2TelemetryEvent) async` and `public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter`.
5. Verify SwiftTuberia surface: grep `/Users/stovak/Projects/pixart-swift-mlx/.build` or the resolved checkout for `TuberiaTensorStat` to confirm the public symbol name and import path before writing `import Tuberia`.

**Exit criteria**:
- [ ] File `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` exists and parses.
- [ ] File `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` exists and parses.
- [ ] `grep -c "^    case " Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns at least 41 (11 top-level cases + 6 + 4 + 3 + 4 + 13 nested = 41 minimum).
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B3: Wire `setTelemetry` seam on every top-level type

**Priority**: 45.25 — establishes the lock-and-emit pattern reused by every emission sortie; touches 7 files, highest single-sortie risk in the foundation layer.

**Entry criteria**:
- [ ] B2 complete: `Flux2TelemetryEvent` and `Flux2TelemetryReporter` defined.

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Sources/PixArtBackbone/PixArtDiT.swift` lines 37–48 for the exact `OSAllocatedUnfairLock` + `setTelemetry` + `currentTelemetry` pattern.
2. Apply the same 8-line pattern to each of these files (locating the class declaration first via grep, treating any embedded line numbers in the requirements doc as advisory):
   - `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` (class `Flux2Pipeline`)
   - `Sources/Flux2Core/Loading/KleinTextEncoder.swift` (class `KleinTextEncoder`)
   - `Sources/Flux2Core/Loading/DevTextEncoder.swift` (class `DevTextEncoder`)
   - `Sources/Flux2Core/Loading/MistralEncoder.swift` (class `Flux2TextEncoder` — grep to confirm class name)
   - `Sources/Flux2Core/Scheduler/FlowMatchEulerScheduler.swift` (class `FlowMatchEulerScheduler`)
   - `Sources/Flux2Core/Loading/WeightLoader.swift` (class `Flux2WeightLoader` — grep to confirm)
   - `Sources/Flux2Core/Transformer/Flux2Transformer.swift` (class `Flux2Transformer2DModel` — grep to confirm)
3. In `Flux2Pipeline.setTelemetry`, after storing the reporter locally, propagate it to every owned subcomponent that has its own setter: text encoders (whichever is non-nil), scheduler, weight loader, transformer. Match by Optional chaining: `kleinEncoder?.setTelemetry(reporter)`, etc.

**Exit criteria**:
- [ ] `grep -l "OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>" Sources/Flux2Core/**/*.swift` returns exactly 7 files (the 7 types above).
- [ ] `grep -c "func setTelemetry" Sources/Flux2Core/**/*.swift` totals exactly 7.
- [ ] `Flux2Pipeline.setTelemetry` body contains at least 5 propagation calls (4 subcomponent types + weight loader; exact count depends on whether Flux2TextEncoder is a separate stored property or shares one).
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B4: Wire pipeline lifecycle events (`pipelineInit`, `pipelineDispose`)

**Priority**: 17.5 — simple two-emit pair; gates B12.

**Entry criteria**:
- [ ] B3 complete: `Flux2Pipeline.setTelemetry` and `currentTelemetry` exist.

**Tasks**:
1. At the end of `Flux2Pipeline.init`, emit `.pipelineInit(model:quantization:vaeConfig:)` via `Task { await currentTelemetry()?.capture(...) }` (init is sync; the Task{} caveat is documented in REQUIREMENTS-instrumentation.md §3.1). Populate `model` from whichever enum/string identifies the model variant; populate `quantization` as a single short string (e.g. `"klein4b-q4-g64"`); populate `vaeConfig` from the VAE configuration.
2. Add a new method `public func dispose() async` to `Flux2Pipeline` that emits `.pipelineDispose` via `await currentTelemetry()?.capture(.pipelineDispose)`. The method does not need to free resources — telemetry only.

**Exit criteria**:
- [ ] `grep "capture(.pipelineInit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1 match.
- [ ] `grep "capture(.pipelineDispose" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1 match.
- [ ] `grep "public func dispose() async" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1 match.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B5: Wire `weightLoadComplete` events

**Priority**: 19.25 — multi-file (6+) and depends on MLX param-count API discovery; only sortie outside Pipeline.swift that can run in parallel with the Pipeline emission stream (B6–B10).

**Entry criteria**:
- [ ] B3 complete (subcomponent setters exist).

**Tasks**:
1. Locate weight-loading sites by *actual* symbol — line numbers in REQUIREMENTS are advisory (P3) and the verbose names in the prior draft do not match the codebase. The real entry points are:
   - `Sources/Flux2Core/Loading/KleinTextEncoder.swift` → `public func load(from:)` (line ~46)
   - `Sources/Flux2Core/Loading/DevTextEncoder.swift` → `public func load(from:)` (line ~38)
   - `Sources/Flux2Core/Loading/MistralEncoder.swift` → `public func load(from:)` (line ~40) — class `Flux2TextEncoder`
   - `Sources/Flux2Core/Loading/TrainingTextEncoder.swift` → `public func load()` (lines ~40 and ~55, two overrides)
   - `Sources/Flux2Core/Loading/WeightLoader.swift` → `public static func loadWeights(from:)` (two overloads, lines ~14 and ~42) and `public static func loadQuantizedTransformer(...)` (line ~694)
   - `Sources/Flux2Core/LoRA/LoRAAdapter.swift` → `public func loadLoRA(_ config:)` (line ~69) — primary LoRA entry; `LoRALoader.load()` and `LoRAConfig.load(from:)` are helpers, do not emit there.
   Confirm each site via `grep -n "func load" Sources/Flux2Core/Loading/ Sources/Flux2Core/LoRA/` before editing.
2. Around each load function (success path only), capture start time at the function entry and emit `.weightLoadComplete(component: <enum>, paramCount: <int>, durationSeconds: <double>)` immediately before the function returns. `component` maps as: `textEncoderKlein` (Klein) / `textEncoderDev` (Dev) / `textEncoderMistral`-or-`textEncoderTraining` (Mistral and TrainingTextEncoder — grep `Flux2TelemetryEvent.swift` for the actual case name from B2) / `transformer` (WeightLoader.loadQuantizedTransformer + transformer-targeted `loadWeights`) / `vae` (vae-targeted `loadWeights`) / `lora` (LoRAAdapter.loadLoRA).
3. Use `Date().timeIntervalSince(start)` for `durationSeconds` (cheap; matches the pattern used by pixart at `PixArtDiT.apply(weights:)`).
4. `paramCount` for text encoders: count parameters in the loaded module (use `.numParameters()` if available on MLX `Module`, else `parameters.flattenedValues().reduce(0) { $0 + $1.size }`). For transformer: same. For VAE: same. For LoRA: count adapter weights.

**Exit criteria**:
- [ ] `grep -rc "capture(.weightLoadComplete" Sources/Flux2Core/` (summed across files) returns at least 5 (Klein + Dev + Mistral/Training + transformer + VAE); LoRA brings it to 6 if the LoRAAdapter site is wired.
- [ ] Each emit site populates `component:`, `paramCount:`, `durationSeconds:` (verify by reading the emit lines; no `0` placeholder for `paramCount`).
- [ ] Every `WeightComponent` enum case declared in B2 is referenced at least once across the emit sites, OR documented in a code comment as "deferred to follow-up iteration" if no current load entry point exists for it.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B6: Wire `textEncodeComplete` events

**Priority**: 20.75 — gates B10 (anomaly check uses textEncode-phase stat) and B12.

**Entry criteria**:
- [ ] B3 complete.

**Tasks**:
1. Locate text-encoder forward call sites in `Flux2Pipeline.swift` via `grep -n "kleinEncoder\?\.encode\|devEncoder\?\.encode\|flux2TextEncoder\?\.encode\|textEncoder\?\.encode" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`.
2. Around each call, capture start time and emit `.textEncodeComplete(encoderName: <string>, finalPromptLength: <int>, embeddingStat: TuberiaTensorStat.sample(embedding), durationSeconds: <double>)`. `encoderName` derived from which encoder branch was taken (e.g. `"klein"`, `"dev"`, `"mistral"`).

**Exit criteria**:
- [ ] `grep -c "capture(.textEncodeComplete" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 1 (more if multiple encoder branches each emit).
- [ ] Each emit site samples `TuberiaTensorStat.sample(...)` exactly once on the embedding tensor.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B7: Wire `schedulerConfigured` event

**Priority**: 17.5 — single emit site; required for B12 boundary-events test.

**Entry criteria**:
- [ ] B3 complete.

**Tasks**:
1. Locate the call site `scheduler.setTimesteps(...)` in `Flux2Pipeline.swift` via grep.
2. Immediately after the call returns, emit `.schedulerConfigured(numInferenceSteps: <int>, shift: <float>, imageSeqLen: <int>, mu: <float>)`. The `mu` value is computed by `FlowMatchEulerScheduler.computeEmpiricalMu(imageSeqLen:numSteps:)` — read whatever the scheduler stores it as after `setTimesteps`.

**Exit criteria**:
- [ ] `grep -c "capture(.schedulerConfigured" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1.
- [ ] All four args populated with non-placeholder values (no `0` literals unless that's the actual runtime value).
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B8: Wire `denoiseLoopStart` / `denoiseLoopEnd` at all four variant sites

**Priority**: 22.0 — highest-priority emission sortie; 4 emit pairs in 1774-line Pipeline.swift; gates B10 (anomaly) and B13 (anomaly test).

**Entry criteria**:
- [ ] B3 complete.

**Tasks**:
1. Locate the four denoise variant sites via `grep -n "for stepIdx in\|forwardKVExtract" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`. There are three `for stepIdx in ...` loops (T2I, I2I KV-cached, I2I full-recompute) plus one non-loop `forwardKVExtract` call (I2I KV-extract step 0). Total: 4 emit pair sites.
2. At each site:
   - Before entering the loop / call: capture the loop start time; emit `.denoiseLoopStart(variant: .<variant>, totalSteps: <int>, latentShape: <[Int]>, latentDtype: <string>)`. For the KV-extract one-shot, `totalSteps: 1`.
   - After the loop exits (success or cancellation `break`) / after the KV-extract call returns: sample `finalLatentStat` and emit `.denoiseLoopEnd(variant:, totalSteps:, completedSteps: <int>, finalLatentStat:, durationSeconds:)`.
3. Inside each loop body, fetch `currentTelemetry()` exactly once at the top of the body and cache it in a local — but per iteration-03, **do not emit per-step events**. The cached lookup is for the Loop Start/End emission only, not for per-step stat sampling.

**Exit criteria**:
- [ ] `grep -c "capture(.denoiseLoopStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 4.
- [ ] `grep -c "capture(.denoiseLoopEnd" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 4.
- [ ] All four `DenoiseVariant` enum cases referenced at least once across the emit sites: `textToImage`, `imageToImageKVExtractStep0`, `imageToImageKVCached`, `imageToImageFullRecompute`.
- [ ] No `denoiseStepComplete` emit anywhere (per iteration-03, that event does not exist in this iteration).
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B9: Wire `vaeDecodeComplete` event

**Priority**: 20.5 — single emit site but gates B10 (anomaly uses pixel stat) and B12.

**Entry criteria**:
- [ ] B3 complete.

**Tasks**:
1. Locate the VAE decode completion site via `grep -n "postprocessVAEOutput\|vae\.decode" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`.
2. After the decode returns (success path), sample the pixel tensor and emit `.vaeDecodeComplete(pixelStat: TuberiaTensorStat.sample(pixels), outputDims: <[Int]>, durationSeconds: <double>)`.

**Exit criteria**:
- [ ] `grep -c "capture(.vaeDecodeComplete" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1.
- [ ] Emit site is in the success path (not in an error-handling branch).
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B10: Add anomaly-check helper and wire `numericalAnomaly` side-channel

**Priority**: 19.0 — coordinates across the three stat-carrying emit sites; touches new helper file plus three edits in Pipeline.swift.

**Entry criteria**:
- [ ] B6, B8, B9 complete (the three boundary emit sites that carry `TuberiaTensorStat` exist).

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Sources/PixArtBackbone/PixArtDiT.swift` around lines 271+ for the reference helper that maps a `TuberiaTensorStat` to optional `AnomalyKind`.
2. Add an internal helper in `Sources/Flux2Core/Telemetry/` (new file `AnomalyCheck.swift` or as a private function on `Flux2TelemetryEvent`) that returns `AnomalyKind?` given a `TuberiaTensorStat`. Logic:
   - `nan` if `stat.hasNaN`
   - `inf` if `stat.hasInf`
   - `outOfRange` if `abs(stat.max) > TuberiaTensorStat.defaultOutOfRangeThreshold` (verify the actual symbol name on the SwiftTuberia type before referencing)
   - `zeroLatent` if `abs(stat.mean) < 1e-6 && stat.std < 1e-6`
   - `nil` otherwise.
3. At each of the three boundary emit sites (textEncodeComplete in B6, denoiseLoopEnd in B8, vaeDecodeComplete in B9), after sampling the stat and before/after the primary emit, call the helper and — if it returns non-nil — emit `.numericalAnomaly(phase: <.textEncode/.denoiseLoopEnd/.vaeDecode>, kind: <kind>, stat: <same stat>)` alongside the primary event.

**Exit criteria**:
- [ ] Anomaly-check helper exists: either `Sources/Flux2Core/Telemetry/AnomalyCheck.swift` exists AND `grep "func .*AnomalyKind?" Sources/Flux2Core/Telemetry/AnomalyCheck.swift` returns at least 1; or a private function in `Flux2TelemetryEvent.swift` with the same signature shape (`grep "private (static )?func .*AnomalyKind?" Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns at least 1). Choose one location and document the choice in a 1-line code comment.
- [ ] `grep -rc "capture(.numericalAnomaly" Sources/Flux2Core/` (summed) returns exactly 3 (one alongside each of the three boundary stat-carrying events).
- [ ] All three `AnomalyPhase` enum cases referenced: `textEncode`, `denoiseLoopEnd`, `vaeDecode`.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B11: Wire `errorThrown` and `generationCancelled` side-channels

**Priority**: 19.25 — tied with B5 for largest emit-site count (20 throws across 4 files); cancellation portion is contingent on grep result (currently zero hits — likely a no-op for cancellation, see Open Questions).

**Entry criteria**:
- [ ] B3 complete.

**Tasks**:
1. Locate every `throw Flux2Error.…` site via `grep -rn "throw Flux2Error\." Sources/Flux2Core/`. As of 2026-05-13 the count is **20 sites across 4 files** (Pipeline.swift: 14, MistralEncoder.swift: 3, KleinTextEncoder.swift: 2, DevTextEncoder.swift: 1). Re-run the grep to confirm the current count before editing — REQUIREMENTS-instrumentation.md only enumerated 14 Pipeline.swift sites; the other 11 are equally important (P3).
2. Immediately before each throw, emit `.errorThrown(phase: <ErrorPhase>, errorDescription: <string>)`. Map each `Flux2Error.<case>` to the matching `ErrorPhase.<case>`. The exhaustive mapping is in the enum definitions in `Flux2TelemetryEvent.swift`. For the throws in `MistralEncoder.swift` / `KleinTextEncoder.swift` / `DevTextEncoder.swift` (encoder load/forward errors), the `ErrorPhase` likely maps to `.textEncoderLoad` or similar — verify the enum case from B2 covers it; if not, file an open question rather than inventing a case.
3. **Cancellation contingency.** Run `grep -rn "Task.isCancelled\|CancellationError\|checkCancellation\|cancellationCheck" Sources/Flux2Core/`. As of 2026-05-13 this returns **zero hits** — no cancellation-check sites currently exist in the codebase. Therefore:
   - **Do NOT add new cancellation-check call sites in this sortie.** That is out of scope for telemetry wiring.
   - If the grep still returns zero, emit no `.generationCancelled` events; note in a single-line code comment near the denoise loops (`// generationCancelled emission deferred: no cancellation-check sites in pipeline as of <date>`).
   - If the grep now returns non-zero (cancellation was added in parallel work), wire `.generationCancelled(stepIndex: <Int?>)` at each site: `nil` for pre-loop sites, `stepIdx` for in-loop sites.

**Exit criteria**:
- [ ] `grep -rc "throw Flux2Error\." Sources/Flux2Core/` (summed) equals `grep -rc "capture(.errorThrown" Sources/Flux2Core/` (summed) — every throw is preceded by an emit. As of 2026-05-13, both should report 20.
- [ ] Cancellation: either (a) `grep -rc "capture(.generationCancelled" Sources/Flux2Core/` matches the cancellation-site count found in task 3, with at least one `nil` and one `stepIdx` value; OR (b) the cancellation grep returns zero and a deferral comment is present near the denoise loops.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds.

---

### Sortie B12: Add `MockReporter` test helper and `Flux2TelemetryBoundaryEventsTests`

**Priority**: 17.75 — establishes the test harness (MockReporter) reused by B13–B15; gates the four downstream test sorties.

**Entry criteria**:
- [ ] B4–B11 complete (full emission surface is wired).

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Tests/PixArtBackboneTests/MockReporter.swift` as the canonical pattern (37 lines; thread-safe array of captured events behind a lock).
2. Add `Tests/TestHelpers/MockFlux2TelemetryReporter.swift` (or analogous file under `TestHelpers` target) implementing `Flux2TelemetryReporter` with a `var captured: [Flux2TelemetryEvent]` and a `withLock`-guarded append. Follow the exact threading discipline of pixart's MockReporter.
3. Create `Tests/Flux2CoreTests/Flux2TelemetryBoundaryEventsTests.swift` using Swift Testing (`@Test`, `#expect`). Include `import TestHelpers`. Run a single mocked T2I generation (use whatever fixtures `Flux2CoreTests` already has for config-only testing — no GPU, no model weights) and assert the event sequence:
   1. `pipelineInit` (1×)
   2. 3+ × `weightLoadComplete` (one per loaded component)
   3. `textEncodeComplete` (1×)
   4. `schedulerConfigured` (1×)
   5. `denoiseLoopStart` (1×)
   6. `denoiseLoopEnd` (1×)
   7. `vaeDecodeComplete` (1×)
4. Verify the API surface of any unfamiliar MLX type before use (P5): if a test needs `MLXArray.zeros(_:type:)`, grep mlx-swift's source to confirm the static-method shape.

**Exit criteria**:
- [ ] `Tests/TestHelpers/MockFlux2TelemetryReporter.swift` exists; `grep -E "(class|struct|actor) MockFlux2TelemetryReporter" Tests/TestHelpers/MockFlux2TelemetryReporter.swift` returns 1.
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryBoundaryEventsTests.swift` exists; declares `import TestHelpers`; `grep -c "@Test" Tests/Flux2CoreTests/Flux2TelemetryBoundaryEventsTests.swift` returns at least 1.
- [ ] `make test-core` succeeds.

---

### Sortie B13: Add `Flux2TelemetryAnomalyTests`

**Priority**: 2.5 — leaf test sortie; can run in parallel with B14, B15 (sub-agent eligible — supervising agent handles build).

**Entry criteria**:
- [ ] B12 complete (MockReporter exists).

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Tests/PixArtBackboneTests/PixArtTelemetryAnomalyTests.swift` as the pattern.
2. Create `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift`. Inject a fake/mock noise predictor (or fixture latent) that returns a tensor containing one NaN. Assert that:
   - The `denoiseLoopEnd` event's `finalLatentStat.hasNaN == true`.
   - A `numericalAnomaly(phase: .denoiseLoopEnd, kind: .nan, ...)` fires alongside the `denoiseLoopEnd` event.
   - The mock reporter contains both events in the expected order.

**Exit criteria**:
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift` exists.
- [ ] `make test-core` succeeds.

---

### Sortie B14: Add `Flux2TelemetryErrorPathTests`

**Priority**: 2.5 — leaf test sortie; can run in parallel with B13, B15 (sub-agent eligible).

**Entry criteria**:
- [ ] B12 complete.

**Tasks**:
1. Create `Tests/Flux2CoreTests/Flux2TelemetryErrorPathTests.swift`. Force a `Flux2Error.invalidConfiguration` throw via configuration that fails validation (use whatever path is cheapest — likely a malformed config struct that triggers a pre-flight check).
2. Assert: `errorThrown(phase: .invalidConfiguration, errorDescription: ...)` fires immediately before the throw is caught by the test harness.

**Exit criteria**:
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryErrorPathTests.swift` exists.
- [ ] `make test-core` succeeds.

---

### Sortie B15: Add `Flux2TelemetryLockContentionTests`

**Priority**: 3.75 — leaf test sortie but elevated risk due to XCTest concurrency / Swift 6 strict mode (F10, F11); can run in parallel with B13, B14 (sub-agent eligible).

**Entry criteria**:
- [ ] B12 complete.

**Tasks**:
1. Read `/Users/stovak/Projects/pixart-swift-mlx/Tests/PixArtBackboneTests/PixArtTelemetryLockContentionTests.swift` (172 lines) as the canonical pattern.
2. Create `Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` using **plain XCTest** (not swift-testing) per F11 / iteration-02 platform-issue carry-over: spawn N concurrent `setTelemetry` toggles plus a running denoise/forward harness; assert no data races and emissions reflect the most-recently-set reporter.
3. Per F10 (Swift 6 strict mode): all private helpers must be non-static OR every call site must use `Self.` qualifier.

**Exit criteria**:
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` exists.
- [ ] File uses XCTest (`grep -c "import XCTest" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 1; `grep -c "import Testing" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 0).
- [ ] F10 enforcement: `grep -E "private static func" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 0 OR every call to such a helper is prefixed with `Self.` (manual verification — file is small enough to eyeball).
- [ ] `make test-core` succeeds.

---

### Sortie B16: Add `Flux2TelemetryNoopOverheadTests` and final cross-library audit

**Priority**: 4.0 — final closer; must run after both A1 and B15.

**Entry criteria**:
- [ ] B12–B15 complete (all wiring + supporting tests in place).
- [ ] A1 complete (pixart REQUIREMENTS doc finalised — the cross-library audit references it).

**Tasks**:
1. Create `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift`. Run two T2I-shaped harness loops over the same fixture: one with `setTelemetry(nil)`, one with `setTelemetry(NoopFlux2TelemetryReporter())`. Take wall-clock medians over 20 iterations of each. Assert the medians are within ±2% of each other.
2. Cross-library audit: produce a short markdown report at `TELEMETRY_AUDIT.md` (root of flux repo) confirming:
   - Both libraries expose `setTelemetry((any <Lib>TelemetryReporter)?)` with the same signature shape.
   - Both libraries' `WeightComponent` / `AnomalyKind` / `ErrorPhase` enum case names use the conventions in AGENTS.md §11.2 (verified by grep against both repos).
   - Both libraries' adapter sink phase strings (when implemented in the Vinetas adapter) would follow `<lib>_<noun>_<lifecycle>` snake_case.
   - List any naming drift found between the two libraries (event case names, enum case names, adapter sink string predictions) — each entry is a follow-up sortie for a later iteration, not a blocker.

**Exit criteria**:
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift` exists; passes with ±2% bound over 20 iters.
- [ ] `TELEMETRY_AUDIT.md` exists at repo root and explicitly compares the two libraries.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds (full suite).

---

## Open Questions & Missing Documentation

The refinement pass surfaced four items that the sortie agent must investigate (not block on). All are tagged with the sortie that owns the resolution so the agent has explicit authority to make the call. Two of them have downstream impact on B16's cross-library audit.

### Unresolved Items

| ID | Sortie | Issue Type | Description | Recommended Resolution |
|----|--------|-----------|-------------|------------------------|
| Q1 | B11 | Open question (auto-handled) | `grep -rn "Task.isCancelled\|CancellationError\|checkCancellation\|cancellationCheck" Sources/Flux2Core/` returns **zero hits** as of 2026-05-13. No existing cancellation-check sites means `.generationCancelled` has no emission targets. | Auto-handled by B11 task 3 contingency: if grep still returns zero, emit no events and add a deferral comment near the denoise loops. Cancellation infrastructure itself is **out of scope** for this iteration — it's a separate work unit. Log finding in B16's `TELEMETRY_AUDIT.md`. |
| Q2 | B3, B5 | Missing decision | `TrainingTextEncoder.swift` is a top-level class in `Sources/Flux2Core/Loading/` but is **not** in B3's "exactly 7 files" list. B2's `WeightComponent` enum (per REQUIREMENTS) includes `textEncoderTraining` — so either B3 must add the 8th setter or the case is dead until training-time instrumentation arrives. | B3 agent decides: (a) include `TrainingTextEncoder` (8 files; update exit criterion from "exactly 7" to "exactly 8"), OR (b) document in a code comment on the `WeightComponent.textEncoderTraining` case that the setter is deferred and the case is unreferenced until a later iteration. Either is acceptable; record the decision in the commit message and in B16's audit. |
| Q3 | B4 | Vague criterion | "Populate `model` from whichever enum/string identifies the model variant" — the canonical model identifier on `Flux2Pipeline` is not specified. | B4 agent inspects `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` for a stored model-variant property (enum or string). If none exists, use the quantization recipe identifier as a proxy and document the choice in a 1-line comment at the emit site. Acceptable values are non-empty, non-placeholder strings. |
| Q4 | B16 | Flakiness risk (carry-over from F11) | "Wall-clock medians over 20 iterations within ±2%" is tight for CI macos-26 runners. Previous iteration-02 carried over a recurring lock-contention flakiness on the same hardware class. | B16 agent: keep ±2% as the *target*, but if observed variance over 20 iters exceeds ±5% in interactive testing, widen the bound to ±5% and add a `// CI-tuned bound` comment. Hard-fail at ±10%. Record actual observed median ratio in `TELEMETRY_AUDIT.md`. |

**None of these are execution-blocking** — each has a defined resolution path that the sortie agent can apply unilaterally. They are flagged so the agent does not get stuck mid-sortie wondering whether to escalate.

### Auto-Fixes Applied During Pass 4

| Sortie | Original | Fix |
|--------|----------|-----|
| A1 | "word count < 50% of current file" (units mixed: KB then word count) | Byte count `wc -c < 11000 bytes` (current is 21,678 bytes). |
| B5 | Grep pattern `loadTextEncoder\|loadTransformer\|loadVAE\|loadLoRA` — none of these names exist in the codebase | Replaced with actual symbol names from `grep -n "func load" Sources/Flux2Core/Loading/ Sources/Flux2Core/LoRA/`. |
| B10 | "Helper exists and returns `AnomalyKind?`" — location unspecified | Two acceptable locations (`AnomalyCheck.swift` OR private fn in `Flux2TelemetryEvent.swift`), each with a concrete grep verification. |
| B11 | "Audited count is ~14 sites in Flux2Pipeline.swift" — undercounts 6 throws in encoder files | Updated to "20 sites across 4 files" (Pipeline 14, Mistral 3, Klein 2, Dev 1); exit criterion compares grep counts so robust to drift. |
| B11 | Cancellation grep pattern with no contingency for zero hits | Added explicit zero-hit handling: emit no events, add deferral comment, log in audit. |
| B12 | "the new test is counted in the suite total" — not directly verifiable | Replaced with `grep -c "@Test" ... returns at least 1` plus `make test-core` succeeds. |
| B15 | F10 rule stated as guidance but not in exit criteria | Added explicit grep verification: `grep -E "private static func" ... returns 0` OR `Self.`-prefixed call site check. |

---

## Parallelism Structure

**Critical Path** (longest dependency chain): `B1 → B2 → B3 → B4 → B6 → B7 → B8 → B9 → B10 → B11 → B12 → B16` (12 sorties). B5 forks off after B3 and rejoins before B12; A1 is in a different repo and runs concurrently with the entire B stream, joining only at B16.

**Parallel Execution Groups**:

- **Group 0 — Layer 0 (different repo, sub-agent eligible)**:
  - A1 (sub-agent): pixart REQUIREMENTS rewrite. Runs concurrently with all of Group 1–5; no contention with the flux build queue. Must complete before B16 (Group 6).
- **Group 1 — Foundation (sequential, supervising agent only)**:
  - B1 → B2 → B3. Each has `make build` + `make test` exit criteria; serialized on the flux build queue.
- **Group 2 — First parallel emission window (after B3)**:
  - B4 (supervising agent — `Flux2Pipeline.swift`).
  - B5 (**sub-agent** — `Loading/` + `LoRA/`; touches disjoint files from B4). Sub-agent writes code only; supervising agent runs `make build && make test` after both edits are present.
  - **Note**: B5 could in principle parallel-run alongside the entire B6–B11 stream, but committing B5 mid-stream forces a context switch on the supervising agent (pause B6/B7/…, build with B5, commit, return). Cleaner to bundle B5 with B4 in a single parallel-write window.
- **Group 3 — Pipeline.swift emission stream (sequential, supervising agent only)**:
  - B6 → B7 → B8 → B9 → B10 → B11. All edit `Flux2Pipeline.swift`; file-level contention forces strict serial order. B10 has explicit entry deps on B6, B8, B9 — ordering preserved.
- **Group 4 — Test infrastructure (supervising agent only)**:
  - B12. Sole sortie in this group; defines the `MockReporter` pattern reused by B13–B15.
- **Group 5 — Parallel test writes (after B12)**:
  - B13 (**sub-agent** — `Flux2TelemetryAnomalyTests.swift`).
  - B14 (**sub-agent** — `Flux2TelemetryErrorPathTests.swift`).
  - B15 (**sub-agent** — `Flux2TelemetryLockContentionTests.swift`, XCTest).
  - All three touch disjoint test files. Sub-agents write code only; supervising agent runs `make test-core` **after each commit individually** to preserve per-sortie attribution if a build/test breaks (do not run a single combined verification — that would obscure which test sortie regressed).
- **Group 6 — Closer (supervising agent only)**:
  - B16. Depends on A1 (Group 0) AND B15 (Group 5). Sole sortie.

**Parallelism Metrics**:
- Maximum concurrent agents at any time: **2 sub-agents + 1 supervising agent = 3** (well under the 4-sub-agent cap).
- Sub-agent eligible sorties: 4 (A1, B5, B13, B14, B15 — actually 5).
- Supervising-agent-only sorties: 12 (B1, B2, B3, B4, B6, B7, B8, B9, B10, B11, B12, B16).
- Critical path length: 12 sorties.

**Build Constraint (enforced)**:
- Every sortie's exit criteria include `make build` and `make test` (or `make test-core`). Per Pass 3 rules, **only the supervising agent runs builds.** Sub-agents (A1, B5, B13, B14, B15) write code and verify file-level grep criteria; the supervising agent owns the build/test verification step before committing the sortie.
- A1 is the lone exception that can build truly concurrently with B-stream work, because its build runs in a different repo (`pixart-swift-mlx`) with no shared build artifacts or queue.

**Missed Opportunities** (intentionally not parallelized):
- B6 + B9 could parallel-write into Pipeline.swift in principle (different call sites), but Pipeline.swift is 1774+ lines and concurrent edits invite merge headaches for negligible savings. Kept sequential.
- B11's per-file work (the 6 non-Pipeline encoder throws) could split out as a sub-agent task, but the cost of coordinating that against the 14 Pipeline.swift throws + cancellation contingency exceeds the savings. Kept as a single supervising-agent sortie.

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 2 |
| Total sorties | 17 (A1, B1–B16) |
| Dependency structure | A1 standalone in pixart repo (Layer 0); B1→B2→B3 foundation in flux; B4 || B5; then B6→B7→B8→B9→B10→B11 sequential on Pipeline.swift; B12 test infra; B13 || B14 || B15 parallel test writes; B16 final closer (gates on A1 + B15) |
| Critical path | 12 sorties: B1→B2→B3→B4→B6→B7→B8→B9→B10→B11→B12→B16 |
| Parallelism | Up to 3 concurrent agents (1 supervising + 2 sub-agents); 5 sub-agent-eligible sorties (A1, B5, B13, B14, B15) |
| Cross-library invariant | All new event names follow [AGENTS.md §11](AGENTS.md#11-telemetry-chokepoint-convention-cross-library); pixart is the canonical pattern source |

**Next step:** `/mission-supervisor start` (after reviewing the Open Questions section below)
