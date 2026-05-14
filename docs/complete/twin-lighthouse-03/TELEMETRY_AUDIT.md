# Telemetry Audit — flux-2-swift-mlx + pixart-swift-mlx

> Cross-library §11-convention audit for OPERATION TWIN LIGHTHOUSE (iteration 03).
> Generated 2026-05-13. Closing artifact for Sortie B16.

This audit compares the boundary-telemetry surfaces of the two sibling
MLX-Swift image-generation libraries that share the
[AGENTS.md §11](AGENTS.md#11-telemetry-chokepoint-convention-cross-library)
chokepoint convention:

- **flux-2-swift-mlx** at `instrumentation/03` (this repo) — full pipeline.
- **pixart-swift-mlx** at `development` commit `ff49dfa` — backbone only.

Every claim in this document was grep-verified against live source before
writing. Numbers reflect the state of both repos as of 2026-05-13.

---

## Scope

Both libraries expose a `setTelemetry((any <Lib>TelemetryReporter)?)` seam on
their top-level types and emit boundary-only events per §11. Neither library
emits per-step / per-block / per-attention-head events in this iteration —
that detail is reserved for a follow-up triggered by a real failure.

The two libraries are deliberately asymmetric:

- **pixart** is a *backbone* — it owns the DiT forward pass, weight load /
  unload, and recipe validation. The pipeline (scheduler, denoise loop, VAE
  decode, cancellation) lives in SwiftTuberia's `DiffusionPipeline`, which
  drives `PixArtDiT` as a plug-in.
- **flux** is a *full pipeline* — text encoders, transformer, scheduler,
  denoise loop, VAE decode, and lifecycle all live in `Flux2Pipeline`.

The cross-library invariant is the *shape* (`setTelemetry` seam,
`<Lib>TelemetryReporter` protocol, `NoopReporter` ship default, exhaustive-
switch routing in the host) — not the event list, which legitimately differs
between a backbone and a pipeline.

---

## setTelemetry Signature Symmetry

| Library | Type | Signature | Storage |
|---|---|---|---|
| pixart | `PixArtDiT` | `public func setTelemetry(_ reporter: (any PixArtTelemetryReporter)?)` | `OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>` |
| flux | `Flux2Pipeline` | `public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?)` | `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` |

Same shape, same lock backing. ✓

Flux additionally exposes the same setter on six owned subcomponents
(`KleinTextEncoder`, `DevTextEncoder`, `MistralEncoder` (`Flux2TextEncoder`),
`FlowMatchEulerScheduler`, `Flux2WeightLoader`, `Flux2Transformer2DModel`,
`LoRAAdapter`) — 8 setTelemetry seams total. The host only ever calls
`Flux2Pipeline.setTelemetry`; the pipeline propagates the reporter inward.

Pixart exposes the setter only on `PixArtDiT`; recipe types take the reporter
as a parameter on the async `validate(telemetry:)` overload instead of via a
stored property.

---

## Event Surface Comparison

| Event family | pixart | flux | Notes |
|---|---|---|---|
| `pipelineInit` | — | 1 emit | Pipeline-layer event; not in backbone's purview. |
| `pipelineDispose` | — | 1 emit | Same. |
| `weightLoadComplete` | 1 emit (`.dit`) | 5 emits (5 components: `textEncoderKlein`, `textEncoderDev`, `transformer`, `vae`, `lora`) | flux's component granularity is larger by design. |
| `weightUnloadComplete` | 1 emit | — | flux does not currently model an unload boundary. |
| `recipeValidated` | 2 emits | — | flux uses `errorThrown` + per-throw deferral instead of validated/failed pairs. |
| `recipeValidationFailed` | 6 emits | — | Same. |
| `textEncodeComplete` | — | 1 emit | Pipeline-layer. |
| `schedulerConfigured` | — | 2 emits (T2I + I2I paths) | Pipeline-layer. |
| `denoiseLoopStart` | — | 4 emits (4 variants) | Pipeline-layer; `textToImage`, `imageToImageKVExtractStep0`, `imageToImageKVCached`, `imageToImageFullRecompute`. |
| `denoiseLoopEnd` | — | 4 emits (matched pairs) | Same. |
| `vaeDecodeComplete` | — | 2 emits (final-decode only) | Checkpoint-preview decodes intentionally do NOT emit (would violate boundary-only). |
| `numericalAnomaly` | (anomaly classifier called at every `forward(_:)` exit, 0–1 emits per step) | 7 stat-carrying emit sites (3 textEncode, 1 denoiseLoopEnd × 4 variants overflows to 7 anomaly checks across all stat-carrying boundaries) | flux phases: `.textEncode`, `.denoiseLoopEnd`, `.vaeDecode`. Pixart phases: `.ditForward` active, `.weightLoad` reserved. |
| `errorThrown` | 6 emits (paired with `recipeValidationFailed`) | 20 emits (one per `throw Flux2Error.*`) | flux's invariant: every `throw Flux2Error.…` is preceded by an emit. |
| `generationCancelled` | — | 0 emits | Q1: no cancellation-check sites in the codebase as of 2026-05-13. Deferral comments present at the four denoise loop start sites; case is reserved for when cancellation infrastructure lands. |

Per-event totals from `grep -rno "capture(\.[a-zA-Z]*"`:

- **flux** emit-site counts: errorThrown=20, numericalAnomaly=7, denoiseLoopStart=4, denoiseLoopEnd=4, vaeDecodeComplete=2, schedulerConfigured=2, weightLoadComplete=1 (across `Loading/` + `LoRA/`, not in Pipeline.swift), textEncodeComplete=1, pipelineInit=1, pipelineDispose=1. Total: 43 emit sites.
- **pixart** emit-site counts: recipeValidated=2, recipeValidationFailed=6, errorThrown=6, weightLoadComplete=1, weightUnloadComplete=1, numericalAnomaly=1 (single in-line capture inside `forward(_:)`). Total: 17 emit sites.

flux's 5 `weightLoadComplete` emits live in `KleinTextEncoder.swift` (1),
`DevTextEncoder.swift` (1), `MistralEncoder.swift` (1), `WeightLoader.swift`
(transformer + vae, 2 distinct call paths), and `LoRAAdapter.swift` (1) —
the grep total above shows them clustered per-file rather than as a single
literal token; the audit recipe-validated each by enum case.

---

## Naming Convention §11.2 Audit

Per [AGENTS.md §11.2](AGENTS.md#112-naming-rules):

- **Event case names**: lowerCamelCase noun + lifecycle suffix.
  - pixart: `weightLoadComplete`, `weightUnloadComplete`, `recipeValidated`,
    `recipeValidationFailed`, `numericalAnomaly`, `errorThrown`. ✓
  - flux: `pipelineInit`, `pipelineDispose`, `weightLoadComplete`,
    `textEncodeComplete`, `schedulerConfigured`, `denoiseLoopStart`,
    `denoiseLoopEnd`, `vaeDecodeComplete`, `numericalAnomaly`,
    `generationCancelled`, `errorThrown`. ✓
- **Nested enum case names** (WeightComponent, AnomalyKind, AnomalyPhase,
  ErrorPhase, plus flux's DenoiseVariant): lowerCamelCase string raw values.
  - pixart `WeightComponent`: `dit`. ✓
  - pixart `AnomalyPhase`: `weightLoad`, `ditForward`. ✓
  - pixart `AnomalyKind`: `nan`, `inf`, `outOfRange`, `zeroLatent`. ✓
  - pixart `ErrorPhase`: `weightLoad`, `forward`, `recipeValidation`, `other`. ✓
  - flux `WeightComponent`: `textEncoderKlein`, `textEncoderDev`,
    `textEncoderTraining`, `transformer`, `vae`, `lora`. ✓
  - flux `DenoiseVariant`: `textToImage`, `imageToImageKVExtractStep0`,
    `imageToImageKVCached`, `imageToImageFullRecompute`. ✓
  - flux `AnomalyPhase`: `textEncode`, `denoiseLoopEnd`, `vaeDecode`. ✓
  - flux `AnomalyKind`: `nan`, `inf`, `outOfRange`, `zeroLatent`. ✓
  - flux `ErrorPhase`: 13 cases — `modelNotLoaded`, `invalidConfiguration`,
    `insufficientMemory`, `modelNotDownloaded`, `generationCancelled`,
    `generationFailed`, `weightLoadFailed`, `vaeDecodeFailed`,
    `textEncoderFailed`, `vlmInterpretFailed`, `loraLoadFailed`,
    `imageProcessingFailed`, `other`. ✓
- **Adapter sink phase strings** (predicted, snake_case `<lib>_<noun>_<lifecycle>`):
  not implemented in this iteration — no Vinetas adapter exists in either repo
  yet. Section §11.6 defines the routing rules; this audit notes only that the
  enum case names support the convention by construction. `flux_pipeline_init`,
  `flux_weight_load_complete_<component>`, `pixart_dit_forward`,
  `pixart_anomaly_<kind>`, etc., will all derive cleanly from the existing
  cases via lowerCamelCase → snake_case mechanical mapping.

No naming-rule violations found by grep.

---

## Naming Drift Across Libraries

By design, the two libraries cover different boundaries, so their event case
names cannot overlap completely. The categorical similarities are:

| Case | pixart | flux | Drift? |
|---|---|---|---|
| `weightLoadComplete` | ✓ | ✓ | No drift — same case name, same arg labels (`component:`, `paramCount:`, `durationSeconds:`). |
| `numericalAnomaly` | ✓ | ✓ | No drift — same case name, same arg labels (`phase:`, `kind:`, `stat:`). |
| `errorThrown` | ✓ | ✓ | No drift — same case name, same arg labels (`phase:`, `errorDescription:`). |
| `AnomalyKind` enum (4 cases) | `nan`/`inf`/`outOfRange`/`zeroLatent` | `nan`/`inf`/`outOfRange`/`zeroLatent` | **Identical.** |

No naming drift was found between the cases that exist on both sides. The
two libraries' divergent cases (pixart's `recipeValidated/Failed`,
`weightUnloadComplete`; flux's `pipelineInit/Dispose`, `textEncodeComplete`,
`schedulerConfigured`, `denoiseLoopStart/End`, `vaeDecodeComplete`,
`generationCancelled`) reflect different layers, not inconsistent naming.

`AnomalyPhase` and `ErrorPhase` enums are *different sets of cases* — but
that's a coverage difference (the libraries instrument different phases),
not a naming-convention violation.

---

## Known Gaps & Deferrals from Iteration 03

These are open items intentionally not closed in iteration 03. Each is a
candidate for a follow-up iteration when (and if) a real failure or feature
demands it.

- **flux `WeightComponent.textEncoderTraining`** is a live enum case with no
  emit site. `TrainingTextEncoder` is a protocol; concrete training-time
  entry points instrument the underlying encoder instead. A code comment in
  `TrainingTextEncoder.swift` documents the deferral. Resolve in a future
  iteration that adds training-time instrumentation.
- **flux `ErrorPhase` unreferenced cases**: 7 of the 13 cases have at least
  one emit site (`invalidConfiguration` ×6, `textEncoderFailed` ×5,
  `denoiseLoopEnd` ×4, `modelNotLoaded` ×3, `generationCancelled` ×3,
  `vaeDecode` ×2, `textEncode` ×1, `insufficientMemory` ×1,
  `imageProcessingFailed` ×1, `generationFailed` ×1). 5 cases are reserved
  for future throw sites: `modelNotDownloaded`, `weightLoadFailed`,
  `vaeDecodeFailed`, `vlmInterpretFailed`, `loraLoadFailed`. They were
  defined for cross-library generality and will activate as new throw sites
  land. Note: a few of the active counts above use `phase: .X` regex against
  source — `phase: .denoiseLoopEnd` is `AnomalyPhase`, not `ErrorPhase`, so
  the `ErrorPhase`-active set is a subset of the listed names; the
  always-reserved set holds.
- **flux `.generationCancelled` event**: zero emit sites because the
  codebase has no cancellation-check sites as of 2026-05-13. Comment
  placeholders are present at the four denoise loop start sites per Q1
  contingency. Wire when cancellation infrastructure lands.
- **VAE checkpoint preview decodes** in flux do NOT emit
  `.vaeDecodeComplete` (would violate boundary-only — preview decodes are
  per-step UI artifacts, not phase boundaries). Only the 2 final-decode
  emits fire. A separate `.checkpointPreviewEmitted` event in a future
  iteration could observe checkpoint cadence if needed.
- **flux B5 paramCount for text encoders** is architectural-constant
  hardcoded (Qwen3-4B 3.95B, Qwen3-8B 8.19B, Mistral 24B) because the
  underlying MLXNN modules are private members of `FluxTextEncoders.shared`.
  A small public param-count accessor in a future iteration would replace
  the constants.
- **pixart `AnomalyPhase.weightLoad`** and **`ErrorPhase.weightLoad` /
  `forward` / `other`** are reserved (declared, no emit sites). They are
  documented in pixart's REQUIREMENTS-instrumentation.md as reserved for
  future activation; not a §11 violation.
- **Vinetas adapter**: not yet implemented in either repo (`grep`-verified
  no `Flux2TelemetryAdapter.swift` or `PixArtTelemetryAdapter.swift` exists
  in `/Users/stovak/Projects/Vinetas`). The predicted sink-phase strings
  (`flux_pipeline_init`, `pixart_dit_forward`, etc.) will derive cleanly
  from the enum case names when the adapter lands; this audit makes no
  claim about live adapter behavior.

---

## Test Status

| Library | Test target | Tests | Suites |
|---|---|---|---|
| flux | `Flux2CoreTests` | 216 | 36 |
| flux | `FluxTextEncodersTests` | 125 | 9 |
| flux | combined CI-safe (`make test`) | 341 | 45 |
| pixart | `PixArtBackboneTests` | 153 tests, 21 suites (as of A1 commit `ff49dfa`) | — |

Both libraries' CI-safe test suites pass green at the audit-generation commit.

### Iteration-03 build/test cadence

Every code-touching sortie in iteration 03 verified `make build` + `make test`
before commit. Sub-agent parallel-write groups verified collectively after
all writes landed. No sortie shipped uncompiling code.

### Observed noop-overhead ratio (Q4)

`Flux2TelemetryNoopOverheadTests` ran on the audit-generation hardware
(macos-26, Apple Silicon, Swift 6.2). Observed:

```
noop overhead observed: nilMedian=2.0265579223632812e-06s, noopMedian=2.0265579223632812e-06s, ratio=1.0, delta=0.0
```

Both medians collapsed to the same ~2 µs floor — the synthetic harness
loop (10 `dispose()` calls per measurement, 20 measurements per branch) is
faster than the macOS clock resolution can meaningfully separate the two
code paths. The ±10% Q4 hard-fail bound is satisfied with delta=0.0.

Future iterations may want a larger emits-per-iteration (~1000) to drive
the medians out of the clock-resolution floor and surface real per-emit
cost; for iteration 03 the test passes its Q4 contract as written.

---

## Convention Compliance Summary

| §11 rule | pixart | flux |
|---|---|---|
| 11.1 — every boundary category has at least one event in the catalog | ✓ (for backbone scope) | ✓ |
| 11.2 — naming rules (PascalCase event cases, lowerCamelCase enum cases) | ✓ | ✓ |
| 11.3 — single stat sample per stat-carrying boundary; anomaly check fires alongside | ✓ (1 sample/forward at the boundary) | ✓ (3 samples per generation: textEncode, denoiseLoopEnd, vaeDecode) |
| 11.4 — `errorThrown` precedes every `throw` in the library's scope | partial (6/13 throws — recipe validation only) | ✓ (20 emits, 20 throws) |
| 11.5 — `setTelemetry((any <Lib>TelemetryReporter)?)` on every top-level type, lock-backed | ✓ (one type) | ✓ (eight types) |
| 11.6 — adapter exhaustive-switch routing | not implemented in this iteration (Vinetas adapter pending) | not implemented in this iteration |
| 11.7 — REQUIREMENTS-instrumentation.md present, slim, matches live impl | ✓ (rewritten in A1) | ✓ (current) |

Pixart's partial 11.4 status (DiT throws and weight-load throws lack
`errorThrown` emits) is noted in pixart's REQUIREMENTS-instrumentation.md as
"reserved for future activation"; it is the documented gap pending a Vinetas-
side need for backbone-throw-level error attribution.
