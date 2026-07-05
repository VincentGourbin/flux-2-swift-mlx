---
type: execution-plan
feature_name: OPERATION THIMBLE TYPHOON
starting_point_commit: 9d94e557dcbc211ef3daa77557b693df4284f611
mission_branch: mission/thimble-typhoon/01
iteration: 1
state: completed
mission: thimble-typhoon-01
updated: 2026-07-05
---

# EXECUTION_PLAN.md — FLUX.2 on iPad

Derived from `requirements/REQUIREMENTS-flux-on-ipad.md`. Ships `flux-2-swift-mlx`
on iPad-class Apple Silicon: a 16 GB iPad Pro path (qint8 Klein 4B, ship-now) and
an 8 GB iPad path (int4 Klein 4B, net-new). Klein 9B is excluded on every tier by
product decision (§0 of the requirements).

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Standing constraint (from requirements §0)

**Klein 9B — all variants (`klein9B`, `klein9BBase`, `klein9BKV`) — is excluded on
every platform and every tier, including 16 GB.** Its refusal is a product
decision, NOT memory-gated. Any sortie that could route to Klein 9B is a defect.
Supported image model for this mission is **Klein 4B only** (Dev/32B is separately
out on memory grounds).

---

## Grounding notes (verified against source)

The requirements doc's file:line hints were checked against the tree. Corrections
folded into the sorties below:

- **VAE tiling already exists but is dead code.** `AutoencoderKLFlux2` in
  `Sources/Flux2Core/VAE/AutoencoderKL.swift` has `VATilingConfig` (lines 12-38),
  `decodeWithTiling(_:tiling:)` (160-172) and `decodeTiled` (177-254) — but the
  pipeline calls the **non-tiled** `vae!.decode(...)` at `Flux2Pipeline.swift`
  1425, 1568, 1623, 1824, 1898. R4 is therefore *wiring*, not implementation
  (Sortie A5).
- **`checkImageSize` error branch is unreachable.** In `MemoryManager.swift`
  183-200 the `>2048²` warn returns before the `>4096²` error branch. Must be
  reordered when adding tier-aware caps (Sortie A4).
- **`Flux2Error` has no model-refusal case.** `Sources/Flux2Core/Flux2Core.swift`
  14-42 has no gating/refusal case; models are never refused in code. R5 must add
  a typed case (Sortie A2).
- **Default steps = 50, not the model recommendation.** `Flux2Pipeline.swift:586`
  defaults `steps = 50`; `Flux2Model.recommendedSteps` (Config:138) returns 4 for
  distilled Klein. iPad tier must use the model recommendation (Sortie A3).
- **Only two pre-quantized repos exist today** — Dev qint8 and
  `aydin99/FLUX.2-klein-4B-int8`. Every other qint8/int4 request falls back to
  bf16 + on-the-fly `quantize()` at `Flux2Pipeline.swift:401`. That fallback is
  the B1 int4 OOM (requirements §3, B1).

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| WU-A — Phase 1 (16 GB iPad, qint8, ship-now) | `.` (Sources/Flux2Core, Sources/FluxTextEncoders, Tests) | 8 | 0 | none |
| WU-B — Phase 2 (8 GB iPad, int4) | `.` (Sources/Flux2Core, Tests) | 5 | 3 | WU-A (needs iPad tier + Phase 1 on-device telemetry) |

**Layer map** (dependency gating; parallelism grouping is refine's Pass 4 job):

| Layer | Sorties |
|-------|---------|
| 0 | A1 (iPad tier foundation), A5 (wire VAE tiling), A6 (phys_footprint telemetry), A7 (CDN verify) |
| 1 | A2 (model gating), A3 (16 GB defaults), A4 (tier-aware resolution cap + bug fix) |
| 2 | A8 (16 GB device smoke test) |
| 3 | B1 (int4 weights on CDN), B3 (working-set recalibration) |
| 4 | B2 (direct int4 load path) |
| 5 | B4 (8 GB defaults) |
| 6 | B5 (8 GB device smoke test / out-of-scope decision) |

---

## Priority & Execution Order (refine Pass 3)

Scores from `priority = depth*3 + foundation*2 + risk*1 + complexity*0.5`. Sortie
IDs are left unchanged — renumbering would ripple cross-references (B4 extends A3/A4;
B1→B2 etc.) for what are near-tie intra-layer swaps. The "intra-layer order" column
is the recommended dispatch order *within* each layer; layer boundaries are never
crossed.

| Sortie | Priority | Layer | Intra-layer order | Justification |
|--------|----------|-------|-------------------|---------------|
| A1 | 23.0 | 0 | 1st | Foundational tier; transitively blocks A2/A3/A4/A8/B4/B5 |
| A6 | 12.75 | 0 | 2nd | Telemetry seam consumed by A8, B2 (R2.3), B3 recalibration |
| A5 | 11.0 | 0 | 3rd | VAE tiling reused on both iPad tiers; blocks A8 |
| A7 | 6.5 | 0 | 4th | CDN verify; external risk but no downstream code dep beyond A8 |
| A2 | 11.0 | 1 | 1st | Typed refusal error reused across tiers; blocks A8 |
| A4 | 10.75 | 1 | 2nd | Resolution cap + unreachable-branch fix; extended by B4 |
| A3 | 10.0 | 1 | 3rd | 16 GB default-knob plumbing; extended by B4 |
| A8 | — | 2 | — | Sole layer-2 sortie (integration gate) |
| B1 | 13.75 | 3 | 1st | int4 variant registration; blocks B2/B4/B5 |
| B3 | 10.75 | 3 | 2nd | Working-set recalibration; blocks B4/B5 |
| B2 | — | 4 | — | Sole layer-4 sortie |
| B4 | — | 5 | — | Sole layer-5 sortie |
| B5 | — | 6 | — | Sole layer-6 sortie |

Reorders vs the original layer listing: **A6 before A5** (layer 0) and **A4 before
A3** (layer 1). Both are intra-layer priority swaps; no layer boundary or dependency
is violated.

---

## Parallelism Structure (refine Pass 4)

**Critical path**: A1 → A4 → A8 → B1 → B2 → B4 → B5 (length: **7 sorties**).

**The build gate dominates.** MLX is arm64-only and nearly every sortie's exit
criteria include a `make build` / `xcodebuild` test gate. Per the parallelism rule
**only the supervising agent builds**, so the 11 build-gated sorties cannot be
farmed to sub-agents for their build/test steps. This repo is build-serialized by
nature; parallelism here is genuinely shallow — stated plainly rather than inflated.

**Parallel execution groups**:
- **Group 1 — Layer 0**: supervising agent builds **A1 → A6 → A5** in priority
  order; **one sub-agent runs A7 concurrently**.
  - A1, A6, A5 — **SUPERVISING AGENT ONLY** (build/test gates)
  - A7 — **NO BUILD (sub-agent eligible)**: `Acervo.availability` checks +
    `CDN_PROVISIONING.md` artifact only
- **Group 2 — Layer 1** (sequential after A1; supervising agent, order A2 → A4 → A3):
  all build-gated
- **Group 3 — Layer 2**: A8 (integration smoke test; supervising agent)
- **Groups 4–7 — WU-B layers 3–6**: B1 → B3 → B2 → B4 → B5. B1's `acervo ship` is
  external ops but its `ModelRegistry` edit builds (supervising). The doc/checklist
  artifacts (B5 jetsam checklist, out-of-scope decision file) are **sub-agent
  eligible**.

**Agent allocation**: 1 supervising agent + up to 1 genuinely useful sub-agent (A7,
plus documentation artifacts). The 4-sub-agent ceiling is **not reachable** here —
the arm64 build gate serializes the code sorties.

**Missed opportunities**: none material. The serialization is inherent to the build
gate, not an artifact of plan structure. A7 and the documentation artifacts are the
only truly parallelizable work.

---

## Work Unit A — Phase 1 (16 GB iPad, qint8, ship-now)

Delivers a working iPad build against **existing code and existing CDN weights** —
no new model weights, no int4 work. Covers R1.2–1.4, R3.1/R3.2, R4, R5, R6, and the
16 GB column of the §5 default table.

### Sortie A1: iPad memory tier foundation

**Layer**: 0

**Entry criteria**:
- [ ] First sortie of WU-A — no prerequisites.

**Tasks**:
1. Introduce an explicit iPad memory tier covering **8 GB** and **12–16 GB**
   unified-memory devices (current thresholds bottom out at 16 GB and treat
   `<32` as one bucket). Add it to `MemoryConfig` (`Sources/Flux2Core/Configuration/MemoryConfig.swift`).
2. Extend `MemoryOptimizationConfig.recommended(forRAMGB:)`
   (`MemoryOptimizationConfig.swift:143-156`, currently `0..<32 → .ultraLowMemory`)
   to route the iPad tier explicitly to `.ultraLowMemory` (eval every 2 blocks +
   clear cache).
3. Extend `ModelRegistry.recommendedConfig(forRAMGB:)`
   (`ModelRegistry.swift:513-526`, currently `0..<32 → .ultraMinimal`) to select
   the iPad tier explicitly.
4. On the iPad tier, force `MemoryConfig.CacheProfile = .conservative`
   (`MemoryConfig.swift:26-41`; `cacheLimitForProfile` conservative = 512 MB, 125-140).
5. Add unit tests asserting tier selection + forced configs at 8, 12, and 16 GB.

**Exit criteria**:
- [ ] `make build` (or `xcodebuild` per CLAUDE.md §1) succeeds.
- [ ] New tests assert `forRAMGB: 8/12/16` resolve to the iPad tier with
      `MemoryOptimizationConfig == .ultraLowMemory` and `CacheProfile == .conservative`.
- [ ] `forRAMGB: 32+` still resolves to the existing Mac tiers (no regression).

### Sortie A2: Model gating with typed error (R5.1, R5.2)

**Layer**: 1

**Entry criteria**:
- [ ] A1 complete — iPad tier exists (`MemoryConfig` exposes tier selection).

**Tasks**:
1. Add a typed refusal case to `Flux2Error` (`Sources/Flux2Core/Flux2Core.swift:14-42`),
   e.g. `case modelNotSupportedOnTier(model:tier:reason:)`.
2. Refuse **all Klein 9B variants** (`klein9B`, `klein9BBase`, `klein9BKV`)
   **unconditionally** — on every tier including 16 GB, NOT memory-gated (§0).
3. On the iPad tier, refuse **Dev (32B)** with the typed error (not an OOM).
4. On the iPad tier, force `model = .klein4B`.
5. Emit `errorThrown` immediately before every new `throw` (CLAUDE.md §5a: one
   `errorThrown` per throw site).
6. Tests: Klein 9B (all three variants) refused on 16 GB **and** 8 GB tiers; Dev
   refused on iPad tier; Klein 4B accepted; each throw emits `errorThrown`.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Test asserts all three Klein 9B variants throw `modelNotSupportedOnTier` on
      the 16 GB tier (proving refusal is not memory-gated).
- [ ] Test asserts Dev throws the typed error on the iPad tier; Klein 4B does not.
- [ ] Grep confirms an `errorThrown` emit precedes each new `throw` site.

### Sortie A3: iPad default knobs — 16 GB column (§5)

**Layer**: 1

**Entry criteria**:
- [ ] A1 complete — iPad tier exists and is queryable during pipeline config.

**Tasks** (apply the §5 "iPad 16 GB" column when the iPad tier is active):
1. Default resolution → **768×768** (`Flux2Pipeline.swift:584-585` and the sibling
   overload defaults at 726-727).
2. Default steps → model `recommendedSteps` (Klein = **4**), replacing the hard
   `steps = 50` at `Flux2Pipeline.swift:586`.
3. Guidance → Klein **1.0** (confirm `Flux2Config.swift:147` already yields 1.0 for
   Klein; wire it through the iPad default path).
4. `memoryProfile` → `.conservative` (`Flux2Pipeline.swift:137`, default `.auto`).
5. `clearCacheEveryNSteps` → **3** (`Flux2Pipeline.swift:140`, default 5).
6. Max reference images → **2** on the iPad 16 GB tier (`Flux2Config.swift:168`,
   Klein = 4).
7. Transformer quant → **qint8** on the iPad tier (`Flux2Pipeline.swift:155`,
   default `.balanced`).
8. Tests asserting each knob resolves to the §5 16 GB value on the iPad tier.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Tests assert, on the iPad-16GB tier: resolution 768², steps 4, guidance 1.0,
      `memoryProfile == .conservative`, `clearCacheEveryNSteps == 3`,
      maxReferenceImages 2, transformer quant qint8.
- [ ] Mac tier defaults are unchanged (regression assertion).

### Sortie A4: Tier-aware resolution cap + fix unreachable error branch (R3, §5 hard max)

**Layer**: 1

**Entry criteria**:
- [ ] A1 complete — iPad tier is available to `checkImageSize`.

**Tasks**:
1. Fix the ordering bug in `checkImageSize` (`MemoryManager.swift:183-200`) so the
   `>4096²` error branch is reachable (currently the `>2048²` warn returns first).
2. Make the hard-max resolution **tier-aware**: iPad 16 GB → error above **1024²**
   (per §5); keep the existing Mac behaviour on Mac tiers. (8 GB → 768² is added in
   B4.)
3. Tests: iPad-16GB tier throws above 1024²; the previously-unreachable error path
   now fires; Mac tier retains existing threshold.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Test asserts a request above 1024² throws on the iPad 16 GB tier.
- [ ] Test asserts the reordered error branch is reachable (a size that should
      error now errors instead of only warning).

### Sortie A5: Wire existing VAE tiling into the pipeline (R4.1, R4.2)

**Layer**: 0

**Entry criteria**:
- [ ] First-layer sortie — no prerequisites (tiling code already exists;
      independent of the tier work).

**Tasks**:
1. Route the pipeline's VAE decode through `AutoencoderKLFlux2.decodeWithTiling(_:tiling:)`
   at the five call sites in `Flux2Pipeline.swift` (1425, 1568, 1623, 1824, 1898),
   replacing the non-tiled `vae!.decode(...)`.
2. Select `VAETilingConfig` on the iPad tier (`.default`/`.aggressive`) and
   single-shot (`.disabled`) on Mac (R4.2). *(Tier read requires A1. Per the
   Parallelism Structure above, A5 is sequenced after A1 within layer 0, so the
   tier is available at decode time — no interim passed-in flag is needed.)*
3. Tests: iPad tier selects the tiled decode path; Mac selects single-shot; a
   decode at 768² returns a non-nil image via the tiled path.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Test asserts the tiled path is invoked on the iPad tier and single-shot on Mac.
- [ ] Test asserts a tiled decode produces a non-nil image (`Flux2GPUTests`).

### Sortie A6: Per-phase phys_footprint telemetry (R6.1)

**Layer**: 0

**Entry criteria**:
- [ ] First-layer sortie — no prerequisites (extends the existing telemetry seam).

**Tasks**:
1. Capture `phys_footprint` (via `task_vm_info`; wire a `FluxProfiler`-style helper
   if one is not already present) and attach it to the four phase-boundary events
   on the `Flux2TelemetryEvent` seam (`Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift`):
   `weightLoadComplete`, `textEncodeComplete`, `denoiseLoopEnd`, `vaeDecodeComplete`.
2. Emit at the existing sites only (WeightLoader:744; Pipeline textEncodeComplete
   1102, denoiseLoopEnd 1312/1441/1585/1845, vaeDecodeComplete 1631/1906) — **no
   new per-step / per-block events** (CLAUDE.md §5a: boundaries, not internals).
3. Tests asserting each of the four boundary events carries a non-nil
   `phys_footprint` value.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Test asserts `weightLoadComplete`, `textEncodeComplete`, `denoiseLoopEnd`,
      and `vaeDecodeComplete` each carry a non-nil `phys_footprint`.
- [ ] No new per-step telemetry event types were added (grep/inspection).

### Sortie A7: Verify CDN provisioning of the Phase-1 weights (R1.2, R1.3, R1.4)

**Layer**: 0

**Entry criteria**:
- [ ] First-layer sortie — no prerequisites.

**Tasks**:
1. Confirm **`aydin99/FLUX.2-klein-4B-int8`** (qint8, ~4 GB) is provisioned on the
   SwiftAcervo CDN via `Acervo.availability(_:)` (SwiftAcervo 0.16 API; CLAUDE.md §5).
2. Confirm **`lmstudio-community/Qwen3-4B-MLX-4bit`** (~2 GB, iPad text encoder) is
   provisioned.
3. Confirm the **VAE** (~3 GB, from `black-forest-labs/FLUX.2-klein-4B` `vae/`) is
   provisioned.
4. If any is missing, ship it with `acervo ship` (external, network action — see
   OQ-1 note; do **not** re-add the retired `ensure-model-cdn.yml` path).
5. Record the availability results in a short artifact (`CDN_PROVISIONING.md`).

**Exit criteria**:
- [ ] All three repos report `.available` (or `.partial` with a documented plan)
      via `Acervo.availability(_:)`.
- [ ] `CDN_PROVISIONING.md` records the state of each of the three components.

### Sortie A8: 16 GB device-matrix smoke test (R6.2, Acceptance)

**Layer**: 2

**Entry criteria**:
- [ ] A1, A2, A3, A4, A5, A6 complete (tier, gating, 16 GB defaults, resolution
      cap, tiled decode, and telemetry all in place).
- [ ] A7 complete (weights available).

**Tasks**:
1. Add a smoke test: Klein 4B **qint8** generation at the iPad-16GB defaults (768²,
   4 steps, guidance 1.0) asserting a non-nil image.
2. Gate on model presence (`XCTSkipUnless` / `@Test(.enabled(if:))`) so it runs
   headless on macOS CI against cached Acervo models (acervo-integration-ci pattern).
3. Document the "no jetsam on real 16 GB iPad hardware" half as a manual on-device
   checklist item (CI cannot assert iPad jetsam — see OQ-3).

**Exit criteria**:
- [ ] `Flux2GPUTests` smoke test produces a non-nil 768² image at the iPad-16GB
      defaults, model-presence-gated.
- [ ] Manual on-device jetsam checklist recorded (target: no jetsam on 16 GB iPad Pro).

---

## Work Unit B — Phase 2 (8 GB iPad, int4)

**Stretch target.** Net-new int4 CDN weights + direct int4 load + working-set
recalibration from Phase 1 telemetry + the §5 8 GB defaults. **WU-B dispatches
unconditionally as soon as its layer dependencies are met** (OQ-2 resolved by user
override — no Phase-1 go/no-go gate). B3/B5 still consume Phase-1 telemetry for the
working-set recalibration, but that is a *data* dependency, not a dispatch gate.

### Sortie B1: Pre-quantized int4 Klein 4B on the CDN (R1.1)

**Layer**: 3

**Entry criteria**:
- [ ] WU-A complete (dependencies met — WU-B dispatches unconditionally, no
      go/no-go gate per OQ-2).
- [ ] int4 weights source resolved: `themindstudio/flux2-klein-4b-mlx-4bit` (OQ-1).

**Tasks**:
1. Use the pre-quantized int4 Klein 4B transformer
   **`themindstudio/flux2-klein-4b-mlx-4bit`** (OQ-1 resolved: verified genuine
   MLX 4-bit — `.scales`/`.biases` per layer, `quantization_level: "4"`,
   `mflux_version 0.15.2`; Apache 2.0; `transformer/` subfolder = 2.18 GB). No
   self-quantization required.
2. Ship it to the SwiftAcervo CDN via
   `acervo ship themindstudio/flux2-klein-4b-mlx-4bit` (shipped during refine —
   confirm `.available`).
3. Register it in `ModelRegistry` as a new `TransformerVariant.klein4B_4bit`
   (`repoId: "themindstudio/flux2-klein-4b-mlx-4bit"`,
   `repoSubfolder: "transformer"`, `estimatedSizeGB: 2.18`;
   `ModelRegistry.swift:12-90`).

**Exit criteria**:
- [ ] `ModelRegistry` exposes a `klein4B_4bit` variant with
      `repoId == "themindstudio/flux2-klein-4b-mlx-4bit"`,
      `repoSubfolder == "transformer"`, and `estimatedSizeGB == 2.18`.
- [ ] `themindstudio/flux2-klein-4b-mlx-4bit` reports `.available` via
      `Acervo.availability(_:)`.

### Sortie B2: Direct pre-quantized int4 load path (R2.1, R2.2, R2.3)

**Layer**: 4

**Entry criteria**:
- [ ] B1 complete — the int4 variant exists and is available on the CDN.
- [ ] A6 complete — phys_footprint telemetry available for the R2.3 assertion.

**Tasks**:
1. Extend `WeightLoader.loadQuantizedTransformer` (`WeightLoader.swift:721`) and the
   qint8→float16 dequant pass (376-399) to load pre-quantized **int4** safetensors
   directly into `QuantizedLinear` (mmap, no bf16 intermediate).
2. `ModelRegistry.variant(for:quantization:)` (`ModelRegistry.swift:214-232`) must
   return `klein4B_4bit` for `(klein4B, int4)` instead of `.klein4B_bf16` +
   on-the-fly `quantize()`.
3. Verify no bf16 materialization occurs on the int4 path (no call into the
   `Flux2Pipeline.swift:395-414` on-the-fly quantize block for `(klein4B, int4)`).
4. R2.3 acceptance: assert peak `phys_footprint` during `(klein4B, int4)`
   transformer load stays under steady-state + working pad (no ≥8 GB spike), using
   the A6 telemetry.
5. Tests covering variant resolution, the direct-load path, and the no-spike assertion.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Test asserts `(klein4B, int4)` resolves to `klein4B_4bit` and loads via
      `QuantizedLinear` without entering the on-the-fly quantize block.
- [ ] Test/measurement asserts no ≥8 GB `phys_footprint` spike during int4 load.

### Sortie B3: Working-set recalibration + 8 GB feature flag (R3.3)

**Layer**: 3

**Entry criteria**:
- [ ] A6 and A8 complete — Phase-1 on-device `phys_footprint` measurements exist.

**Tasks**:
1. Replace the magic `+8` in `estimatedTotalMemoryGB`
   (`QuantizationConfig.swift:82-85`, "VAE 3 + working 5") with a named, tier-aware
   working-set constant derived from measured on-device `phys_footprint` for
   Klein 4B (OQ-4).
2. Add an `enable8GBTier` feature flag defaulting **OFF** until the measurement
   confirms the 8 GB floor is recoverable (requirements §3 B2, §8).
3. Tests asserting the recalibrated estimate for Klein 4B and that the 8 GB tier is
   unreachable while the flag is OFF.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] The working-set constant is a named value sourced from measured data, not the
      Mac-shaped `+8` guess.
- [ ] Test asserts the 8 GB tier is gated OFF by default and enabled only via the flag.

### Sortie B4: iPad default knobs — 8 GB column (§5)

**Layer**: 5

**Entry criteria**:
- [ ] B1, B2 complete (int4 weights + direct load available).
- [ ] B3 complete (8 GB tier + feature flag exist).
- [ ] A3 complete (16 GB default-knob plumbing to extend).

**Tasks** (apply the §5 "iPad 8 GB" column when the 8 GB tier is active):
1. Default resolution → **512×512**.
2. Hard max resolution → error above **768²** (extend A4's tier-aware cap).
3. `clearCacheEveryNSteps` → **2**.
4. Transformer quant → **int4** (pre-quantized, via B1/B2).
5. Max reference images → **1**.
6. Confirm steps (Klein 4), guidance (1.0), `memoryProfile` (`.conservative`), and
   `MemoryOptimizationConfig` (`.ultraLowMemory`) all resolve for the 8 GB tier.
7. Tests asserting each knob resolves to the §5 8 GB value on the 8 GB tier.

**Exit criteria**:
- [ ] Build succeeds.
- [ ] Tests assert, on the 8 GB tier: resolution 512², hard-max 768², clearCache 2,
      transformer quant int4, maxReferenceImages 1.

### Sortie B5: 8 GB device smoke test / out-of-scope decision (R6.2, Acceptance)

**Layer**: 6

**Entry criteria**:
- [ ] B1, B2, B3, B4 complete.
- [ ] A8 complete (16 GB smoke test pattern established).

**Tasks**:
1. Add a smoke test: Klein 4B **int4** generation at 512² on the 8 GB tier,
   asserting a non-nil image, model-presence-gated for macOS CI.
2. Document the "no jetsam on real 8 GB iPad hardware" half as a manual on-device
   checklist item (OQ-3).
3. **OR**, if Phase-1 telemetry / B3 measurement shows the honest floor stays near
   ~10 GB even after tiling, record a documented decision that the 8 GB tier is out
   of scope (requirements §8, Acceptance) and mark WU-B closed accordingly.

**Exit criteria**:
- [ ] Either: the int4 512² smoke test produces a non-nil image (model-gated) **and**
      the on-device jetsam checklist is recorded; **or**: a documented "8 GB out of
      scope" decision file exists with the measured floor that justifies it.

---

## Resolved Decisions (refine Pass 1)

All five blocking open questions from `breakdown` were resolved during refinement.
Recorded here for the audit trail; none remain open.

| # | Decision | Affects | Source |
|---|----------|---------|--------|
| OQ-1 | int4 weights = **`themindstudio/flux2-klein-4b-mlx-4bit`** — verified genuine MLX 4-bit (`.scales`/`.biases` per layer, `quantization_level: "4"`, `mflux_version 0.15.2`), Apache 2.0, `transformer/` subfolder = 2.18 GB. **No self-quantization.** Shipped to the SwiftAcervo CDN via `acervo ship` during refine. | B1 → B2, B4, B5 | user (research-driven) |
| OQ-2 | **WU-B dispatches unconditionally** as soon as its layer dependencies are met — **no** Phase-1 go/no-go gate. B3/B5 still consume Phase-1 `phys_footprint` telemetry as a *data* dependency, not a dispatch gate. | all of WU-B | **user override** |
| OQ-3 | Device-matrix smoke tests are **split**: the CI-runnable "non-nil image at iPad defaults" half runs headless on macOS CI against cached Acervo models (acervo-integration-ci pattern); the "no jetsam on device" half is a manual on-device checklist item tracked in the mission brief. | A8, B5 | recommendation |
| OQ-4 | B3 ships the current `+8` as a **named, documented** working-set constant plus an `enable8GBTier` flag defaulting **OFF**; the real value is set from A6/A8 telemetry (16 GB hardware, extrapolated via the §2 max-phase model) in a follow-up once the measurement lands. | B3, B5 | recommendation |
| OQ-5 | §7 VinetasIOS app-level items (UI resolution cap, device gating, app-group env) are **out of scope** for this library repo. This mission delivers the tier, typed refusal errors, and tier-aware caps as **library API**; VinetasIOS consumes them in a separate app-side mission. | scope boundary | recommendation |

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 2 |
| Total sorties | 13 |
| Open questions | 0 (all resolved in refine Pass 1) |
| Dependency structure | 7 layers (WU-A layers 0–2, WU-B layers 3–6; WU-B depends on WU-A completion, dispatched unconditionally) |
