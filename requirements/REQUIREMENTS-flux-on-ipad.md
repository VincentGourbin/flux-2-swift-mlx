---
type: reference
updated: 2026-07-04
---

# REQUIREMENTS — FLUX.2 on iPad

Scope: make `flux-2-swift-mlx` run FLUX.2 image generation on iPad-class Apple
Silicon (8 GB and 16 GB), and define the iPad-specific default changes required
in both the library and the consuming app (`VinetasIOS`).

This document is grounded in the current source. File:line references are to
`flux-2-swift-mlx` unless noted. Numbers are the package author's own figures
(README quant tables, `ModelRegistry.estimatedSizeGB`) plus the in-code memory
estimator.

---

## 0. Standing constraint — Klein 9B is excluded

> **Klein 9B (all variants: `klein9B`, `klein9BBase`, `klein9BKV`) is NOT an
> option and MUST NOT be considered, targeted, or offered on any platform — not
> as a fallback, not as a "16 GB unlocks more" tier, not anywhere.** This is a
> product decision independent of memory footprint and holds until the decision
> owner explicitly reverses it. Any requirement, default, or gate below that
> could route to Klein 9B is a defect. Supported image models for this effort
> are **Klein 4B only** (Dev/32B is separately out on memory grounds).

---

## 1. Verdict

| Target device | Config | Status |
|---|---|---|
| **16 GB iPad Pro (M-series)** | **qint8 Klein 4B**, pre-quantized, ≤768 px | **Shippable with existing code** (no bf16 load spike, no new CDN weights). This is the first target. |
| **8 GB iPad (Air / non-1TB Pro)** | int4 Klein 4B, pre-quantized, ≤512 px | **Requires net-new work** — pre-quantized int4 CDN weights, working-set recalibration, VAE tiling. Stretch target. |
| **Klein 9B (any variant), any platform** | — | **Excluded by product decision (§0).** Never target until reversed. |
| Dev (32B), any iPad | — | **Not viable on memory grounds.** Do not target. |

**Why the smaller device is hard is not the weights** — int4 Klein 4B is only
~2.1 GB (transformer) + ~2 GB (Qwen3-4B-4bit) + ~3 GB (VAE). It's (a) a
load-time bf16 spike unique to int4, (b) a 5 GB working-set assumption baked
into the estimator, and (c) an uncapped VAE decode at the 1024² default.

---

## 2. Architecture facts that make this feasible

These already exist — no change required, listed so the requirements below don't
re-derive them.

- **Peak memory = max phase, not sum.** The pipeline loads and explicitly frees
  each component in sequence: `unloadTextEncoder()` sets the encoder to `nil`
  before the transformer loads (`Flux2Pipeline.swift:286-296`, comment at :311
  "prevents the text encoder and transformer from overlapping in memory");
  `unloadTransformer()` frees the transformer (`:561-563`); VAE freed on cleanup
  (`:2324-2325`). The estimator encodes this directly:
  `estimatedTotalMemoryGB = max(textEncoder, transformer) + 8`
  (`QuantizationConfig.swift:82-85`).
- **Sequential-in-unified-RAM, no disk staging.** Weights are freed (`= nil`) and
  GPU cache cleared between phases; macOS/iPadOS reclaims. There is no
  disk-offload-between-stages (the only `offload` flag is training-only,
  `LoRATrainingConfig.swift:312`).
- **The encoder phase is already lean.** The Klein text encoder hard-caps GPU
  cache to 512 MB and forces `.aggressive` graph-eval
  (`FluxTextEncoders.swift:197-202`). The encoder is not the constraint.
- **A pre-quantized 8-bit Klein 4B transformer already ships.**
  `aydin99/FLUX.2-klein-4B-int8` (~4 GB) is in `ModelRegistry.swift`. qint8 loads
  directly with **no bf16 materialization**.
- **App entitlements are already in place.** `VinetasIOS.entitlements` declares
  `com.apple.developer.kernel.increased-memory-limit` and
  `extended-virtual-addressing`. The physical-RAM gate that blocked 8 GB iPads
  was already removed (Vinetas commit `3eed969`).

---

## 3. Blockers (what actually stands in the way)

### B1 — int4 materializes full bf16 weights at load (int4-only)
`ModelRegistry.swift:214-232` returns the **bf16** file for int4 and quantizes
in memory (`Flux2Pipeline.swift:395-414`, `quantize(...)` at :401). This
transiently holds the full ~8 GB bf16 Klein 4B (and briefly both bf16 + int4
during conversion) before shrinking to ~2.1 GB. **On an 8 GB iPad this OOMs
before generation starts.** qint8 is unaffected (pre-quantized on disk).

### B2 — 5 GB working-set assumption
`QuantizationConfig.swift:82-85`: the `+8` in the estimator is `VAE (3) +
working (5)`. By the package's own math, Klein 4B int4 ≈ `max(2, 2.1) + 8` ≈
**~10 GB** — which exceeds an 8 GB device. The 5 GB pad is a conservative
Mac-shaped guess, not a measured iPad `phys_footprint`. It must be recalibrated
against real on-device measurement.

### B3 — VAE decode is the largest single spike, and it's uncapped
Default resolution is **1024×1024** (`Flux2Pipeline.swift:584-585`) and
`checkImageSize` only *errors above 4096²* / warns above 2048²
(`MemoryManager.swift:183-200`). VAE decode activations at 1024² are the biggest
transient in the pipeline. There is no VAE tiling.

### B4 — memory-tier logic is Mac-shaped
Thresholds start at 16 GB and treat <32 GB as the smallest tier
(`MemoryOptimizationConfig.recommended(forRAMGB:)` → `<32 → ultraLowMemory`;
`ModelRegistry` auto-config `<32 → ultraMinimal`). There is **no iPad / mobile /
`lowMemory` code path** and no wired-residency tuning. This is net-new logic, not
a config flip.

---

## 4. Requirements

### R1 — Pre-quantized weights on the CDN (unblocks 8 GB)
- **R1.1** Provision a **pre-quantized int4 Klein 4B transformer** on the
  SwiftAcervo CDN (analogous to the existing `aydin99/FLUX.2-klein-4B-int8`), so
  int4 loads directly and **never materializes bf16**. Blocks B1.
- **R1.2** Confirm **`aydin99/FLUX.2-klein-4B-int8` (qint8)** is provisioned on
  the CDN today — this is the 16 GB ship-now path. If not provisioned, provision
  it. (v3.0.0+ throws `notProvisionedOnCDN` rather than falling back to HF.)
- **R1.3** Confirm **Qwen3-4B-4bit** (`lmstudio-community/Qwen3-4B-MLX-4bit`,
  ~2 GB) is provisioned. This is the iPad text encoder.
- **R1.4** VAE (~3 GB, from `black-forest-labs/FLUX.2-klein-4B` `vae/`) provisioned.

### R2 — Direct pre-quantized int4 load path
- **R2.1** Extend `WeightLoader.loadQuantizedTransformer`
  (`WeightLoader.swift:721`) — which already handles pre-quantized qint8
  safetensors — to load pre-quantized **int4** safetensors directly into
  `QuantizedLinear` (mmap, no bf16 intermediate).
- **R2.2** `ModelRegistry.TransformerVariant.variant(for:quantization:)`
  (`ModelRegistry.swift:214-232`) must return the pre-quantized int4 repo for
  `(klein4B, int4)` instead of the bf16 file + on-the-fly quantize.
- **R2.3** Acceptance: peak `phys_footprint` during transformer load for
  `(klein4B, int4)` must stay under the steady-state ceiling (no ≥8 GB spike),
  verified via `FluxProfiler` `task_vm_info`.

### R3 — iPad memory tier
- **R3.1** Add an explicit iPad tier to `MemoryConfig` /
  `MemoryOptimizationConfig` / `ModelRegistry` auto-selection covering **8 GB**
  and **12–16 GB** unified-memory devices (current thresholds bottom out at
  16 GB and treat <32 GB as one bucket).
- **R3.2** On the iPad tier, force `MemoryOptimizationConfig = .ultraLowMemory`
  (eval every 2 blocks + clear cache) and `CacheProfile = .conservative`.
- **R3.3** Recalibrate the working-set constant in `QuantizationConfig.swift:82-85`
  (currently `+8` = VAE 3 + working 5) against measured on-device
  `phys_footprint` for Klein 4B. Blocks B2. Until measured, gate 8 GB behind a
  feature flag.

### R4 — VAE tiling
- **R4.1** Implement tiled VAE decode (decode the latent in overlapping tiles,
  blend) so peak decode activation is bounded independent of output resolution.
  Blocks B3.
- **R4.2** Enable tiling automatically on the iPad tier; keep single-shot decode
  on Mac.

### R5 — Model gating
- **R5.1** On the iPad tier, force model = **Klein 4B**. Refuse Klein 9B (all
  variants, per §0) and Dev with a typed error — Klein 9B refusal is
  unconditional and not memory-gated (a 16 GB device must still refuse it), so it
  can never be reached even if a future tier has the RAM.
- **R5.2** 8 GB → int4 (post-R1.1/R2); 12–16 GB → qint8. Neither branch, nor any
  future higher-memory tier, may select Klein 9B.

### R6 — On-device validation & telemetry
- **R6.1** Emit `phys_footprint` at each phase boundary (`weightLoadComplete`,
  `textEncodeComplete`, `denoiseLoopEnd`, `vaeDecodeComplete`) via the existing
  `Flux2TelemetryEvent` seam so the working-set recalibration (R3.3) uses real
  numbers.
- **R6.2** Add a device-matrix smoke test: Klein 4B generation at the iPad
  defaults on 8 GB and 16 GB hardware, asserting no jetsam and a non-nil image.

---

## 5. Default changes for the iPad version

The single most important deliverable the caller asked for. "Current" = today's
Mac-shaped defaults; "iPad" = the value the iPad tier must apply.

| Knob | Location | Current (Mac) | iPad 16 GB | iPad 8 GB |
|---|---|---|---|---|
| Model | pipeline / auto | auto | Klein 4B (forced) | Klein 4B (forced) |
| Transformer quant | `Flux2Pipeline.swift:155` (`.balanced`) | qint8 | **qint8** (pre-quantized) | **int4** (pre-quantized, needs R1.1/R2) |
| Text encoder | pipeline | Mistral 8bit (Dev) | Qwen3-4B-4bit | Qwen3-4B-4bit |
| Default resolution | `Flux2Pipeline.swift:584-585` | 1024×1024 | 768×768 | **512×512** |
| Hard max resolution | `MemoryManager.swift:183-200` | error >4096² | error >1024² | error >768² |
| Default steps | `Flux2Pipeline.swift:586` (generate API = 50) | 50 | model `defaultSteps` (**Klein = 4**) | model `defaultSteps` (**Klein = 4**) |
| Guidance | `Flux2Config.swift:144-149` | Dev 4.0 | Klein **1.0** | Klein **1.0** |
| `memoryProfile` | `Flux2Pipeline.swift:137` (`.auto`) | auto | `.conservative` | `.conservative` |
| `MemoryOptimizationConfig` | auto by RAM | moderate/light | `.ultraLowMemory` | `.ultraLowMemory` |
| `clearCacheEveryNSteps` | `Flux2Pipeline.swift:140` | 5 | 3 | 2 |
| Working-set pad | `QuantizationConfig.swift:82-85` | +8 GB (VAE 3 + work 5) | recalibrated (R3.3) | recalibrated (R3.3) |
| VAE decode | pipeline | single-shot | tiled (R4) | tiled (R4) |
| Max reference images | `Flux2Config.swift:165-170` | Klein 4 | 2 | 1 |

---

## 6. Phasing

1. **Phase 1 — 16 GB iPad Pro, ship-now (qint8).**
   R1.2/R1.3/R1.4 (verify CDN) + R3.1/R3.2 (iPad tier) + R5 (gating) + §5
   defaults for the 16 GB column + R4 (VAE tiling) + R6. **No new model weights,
   no int4 work.** Delivers a working iPad build against existing code.
2. **Phase 2 — 8 GB iPad (int4).**
   R1.1 (pre-quantized int4 on CDN) + R2 (direct int4 load) + R3.3 (working-set
   recalibration from Phase 1 telemetry) + §5 defaults for the 8 GB column.

---

## 7. App-level (VinetasIOS) items

Outside the library, tracked here for completeness:

- Entitlements: **already done** (`increased-memory-limit`,
  `extended-virtual-addressing`); RAM gate already removed (commit `3eed969`).
- Surface the resolution cap in the UI so users can't request >max for their tier.
- Device gating: read unified memory, select the 8 GB vs 16 GB tier, and present
  a clear "not supported on this device" state for sub-8 GB devices. Klein 9B and
  Dev are never offered in the model picker at all (§0 / R5.1).
- `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the iOS launch
  environment (required by SwiftAcervo v0.10+; there is no silent fallback).

---

## 8. Open questions / risks

- **int4 quality on a 4B model.** int4 Klein 4B is the *only* config that fits
  8 GB — a real output-quality tradeoff. A/B int4 vs qint8 Klein 4B on 16 GB
  before locking the 8 GB tier. Dovetails with the freemium plan (PixArt free /
  FLUX.2 PRO).
- **Is 8 GB worth it at all?** If R3.3 measurement shows the honest floor stays
  near ~10 GB even after tiling, the 8 GB tier may not be recoverable and the
  product line should be 16 GB-iPad-Pro-and-up. Decide after Phase 1 telemetry.
- **iPadOS jetsam headroom is not the physical RAM.** Even with the entitlements,
  usable budget on an 8 GB device is a fraction of 8 GB and varies with
  foreground/background state. R6 must validate against real jetsam behavior, not
  the estimator.

---

## 9. Acceptance criteria

- [ ] Klein 4B qint8 generates a 768² image on a 16 GB iPad Pro with no jetsam
      (Phase 1).
- [ ] Klein 4B int4 loads with no bf16 spike (peak load `phys_footprint` <
      steady-state + working pad) (Phase 2).
- [ ] Klein 4B int4 generates a 512² image on an 8 GB iPad with no jetsam
      (Phase 2), OR a documented decision that 8 GB is out of scope.
- [ ] Klein 9B (all variants) is unreachable in the model picker and refused with
      a typed error on every tier including 16 GB (§0), not memory-gated.
- [ ] Dev is refused on iPad with a typed error, not an OOM.
- [ ] Per-phase `phys_footprint` telemetry is emitted and the working-set
      constant is set from measured data, not the Mac-shaped guess.
