---
type: doc
---

# On-Device Manual Checklist — FLUX.2 on iPad

**Mission**: OPERATION THIMBLE TYPHOON (`mission/thimble-typhoon/01`)

CI (`.github/workflows/integration-tests.yml`) exercises the **CI-runnable half**
of the device-matrix acceptance test: Klein 4B qint8 generation at the iPad-16GB
defaults produces a non-nil 768² image, headless on macOS arm64 against cached
SwiftAcervo models (`Tests/Flux2GPUTests/IPadDeviceMatrixGPUTests.swift`).

CI **cannot** assert the other half: that a real 16 GB iPad does not get its app
process **jetsammed** (killed by the OS memory pressure daemon) during
generation. Jetsam is an iOS/iPadOS runtime behavior with no headless-macOS
equivalent. Per **OQ-3** of `EXECUTION_PLAN.md`, that half is tracked here as a
**manual on-device checklist**, run by a human on physical hardware before ship.

---

## A8 — 16 GB iPad Pro jetsam checklist

**Target device**: iPad Pro with **16 GB** unified memory (M-series). The
consuming app (e.g. VinetasIOS) built against this library with the
`group.intrusive-memory.models` App Group configured so SwiftAcervo resolves the
shared models container.

**Config under test** (§5 "iPad 16 GB" column, resolved by the Sortie A3
`forRAMGB:` helpers): Klein 4B, **qint8** transformer, 768×768, **4** steps,
guidance **1.0**, `memoryProfile == .conservative`, `clearCacheEveryNSteps == 3`,
tiled VAE decode (Sortie A5), max 2 reference images.

**Pass target**: **no jetsam** — the app process survives the full
generate → VAE decode → image return cycle without an OS kill, on a device with
other typical foreground/background apps resident.

| # | Step | Expected | Result (✅ / ❌ + notes) |
|---|------|----------|--------------------------|
| 1 | Cold-launch the app; confirm it reports the **iPad tier** (≤16 GB) and forces Klein 4B (A1/A2). | iPad tier selected; Dev / Klein 9B refused with the typed error, not an OOM. | |
| 2 | Ensure the three Phase-1 weights are present in the App Group container (transformer qint8, VAE, Qwen3-4B encoder). | `Acervo.isModelAvailable` true for all three. | |
| 3 | Run **text→image** at the iPad-16GB defaults (768², 4 steps, guidance 1.0), seed fixed. | A non-nil 768² image is returned. | |
| 4 | Watch `phys_footprint` (A6 telemetry: `weightLoadComplete`, `textEncodeComplete`, `denoiseLoopEnd`, `vaeDecodeComplete`). Note the **peak**. | Peak stays within the 16 GB working set; no monotonic climb across steps. | |
| 5 | Confirm the app process was **not jetsammed** at any phase (check for `EXC_RESOURCE` / `JetsamEvent` in the device console / crash reports). | **No jetsam.** No `JetsamEvent-*.ips` generated for the app during the run. | |
| 6 | Repeat step 3 **5×** back-to-back (no relaunch) to surface a slow leak / fragmentation-driven kill. | All 5 runs complete; no jetsam; peak `phys_footprint` does not trend upward run-over-run. | |
| 7 | Run **image→image** with **2** reference images (the iPad-16GB `maxReferenceImages` cap). | Completes; no jetsam; reference count is honored (a 3rd image is rejected/capped). | |
| 8 | Background the app mid-generation, then foreground it. | No crash; generation resumes or fails gracefully (no jetsam-on-return). | |

### How to capture jetsam evidence

- **Xcode**: Devices & Simulators → select device → **View Device Logs**; filter
  for `JetsamEvent` / the app's process name. A jetsam produces a
  `JetsamEvent-<timestamp>.ips` report.
- **On-device**: Settings → Privacy & Security → Analytics & Improvements →
  Analytics Data → look for `JetsamEvent-*` entries around the test window.
- Correlate the A6 `phys_footprint` peak (from step 4) against the device's
  per-process jetsam limit to see the margin.

### Recording a run

Copy the table above into a dated run log (PR description, or an appended
section here) with the device model + iPadOS version, and mark each row. A run
is a **PASS** only if rows 5, 6, 7 all show **no jetsam**.

> If a 16 GB device jetsams even at these defaults, that is a **blocker** for the
> Phase-1 ship — feed the measured peak `phys_footprint` back into the
> working-set recalibration (Sortie B3, OQ-4) before adjusting defaults.

---

## B5 — 8 GB iPad jetsam checklist

**Honest status first** (requirements §8 / EXECUTION_PLAN.md OQ-4): the 8 GB
sub-tier (`MemoryConfig.MemoryTier.iPad8GB`) and its §5 "iPad 8 GB" knobs —
512² default resolution, 768² hard-max, int4 transformer (pre-quantized,
B1/B2), `clearCacheEveryNSteps == 2`, max 1 reference image (B3/B4) — are
**fully implemented**. `MemoryConfig.enable8GBTier` nonetheless **defaults
OFF**, because B3's conservative working-set estimate currently exceeds the
8 GB physical floor:

| Component | Estimated size |
|---|---|
| int4 transformer (themindstudio, `klein4B_4bit`) | ~2.18 GB |
| text encoder (Qwen3-4B, 8-bit preferred / 4-bit fallback) | ~2.28 GB |
| VAE | ~0.168 GB |
| Working-set pad (activations, KV, GPU cache headroom) | ~6 GB |
| **Conservative total** | **~10 GB** |

~10 GB > 8 GB physical RAM, so the flag stays OFF until a real device proves
the estimate is pessimistic. **This is not a claim that 8 GB is out of
scope** — the 8 GB path is *built and gated*, not rejected (EXECUTION_PLAN.md
Sortie B5 explicitly frames this as "gated OFF pending measured on-device
confirmation," not a permanent decision). The CI-runnable smoke test
(`Tests/Flux2GPUTests/IPad8GBDeviceMatrixGPUTests.swift`) exercises the
`.iPad8GB` config via the `enable8GBTier:` **parameter** on the `forRAMGB:`
helpers, without flipping the shared global flag — so the CI-covered half of
the acceptance criterion (non-nil 512² image, int4, model-gated) is provable
today. This section is the OTHER half: the manual, on-device measurement that
would let a follow-up sortie flip the flag ON.

**Target device**: iPad with **8 GB** unified memory (M-series or A-series
with 8 GB). The consuming app (e.g. VinetasIOS) built against this library
with the `group.intrusive-memory.models` App Group configured, and with
`MemoryConfig.enable8GBTier` flipped ON locally for the duration of this test
(it stays OFF in the shipped default).

**Config under test** (§5 "iPad 8 GB" column, resolved by the Sortie B4
`forRAMGB:` helpers with `enable8GBTier: true`): Klein 4B, **int4**
transformer (pre-quantized, `themindstudio/flux2-klein-4b-mlx-4bit`), 512×512,
**4** steps, guidance **1.0**, `memoryProfile == .conservative`,
`clearCacheEveryNSteps == 2`, tiled VAE decode (Sortie A5), max 1 reference
image.

**Pass target**: **no jetsam** — the app process survives the full
generate → VAE decode → image return cycle without an OS kill, on a device
with other typical foreground/background apps resident. This IS the
measurement that resolves OQ-4: if the device clears this checklist with
margin, feed the observed peak `phys_footprint` back into B3's working-set
constant and flip `enable8GBTier` ON in a follow-up; if it jetsams, the ~10 GB
conservative estimate above was directionally correct and the flag should
stay OFF (or the working-set pad needs to shrink further before retrying).

| # | Step | Expected | Result (✅ / ❌ + notes) |
|---|------|----------|--------------------------|
| 1 | Cold-launch the app with `enable8GBTier` forced ON; confirm it reports the **iPad 8GB sub-tier** and forces Klein 4B int4 (A1/A2/B3/B4). | `.iPad8GB` sub-tier selected; Dev / Klein 9B refused with the typed error, not an OOM. | |
| 2 | Ensure the three 8 GB-tier weights are present in the App Group container (int4 transformer, VAE, Qwen3-4B encoder). | `Acervo.isModelAvailable` true for all three. | |
| 3 | Run **text→image** at the iPad-8GB defaults (512², 4 steps, guidance 1.0), seed fixed. | A non-nil 512² image is returned. | |
| 4 | Watch `phys_footprint` (A6 telemetry: `weightLoadComplete`, `textEncodeComplete`, `denoiseLoopEnd`, `vaeDecodeComplete`). Note the **peak**. | Peak stays under 8 GB with margin; no monotonic climb across steps. | |
| 5 | Confirm the app process was **not jetsammed** at any phase (check for `EXC_RESOURCE` / `JetsamEvent` in the device console / crash reports). | **No jetsam.** No `JetsamEvent-*.ips` generated for the app during the run. | |
| 6 | Repeat step 3 **5×** back-to-back (no relaunch) to surface a slow leak / fragmentation-driven kill. | All 5 runs complete; no jetsam; peak `phys_footprint` does not trend upward run-over-run. | |
| 7 | Run **image→image** with **1** reference image (the iPad-8GB `maxReferenceImages` cap). | Completes; no jetsam; a 2nd reference image is rejected/capped. | |
| 8 | Background the app mid-generation, then foreground it. | No crash; generation resumes or fails gracefully (no jetsam-on-return). | |

### How to capture jetsam evidence

Same procedure as the 16 GB checklist above:

- **Xcode**: Devices & Simulators → select device → **View Device Logs**; filter
  for `JetsamEvent` / the app's process name.
- **On-device**: Settings → Privacy & Security → Analytics & Improvements →
  Analytics Data → look for `JetsamEvent-*` entries around the test window.
- Correlate the A6 `phys_footprint` peak (from step 4) against the device's
  per-process jetsam limit to see the margin (or deficit) against 8 GB.

### Recording a run

Copy the table above into a dated run log (PR description, or an appended
section here) with the device model + iPadOS version, and mark each row. A run
is a **PASS** only if rows 5, 6, 7 all show **no jetsam**. Record the observed
peak `phys_footprint` explicitly — that number is the input to the B3
working-set recalibration follow-up, whether the run passes or fails.

> A PASS here is the trigger to open a follow-up sortie that (a) sets B3's
> working-set constant from the measured peak instead of the ~10 GB
> conservative estimate, and (b) flips `MemoryConfig.enable8GBTier` to `true`.
> A FAIL (jetsam) means the ~10 GB conservative estimate was directionally
> right and `enable8GBTier` should stay OFF pending further memory reduction
> work (e.g. a smaller working-set pad, more aggressive VAE tiling).

---

## Provenance

- CI half (16 GB): `Tests/Flux2GPUTests/IPadDeviceMatrixGPUTests.swift` +
  `.github/workflows/integration-tests.yml`.
- CI half (8 GB): `Tests/Flux2GPUTests/IPad8GBDeviceMatrixGPUTests.swift`
  (Sortie B5) — model-presence-gated, exercises `.iPad8GB` via the
  `enable8GBTier:` parameter without changing the shipped global default.
- Split rationale: `EXECUTION_PLAN.md` OQ-3 (device-matrix tests are split;
  jetsam is manual-only).
- 8 GB gating rationale: `EXECUTION_PLAN.md` OQ-4 /
  `requirements/REQUIREMENTS-flux-on-ipad.md` §8 ("Is 8 GB worth it at all?")
  and §3 R3.3 — `MemoryConfig.enable8GBTier` defaults OFF pending the
  measurement this checklist performs.
