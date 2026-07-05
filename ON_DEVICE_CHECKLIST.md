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

## Provenance

- CI half: `Tests/Flux2GPUTests/IPadDeviceMatrixGPUTests.swift` +
  `.github/workflows/integration-tests.yml`.
- Split rationale: `EXECUTION_PLAN.md` OQ-3 (device-matrix tests are split;
  jetsam is manual-only).
- The 8 GB iPad equivalent of this checklist is added by Sortie **B5**.
