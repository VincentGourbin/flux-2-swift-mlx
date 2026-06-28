# Handoff: Upsize control (shipped) + session context

**Date:** 2026-06-28
**Branch:** `mix/v2.4.0`
**Head commit:** `9437b75` (`feat: Add Upsize control (Bicubic/Lanczos resample + FLUX BF16 enlarge)`) — pushed to `origin`.
**Status of the Upsize feature:** **SHIPPED, builds clean, NOT yet runtime-verified.** The operator said "it's a simple edit, no need to test." Treat it as compiled-but-unexercised.
**Operator's stated next task:** **barn doors** ("we still haven't done barn doors, but that will be after the refactor is complete" — the sizing-consolidation refactor is now complete). See [§ Next task](#next-task-barn-doors-scope-to-confirm).

---

## TL;DR

This session replaced an automated "decide how to scale / when to FLUX-enlarge" pipeline (SCUNet cleanup + a no-reference quality gate) with **one explicit, human-in-the-loop control**: an **Upsize** dropdown + Apply button on the Images palette. The operator decides the method per image; nothing auto-runs.

- Two paths behind one button: **Bicubic / Lanczos** (instant Core Image resample) and **FLUX.2 4B / 9B / Dev** (a bounded I2I enlarge at the megapixel budget, BF16, faithful prompt).
- The earlier SCUNet + NR-IQA exploration is **parked, not the path forward** — don't resurrect it without the operator asking. See [§ The detour](#the-detour-scunet--quality-gate-parked).
- A git push gotcha bit us twice; the fix is documented in [§ Git state & the push gotcha](#git-state--the-push-gotcha). Read it before your first push — the 68.5 MB SCUNet blob is on this branch.

---

## What shipped: the Upsize control

### Operator requirements (the decisions that drove the shape)

1. **Explicit choice, not automation.** A dropdown of scaling methods, applied on demand — *not* an ML quality gate deciding for the user.
2. **Methods (in dropdown order):** Bicubic, Lanczos, FLUX.2 4B, FLUX.2 9B, FLUX.2 Dev.
   - **Lanczos is the default.**
   - **Bicubic kept deliberately** — the operator wants it "for other reasons" even though Lanczos is sharper for photos. Do not remove it as "redundant."
3. **BF16 everywhere.** All three FLUX enlarge models run with the **transformer forced to `.bf16`**, regardless of the app's normal `transformerQuantization` (which defaults to `.qint8`). Max quality for the enlarge; no quantization.
4. **4B = `klein4B` with the BF16 transformer.** **`klein9BKV` is intentionally excluded** — KV is tuned for faster multi-reference I2I, not single-image enlargement.
5. **Explicit Apply, not on-load.** A FLUX enlarge is a full generation (slow); it must be a deliberate button press, never automatic when an image loads.
6. **Faithful prompt, editable in Settings.** The enlarge uses a neutral "reproduce, don't reinvent" instruction so it adds resolution without changing content. The operator can edit it in **Settings → Upsize**.
7. **Grey out FLUX when downscaling.** If the source already meets/exceeds the budget, generative enlarge has nothing to invent — only resampling is offered.

### Files

| File | Change |
| --- | --- |
| `Sources/Flux2Core/ImageScaler.swift` | Added `ImageScaler.bicubic(_:to:)`; refactored `lanczos` + `bicubic` onto a shared private `scale(...)` helper (filter name + optional `configure` closure). Bicubic uses `CIBicubicScaleTransform` with `inputB = 0`, `inputC = 0.75` (Catmull-Rom). Header comment updated: consumers are the save bus (`ImageSaveService`) and the Upsize control. |
| `Sources/Flux2App/ViewModels/ImageGenerationViewModel.swift` | Added `@Published var upsizeMethod: UpsizeMethod = .lanczos`. |
| `Sources/Flux2App/ViewModels/ImageGenerationViewModel+Upsize.swift` | **New.** The `UpsizeMethod` enum, the Settings-backed faithful prompt, and all Upsize logic (`performUpsize`, resample path, FLUX path). |
| `Sources/Flux2App/Views/ImagesPaletteView.swift` | Added `UpsizeControlView` (dropdown + Apply + hint) and mounted it for the **primary reference** slot only (`slot.isPrimary && slot.role == .reference && slot.hasImage`). |
| `Sources/Flux2App/Views/SettingsView.swift` | Added `@AppStorage(ImageGenerationViewModel.upsizeFaithfulPromptKey)` and a **"Upsize"** section: a multi-line faithful-prompt field + a **Reset** button to restore the default. |

### How it works

`UpsizeMethod` (in `…+Upsize.swift`) maps each entry to an optional `Flux2Model`:

```swift
enum UpsizeMethod: String, CaseIterable, Identifiable, Hashable {
    case bicubic, lanczos, flux4B, flux9B, fluxDev
    var fluxModel: Flux2Model? {
        switch self {
        case .bicubic, .lanczos: return nil
        case .flux4B: return .klein4B
        case .flux9B: return .klein9B
        case .fluxDev: return .dev
        }
    }
    var isGenerative: Bool { fluxModel != nil }
}
```

Gating:

- `isUpsizeDownscaling` — `true` when `primaryReferenceImage.width * height >= conditioningPixelBudget`. When true, generative entries are disabled.
- `canApplyUpsize` — requires `hasPrimaryReference && !isPipelineBusy`, and forbids a generative method while downscaling.

`performUpsize()` routes to one of two paths:

**Resample path (`applyResampleUpsize`)** — instant, synchronous:

1. Compute the budget-filled target via `upsizeTargetSize(for:)`, which builds an `ImagePreparationSettings` (`sizingFavor = .original`, `.crop`, the app's `megapixelBudget` + `pixelAlignment`, full-frame `contextArea`) and calls `ImagePreparation.generationSize(...)`.
2. `ImageScaler.bicubic` or `ImageScaler.lanczos` to that target.
3. `replacePrimaryReference(with:preservingSpatialWorkflow: true)` + `appendEditHistory(image:kind: .adopt, label: "Upsize (…)")`.

**FLUX enlarge path (`startFluxUpsize` → `runFluxUpsize`)** — a bounded generation on its **own pipeline**:

1. Sets `isGenerating` / `isPipelineBusy`; runs in `generationTask` (so the existing cancel/progress UI applies — **verify cancel actually reaches it**, see loose ends).
2. **Frees the edit pipeline first** (`if pipeline != nil { await clearPipeline() }`) so the enlarge model and the edit model are **never co-resident** — sequential by design.
3. Builds a fresh `Flux2Pipeline(model:, quantization: Flux2QuantizationConfig(textEncoder: textQuantization, transformer: .bf16), hfToken:)`. **Transformer is hard-coded `.bf16`.** HF token from `HF_TOKEN` env or the `hfToken` default.
4. Preps full-frame at budget (`compositeBack = false` → output *is* the budget-sized enlargement) via `ImagePreparation.prepare(referenceImages: [source], settings:)`.
5. `generateImageToImageWithResult(prompt: resolvedFaithfulUpscalePrompt, …, seed: nil, upsamplePrompt: false, …)` at `model.defaultSteps` / `model.defaultGuidance`.
6. On success: `replacePrimaryReference` + `appendEditHistory(… kind: .adopt …)`.
7. Always: `await upscalePipeline.clearAll()` + `Memory.clearCache()`.

The faithful prompt:

```swift
static let upsizeFaithfulPromptKey = "upsizeFaithfulPrompt"
static let defaultFaithfulUpscalePrompt =
    "Reproduce this image exactly — same composition, subjects, colors, and detail — "
    + "at higher resolution with sharper, finer detail. Add nothing and change nothing."
// resolvedFaithfulUpscalePrompt: stored (trimmed, non-empty) else default.
```

### Design choices worth preserving

- **Reused the edit-history `.adopt` kind** with a custom label — no schema change to `EditHistoryStore`. The Upsize result is a normal undoable document step (⌃⌘Z works).
- **One target-size source of truth.** Both paths size through `ImagePreparation` (the sizing-consolidation refactor's public API), so Upsize honors the same budget/alignment math as generation. Don't fork that math.
- **Enlarge BF16 is independent of the edit model's quantization.** Intentional per the operator. Don't "simplify" it to reuse `transformerQuantization`.
- **Own pipeline + clear-first** keeps peak memory bounded. Don't make the enlarge share the live edit pipeline.

### Loose ends / not-yet-verified (your checklist if you touch this)

1. **Runtime unexercised.** It compiles; it has not been clicked. First real run should confirm: dropdown shows all 5, Lanczos default, FLUX rows grey when the loaded image is ≥ budget, Apply produces a history step, status messages read sensibly.
2. **Cancel wiring.** The FLUX path sets `generationTask`/`isGenerating`. Confirm the app's existing Cancel control actually cancels an in-flight *Upsize* enlarge (it should, since it cancels `generationTask`), and that `clearAll()` still runs on cancel (it's after the `do/catch`, so yes — but verify no leaked pipeline).
3. **Re-entrancy.** `canApplyUpsize` blocks Apply while `isPipelineBusy`, and the Apply button is disabled during generation. Confirm you can't start an Upsize mid-edit-generation or vice versa.
4. **Downscaling + resampler is allowed.** Bicubic/Lanczos remain enabled when the source exceeds budget (that's the intended "shrink to budget" path). Only the *generative* rows grey out.
5. **`UpsizeControlView` lives in `ImagesPaletteView.swift`.** It's only mounted for the primary reference slot. If barn-door work reorganizes the Images palette, keep that mount condition.

---

## The detour: SCUNet + quality gate (parked)

A large part of this session explored an **automated** path that the operator ultimately rejected. Capturing it so you don't re-derive it:

- **SCUNet** (`scunet_color_real_psnr`) was ported to mlx-swift (`SCUNetModel`), parity-verified vs the Python MLX reference (PSNR ~120 dB), bundled as a 68.5 MB Flux2Core resource, and given `ImageCleanup.scunet(_:)` + a `flux2 clean-image` CLI. **It works and is committed** (`3ffaad4`, `090e226`).
  - **But:** on the operator's 15 test images, SCUNet mostly **blurred** them — worse the higher the input quality/resolution. Useful on ~2/15. The operator's verdict: not a default cleanup step. SCUNet has **no strength parameter** to dial back.
  - It is **not wired into any default flow.** `ImageCleanup` / the CLI remain available as a standalone Flux2Core seam if ever wanted, but the operator is not asking for it. **Don't auto-insert SCUNet into Upsize or prep.**
- **Automated quality assessment** (TOPIQ / NR-IQA / JPEG-artifact detection) was explored to decide "resample vs FLUX-enlarge" automatically. TOPIQ was even converted to Core ML FP16. The operator **abandoned the automated gate**: an evaluator that's inconsistent is worse than showing the image and letting the human choose. That rejection is *why* the Upsize control is a manual dropdown. **Do not build an automated scaling/quality decision** unless explicitly asked.
- **MLX vs Core ML:** settled — MLX is the native runtime here (no tensor-rank ceiling; Core ML capped at rank 5, which broke 6D SCUNet tensors). Diffusion stays MLX. This is not an open question.

---

## Plan doc status

`clean_size_invent_pipeline_b9a92047.plan.md` (a Cursor plan; not in `docs/`) drove the *automated* clean/size/invent pipeline and is now **largely obsolete** — its SCUNet-routing and NR-IQA stages were dropped in favor of the manual Upsize control. Don't treat it as the current spec. The current intent for scaling is the Upsize control documented here.

(`docs/plans/` contains only the unrelated `qwen-edit` port brief.)

---

## Next task: barn doors (scope to confirm)

The operator named **barn doors** as the next task, gated on the refactor — which is done. **Before building, confirm scope with the operator**, because the docs and the operator's wording disagree:

- `docs/ImagePreparation.md` (§ "Step 2 — Live Area Definition" and § "Operator intent") describes a **fully-specified, apparently-implemented** barn-door / Live Area UI: drag edges to close doors, drag a rectangle to define a region, Reset to reopen, darkened surround, ~20px edge-handle priority.
- The operator said **"we still haven't done barn doors."**

So either the documented barn-door UI is **not actually implemented / was removed in the refactor**, or "barn doors" means a **specific rework** of the existing Live Area. **Ask first** (the operator explicitly prefers questions — see working style). Use `docs/ImagePreparation.md` § "Operator intent (how barn doors are actually used)" as the design north star:

- Barn doors define the **minimum sufficient scene** (light path, room volume, shadow landing zones) for conditioning **and** paste-back — **not** a tight marquee on the edit target, **not** a brush mask.
- They set **conditioning crop + paste-back boundary together**; outside the doors stays bit-exact original.
- **Aspect ratio + megapixel economics** are the point: spend the budget on the scene that matters.

Related constraint from `AGENTS.md`: "Dormant `processArea` plumbing in prompt-edit mode is intentional — reserved for future paste-back UI, **not** barn-door Live Area." Don't conflate `processArea` with the barn-door region.

---

## Git state & the push gotcha

- Branch `mix/v2.4.0` is in sync with `origin/mix/v2.4.0` at `9437b75`. Three commits went up this session: `3ffaad4` (SCUNet model), `090e226` (SCUNet weights + CLI), `9437b75` (Upsize).
- **The 68.5 MB `Sources/Flux2Core/Resources/scunet_color_real_psnr_mlx.safetensors` is tracked on this branch** (a `.gitignore` negation past the blanket `*.safetensors` ignore). GitHub warns it's over the 50 MB advisory — that's a **warning, not a failure**.

**Push gotcha (read before pushing):** `git push origin HEAD` failed **twice** with:

```
remote: error: inflate: data stream error (too many length or distance symbols)
remote: fatal: pack has bad object at offset …: inflate returned -3
error: remote unpack failed: index-pack failed
```

`git fsck --full --strict` was **clean** (local objects fine), so the corruption was in the **pack git generated for the push**, via the pack-reuse optimization copying a bad compressed stream. The fix that worked:

```bash
git -c pack.allowPackReuse=false push origin HEAD
```

If a push fails this way again (likely while the big blob is in the sent pack), disable pack reuse rather than retrying naively or assuming network flakiness. Don't `git gc`/repack-thrash first — `fsck` clean means local is fine.

---

## Operator working style (standing preferences surfaced this session)

- **Loves questions; hates silent autonomy.** "ask all the questions you want — please." Confirm non-trivial decisions before acting. For small reversible choices, decide and note it.
- **Test only when asked.** The operator will often say "no need to test" for simple edits. Don't over-index on test scaffolding.
- **VM rule (from `AGENTS.md`):** before running anything in the Tart VM / `circus`, **say what you intend to run and stop for the OK.** If the VM misbehaves, **stop and report — do not thrash** (no retry loops, no host-only "verification" claimed as VM).
- **Repo boundary (from `AGENTS.md`):** this agent writes **only** to `flux-2-swift-mix`. Never touch `comfy-be-nodes`, `utility-be-circus`, `wp-be-*`. Cross-app coordination goes through the human.
- **Upstream is opt-in.** Default work stays on private `origin`. Don't auto-open PRs or push to `fork`.

---

## Build / verify

```bash
swift build --product Flux2App
swift build --product Flux2CLI
bin/build-mlx-metallib.sh          # after any clean build — MLX Metal shaders
bin/package-flux2app.sh && open Flux2App.app   # Dock-icon .app bundle for visual checks
```

Both products built clean at `9437b75`.

---

## File map (Upsize)

| File | Role |
| --- | --- |
| `Sources/Flux2App/ViewModels/ImageGenerationViewModel+Upsize.swift` | `UpsizeMethod`, faithful prompt, all Upsize logic |
| `Sources/Flux2App/ViewModels/ImageGenerationViewModel.swift` | `@Published upsizeMethod` (default `.lanczos`) |
| `Sources/Flux2App/Views/ImagesPaletteView.swift` | `UpsizeControlView`; mounted on the primary reference slot |
| `Sources/Flux2App/Views/SettingsView.swift` | Settings → Upsize (faithful prompt + Reset) |
| `Sources/Flux2Core/ImageScaler.swift` | `bicubic` + `lanczos` on a shared `scale(...)` helper |
| `Sources/Flux2Core/Cleanup/ImageCleanup.swift` + `Cleanup/SCUNetModel.swift` *(SCUNet, parked)* | `ImageCleanup.scunet(_:)` — available, not wired into any default flow |
| `docs/ImagePreparation.md` | Headline workflow + barn-door **operator intent** (north star for the next task) |
| `AGENTS.md` | Repo boundary, fork/upstream policy, VM rule, verification |

---

*Coded with [Cursor](https://cursor.com).*
