# Inpainting & Outpainting — By-the-Book Implementation Guide

How to get *good* results from `Flux2MaskedInpaintingChain` / `Flux2OutpaintingChain`,
what the framework does under the hood, and where it differs from the reference
implementation (diffusers `Flux2KleinInpaintPipeline`, merged upstream in
[PR #13050](https://github.com/huggingface/diffusers/pull/13050)).

Written against the empirical studies in this repo (cat→duck replacement, 2CV
outpainting) and the diffusers reference. If your results are poor, work through
the [checklist](#0-tldr--the-checklist) first — in practice most quality problems
are caused by the host app's mask, prompt, or resolution handling, not by the chain.

---

## 0. TL;DR — the checklist

Every item here has produced a *visibly bad* result when missed. In rough order
of impact:

1. **Prompt describes the WHOLE scene, not the edit.** 30–80 words, BFL
   structure (Subject + Action + Style + Context). `"a duck"` gives a floating,
   unshaded duck; a full scene description gives correct scale + cast shadow.
   Best: load the Qwen3.5 VLM and set `enrichPromptWithVLM: true` + the right
   `intent`.
2. **Soft mask edges.** Gaussian-blur the mask edge ≈ `image_width / 30`
   (the diffusers example uses `blur_factor=12` at 1024 px — same ballpark).
   Hard masks produce visible seams.
3. **Right conditioning for the right job** (see the
   [decision table](#2-choose-the-workflow--decision-table)):
   - *replace* → **no** reference (`useImageAsReference: false`, default),
     or the **new object's photo** as `referenceImages`.
   - *repair / extend / outpaint* → source as reference.
   - Passing the source as reference during a *replace* bleeds the old subject
     into the new one (verified: duck inherits the cat's tabby head).
4. **Resolution: don't let the mask region shrink to nothing.** The chain
   works at ≤ `maxPixels` (default 1024²). A small mask in a 12 MP photo ends
   up covered by a handful of latent tokens → mushy result. Set
   **`maskCropPadding: 32...64`** — the chain then inpaints only a crop around
   the mask (full token budget on the edit) and pastes the result back onto
   the untouched original. See [§5](#5-resolution-maxpixels-and-crop-and-stitch).
5. **Keep the rest of the photo bit-identical.** Without compositing, the
   chain's output is the *whole* decoded canvas at working resolution: the
   kept region has been VAE-roundtripped and possibly downscaled. Set
   **`compositeOnOriginal: true`** (implied by `maskCropPadding`) so kept
   pixels come straight from the user's original.
6. **If you use `enrichPromptWithVLM`, actually load the VLM.** The chain
   *silently falls back* to the verbatim prompt when
   `FluxTextEncoders.shared.isQwen35VLMLoaded == false` (a `FluxDebug` warning
   is logged, nothing throws). Easy to miss in an app.
7. **Seed everything** while iterating, so you can attribute changes to your
   changes.

---

## 1. How the chain works (and why there is no "Fill" model)

FLUX.2 has **no dedicated Fill checkpoint** (unlike FLUX.1-Fill). Both this
framework and diffusers implement inpainting as **RePaint-style latent
blending** on top of the standard generation loop:

```
after each scheduler step:
    originalNoised = (1 - σ_next)·imageLatents + σ_next·noise
    latents        = (1 - mask)·originalNoised + mask·latents
```

- Outside the mask, the latents are *forced back* to the original image
  re-noised to the current σ — the model "sees" the true surroundings at every
  step and can harmonize the masked content against them.
- On the final step `σ_next == 0`, so the outside region is restored to the
  clean original latent (no drift outside the mask — in latent space).
- The mask lives at **latent resolution: 1 token = 16×16 px** (VAE 8× ×
  2×2 packing). Sub-16 px mask detail cannot be represented; soft values are
  honoured and become per-token blend weights.

Both mask conventions end up in the same internal form
(`1.0 = inpaint, 0.0 = keep`):

| `maskConvention` | You provide | Typical source |
|---|---|---|
| `.grayscaleWhiteInpaint` (default) | separate grayscale image, white = repaint | segmentation model, threshold |
| `.alphaTransparentInpaint` | an *erased copy of the source* (alpha 0 = repaint) | Photoshop / Procreate eraser |

## 2. Choose the workflow — decision table

The single most common integration mistake is using the wrong conditioning
mode for the job. FLUX.2 has no negative prompts and no edit-instruction
channel: conditioning = prompt + optional reference images, nothing else.

| Goal | `referenceImages` | `useImageAsReference` | `strength` | VLM `intent` | Notes |
|---|---|---|---|---|---|
| **Replace** X with Y (prompt-driven) | `nil` | `false` (default) | `1.0` | `.replace` | Prompt must describe the whole scene with Y in it |
| **Replace** X with a *specific* Y (you have a photo of Y) | `[photoOfY]` | `false` | `1.0` | `.replace` | The diffusers reference example does exactly this (`"Replace this ball"` + ball photo + blurred mask). Short instruction prompts work when a reference carries the appearance |
| **Remove** X, continue background | `nil` | `false` | `1.0` | `.remove` | **Never name X in the prompt** — FLUX.2 has no negatives; naming re-introduces it. Describe the background that should exist |
| **Modify** X (color, material, pose) | `nil` | `false` | `0.5–0.75` | `.modify` | Describe X in its final state, in-scene. `strength < 1` starts from the noised original, preserving X's layout while restyling it |
| **Repair** damaged/empty region | `nil` | `true` | `1.0` | `.modify` | Masked region carries no subject to leak, so source-as-reference safely provides palette/lighting |
| **Outpaint** | handled by `Flux2OutpaintingChain` | — | `1.0` | — | Chain passes the original as reference + smart mask automatically |

### `strength` — img2img init (diffusers parity)

At `strength: 1.0` (default) the masked region starts from pure noise. At
`< 1.0` the denoising starts from the **original image noised to σ₀** and the
first `steps·(1-strength)` timesteps are skipped — the masked region keeps the
original's low-frequency structure (layout, pose, palette) and the model only
re-details it. Right tool for *modify* edits; wrong tool for *replace*/*remove*
(the old subject survives as a ghost).

Granularity caveat: with 4-step distilled models only `strength ≤ 0.75`
actually skips a step (0.75 → 3 steps from σ≈0.75). On base models at 28–50
steps, strength behaves continuously like SD img2img. Ignored (with a log)
when LoRA custom sigmas are active.

Why source-as-reference is default-OFF: with the source as an I2I reference,
the transformer attends to the reference tokens — which still contain the
to-be-replaced subject under the mask. Verified on cat→duck: the duck inherits
the tabby's grey-striped head, and the run costs ~2× (concatenated sequence).

## 3. Masks by the book

- **Convention**: white (or transparent, in alpha mode) = inpaint. Double-check
  you're not inverted — an inverted mask "works" but regenerates everything
  *except* your target, which reads as "the model ignored my mask".
- **Coverage**: make the mask *generously* larger than the object for
  `.replace`/`.remove` — include the object's shadow and reflections, otherwise
  the old shadow stays and the new subject looks pasted.
- **Soft edge**: Gaussian blur radius ≈ `image_width / 30` (≈ 32 px at 1024).
  This is the validated value from the 2CV study; diffusers' example
  `blur_factor=12` is equivalent in spirit. Pre-binarise only if you
  explicitly want a hard seam.
- **Granularity**: 1 latent token = 16×16 px. Don't bother with pixel-perfect
  mask contours; they're bilinearly reduced to the token grid anyway.
- **Outpainting masks** are built automatically by `Flux2OutpaintingChain`
  (pure white strips + 32 px inner ramp + black keep). Don't hand-build a
  full-canvas gradient — a gradient over the strips lets the seed noise bleed
  through (verified: bands/stripes artifacts).

## 4. Prompting by the book

Reference: [BFL prompting guide](https://docs.bfl.ml/guides/prompting_guide_flux2).

- **Structure**: Subject + Action + Style + Context, leading tokens weigh more.
- **Length**: 30–80 words. Describe the **entire target image**, including the
  kept surroundings (lighting direction, camera height, materials, palette) —
  this is what lets the model integrate the new content into your photo.
- **No negatives.** Describe what should exist, never what shouldn't.
- **For `.remove`**: describe the *background* as if the object never existed.
  Any mention of the removed object re-summons it.

### The VLM path (recommended for apps)

Hand-writing a 50-word scene-aware prompt for every user edit is unrealistic.
The framework bundles an image-aware prompt builder:

```swift
try await FluxTextEncoders.shared.loadQwen35VLM(from: vlmPath)  // caller owns lifecycle

let chain = Flux2MaskedInpaintingChain(
    pipeline: pipeline,
    prompt: userShortInstruction,       // e.g. "make it a duck"
    image: source,
    mask: mask,
    enrichPromptWithVLM: true,
    intent: .replace,                   // .replace / .remove / .modify — REQUIRED to be right
    seed: seed
)
```

Contract to know when integrating:
- VLM not loaded → **silent fallback** to the verbatim prompt (warning in
  `FluxDebug` only). If your app's results look like the enrichment "does
  nothing", check the VLM is loaded.
- `enrichPromptWithVLM` + `upsamplePrompt` both true → VLM wins, warning logged.
- The final prompt used is returned in the result's `usedPrompt` — surface it
  in debug UI; it's the fastest way to diagnose a bad edit.

## 5. Resolution, `maxPixels`, and crop-and-stitch

The chain resolves working dimensions as
`min(image, maxPixels)` floored to multiples of 32. Two traps:

1. **Whole-canvas budget.** Default `maxPixels = 1024²`. A 4000×3000 photo is
   downscaled to ~1170×864 *before* anything happens. A 300 px mask region in
   that photo becomes ~90 px ≈ 5×5 latent tokens — far too few to paint a
   detailed subject.
2. **Output ≠ original.** The raw result is the full decoded canvas at working
   resolution: kept pixels are VAE-roundtripped (slightly softened) and
   downscaled. Users notice their photo "lost quality everywhere" even though
   the edit itself is fine.

Both are solved natively — these are the framework equivalents of diffusers'
`padding_mask_crop` + `apply_overlay`:

```swift
let chain = Flux2MaskedInpaintingChain(
    pipeline: pipeline,
    prompt: prompt,
    image: photo,               // full-resolution original
    mask: mask,
    maskCropPadding: 48,        // crop-and-stitch: inpaint only around the mask
    seed: seed
)
// result.image has the ORIGINAL resolution; pixels outside the mask are
// bit-identical to `photo` (no VAE roundtrip, no downscale).
```

- **`maskCropPadding` (recommended: 32–64)** — the chain finds the mask's
  bounding box, expands it by the padding then to the image's aspect ratio,
  inpaints *that crop only* (full token budget on the edit), and pastes the
  result back onto the untouched original with the soft mask as per-pixel
  alpha. CLI: `--mask-crop-padding 48`.
- **`compositeOnOriginal: true`** — pixel composite without cropping (implied
  by `maskCropPadding`). Use when the mask covers most of the image but you
  still want kept pixels bit-exact. CLI: `--composite-on-original`.

Rule of thumb: mask bbox < ~half the image area → use `maskCropPadding`;
otherwise `compositeOnOriginal` alone.

Notes:
- The composite assumes an **opaque** source (photos). Images with
  alpha < 255 are flattened to opaque — semi-transparent regions won't be
  bit-exact.
- If the crop can't be established (e.g. the mask is empty), the chain falls
  back to full-canvas inpainting but still composites onto the original, so
  the output contract (original resolution, kept pixels bit-exact) holds.
- `strength < 1/steps` yields zero denoising steps and throws
  `invalidConfiguration` instead of silently returning the unedited image.

For **outpainting**, the opposite trap: set `maxPixels ≥ canvas_w × canvas_h`
or the extended canvas is downscaled below its native size
(`Flux2OutpaintingChain` raises it internally — don't fight it by passing a
smaller value).

## 6. Model & sampling choice

| Setup | Steps / guidance | When |
|---|---|---|
| **klein-9b distilled** (default) | 4 steps, guidance 1.0 | Fast path (~1 min at 1 MP on M2 Ultra). The 4-step count is by design (distillation) — don't change it |
| **klein-base-9B + classical CFG** | 28–50 steps, guidance ≈ 4–8, optional negative prompt "" | Quality path. This is what the diffusers inpaint pipeline defaults to (`guidance_scale=8.0`). The framework's CFG path activates automatically for `.klein9BBase` when `guidance > 1` |

Also relevant to perceived quality:
- **Transformer quantization**: qint8 is visually lossless on Klein 9B; avoid
  mxfp4/nvfp4 for inpainting (color fidelity caveats — see the
  [quantization benchmark](examples/quantization-benchmark/README.md)).
- **Determinism**: same seed + same inputs → identical output. Iterate on one
  variable at a time.

## 7. Parity with the diffusers reference

Our core blend is algorithm-identical to `Flux2KleinInpaintPipeline`
(the mask preparation, per-step blend, and final-step restore match). The
reference's four additional mechanisms are now covered:

| diffusers | Framework equivalent | Status |
|---|---|---|
| `padding_mask_crop` + `apply_overlay` | `maskCropPadding` / `compositeOnOriginal` on the chain (§5) | ✅ native |
| `strength` (img2img init via `scale_noise`) | `strength` on the chain, plumbed through `generateWithResult(initLatents:strength:)` (§2) | ✅ native |
| Fixed blend noise (drawn once, reused each step) | The chain's RePaint hook draws the blend noise once per run — the outside-mask region follows one consistent diffusion trajectory across the 4 steps instead of jittering | ✅ native (automatic) |
| Reference-guided replacement (`image_reference`) | `referenceImages: [photoOfNewObject]` — see the decision table (§2) | ✅ documented pattern |

Residual differences, deliberate:
- diffusers' inpaint pipeline defaults to klein-**base** + `guidance_scale=8.0`
  + 50 steps; our chain defaults to klein distilled (4 steps, guidance 1.0)
  for speed. Both paths are available (§6).
- diffusers reuses the *same* tensor for the img2img init noise and the blend
  noise; ours are independent draws from the same seeded RNG. RePaint forces
  the outside region every step, so only within-run consistency matters — which
  both implementations have.

## 8. App integration checklist (FluxForge-style hosts)

- [ ] Mask convention matches what the UI produces (eraser → `.alphaTransparentInpaint`)
- [ ] Mask includes the object's shadow/reflection for replace/remove
- [ ] Mask edge blurred ≈ `width/30`; no full-canvas gradients
- [ ] `intent` wired to the UI's mode (replace/remove/modify) — not hardcoded
- [ ] VLM loaded before enabling `enrichPromptWithVLM`; `usedPrompt` surfaced in debug
- [ ] No source-as-reference on replace; reference = new-object photo when the user provides one
- [ ] `maskCropPadding: 32...64` set for photos where the mask is small relative to the image (§5)
- [ ] `compositeOnOriginal: true` when not using `maskCropPadding` — kept pixels bit-exact (§5)
- [ ] `strength` wired to the edit mode: 1.0 for replace/remove, 0.5–0.75 for modify (§2)
- [ ] Seed surfaced/loggable; steps/guidance left at model defaults
- [ ] Outpainting goes through `Flux2OutpaintingChain` (not hand-built masks)

---

*Empirical sources: cat→duck prompt & reference study (PR #87/#88), 2CV
outpainting iterations v1–v6, VLM prompt-builder design (2026-05-29),
diffusers `Flux2KleinInpaintPipeline` (PR #13050).*
