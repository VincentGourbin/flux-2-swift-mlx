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
   up covered by a handful of latent tokens → mushy result. Use the
   [crop-and-stitch recipe](#5-resolution-maxpixels-and-the-crop-and-stitch-recipe).
5. **Composite the result back onto the original in pixel space.** The chain's
   output is the *whole* decoded canvas at working resolution: the kept region
   has been VAE-roundtripped and possibly downscaled. Paste the generated
   masked region into your original instead of replacing the full image.
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

| Goal | `referenceImages` | `useImageAsReference` | VLM `intent` | Notes |
|---|---|---|---|---|
| **Replace** X with Y (prompt-driven) | `nil` | `false` (default) | `.replace` | Prompt must describe the whole scene with Y in it |
| **Replace** X with a *specific* Y (you have a photo of Y) | `[photoOfY]` | `false` | `.replace` | The diffusers reference example does exactly this (`"Replace this ball"` + ball photo + blurred mask). Short instruction prompts work when a reference carries the appearance |
| **Remove** X, continue background | `nil` | `false` | `.remove` | **Never name X in the prompt** — FLUX.2 has no negatives; naming re-introduces it. Describe the background that should exist |
| **Modify** X (color, material, pose) | `nil` | `false` | `.modify` | Describe X in its final state, in-scene |
| **Repair** damaged/empty region | `nil` | `true` | `.modify` | Masked region carries no subject to leak, so source-as-reference safely provides palette/lighting |
| **Outpaint** | handled by `Flux2OutpaintingChain` | — | — | Chain passes the original as reference + smart mask automatically |

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

## 5. Resolution, `maxPixels`, and the crop-and-stitch recipe

The chain resolves working dimensions as
`min(image, maxPixels)` floored to multiples of 32. Two traps:

1. **Whole-canvas budget.** Default `maxPixels = 1024²`. A 4000×3000 photo is
   downscaled to ~1170×864 *before* anything happens. A 300 px mask region in
   that photo becomes ~90 px ≈ 5×5 latent tokens — far too few to paint a
   detailed subject.
2. **Output ≠ original.** The result is the full decoded canvas at working
   resolution: kept pixels are VAE-roundtripped (slightly softened) and
   downscaled. Users notice their photo "lost quality everywhere" even though
   the edit itself is fine.

**By-the-book host-side recipe (crop-and-stitch)** — this is what diffusers'
`padding_mask_crop` + `apply_overlay` do, and what your app should do until
the framework provides it natively:

```
1. bbox   = bounding box of mask > 0
2. crop   = bbox expanded by ~32–64 px padding, adjusted to the image aspect
            ratio, clamped to image bounds
3. run the chain on (imageCrop, maskCrop) with maxPixels = 1024²
   → the masked region now gets the FULL token budget
4. resize the generated crop back to the crop's native size
5. composite in pixel space onto the UNTOUCHED original:
   out = original·(1 - blurredMask) + generated·blurredMask
   (use the same soft mask; only pixels under the mask change)
```

Step 5 alone (pixel composite, even without cropping) already guarantees the
kept region is bit-identical to the user's photo. Do it in every integration.

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

## 7. Known gaps vs the diffusers reference (framework roadmap)

Our core blend is algorithm-identical to `Flux2KleinInpaintPipeline`
(the mask preparation, per-step blend, and final-step restore match). The
reference has four mechanisms we don't have yet — candidates for framework
work, roughly by expected impact:

1. **`padding_mask_crop` + `apply_overlay`** — native crop-and-stitch and
   pixel-space compositing (§5 recipe, but inside the chain). Biggest win for
   real-photo editing apps.
2. **`strength` (img2img init)** — diffusers starts the masked region from the
   *noised original* (`scale_noise(imageLatents, t₀)`) and runs only the last
   `steps × strength` timesteps (default 0.8). At strength < 1 the masked
   region keeps the original's low-frequency structure — much better for
   `.modify` edits (recolor, retexture) where you want the layout preserved.
   Our chain always starts from pure noise (≡ strength 1.0, which is also what
   the diffusers replacement example uses).
3. **Fixed blend noise** — diffusers draws the RePaint blend noise **once**
   and reuses the same tensor at every step, so the outside-mask region follows
   one consistent diffusion trajectory across steps. Our hook draws fresh
   noise per step, which makes the model's context jitter between steps —
   with only 4 steps, each of the few context views is different. Cheap fix,
   plausible quality gain at the mask boundary.
4. **Reference-guided replacement UX** — we support `referenceImages`, but the
   documented pattern "pass a photo of the NEW object + short instruction
   prompt + strength 1.0 + blurred mask" (the reference's flagship example)
   should be surfaced as a first-class recipe.

## 8. App integration checklist (FluxForge-style hosts)

- [ ] Mask convention matches what the UI produces (eraser → `.alphaTransparentInpaint`)
- [ ] Mask includes the object's shadow/reflection for replace/remove
- [ ] Mask edge blurred ≈ `width/30`; no full-canvas gradients
- [ ] `intent` wired to the UI's mode (replace/remove/modify) — not hardcoded
- [ ] VLM loaded before enabling `enrichPromptWithVLM`; `usedPrompt` surfaced in debug
- [ ] No source-as-reference on replace; reference = new-object photo when the user provides one
- [ ] Crop-and-stitch around the mask for photos > `maxPixels` (§5)
- [ ] Final pixel composite onto the original (§5 step 5) — always
- [ ] Seed surfaced/loggable; steps/guidance left at model defaults
- [ ] Outpainting goes through `Flux2OutpaintingChain` (not hand-built masks)

---

*Empirical sources: cat→duck prompt & reference study (PR #87/#88), 2CV
outpainting iterations v1–v6, VLM prompt-builder design (2026-05-29),
diffusers `Flux2KleinInpaintPipeline` (PR #13050).*
