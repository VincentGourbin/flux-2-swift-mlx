# Flux2App Fork — Single-Image Prompt-Based Editing

This fork extends [flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) toward **single-image, prompt-based image editing**: load a reference photo, describe the change you want, and get a localized result composited back into the original.

It is **not** a mask-and-inpaint workflow. I think brush masks and dedicated inpaint pipelines are dead tech for this use case — they add UI friction and model complexity without matching how FLUX.2 Dev actually conditions on a reference image. This fork instead uses **Image Preparation**: format the image for the model, optionally define a **Live Area**, generate, and paste the result back.

Upstream already merged two of my earlier contributions ([#94](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/94) I2I layout overflow, [#95](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/95) ResumableAdamW). The feature work lives in [PR #98](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/98).

---

## Image Preparation

The only feature here that needs real documentation. Everything else in the fork is studio plumbing (projects, saves, downloads) or ergonomics. **Image Preparation** is the editing workflow.

It has two steps, in order.

### Step 1 — Image Formatting (mandatory)

Before generation, the app fits your reference image to the **optimal step size** for the selected model family (FLUX.2 snaps to a 32-pixel grid).

Controls:

| Control | What it does |
|---|---|
| **Favour** | Bias toward original aspect, horizontal, or vertical when choosing crop geometry |
| **Method** | **Crop** discards pixels outside the target frame; **Pad** letterboxes to preserve the full image |
| **Scale** | Fine-tune how aggressively the image is scaled before crop/pad |

A **light translucent overlay** on the preview shows exactly what will be cropped away or where padding will land. This replaces manual prep in an external editor.

Image Formatting always runs. You cannot skip it — the model needs correctly dimensioned input.

### Step 2 — Live Area Definition (optional)

After formatting, you can narrow processing to a **sub-region** of the image.

**What Live Area is**

- Defines what the model **sees** for conditioning (context for inferring lighting, materials, geometry, etc.)
- Defines where the **generated result is rendered and pasted back**

**What Live Area is not**

- Not a brush mask
- Not “only change these pixels, freeze everything else at the pixel level”
- Not a separate paste-back rectangle (an earlier UI experiment; removed)

The model still receives a coherent crop of the live region. The app composites the generated patch back into that same region on the full-resolution original.

**Why bother**

- **Save prep steps** — crop to the subject in-app instead of Photoshop first
- **Raise effective resolution** — pair Live Area with the **Megapixel Budget** (below) so a small region can still render at full output size without shrinking the whole canvas

**UI — barn doors**

In the Image-to-Image preview:

1. **Drag the barn-door edges** (top, bottom, left, right) to pull the “doors” closed and expose only the live region. The area outside the doors is darkened.
2. **Or drag a rectangle** anywhere on the image to draw a new live region from scratch.
3. **Reset** restores the full frame (doors fully open).

Edge handles take priority within ~20px so you don’t accidentally start a new region when adjusting a door.

### Megapixel Budget

Separate from Live Area, but they work together.

The **Megapixel Budget** is the maximum total pixel count for generation. Live Area sets the **aspect ratio**; the budget sets **how many pixels** fill that ratio.

Example: a small live region (say 30% of the frame) with a 1 MP budget still generates at ~1 MP in that aspect — the conditioning crop is upscaled to hit the budget. You get local editing without throwing away output resolution.

---

## End-to-end flow (Image to Image)

1. Load reference image(s)
2. **Image Formatting** — crop/pad to model step size (overlay shows the effect)
3. **Live Area** (optional) — barn doors or drag-to-define region
4. Set prompt, steps, guidance, megapixel budget
5. Generate
6. Composite result back into the original at the live region
7. Save (optional Lanczos upscale, companion “-input” file, project file)

---

## Other additions (brief)

Not documented in depth here; they support day-to-day use:

- **Project files** — New / Open / Save / Save As (macOS File menu); persists settings across sessions
- **Image save service** — configurable output folder, format, naming (timestamp, auto-increment), presets
- **Model downloads** — clearer errors and UI for Hugging Face gated models (FLUX.2 Dev bf16, Klein 9B)
- **Generation ergonomics** — cooperative cancel, clear preview vs unload models, guidance default, live upsampled-prompt display

---

## Roadmap and maintenance

I will keep adding features and workflows, aggressively.

If you find a bug in this fork, I will fix it.

---

## Relationship to upstream

| Item | Status |
|---|---|
| I2I layout overflow fix | Merged upstream ([#94](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/94)) |
| ResumableAdamW / MLX 0.31.4 fix | Merged upstream ([#95](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/95)) |
| Image Preparation + studio features | Open ([#98](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/98)) — may need rebase onto current `main` |

Upstream `main` has moved ahead (chains, VLM enrichment, vision masks, etc.). This fork is on a **different product track**: prompt-based regional editing without mask channels.

---

*Coded with [Cursor](https://cursor.com) and 17 live ferrets.*
