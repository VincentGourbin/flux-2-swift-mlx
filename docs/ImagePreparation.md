# Flux2App Fork — Single-Image Editing

This fork extends [flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) toward **single-image editing** from a reference photo. Two workflows ship in Image to Image:

1. **Prompt edit** — **Image Preparation**: format the image for the model, optionally define a **Live Area** (barn doors), generate with I2I conditioning, and composite the result back into the full image.
2. **Generative fill** — draw a rectangle over a blemish or bad patch; RePaint-style local repair via Flux2Chains (optional Qwen3.5 VLM prompt enrichment).

Upstream already merged two of my earlier contributions ([#94](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/94) I2I layout overflow, [#95](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/95) ResumableAdamW). The feature work lives in [PR #98](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/98).

---

## Prompt edit (Image Preparation)

The barn-door / megapixel-budget workflow. Everything below through **End-to-end flow** applies to **Prompt edit** mode only.

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

- **Megapixel economics** — exclude parts of the frame that are irrelevant to inference and output (e.g. trees on both sides of a house) so the **Megapixel Budget** is spent on the scene that matters, not spread across a wasteful panorama
- **Raise effective resolution** — pair Live Area with the budget so a smaller geographic slice still generates at full output size (the conditioning crop is upscaled to hit the cap)
- **Reframe aspect ratio** — on a very wide master, draw doors around a region closer to square (room + light + subjects) instead of forcing a long skinny crop through the budget
- **Save prep steps** — crop in-app instead of Photoshop first

### Operator intent (how barn doors are actually used)

Barn doors are **not** for selecting edit targets (“draw around the three women”). They define the **minimum sufficient scene**: big enough for the model to infer lighting, geometry, and relationships, and big enough for the **composite** to land knock-on effects (e.g. shadows on a far wall), while dropping dead weight that will never change.

The operator uses judgment about what Mistral / FLUX / the VAE need — there is no published spec. Typical framing includes light entry, subjects, and the volume they inhabit (a whole room), **not** a tight marquee on the object being edited. Anything outside the doors on the full-resolution image stays **bit-exact original**; inside, the whole crop is reinterpreted toward the prompt and pasted back on a **hard edge**.

**Draw generously for output, not just input.** If a shadow or reflection should differ after the edit, that surface must be **inside** the doors. Pixels outside never update.

**Aspect ratio is a first-class lever.** Wide photos often need a live region closer to square; include extra context above/below if that carries lighting information the model needs.

**What barn doors are not in this workflow**

- Not a brush mask or per-pixel edit map (that is **Generative fill** or `flux2 inpaint`, or an external editor)
- Not “only where to look” with the full frame still composited — doors set **conditioning crop** and **paste-back boundary** together
- Not object referents for the prompt (“these three women”) — the **text prompt** carries the edit; doors carry scene volume and megapixel economics

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

## Generative fill

For **local repair** of a small bad patch (sensor dust, a wrong texture, a torn corner) — not for scene-wide prompt edits. Uses upstream `Flux2MaskedInpaintingChain` (RePaint per-step latent blending): the model may only write inside the fill rectangle; pixels outside are forced back to the source at every denoising step.

**When to use which**

| Goal | Mode |
| --- | --- |
| Change lighting, subjects, or scene interpretation inside barn doors | **Prompt edit** |
| Fix an undescribed blemish in one area | **Generative fill** |
| Named object swap / remove / scene change with full control | **CLI** `flux2 inpaint` (intent + mask file) |

**UI**

1. Choose **Generative fill** in **Edit workflow**.
2. **Drag a rectangle** on the preview over the patch — dashed accent border, dimmed surround.
3. Optional short prompt (e.g. “fill it in”); with **Enrich prompt with Qwen3.5 VLM** on (default), the app downloads/loads Qwen3.5 4B 4-bit and rewrites a Flux 2 prompt from the image. With enrichment off, your prompt is passed through verbatim.
4. **Resolution cap** (megapixel budget) limits working size; large inputs scale down before filling.
5. Generate — output is the full image with only the fill region changed.

**CLI equivalent** (mask file, all intents):

```bash
flux2 inpaint "fill in the damaged wall texture…" -i photo.jpg -m mask.png -o out.png \
  --enrich-prompt-with-vlm --qwen35-variant 4bit
```

See `flux2 inpaint --help` for intent, VLM enrichment, and mask conventions.

**Projects** persist `editMode`, `processArea` (fill rectangle), `inpaintIntent`, and `enrichInpaintPromptWithVLM` alongside the existing Image Preparation fields.

---

## Flux2CLI

`flux2 i2i` supports the same Image Preparation pipeline when you pass any prep
flag, or load a Flux2App project file.

**Legacy (unchanged):** plain reference image + optional `-w`/`-h` — no
formatting, live area, or composite.

**Prepared:**

```bash
flux2 i2i "cyan studio backdrop" -i photo.jpg -o edited.png \
  --prepared \
  --method pad --favour original \
  --live-area 0.1,0.1,0.8,0.8 \
  --megapixels 1.0
```

**From project JSON** (same shape as Flux2App saves / `F2SM_PROJECT` smoke
fixture):

```bash
flux2 i2i --project path/to/project.json -o edited.png --model klein-4b
```

| Flag | App equivalent |
| --- | --- |
| `--favour` | Image Formatting → Favour |
| `--method` | Crop / Pad |
| `--scale` | Preparation scale |
| `--live-area x,y,w,h` | Barn-door / Live Area (normalized) |
| `--process-area x,y,w,h` | Process selection for paste-back (optional) |
| `--megapixels` | Megapixel budget |
| `--no-composite` | Save raw model canvas only |
| `--prepared` | Force prep with defaults (full frame, 1 MP) |

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
