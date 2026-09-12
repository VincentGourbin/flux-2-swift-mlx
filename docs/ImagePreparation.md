# Image Preparation

Barn-door **Live Area** editing and megapixel-budget formatting for I2I —
contributed by Bill Evans ([@realnotsteve](https://github.com/realnotsteve))
in [PR #99](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/99), carved
down to the Core formatting/Live Area/composite workflow for this merge (see
that PR's history for the fuller fork, including a Generative Fill mode and
Flux2App project-file UI, not part of this scope).

## Step 1 — Image Formatting (mandatory)

Before generation, the reference image is fit to the **optimal step size**
for the model (FLUX.2 snaps to a 32-pixel grid).

| Control | What it does |
|---|---|
| **Favour** (`--favour`) | Bias toward original aspect, horizontal, or vertical when choosing crop geometry |
| **Method** (`--method`) | **Crop** discards pixels outside the target frame; **Pad** letterboxes to preserve the full image |
| **Scale** (`--prep-scale`) | Fine-tune how aggressively the image is scaled before crop/pad |

Image Formatting always runs once any prep flag is set — the model needs
correctly dimensioned input.

With multiple `--images`, Live Area and `--process-area` only apply to the
**first** reference; every additional reference is formatted full-frame
regardless of a Live Area on the first.

## Step 2 — Live Area (optional)

After formatting, generation can be narrowed to a **sub-region** of the image
via `--live-area x,y,width,height` (normalized `0…1`, top-left origin, against
the full original image). `--process-area` uses that same coordinate space
(not relative to the Live Area) and must overlap it.

**What Live Area is**

- Defines what the model **sees** for conditioning (context for inferring
  lighting, materials, geometry, etc.)
- Defines where the **generated result is rendered and pasted back**

**What Live Area is not**

- Not a brush mask — anything outside the Live Area on the full-resolution
  image stays effectively bit-exact original (the compositing round-trip
  through the generation-resolution canvas can shift the paste-back edge by
  a few pixels — more for a Live Area drawn much larger than the megapixel
  budget, since the round-trip rounding error scales with that ratio);
  inside, the whole crop is reinterpreted toward the prompt and pasted back
  on a hard edge.
- Not object referents for the prompt ("these three women") — the **text
  prompt** carries the edit; Live Area carries scene volume and megapixel
  economics.

**Why bother**

- **Megapixel economics** — exclude parts of the frame irrelevant to
  inference so the megapixel budget is spent on the scene that matters.
- **Raise effective resolution** — pair Live Area with the budget so a
  smaller geographic slice still generates at full output size.
- **Reframe aspect ratio** — on a very wide master, draw a region closer to
  square instead of forcing a long skinny crop through the budget.

**Operator intent.** Draw generously: big enough for the model to infer
lighting, geometry, and relationships, and big enough for the composite to
land knock-on effects (e.g. shadows on a far wall), while dropping dead
weight that will never change. If a shadow or reflection should differ after
the edit, that surface must be *inside* the Live Area — pixels outside never
update.

## Megapixel Budget (`--megapixels`)

Separate from Live Area, but they work together. The budget is the maximum
total pixel count for generation (`0.25`–`4.0` MP, default `1.0`). Live Area
sets the **aspect ratio**; the budget sets **how many pixels** the model
*generates* — a small live region with a 1 MP budget still generates at ~1 MP
in that aspect, giving local editing without throwing away output
resolution.

That's the generation canvas, not the reference the VAE encodes: the
reference is deliberately never upsampled past its native resolution (an
interpolated, upsampled JPEG smears into fuzzy gradients FLUX.2 reads as
real structure) — it's rendered at native scale and the model enlarges it
generatively to fill the budget, rather than the framework enlarging the
pixels first.

## End-to-end flow

1. Load reference image
2. **Image Formatting** — crop/pad to model step size
3. **Live Area** (optional) — narrow to a sub-region
4. Set prompt, steps, guidance, megapixel budget
5. Generate
6. Composite the result back into the original at the Live Area (skipped for
   a full-frame edit, or with `--no-composite`)

## Flux2CLI

```bash
flux2 i2i "cyan studio backdrop" -i photo.jpg -o edited.png \
  --prepared --method pad --favour original --megapixels 1.0

flux2 i2i "warmer light, add a lamp" -i room.jpg -o edited.png \
  --live-area 0.15,0.1,0.7,0.85 --megapixels 1.5
```

See `flux2 i2i --help` for the full flag list, or
[docs/CLI.md](CLI.md#image-preparation-barn-door-live-area--megapixel-budget).
