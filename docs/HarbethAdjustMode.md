# Procedural Adjust Mode (Harbeth) — deferred plan

**Status:** design only — not implemented. Pick up when we want a procedural counterpart to genAI alongside Text to Image and Image to Image.

---

## Intent

Add a third **purple sidebar mode** (same family as Text to Image and Image to Image) powered by [Harbeth](https://github.com/yangKJ/Harbeth): GPU Metal filters for **parametric** edits (exposure, contrast, saturation, sharpen, etc.).

| Layer | Role |
| --- | --- |
| **Harbeth (Adjust)** | Deterministic, repeatable pixel math |
| **FLUX T2I / I2I** | Semantic, prompt-driven change |

Typical pipeline: **T2I → Adjust → I2I → Adjust → save**, with Adjust as optional polish between generative steps.

---

## Tab handoff

When switching between the three image modes, **copy whatever is in the preview pane** into the next mode as its starting image — except Text to Image, which only **exports** (never accepts an inbound handoff).

| From → To | Handoff |
| --- | --- |
| Text to Image → Image to Image | Preview/generated image → **primary reference** in I2I |
| Text to Image → Adjust | Same |
| Image to Image → Adjust | Current preview image → Adjust canvas |
| Adjust → Image to Image | Adjusted image → **primary reference** in I2I |
| Any → Text to Image | **No import** (T2I starts without a reference image) |

**Preview source (per mode):** prefer `previewDisplayImage ?? generatedImage ?? previewSourceImage` for T2I/I2I; for Adjust, the current adjusted bitmap.

**Defaults:** hand off **full-resolution** pixels (not a zoomed preview texture). For I2I, land on the **primary reference** slot, not an arbitrary tab. Default to **what the preview shows** (formatted/prepared view), since that is what the operator is looking at.

---

## Why a separate tab (not a palette inside I2I)

- Different workflow: no prompt, barn doors, or diffusion — sliders and filter chains only.
- **Memory:** Adjust mode need not load the FLUX transformer; entering/leaving can unload MLX when appropriate.
- Keeps Image Preparation focused on genAI; Adjust stays the procedural studio.

---

## Architecture sketch

1. **`GenerationWorkflow.adjust`** (name TBD) + new sidebar entry (e.g. **Adjust**, purple styling like T2I/I2I).
2. **`ModeHandoffStore`** — small shared holder (`CGImage?` + optional metadata) owned at app shell level (`ContentView` or session store).
3. **`onChange(selectedTab)`** — leaving tab writes preview to bus; arriving tab reads bus (T2I skips inbound read).
4. **`HarbethAdjustView`** — preview, filter strip, reset; Harbeth via SPM (**pin a commit**, not `master`).
5. **Bridge** — `CGImage` ↔ Harbeth filter pipeline (`HarbethIO` or equivalent).

Today T2I and I2I each own a separate `@StateObject` `ImageGenerationViewModel`; handoff is **new** and does not exist on tab switch yet.

---

## Suggested v1 scope

- Handoff bus + third tab shell.
- Harbeth dependency (pinned revision).
- Small fixed slider set (e.g. brightness, contrast, saturation).
- Non-destructive params in memory; optional project JSON later.
- **Defer:** full filter catalog, before/after split, project persistence for adjustments, custom Metal shaders.

---

## Open decisions

- **Tab label and icon** — Adjust, Procedural, Develop, etc.
- **Sidebar tag renumbering** — adding a tab shifts numeric tags unless we move to stable tab IDs in session restore.
- **Project file** — session-only v1 vs `adjustments` block in generation project JSON.
- **Save path** — apply Harbeth chain only in Adjust, on export from I2I, or both.
- **Overlap with Core Image** — Lanczos upscale already uses `CILanczosScaleTransform` on save; decide whether Adjust replaces or complements that for tone/color.

---

## References

- Harbeth: https://github.com/yangKJ/Harbeth (MIT, macOS + Metal)
- Current modes: `ContentView` sidebar tags 4 (T2I), 5 (I2I); separate `ImageGenerationViewModel` per view
- Related product doc: [ImagePreparation.md](ImagePreparation.md)
