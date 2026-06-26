# Agent handoff — `flux-2-swift-mix`

**Read this first** before editing the app. Prior chat (intents terminology deep-dive): [ccd009b4-68e0-4034-937b-dff51c4c475d](ccd009b4-68e0-4034-937b-dff51c4c475d).

| | |
|---|---|
| **Repo** | `/Users/drwevans/GitHub/flux-2-swift-mix` |
| **Branch** | `mix/v2.4.0` (tracks `origin`) |
| **User** | Single developer, macOS only, one machine |
| **Canonical agent rules** | [`AGENTS.md`](../AGENTS.md) at repo root |

---

## 0. Current task (start here)

**This commit ships Images palette v2 + project JSON v2.** Your job is to **verify, fix regressions, and continue I2I polish** — not to re-litigate intent theory unless the user asks.

### Do now

1. **Build:** `swift build --product Flux2App && swift build --product Flux2CLI` then `bin/build-mlx-metallib.sh` if needed.
2. **Smoke (optional):** `bin/vm-smoke.sh` with Circus + model cache mount.
3. **Manual check in Flux2App → Image to Image:**
   - Images palette: add tabs (max 16), assign Reference / Interpret (VLM), set Primary
   - Per-tab formatting (favour, crop/pad, scale)
   - Megapixel budget in Generation Parameters
   - Live Area + selections only on **Primary** reference tab
   - Save / reload project v2 JSON; confirm VMSmoke fixture loads (`F2SM_PROJECT=Tests/Fixtures/VMSmoke/project.json`)
4. **Fix anything broken** from the palette migration (slot ordering, interpret paths, composite, generative fill).

### Do next (when user directs)

- Intent picker / `fillHelp` copy: use **selection** not “mask”; clarify Replace background needs Vision subject tool
- Intent taxonomy simplification (user rethinking whether five intents should stay)
- Fill UX redesign (discussion only until spec): selection vs edit region, context mask slider

### Do not

- Rename `changeScene` enum/API without explicit request (display name “Replace background” is already done)
- Auto-open upstream PRs or push to `fork`
- Edit other repos (`utility-be-circus`, etc.)

---

## 1. What this application is

**Flux2App** is a macOS SwiftUI app for **single-image, prompt-based editing** built on top of Vincent Gourbin’s MIT **flux-2-swift-mlx** framework (vendored as SwiftPM targets in this repo).

**Product focus (fork):** Image Preparation — load a reference photo, format it for FLUX.2, optionally frame a **Live Area** (barn doors), run image-to-image with a text prompt, composite the result back into the full-resolution original.

**Secondary in-app path:** **Generative fill** — draw a **selection** on a blemish; RePaint-style local inpaint via `Flux2MaskedInpaintingChain` (optional Qwen3.5 VLM prompt enrichment).

**Not in scope unless the user asks:** FluxForge Studio parity, Apple Vision-heavy product surface, polygon mask UI expansion, auto-upstream PRs. See upstream policy in `AGENTS.md`.

**Headline doc:** [`docs/ImagePreparation.md`](ImagePreparation.md) — especially [Operator intent](ImagePreparation.md#operator-intent-how-barn-doors-are-actually-used) (barn doors = scene volume + megapixel economics, **not** tight object selection).

---

## 2. Repos and remotes

| Remote | Repo | Role |
|--------|------|------|
| `origin` | `realnotsteve/flux-2-swift-mix` | **Private** working copy |
| `fork` | `realnotsteve/flux-2-swift-mlx-1` | Public fork for upstream PRs only |
| `upstream` | `VincentGourbin/flux-2-swift-mlx` | Rebase source (`no_push`) |

Active upstream offer: **PR #99** (Image Preparation). Do not add studio/VM plumbing to upstream offers without explicit user intent.

**Repo boundary:** This agent writes **only** to `flux-2-swift-mix`. No edits in `utility-be-circus`, `comfy-be-nodes`, `wp-be-*`, etc.

---

## 3. SwiftPM layout

```
Package.swift
├── FluxTextEncoders    # Mistral, Qwen3, Qwen3.5 VLM, tokenizers
├── Flux2Core           # Pipeline, ImagePreparation, project JSON, mask types
├── Flux2Chains         # Inpaint, outpaint, VLM prompt builder, Vision subject masks
├── Flux2CLI            # `flux2` command (i2i, inpaint, outpaint, train, …)
├── FluxEncodersCLI     # Encoder utilities
└── Flux2App            # macOS SwiftUI application
```

- **Platform:** macOS 15+
- **MLX pin:** `mlx-swift` exact `0.31.4`
- **Metal:** `swift build` does **not** produce `mlx.metallib` — run `bin/build-mlx-metallib.sh` after clean builds
- **App bundle:** `bin/package-flux2app.sh` → `Flux2App.app` (icon in `Assets/AppIcon/`)

---

## 4. Flux2App — navigation and major types

### 4.1 Shell

| File | Role |
|------|------|
| `Sources/Flux2App/Flux2App.swift` | `@main`, `ModelManager` env, project File menu |
| `Sources/Flux2App/Views/ContentView.swift` | Sidebar tabs: Chat, Generate, Vision, Qwen3 Chat, **T2I**, **I2I**, Tools, Models |
| `Sources/Flux2App/ViewModels/ModelManager.swift` | Download/load text encoders + diffusion weights |
| `Sources/Flux2App/Flux2AppSessionStore.swift` | Last tab, lightweight session restore |
| `Sources/Flux2App/ViewModels/ImageSaveService.swift` | Save output + optional Lanczos upscale, companion `-input` file |

**Smoke launch:** `F2SM_PROJECT` env opens **Image to Image** tab (tag 5) and loads project JSON.

### 4.2 Image to Image UI (primary editing surface)

| File | Role |
|------|------|
| `Views/ImageToImageView.swift` | Layout: palette column + preview/output; canvas overlay; generate wiring |
| `Views/ImageGenerationHeaderBar.swift` | Model family, model, quantizations, Generate, download status |
| `Views/ImageGenerationPromptSection.swift` | Prompt field, upsample / VLM toggles |
| `Views/ImageToImageCanvasToolsSidebar.swift` | Sidebar canvas tools when I2I tab selected |
| `Views/SelectionToolBar.swift` | Tool buttons (Live Area, rectangle, polygon, subject, crop) |
| `Views/PalettePanel.swift` | Collapsible / detachable palette chrome |
| `Views/ImagesPaletteView.swift` | **New** tabbed images palette (see §6) |
| `ViewModels/ImageGenerationViewModel.swift` | All generation state, routes, prepare/composite/generate |
| `ViewModels/ImageGenerationViewModel+ImageSlots.swift` | Image slot CRUD, project v2 load/save for images |
| `Models/GenerationImageSlot.swift` | In-memory slot model |

**T2I** reuses `ImageGenerationViewModel(workflow: .textToImage)` via `TextToImageView.swift` with a smaller palette set.

### 4.3 I2I generate routes (inferred, not user-picked)

Defined in `Sources/Flux2Core/I2IGenerateRoute.swift`. `ImageGenerationViewModel.generateRoute`:

| Route | When | Pipeline |
|-------|------|----------|
| `fullImage` | No active selection; Live Area / barn doors optional | `ImagePreparation` → `pipeline.generateImageToImageWithResult` → optional composite back via `contextArea` |
| `localFill` | User drew a selection (`inpaintMaskLayers` non-empty, etc.) | `buildGenerativeFillMask` → `Flux2MaskedInpaintingChain` |
| `outpaint` | Crop canvas tool active | `Flux2OutpaintingChain` |

Legacy project field `editMode` maps via `I2IGenerateRoute.fromLegacyProjectValue` (`promptEdit` → `fullImage`, `generativeFill` → `localFill`).

### 4.4 Canvas tools

`InpaintMaskTool` (`Sources/Flux2Core/InpaintMaskTypes.swift`):

| Tool | Purpose |
|------|---------|
| `liveArea` | Barn doors — sets `contextArea` (conditioning + paste-back for **fullImage** route) |
| `rectangle` / `polygon` | Draw **selection** for generative fill |
| `visionSubject` | Lasso/box hint → Apple Vision foreground mask |
| `cropCanvas` | Outpaint padding handles |

**Spatial editing** only on **Primary** reference tab (`isSpatialEditingActive` in `+ImageSlots`).

Modifier keys while drawing: ⇧ add to selection, ⌥ subtract (`SelectionMode`).

### 4.5 Prompt upsampling vs VLM enrichment

| Feature | Applies to | Mechanism |
|---------|------------|-----------|
| `upsamplePrompt` (Mistral) | `fullImage`, `outpaint` | Text-only prompt rewrite |
| `enrichInpaintPromptWithVLM` (Qwen3.5) | `localFill` only | Image-aware; `Flux2VLMPromptBuilder` + `vlmContextArea` crop for Qwen |

On first selection draw, app defaults: `inpaintIntent = .fill`, `enrichInpaintPromptWithVLM = true`, auto `syncAutoVLMContextArea()`.

---

## 5. Image Preparation pipeline (fullImage route)

Core logic in `Sources/Flux2Core/ImagePreparation.swift` (and types in `ImagePreparationTypes.swift`).

1. **Image Formatting** (per slot): favour (original / horizontal / vertical), crop vs pad, scale → model step grid (32px for FLUX.2).
2. **Live Area** (`contextArea`): optional barn-door rectangle. Defines conditioning crop **and** paste-back region. Outside stays bit-exact on composite.
3. **Megapixel budget** (`megapixelBudget`): caps total pixels for generation; works with Live Area aspect.
4. **Generate** I2I with reference image(s) + optional interpret images for VLM.
5. **Composite** generated patch back into full-res original when `compositionPlan` is set.

`ImageCoordinateMapper` + `ImageSaveService` handle display vs pixel coordinates (top-left image origin).

**Dormant:** `processArea` paste-back marquee for prompt-edit mode — reserved, not active UI.

---

## 6. Images palette + project v2 (shipped in this commit)

Large change on `mix/v2.4.0`. Verify build + I2I smoke after pull.

### What changed

Replaced separate “Reference Images / Image Formatting / Interpret Images” UI with unified **Images** palette:

- Up to **16 tabs** (`FluxGenerationProject.maxImageSlots`)
- Per-tab **role:** Unassigned / Reference / Interpret (VLM)
- One **Primary** reference (checkbox); spatial tools only on Primary
- Per-tab **formatting** (favour, method, scale)
- **Megapixel budget** moved to Generation Parameters palette
- **Processing Area** palette removed (help text absorbed elsewhere)
- UI polish: pill tab bar, `+` circle add tab, larger preview (`ImagesPaletteImagePreview`)

### New / modified files

```
Sources/Flux2Core/GenerationImageRole.swift          # role + ImageSlotFormatting + GenerationImageRecord
Sources/Flux2App/Models/GenerationImageSlot.swift
Sources/Flux2App/ViewModels/ImageGenerationViewModel+ImageSlots.swift
Sources/Flux2App/Views/ImagesPaletteView.swift
Sources/Flux2Core/FluxGenerationProject.swift        # version 2 schema
Tests/Fixtures/VMSmoke/project.json                # converted to v2
Sources/Flux2Core/Pipeline/Flux2Pipeline.swift     # I2I ref cap 3 → 6
```

### Project v2 shape (`FluxGenerationProject`)

- `version: 2`
- `images: [GenerationImageRecord]` with embedded PNG or `sourcePath`
- `selectedImageSlotID`
- Still carries spatial fields: `contextArea`, `inpaintMaskLayers`, `inpaintIntent`, `inpaintMaskTool`, `outpaintPadding`, etc.
- v1 projects fail load with version error — **clean break** (user accepted for v2)

### Small committed-adjacent change

- `changeScene` intent **display name** → `"Replace background"` (`Flux2InpaintIntent.swift` only; enum case unchanged)

---

## 7. Generative fill (localFill route)

1. User draws **selection** → stored in `inpaintMaskLayers` (+ optional `processArea` legacy rect).
2. `buildGenerativeFillMask` → `ImageMaskBuilder.buildInpaintMask` (layer union/subtract + 2–12px feather).
3. Vision layers resolved in `resolveVisionSubjectMask` (`pickSubjectInpaintMask` or `pickChangeSceneMask` when intent is `changeScene`).
4. `Flux2MaskedInpaintingChain` runs RePaint (latent blend keeps pixels outside edit region each step).
5. `inpaintIntent` selects VLM system prompt when enrichment on.

**Full intent control** also exposed on CLI: `flux2 inpaint --intent replace|remove|fill|modify|change-scene`.

---

## 8. User-agreed terminology (mandatory in discussion)

| Term | Meaning |
|------|---------|
| **Selection** | What the user draws (rectangle, polygon, Vision subject hint). `inpaintMaskLayers`, marching ants. |
| **Mask** | **Context frame around the selection** — barn doors / Live Area / `contextArea`. What Qwen sees for prompt framing. |

Code still names the FLUX grayscale raster `inpaintMask` / `buildGenerativeFillMask`. When talking to the user, call that the **edit region** or **selection as FLUX sees it** — not “mask.”

### Intents — what they actually do (user validated)

For rectangle/polygon, intents differ only in **VLM prompt strategy**, not geometry:

| Intent | UI label | Steers prompt toward |
|--------|----------|----------------------|
| `replace` | Replace | New subject in edit region |
| `remove` | Remove | Surface continuation (generative, not flat fill) |
| `fill` | Fill | Patch from surrounding context |
| `modify` | Modify | Same subject, changed detail |
| `changeScene` | Replace background | New background only; don’t name subject |

**Replace background** inverts edit region (keep subject, paint around) **only** with **Vision subject** tool + `changeScene`. Rectangle + Replace background still repaints inside the box — help text is misleading.

**Replace** cannot change footprint: car-in-cat-sized selection fails geometrically. User may rethink intent picker UX.

---

## 9. Key framework files (shared app + CLI)

| Area | Files |
|------|-------|
| Pipeline | `Flux2Core/Pipeline/Flux2Pipeline.swift` |
| Image prep | `Flux2Core/ImagePreparation.swift`, `ImagePreparationTypes.swift` |
| Masks | `Flux2Core/ImageMaskBuilder.swift`, `InpaintMaskTypes.swift` |
| Chains | `Flux2Chains/Flux2MaskedInpaintingChain.swift`, `Flux2OutpaintingChain.swift` |
| Intents / VLM prompts | `Flux2Chains/Flux2InpaintIntent.swift`, `Flux2VLMPromptBuilder.swift` |
| Vision masks | `Flux2Chains/Flux2SubjectMask.swift` |
| CLI i2i | `Flux2CLI/ImageToImagePreparation.swift` |
| CLI inpaint | `Flux2CLI/InpaintCommand.swift`, `MaskSubjectCommand.swift` |

---

## 10. Projects, session, save

- **Project JSON:** `FluxGenerationProject` — File → New/Open/Save (`ProjectCommands.swift`, methods on `ImageGenerationViewModel`)
- **Session:** `Flux2AppSessionStore` persists tab + partial I2I state between launches
- **Output save:** palettes in I2I view + `ImageSaveService` (format, upscale, filename segments)
- **Env:** `F2SM_MODELS_DIR` overrides `~/Library/Caches/models`; Tart guest path auto-detected

---

## 11. Verification and smoke

```bash
cd ~/GitHub/flux-2-swift-mix
swift package resolve
swift build --product Flux2App
swift build --product Flux2CLI
bin/build-mlx-metallib.sh
```

**VM smoke** (requires Circus.app, profile `tart-av-dev`):

```bash
bin/vm-smoke.sh                    # UI load fixture project
bin/vm-smoke-generate.sh           # CLI klein i2i
bin/vm-smoke-generate-fill.sh      # CLI inpaint + Qwen VLM
```

Contract: [`docs/CIRCUS-TART-DEV-ENV-API.md`](CIRCUS-TART-DEV-ENV-API.md). **Stop and report** VM failures — do not retry SSH in a loop (`AGENTS.md`).

Fixture: `Tests/Fixtures/VMSmoke/project.json` (v2, embedded reference PNG).

---

## 12. Recent commit history (context)

```
8aefa8a Refactor I2I/T2I layout: sidebar tools, header generate, prompt bar, palettes.
730faf5 feat: Unify I2I workflow UI and inline model controls in the header.
998ee89 feat: Add layered fill masks, VLM barn-door context, and preview UX.
72c0ccf feat: Add Flux2App icon, .app packaging, and fill prompt UX fix.
```

Images palette v2 landed in the commit after `8aefa8a`.

---

## 13. Backlog (ask user first)

1. ~~**Commit** Images palette + project v2~~ — done
2. **Intent picker / help text** — align copy with selection vs context-mask terminology; clarify Replace background needs Vision subject
3. **Intent taxonomy** — user may reduce or relabel intents (prompt modes vs fake operations)
4. **Fill UX redesign** (discussion only): selection ≠ immediate edit region; workflow slider; Qwen always sees full image
5. **`fillHelp` strings** still say “mask” — user wants “selection” in UI copy

---

## 14. Agent discipline

- **Simple renames** → `displayName` / UI strings only unless user asks for API breaks
- **Do not** conflate intent (prompting) with tool (rectangle vs Vision) when explaining behavior
- **Do not** auto-commit, auto-push to `fork`, or open upstream PRs without user request
- **Bump version** only if project convention requires (app has no strict version header in plugin sense; describe in commit messages)
- User does not edit files manually — agent makes changes
- Commits may include `Co-authored-by: Cursor <cursoragent@cursor.com>`

---

## 15. Doc index

| Doc | Contents |
|-----|----------|
| [`AGENTS.md`](../AGENTS.md) | Build, VM, upstream policy, fork boundaries |
| [`docs/ImagePreparation.md`](ImagePreparation.md) | Product workflow, barn doors, generative fill |
| [`docs/Flux2App.md`](Flux2App.md) | Legacy feature overview + screenshots |
| [`docs/CLI.md`](CLI.md) | CLI reference |
| [`docs/CIRCUS-TART-DEV-ENV-API.md`](CIRCUS-TART-DEV-ENV-API.md) | VM smoke contract |
