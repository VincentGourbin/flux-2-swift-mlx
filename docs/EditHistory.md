# Edit history and project bundle — deferred plan

**Status:** phases 0–3 implemented. Phase 4 (import entry + prune) implemented.

**Baseline:** macOS **Tahoe only**, single machine. No legacy-OS format fallbacks, no Quick Look requirements, no cross-platform export concerns.

**Pixel format:** **JPEG XL for everything** inside the project document — lossless masters, slot rasters, preview, and strip thumbnails (lossy JXL at high quality for thumbs is fine). Speed is not a constraint; bundle size and a single encode path matter more than hardware HEIC.

---

## Intent

Persist a **linear edit history** with undo/redo that survives **Save / Open Project**. Each meaningful step stores a full raster plus the recipe (prompt, model, spatial settings) needed to restore that state.

History is **not** a low-level undo of every barn-door drag or slider tick. It records **document states**:

| Creates a history entry | Does not |
| --- | --- |
| Successful generate (prompt edit, generative fill, outpaint) | Barn-door nudge, context-mask slider, selection draw before generate |
| Explicit **Use as Reference** / adopt output (if we expose that as a step) | In-flight denoise checkpoints (stay ephemeral) |
| Initial import / open of primary reference (optional first entry) | Mask polygon point added |

**Document history undo** is a pointer on the persisted list (History column, or dedicated shortcuts — see below). A new generate after stepping back truncates the redo tail.

**Restore semantics:** jumping to a history entry reloads **preview and primary reference** from that step’s master image, plus spatial settings from the sidecar. The next generate continues from that frame.

---

## Selection undo / redo (lightweight, separate stack)

Selection edits are **cheap to undo**: a snapshot is a few kilobytes of JSON (mask layers, optional draft geometry, fill context scale) — no JXL encode, no bundle write.

Use a **second stack**, independent of document history:

| | **Selection undo** | **Document history** |
| --- | --- | --- |
| **Stores** | `inpaintMaskLayers`, `processArea`, draft polygon/lasso points, `fillContextMaskScale`, `inpaintIntent` | Lossless JXL master + full recipe |
| **When** | On each committed selection change | On successful generate / adopt |
| **Persisted** | **Session only** — current selection is saved in `project.json`; the undo *stack* is not |
| **Footprint** | Tiny | Large (per step) |

**Why separate:** accidental **Reset Selection**, a bad polygon, or ⇧/⌥ combine mistakes are common; restoring the previous mask should not require a saved generate frame. Conversely, stepping document history should not be blocked by twenty selection micro-states.

### Snapshot shape (conceptual)

```json
{
  "inpaintMaskLayers": [ … ],
  "processArea": null,
  "draftPolygonPoints": [],
  "draftLassoPoints": [],
  "fillContextMaskScale": 0,
  "inpaintIntent": "fill"
}
```

`visionSubjectMasks` are **not** stored in the snapshot — on restore, re-resolve from layers + primary image (same as today after project load). Barn doors (`contextArea`) are **not** in the selection stack; they stay untouched during fill (see generative-fill context-mask behavior).

### Vision subject tool (selection, not highlight)

The Subject tool **creates mask layers** like Rectangle and Polygon. After Vision resolves the hint, the committed overlay is **marching ants on the silhouette boundary** — not a white interior highlight. That matches “this is the edit region” and pairs with selection undo (bad subject pick → Cmd+Z).

| State | Preview |
| --- | --- |
| Lasso in progress | Marching ants on draft lasso |
| Vision resolving | Spinner |
| Committed | Marching ants on traced mask outline (fallback: hint rect/polygon) |

The raster mask still drives generate; only the **chrome** changes.

### When to push an undo point

One entry per **committed** change, not per pointer move:

- Rectangle committed (replace / add / subtract)
- Polygon closed
- Vision subject resolved
- **Reset Selection** / clear (so undo brings the prior mask back)
- Optional: context-mask slider **mouseup** or debounced stop (single entry per drag, not per tick)

Do **not** push on: individual polygon corners before close, in-progress rectangle drag, barn-door drag, megapixel slider.

### Keyboard and menu

- **Cmd+Z / Cmd+Shift+Z** — selection undo / redo when the I2I canvas is active and the selection stack has entries.
- **Document history** — primarily the **History** column (click to jump). Optional later: separate shortcuts (e.g. ⌃⌘Z) if stepping milestones from the keyboard feels necessary.

If both stacks could apply, **selection undo wins** — document history is for milestones, not marquee edits.

### Implementation sketch

- `SelectionUndoStore` — `[SelectionSnapshot]` + `index`, owned by `ImageGenerationViewModel`.
- Mutations go through helpers (`commitFillRectangle`, `deselectSelections`, …) that call `pushSelectionSnapshot()` before applying change.
- `restoreSelectionSnapshot(_:)` writes mask fields and triggers vision mask refresh for subject layers.
- Clear selection stack on **project load**, **history jump** (document restore replaces selection from sidecar), and **new primary image**.

`NSUndoManager` is optional; a small explicit stack matches the bounded-command style and is easier to reason about than wiring every `@Published` field into AppKit undo.

---

## Why a bundle (not JSON + base64)

Today `flux_project.json` embeds slot images as `pngBase64` and drops `generatedImage` on load. History multiplies that problem.

A **package document** keeps JSON as manifest only and stores pixels as files:

```text
MyEdit.flux2project/
  project.json              # manifest, settings, history index (no embedded rasters)
  preview.jxl               # current canvas (processed / B side)
  slots/
    primary.jxl
    ref-2.jxl
  history/
    0001.jxl                  # lossless master
    0001.json                 # recipe sidecar (or inline in manifest)
    0002.jxl
    0002.json
  thumbs/
    0001.jxl                  # small strip image for left column
    0002.jxl
```

**Migration:** continue opening flat **v2** `flux_project.json`; **Save / Save As** writes **v3** bundle. Smoke fixtures can be `project.json` + sibling assets in a folder.

---

## JPEG XL encoding policy

All in-bundle writes use **Image I/O** (`CGImageDestination`) with `public.jpeg-xl`.

| Asset | JXL mode |
| --- | --- |
| `history/*.jxl`, `slots/*.jxl`, `preview.jxl` | **Lossless** |
| `thumbs/*.jxl` | **Lossy**, high quality (~same pipeline, lower resolution) |

Implementation lives in a **document encoder** (e.g. `ProjectBundleImageWriter`) — separate from `ImageSaveService` export preferences (Pictures folder can stay PNG/JPEG/HEIC as today).

No PNG/TIFF/HEIC fallbacks in the bundle path. If JXL encode fails on dev machine, fail loudly during development rather than silently switching format.

---

## History model (`project.json`)

```json
{
  "version": 3,
  "currentHistoryIndex": 2,
  "history": [
    {
      "id": "…",
      "label": "Generate",
      "master": "history/0001.jxl",
      "thumb": "thumbs/0001.jxl",
      "kind": "generate",
      "prompt": "…",
      "settings": { "selectedModel": "…", "steps": 4, "megapixelBudget": 1.0 },
      "spatial": { "contextArea": {…}, "inpaintMaskLayers": […], "fillContextMaskScale": 0 }
    }
  ],
  "images": [ … slot records with relative paths, not base64 … ]
}
```

- **`currentHistoryIndex`** — undo/redo pointer.
- **`kind`** — `import` | `generate` | `adopt` (extensible).
- **`spatial`** — everything needed to redraw overlays and run the next op from that state.

Prune policy (optional later): cap depth (e.g. 30) or **Prune History…** command — single user, but unbounded 4 MP × 30 lossless JXL is still large.

---

## UI — app shell sidebar

History belongs in the **far-left app sidebar** (`ContentView` `NavigationSplitView`), directly under the **Mode** section — not in the I2I palette column.

```text
┌─────────────────────────┐
│ Mode                    │
│  Chat                   │
│  …                      │
│  Image to Image  ●      │
├─────────────────────────┤
│ History                 │  grows to fill sidebar downward (I2I only)
│  ● step 3  [thumb]      │
│  ○ step 2               │
│  ○ step 1               │
└─────────────────────────┘
        │ detail → I2I palettes + canvas
```

The I2I palette column keeps canvas tools and Images / Workflow / Parameters only.

- Click row → restore that index (moves pointer).
- Current step highlighted; keyboard shortcuts wired to same store.
- Thumbnails from `thumbs/*.jxl` only — decode small, not full master.

---

## Code touchpoints (when implementing)

| Area | Change |
| --- | --- |
| `FluxGenerationProject` | v3 manifest; `history[]`; slot `relativePath` replaces `pngBase64` |
| `ImageGenerationViewModel` | `EditHistoryStore` (document) + `SelectionUndoStore` (session) |
| Save / load | `FileWrapper` / `.flux2project` UTType; path-relative asset resolution |
| `ImageToImageView` | I2I palettes + canvas (history is **not** here) |
| `ContentView` | Mode sidebar `List` + `EditHistorySidebarSection` under Mode when I2I is active |
| Encode | `ProjectBundleImageWriter` — JXL lossless + thumb helper |
| Checkpoints | unchanged — ephemeral, not history |

**Bounded commands:** append history **once** on successful generate (or explicit adopt). No background reconciler, mutation observers, or auto-sync loops.

---

## Phasing

0. **Selection undo/redo** — snapshot stack + Cmd+Z; no bundle work required; ships value early. **(implemented)**
1. **Bundle + JXL slots/preview** — prove save/load without history UI. **(implemented)**
2. **Linear history** — append on generate/adopt; left-column list; click to restore. **(implemented)**
3. **Document history shortcuts** — History menu, ⌃⌘Z / ⌃⌘⇧Z step back/forward. **(implemented)**
4. **Import + prune** — first primary import as step 1; cap at 30 entries; Clear History in panel. **(implemented)**

---

## Relationship to other docs

- **Image Preparation** (`docs/ImagePreparation.md`) — prompt-edit barn doors and compositing; history restores `contextArea` and formatting state per entry.
- **Generative fill** — history restores selection + `fillContextMaskScale` from `spatial`; Live Area barn doors remain in `contextArea` unchanged across fill steps (see recent fill context-mask work).
- **Export** (`ImageSaveService`) — unchanged; bundle is the document, Pictures folder is deliberate export.
