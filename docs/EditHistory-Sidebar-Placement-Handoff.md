# Handoff: Move Edit History to the app shell sidebar (under Mode)

**Date:** 2026-06-26  
**Branch:** `mix/v2.4.0`  
**Status:** **FIXED** (2026-06-26) — sidebar placement confirmed; history save/restore bugs addressed in follow-up commits.  
**Scope:** **Layout only.** History store, bundle I/O, restore semantics, and keyboard shortcuts are done and must not be reworked.

---

## Operator requirement (read this first)

History belongs in the **far-left column of the app window**, in a **new section directly under the Mode section** (Chat, Generate, Vision, …, **Image to Image**).

It does **not** belong in the Image-to-Image **palette column** (the second column from the left: canvas tools, Images, Workflow, Generation Parameters).

### Window columns (left → right)

```text
┌─ COL 1: App sidebar ────┬─ COL 2: I2I palettes ────┬─ COL 3: Preview ─────────┐
│ Mode                    │ Canvas tools (selection)  │ Prompt bar              │
│  Chat                   │ Images palette            │ Generated image         │
│  Generate               │ Workflow palette          │                         │
│  Vision                 │ Generation Parameters     │                         │
│  …                      │ Output Options            │                         │
│  Image to Image ●       │                           │                         │
├─────────────────────────┤                           │                         │
│ History  ← HERE         │  History must NOT appear  │                         │
│  ● step 3 [thumb]       │  anywhere in this column  │                         │
│  ○ step 2               │                           │                         │
│  ○ step 1               │                           │                         │
└─────────────────────────┴───────────────────────────┴─────────────────────────┘
```

**COL 1** = `ContentView` → `NavigationSplitView` sidebar (`List` with `Section("Mode")`).  
**COL 2** = `ImageToImageView` → `HSplitView` leading `VStack` (canvas tools + palette `ScrollView`).  
**COL 3** = `ImageToImageView` → `outputSection` (preview).

The operator has stated this multiple times. Do not interpret “left column” as the I2I palette column.

---

## Git state

| Commit | History panel location |
| --- | --- |
| `6a72e00` (current **origin**) | **Wrong:** `ImageToImageView.swift` — between `ImageToImageCanvasToolsSidebar` and the Images `ScrollView` |
| **Uncommitted local changes** | **Attempted fix:** `ContentView.swift` — `EditHistoryPanel` in sidebar `VStack` below Mode `List`; removed from `ImageToImageView` |

Local diff (not pushed):

- `Sources/Flux2App/Views/ContentView.swift`
- `Sources/Flux2App/Views/ImageToImageView.swift`
- `Sources/Flux2App/Views/EditHistoryPanel.swift`
- `docs/EditHistory.md`

**Operator tested and says History is still in the same (wrong) place.** Either:

1. They ran the **committed** build (`6a72e00`) without the local sidebar move, or  
2. The local `ContentView` approach **does not render** History in the visible sidebar (likely — see below).

**Your first job:** confirm which binary they ran, then fix layout with **visual verification** (screenshot), not “build passed.”

---

## What’s wrong on origin (`6a72e00`)

`ImageToImageView.swift` (~lines 31–45):

```swift
ImageToImageCanvasToolsSidebar(...)
Divider()
EditHistoryPanel(viewModel: ..., historyStore: ...)  // ← WRONG COLUMN
Divider()
ScrollView { /* Images, Workflow, Parameters, Output Options */ }
```

This is COL 2. The operator sees History sandwiched between canvas tools and the Images palette — exactly where they do **not** want it.

---

## What the previous agent tried (uncommitted)

### 1. Lift I2I view model to `ContentView`

```swift
@StateObject private var imageToImageViewModel = ImageGenerationViewModel(
    loadsEnvironmentProject: true,
    workflow: .imageToImage
)
// ...
case 5: ImageToImageView(viewModel: imageToImageViewModel)
```

`ImageToImageView` changed from `@StateObject private var viewModel` to `@ObservedObject var viewModel` so sidebar and detail share one `EditHistoryStore`.

**Keep this pattern** — sidebar cannot show live history without shared view model.

### 2. Put `EditHistoryPanel` in `NavigationSplitView` sidebar

```swift
NavigationSplitView {
    VStack(spacing: 0) {
        List(selection: ...) { Section("Mode") { ... } }
            .listStyle(.sidebar)
            .fixedSize(horizontal: false, vertical: true)

        if selectedTab == 5 {
            Divider()
            EditHistoryPanel(...)
        }
    }
    .frame(minWidth: 220, idealWidth: 240)
} detail: { ... }
```

### 3. Removed `EditHistoryPanel` from `ImageToImageView`

Palette column is back to: canvas tools → Divider → palette ScrollView only.

---

## Why the uncommitted fix may still look “unchanged”

Investigate these before assuming the code is correct:

| Hypothesis | What to check |
| --- | --- |
| **Stale binary** | Operator may run `Flux2App.app` or `.build/.../Flux2App` from before local edits. Rebuild: `swift build --product Flux2App` + `bin/build-mlx-metallib.sh`; prefer `bin/package-flux2app.sh` + `open Flux2App.app`. |
| **Sidebar hidden** | macOS `NavigationSplitView` sidebar can collapse. Ensure sidebar is visible (View → Sidebar or drag splitter). Consider `NavigationSplitView(columnVisibility: .constant(.all))` during dev. |
| **VStack + List doesn’t paint History** | Putting custom views *below* a sidebar `List` inside the split column is a known fragile pattern. History may be clipped, zero-height, or off-screen. **Prefer History inside the `List` as its own section.** |
| **`.fixedSize(vertical: true)` on List** | May prevent the sidebar column from allocating space to the History block below. |
| **Wrong column mistaken for success** | If History is removed from COL 2 but invisible in COL 1, operator still sees “no change” if they’re looking at an old build that still has COL 2 placement. |

---

## Recommended fix (for the next agent)

### Primary approach: `Section("History")` inside the sidebar `List`

Keep one `List` in the sidebar. When `selectedTab == 5` (Image to Image), append a second section:

```swift
List(selection: $selectedTab) {
    Section("Mode") {
        // tags 0…7 (unchanged)
    }
    if selectedTab == 5 {
        Section("History") {
            // Option A: custom rows via ForEach on history entries
            // Option B: one row containing EditHistoryPanel with listRowInsets / listRowSeparator(.hidden)
        }
    }
}
.listStyle(.sidebar)
```

Use `listRowBackground`, `listRowInsets(EdgeInsets())`, and `listRowSeparator(.hidden)` so the panel fills the section cleanly.

**Do not** rely on a sibling `VStack` below the `List` unless you verify it visually on macOS.

### Sidebar width

Mode list was `minWidth: 200`; local attempt used `220–240`. History rows use 44×44 thumbs + labels — **min ~220**, ideal **240–260**.

### Visibility rule

Show History section **only when Image to Image is selected** (`selectedTab == 5`). Empty state copy is already in `EditHistoryPanel`.

### What not to change

- `EditHistoryStore.swift` — append, prune, load, save
- `ImageGenerationViewModel+EditHistory.swift` — jump/restore, import milestone, manifest fields
- `FluxGenerationProjectBundle` history assets
- `DocumentHistoryCommands` — ⌃⌘Z / ⌃⌘⇧Z (keep `focusedSceneValue` on `ImageToImageView`)
- Selection undo stack (separate from document history)

---

## File map

| File | Role |
| --- | --- |
| `Sources/Flux2App/Views/ContentView.swift` | **Mount point:** `NavigationSplitView` sidebar; Mode `List`; History section goes here |
| `Sources/Flux2App/Views/ImageToImageView.swift` | I2I detail only; **must not** contain `EditHistoryPanel` |
| `Sources/Flux2App/Views/EditHistoryPanel.swift` | UI component (header, scroll list, Clear…, rows) |
| `Sources/Flux2App/ViewModels/EditHistoryStore.swift` | In-memory entries + thumbs; `@Published` for UI |
| `Sources/Flux2App/ViewModels/ImageGenerationViewModel.swift` | Owns `let editHistoryStore`; save/load/generate hooks |
| `Sources/Flux2App/ViewModels/ImageGenerationViewModel+EditHistory.swift` | `jumpToHistory`, `appendEditHistoryAfterGenerate`, etc. |
| `Sources/Flux2App/Views/DocumentHistoryCommands.swift` | History menu shortcuts |
| `docs/EditHistory.md` | Feature plan (logic complete); UI section updated to say app sidebar — placement still wrong in practice |

---

## View model ownership (required)

Today on **origin**, `ImageToImageView` owns `@StateObject private var viewModel`. The sidebar in `ContentView` cannot observe that store.

**Required:** single `ImageGenerationViewModel` for I2I shared by:

- `ContentView` sidebar (`EditHistoryPanel`)
- `ImageToImageView(viewModel:)` detail

Local uncommitted code already does this via `@StateObject` on `ContentView` + `@ObservedObject` on `ImageToImageView`. Preserve or equivalent (`@EnvironmentObject`).

`TextToImageView` keeps its **own** view model — do not merge.

---

## Verification checklist (mandatory)

1. `swift build --product Flux2App` && `bin/build-mlx-metallib.sh`
2. `bin/package-flux2app.sh` && `open Flux2App.app` (or run fresh debug binary — not an old copy)
3. Select **Image to Image**
4. Ensure **sidebar is visible** (left edge of window)
5. **Screenshot** — annotate three columns; confirm History is under Mode in COL 1
6. Confirm COL 2 has **no** History block between canvas tools and Images
7. Load/generate once — History rows appear in COL 1; click row restores preview
8. ⌃⌘Z / ⌃⌘⇧Z still step document history when I2I focused

Do not mark this task done without operator-visible confirmation in COL 1.

---

## Doc correction

`docs/EditHistory.md` was wrongly written during phase 2 as “I2I palette column.” It was updated in the uncommitted diff to say app sidebar. **Trust the operator, not the original phase-2 agent text.**

---

## Related commits (history feature — already shipped)

| Commit | Summary |
| --- | --- |
| `98ebed8` | Linear history, panel (wrong column), bundle persist, ⌃⌘Z shortcuts |
| `6a72e00` | Import milestone, 30-step cap, Clear History |

Only **placement** remains broken.

---

## Suggested commit message (when fixed)

```text
fix: Mount edit history under Mode in app sidebar

Move EditHistoryPanel from the I2I palette column to the NavigationSplitView
sidebar below Mode. Share ImageGenerationViewModel between ContentView and
ImageToImageView so the sidebar reflects live history.
```

---

## Out of scope

- Harbeth Adjust Mode (`docs/HarbethAdjustMode.md`)
- Upstream PR offers
- VM smoke unless operator asks
