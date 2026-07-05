# Edit history — agent architecture guide

**Scope:** Image to Image document history only. Selection undo (⌘Z) is a separate stack in `SelectionUndoStore`.

**Operator intent:** [EditHistory.md](EditHistory.md)  
**Sidebar placement:** [EditHistory-Sidebar-Placement-Handoff.md](EditHistory-Sidebar-Placement-Handoff.md)

---

## Folder map (`Sources/Flux2App/EditHistory/`)

| File | Role | Change when… |
| --- | --- | --- |
| `EditHistoryStore.swift` | In-memory list, pointer, JXL caches, append/prune/load/save assets | Adjust cap, prune policy, bundle asset resolution |
| `EditHistorySidebarSection.swift` | Sidebar `List` section under Mode (COL 1) | UI layout, empty state, Clear dialog |
| `DocumentHistoryCommands.swift` | History menu, ⌃⌘Z / ⌃⌘⇧Z | Shortcut wiring |
| `ImageGenerationViewModel+EditHistory.swift` | **Public API** — jump, load, clear, adopt hooks, bundle URL | New entry points callers use |
| `ImageGenerationViewModel+EditHistoryCapture.swift` | Snapshot recipe + labels on append | New fields stored per step |
| `ImageGenerationViewModel+EditHistoryRestore.swift` | Apply entry to canvas; `EditHistoryRouteResolver` | Restore semantics, route/tool after jump |
| `../ViewModels/ImageGenerationViewModel.swift` | Owns `editHistoryStore`, restore flags, project load/save | Shared VM lifetime, smoke marker |

**Core types (do not duplicate):** `Sources/Flux2Core/EditHistoryTypes.swift`, bundle I/O in `FluxGenerationProjectBundle.swift`.

**Not here:** I2I palette column (`ImageToImageView`), selection undo (`SelectionUndoStore`), project shell (`applyProjectShell`).

---

## Data flow

```text
Append (generate / adopt / import)
  → capture settings + spatial + master JXL
  → EditHistoryStore.append (truncate redo tail, prune at 30)
  → sidebar + manifest on save

Jump (sidebar row / ⌃⌘Z)
  → load master (memory or bundle)
  → restoreFromHistoryEntry (preservingSpatialWorkflow on primary replace)
  → clear selection undo stack

Save
  → editHistoryManifestFields() → project.json history[]
  → historyAssetsForSave() → every entry’s master+thumb (memory OR existing bundle disk)
  → atomic bundle replace

Load
  → EditHistoryStore.load (thumbs only cached)
  → applyProjectShell + slots + preview
  → applyLoadedHistoryPointer (sync canvas to currentHistoryIndex when valid)
```

---

## Invariants (do not break)

1. **Single I2I view model** — `ContentView` `@StateObject` + `ImageToImageView` `@ObservedObject` share one `editHistoryStore`.
2. **Save must write every manifest entry’s JXL** — use `historyAssets(bundleRoot:)`; never write manifest entries without files.
3. **History restore must not reset spatial workflow** — `replacePrimaryReference(..., preservingSpatialWorkflow: true)`; then `updateOutpaintPadding` when needed.
4. **Guard programmatic restore** — `isRestoringEditHistory` prevents `hasActiveSelection` onChange from clobbering fill context scale.
5. **Document history ≠ selection undo** — jumping history clears selection undo; shortcuts are ⌃⌘Z vs ⌘Z.
6. **Sidebar placement** — `EditHistorySidebarSection` inside `ContentView` sidebar `List`, `selectedTab == 5` only.

---

## Smoke / verification

```bash
swift build --product Flux2App
swift test --filter EditHistoryStoreTests
bin/vm-smoke.sh   # marker must include history_steps= and history_index=
```

`F2SM_SMOKE_MARKER` detail includes `history_steps=` / `history_index=` via `editHistorySmokeSummary()`.

---

## Common tasks

| Task | Where |
| --- | --- |
| New field in history recipe | `EditHistorySpatial` / `EditHistorySettings` in Flux2Core + capture + restore |
| New append trigger | Call `appendEditHistoryAfterGenerate` / `appendEditHistoryAfterAdopt` / `maybeRecordImportHistory` |
| Change max depth | `EditHistoryStore.maxEntryCount` + sidebar footer copy |
| Fix wrong column | `ContentView` sidebar `List`, **not** `ImageToImageView` palette |
| Save loses steps | `EditHistoryStore.historyAssets(bundleRoot:)` |
| Outpaint wrong after jump | `EditHistoryRouteResolver` + `applyGenerateRouteFromHistory` |

---

## Tests

`Tests/Flux2AppTests/EditHistoryStoreTests.swift` — bundle round-trip (needs JXL), index validation, route resolver.

---

## Related docs

- [EditHistory.md](EditHistory.md) — feature plan and operator semantics  
- [AGENTS.md](../AGENTS.md) — repo-wide verification and VM smoke
