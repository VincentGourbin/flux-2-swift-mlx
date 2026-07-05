# PR #99 update — changes since Vincent's review (July 2026)

Branch: `mix/v2.4.0` on [`realnotsteve/flux-2-swift-mlx-1`](https://github.com/realnotsteve/flux-2-swift-mlx-1/tree/mix/v2.4.0)  
Upstream PR: [#99 — I2I Image Preparation](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/99)  
Tip commit: `a0f7920` (version **2.5.6.279** in private fork; review-fix commit bumps `Flux2Core.version` only)

This document separates **what we are offering upstream** from **fork-only work** that lives on the same branch but is **not** part of the Image Preparation merge proposal unless you explicitly want it split out.

---

## 1. Upstream offer — review fixes (`a0f7920`)

These address your [June 28 review](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/99#issuecomment-4825376843).

### CLI crash (blocker)

**File:** `Sources/Flux2CLI/Flux2CLI.swift`

- After loading `--images` / `--project` references, validate `refImages` is non-empty and within the model's reference cap **before** inferring output dimensions.
- Fixes `flux2 i2i "prompt" --project empty.json` (no images, no `--width`) trapping on `refImages[0]` instead of returning a validation error.

### Reference encode budget (question)

**File:** `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`

- **Primary reference (index 0):** still tracks output size for paste-back alignment, floored at 1024², now **capped at 2048²** so very large outputs do not grow VRAM ~4× vs. the old fixed 1 M-pixel budget.
- **Additional references:** stay at the historical **1024²** cap (no longer inherit the output-sized budget).

Happy to change the 2048² ceiling or gate it behind a flag if you prefer a narrower default upstream.

### Download progress (minor)

**File:** `Sources/Flux2Core/Loading/ModelDownloader.swift`

- When Hugging Face omits file sizes (`totalBytes` would be 0 or 1), progress uses **per-file weighting** instead of byte ratios that saturate at 99.9% after the first kilobyte.

### Paste-back crop (minor)

**File:** `Sources/Flux2Core/ImagePreparation.swift`

- Crop context dimensions use **rounded integral** pixel sizes when compositing the generated patch back, avoiding 1px truncation seams under float drift.

### How to verify (host)

```bash
swift build --product Flux2CLI
swift build --product Flux2App
bin/build-mlx-metallib.sh   # or CONFIG=Release bin/package-flux2app.sh

# Blocker regression: should print validation error, not trap
flux2 i2i "test" --project /path/to/project-with-no-images.json --model klein-4b
```

---

## 2. Fork-only — not in upstream offer (same branch, private intent)

Per `AGENTS.md` upstream policy, the following landed on `mix/v2.4.0` **after** the PR was opened but are **not** proposed for `VincentGourbin/flux-2-swift-mlx` unless we agree to split them:

| Commit area | Summary | Why fork-only |
|---|---|---|
| `9437b75` Upsize control | Bicubic/Lanczos resample + optional FLUX BF16 enlarge in Images palette | Personal editing workflow |
| `090e226` / `3ffaad4` SCUNet | Bundled MLX denoise/JPEG cleanup + `clean-image` CLI | Detour from core Image Preparation |
| `f839414` Outpaint fixes | Non-32 canvas padding inside `Flux2OutpaintingChain`; `Flux2ChainError` → `LocalizedError` | Could be a **separate small chains PR** if you want it |
| `bd933e6`–`c8dc8a4` Packaging | `build/Flux2App.app` staging, Project Builder relocate, single Launch Services registration | Dev environment / install plumbing |
| `98c2a36`–`0bc004a` AGENTS | Project Builder `build` workflow, commit policy | Agent docs |
| `f04c194` | Cursor rule for Project Builder | Local agent config |

None of the above is required to review Image Preparation itself.

---

## 3. What remains in PR #99 (unchanged scope)

Still the original offer documented in [docs/ImagePreparation.md](ImagePreparation.md):

- **Image Preparation** — mandatory formatting + optional barn-door Live Area + megapixel budget + composited paste-back
- **Flux2CLI** — `flux2 i2i --project` / preparation flags mirroring the app
- **Generative fill** (in-app) — rectangle repair via Flux2Chains (orthogonal to prompt edit)
- Studio ergonomics carried from the first offer (project bundles, save service, gated download UX) — happy to narrow per your scope note

VM smoke scripts (`bin/vm-smoke*.sh`) and `docs/CIRCUS-TART-DEV-ENV-API.md` remain optional reviewer plumbing; not required to run the app.

---

## 4. Suggested merge path

1. **Merge review fixes** (`a0f7920`) — blocker + progress + encode cap + crop polish.
2. **Discuss** 2048² primary-ref cap default (or CLI flag).
3. **Optional follow-ups** (separate PRs): outpaint chain hardening (`f839414`), Upsize, SCUNet — only if you want them upstream.

---

## 5. Test plan (updated)

- [x] `swift build --product Flux2App`
- [x] `swift build --product Flux2CLI`
- [x] ImagePreparation unit tests (`swift test --filter ImagePreparation`)
- [ ] Reviewer: empty-project CLI validation (see §1)
- [ ] Reviewer: large-output I2I memory behaviour with multi-reference images
- [ ] Reviewer: model download with gated repo + progress bar sanity
