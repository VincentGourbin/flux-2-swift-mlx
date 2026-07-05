---
type: doc
---

# CDN Provisioning Verification — Sortie A7

**Mission**: OPERATION THIMBLE TYPHOON (`mission/thimble-typhoon/01`)
**Sortie**: A7 — Verify CDN provisioning of the Phase-1 weights (R1.2, R1.3, R1.4)
**Verified**: 2026-07-05T04:58:29Z

## Mechanism used

The `acervo` CLI (v… installed at `/opt/homebrew/bin/acervo`) does not expose a
bare "does this exist on the CDN" subcommand — `acervo verify <model-id>`
(CDN mode) requires a pre-existing local staging directory to diff against,
and none exists in this sandbox for any of the three targets. Building the
Swift package to call `Acervo.availability(_:)` directly was out of scope for
this sortie (no-build sub-agent per the Parallelism Structure in
`EXECUTION_PLAN.md`).

Instead, this verification fetches the same artifact `Acervo.availability`
consults under the hood: the CDN-hosted `manifest.json` for each model, at
the path SwiftAcervo's own `VerifyCommand`/`UploadCommand`/`ShipCommand` use
(`Sources/CLI/VerifyCommand.swift` in the SwiftAcervo source checkout at
`/Users/stovak/Projects/package-collection/pkg/SwiftAcervo`):

```
<$R2_PUBLIC_URL>/models/<org>_<repo>/manifest.json
```

i.e. `curl` against `$R2_PUBLIC_URL/models/<slug>/manifest.json`, where
`<slug>` is the HF repo id with `/` replaced by `_`. `$R2_PUBLIC_URL` was
already set in the shell environment (existence-checked only, per this
repo's secrets policy — its value was never echoed). A `manifest.json` that
resolves (HTTP 200) with a well-formed `files[]`/`sha256` payload is the same
signal `Acervo.availability(_:)` treats as `.available` (manifest present +
resolvable on the CDN); this sortie does not additionally verify per-file
byte-for-byte staleness, which `Acervo.availability` also does not do at the
"is it there" layer.

Sanity check: a known-unshipped slug (`black-forest-labs_FLUX.2-dev`, the
full bf16 Dev repo — only its `qint8` subfolder is separately staged per the
EXECUTION_PLAN grounding notes) returned `HTTP 404` under the same mechanism,
confirming the check discriminates present vs. absent rather than always
returning 200.

## Results

| # | Component | Repo ID | Mechanism | Observed state | Manifest size (actual) |
|---|-----------|---------|-----------|-----------------|--------------------------|
| 1 | Transformer (qint8, Klein 4B) | `aydin99/FLUX.2-klein-4B-int8` | `curl $R2_PUBLIC_URL/models/aydin99_FLUX.2-klein-4B-int8/manifest.json` → HTTP 200 | **`.available`** | Repo total 8.86 GB (18 files); the qint8 transformer weight itself (`diffusion_pytorch_model.safetensors`) is 3.878 GB — matches the plan's "~4 GB" estimate. Repo also bundles its own text_encoder (4.80 GB) and vae (0.168 GB), which this pipeline does not consume from this repo. |
| 2 | Text encoder (iPad) | `lmstudio-community/Qwen3-4B-MLX-4bit` | `curl $R2_PUBLIC_URL/models/lmstudio-community_Qwen3-4B-MLX-4bit/manifest.json` → HTTP 200 | **`.available`** | 2.28 GB (10 files) — matches the plan's "~2 GB" estimate. |
| 3 | VAE | `black-forest-labs/FLUX.2-klein-4B` (`vae/` subfolder) | `curl $R2_PUBLIC_URL/models/black-forest-labs_FLUX.2-klein-4B/manifest.json` → HTTP 200 | **`.available`** | Full repo manifest is 15.99 GB (83 files: text_encoder 8.05 GB, transformer 7.75 GB, vae 0.168 GB, tokenizer/scheduler negligible). The `vae/` subfolder itself is **0.168 GB, not ~3 GB** — see discrepancy note below. |

All three manifests returned well-formed JSON with `primaryRepo`,
`components`, and a non-empty `files[]` array carrying real `sha256` /
`sizeBytes` entries (not placeholders) — consistent with the SwiftAcervo
0.16+ `CDNManifest` wire contract described in `CLAUDE.md` §5.

## Discrepancy note (VAE size estimate)

The requirements doc and `EXECUTION_PLAN.md` estimate the VAE at **~3 GB**.
The actual `vae/` subfolder inside `black-forest-labs/FLUX.2-klein-4B` on the
CDN is **0.168 GB** (a single `diffusion_pytorch_model.safetensors` +
`config.json`). This is **not** a provisioning gap — the manifest is present,
resolvable, and the files exist with valid checksums — it is a size-estimate
correction. Downstream sorties that budget memory/download time against a
"~3 GB VAE" figure should use 0.168 GB instead. Flagging for the supervisor;
this sortie does not modify `EXECUTION_PLAN.md` per its boundaries.

## Overall result

**COMPLETE** — all three Phase-1 components (`aydin99/FLUX.2-klein-4B-int8`,
`lmstudio-community/Qwen3-4B-MLX-4bit`, and the `vae/` subfolder of
`black-forest-labs/FLUX.2-klein-4B`) report `.available` (manifest resolves
on the CDN with well-formed contents). No `acervo ship` action was required
or taken.
