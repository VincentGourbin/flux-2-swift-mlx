# Agent guide for `flux-2-swift-mix`

Fork of [flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) focused on **single-image, prompt-based editing** in Flux2App. See [docs/ImagePreparation.md](docs/ImagePreparation.md) for the headline workflow and **Operator intent** (how barn doors are actually used).

**Private personal repo** (`origin`). Day-to-day work lands here first. Vincent shares the MIT framework; we share **selected** work back — reciprocal, but not on autopilot (not unpaid FluxForge product development).

Single developer, macOS, one machine.

## Working style (collaboration)

**Ask freely; confirm before any non-trivial decision (rule, 2026-06-27).** This
developer *prefers* to be consulted — surface choices as questions and wait for
an answer rather than picking a direction and running with it. Lots of small
questions are welcome. Silently making a string of decisions — especially
mid-task, e.g. pushing through repeated failed attempts — is the thing to avoid.
When something is ambiguous, has trade-offs, or would commit real time/effort,
stop and ask. Default to *less* autonomous action here than the usual agent
default.

---

## Repos and remotes

| Remote | Repo | Role |
| --- | --- | --- |
| `origin` | `realnotsteve/flux-2-swift-mix` | **Private** working copy and backup |
| `fork` | `realnotsteve/flux-2-swift-mlx-1` | **Public GitHub fork** of Vincent's repo — push here only when deliberately opening/updating an upstream PR |
| `upstream` | `VincentGourbin/flux-2-swift-mlx` | Pull/rebase for framework updates. Push disabled locally (`no_push`) — use `fork` + PR instead |

**Legacy:** `realnotsteve/flux-2-swift-mlx` still exists but is **not** attached to GitHub's fork network (`parent: null`), so cross-repo PRs from it fail. Upstream offers use **`flux-2-swift-mlx-1`** instead (created 2026-06-23).

Feature branch: `mix/v2.4.0`. Active upstream offer: **[PR #99](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/99)** (supersedes closed #98).

### Upstream policy (2026-06-22)

Reciprocal open source, bounded scope: pull his framework improvements freely; push **only what you choose** to offer. The goal is good citizenship in a shared MIT tree — not building his commercial app surface for free.

- **Default:** new work stays on private `origin`. Agents do **not** auto-open PRs, auto-push to `fork`, or withdraw offers without you saying so.
- **Offer upstream when you ask:** framework fixes, general-purpose library/CLI improvements, tests, docs that belong in the shared repo.
- **Keep private unless you ask:** personal editing workflow polish, app UX tuned to your habits.
- **Do not offer upstream** (duplicates Vincent's or parallel product work — even if the code is good):
  - **FluxForge Studio parity** — project files, save/export services, app shell / window workflow, download UX tuned like a shipped studio app, anything that is product surface rather than framework.
  - **Circus / Tart dev environment** — `circus` CLI integration, VM smoke orchestration, `docs/CIRCUS-TART-DEV-ENV-API.md`, profile/mount contracts. Circus owns that stack (`utility-be-circus`); this repo only *consumes* it for smoke tests.
- **PR #99** ([Image Preparation](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/99)) supersedes closed #98. Offer branch `mix/v2.4.0` lives on `fork` (`flux-2-swift-mlx-1`). It predates the carve-outs below and includes some studio plumbing — do not add more of that category to upstream offers without explicit intent.
- **Rebase freely** onto `upstream/main` for his MIT changes (`Flux2Chains`, etc.). Do not remove upstream library features when extending the fork.

---

## Repo boundary

**This agent writes only to `flux-2-swift-mix`.** No files, commits, or handoff docs in
`utility-be-circus`, `comfy-be-nodes`, `wp-be-*`, or any other repo — ever.

Cross-app coordination goes through the human (relay a copy block), or read the other
repo on ORIGIN after its owning agent commits. Do not build bash shims for another app's
CLI while waiting for the real thing (e.g. no `bin/circus` in this repo).

| Repo | Owner |
| --- | --- |
| `flux-2-swift-mix` | This agent |
| `utility-be-circus` | Circus agent |
| Everything else | Its own agent or the human |

Circus / Tart VM environment is documented here for smoke tests; Circus source code is
not in scope. Frozen consumer contract: [docs/CIRCUS-TART-DEV-ENV-API.md](docs/CIRCUS-TART-DEV-ENV-API.md).

---

## Verification

### Project Builder / daily build

This repo is registered in **Project Builder** (`utility-be-project-builder`).

| User says | Agent does |
| --- | --- |
| **`build`** | Read `~/GitHub/utility-be-project-builder/Configs/flux-2-swift-mix.json`, honor `run_preferences` (clamped by `capabilities`), run enabled `commands` from repo root |
| **`compile`** | Normal Swift compile only — **do not** read the Project Builder config |

Full config schema and agent contract: [utility-be-project-builder/Configs/AGENTS.md](../utility-be-project-builder/Configs/AGENTS.md).

**Typical `build` pipeline** (when all four `run_preferences` are enabled):

1. **Compile:** `CONFIG=Release bin/package-flux2app.sh` → `build/Flux2App.app` (gitignored staging)
2. **Relocate:** copy to `/Applications/Flux2App.app` (remove destination first when overwrite is on)
3. **Launch:** `open /Applications/Flux2App.app`

**Staging vs canonical install:**

| Path | Role |
| --- | --- |
| `build/Flux2App.app` | Gitignored compile output — exists only between compile and relocate |
| `/Applications/Flux2App.app` | **Canonical installed app** — the only copy to run day to day |

The repo keeps the icon source (`Assets/AppIcon/Flux2App-1024.png`), a committed `Supporting/Flux2App/Info.plist` template, and the thin `bin/package-flux2app.sh` wrapper. There is **no** `Flux2App.app` in the repo root. **Relocate deletes `build/Flux2App.app` after install** so macOS only registers one copy in `/Applications`.

**Agent notes:**

- If `run_preferences` is absent → default to **compile only** when `capabilities.compile` is true.
- If every `run_preferences` flag is `false` → `build` runs nothing; say so and ask whether to fall back to compile-only or have the user re-check Project Builder.
- Before relocate with overwrite, **quit** a running `Flux2App` (`⌘Q` or `pkill -x Flux2App`) so the new binary loads on launch.
- `bin/package-flux2app.sh` already runs `bin/build-mlx-metallib.sh` and `bin/build-app-icon.sh` — no separate metallib step needed for a packaged app.

Local Cursor rule (when present): `.cursor/rules/project-builder-build.mdc`.

### Compile (minimum bar)

Use this section when the user says **`compile`**, or when Project Builder compile is the only enabled step.

```bash
swift package resolve
swift build --product Flux2App
swift build --product Flux2CLI
```

**MLX Metal shaders:** bare `swift build` does **not** compile `mlx.metallib` ([mlx-swift README](https://github.com/ml-explore/mlx-swift#swiftpm)). After every clean build (host or VM), run:

```bash
bin/build-mlx-metallib.sh
```

That `xcodebuild`s mlx-swift’s `Cmlx` scheme and copies `default.metallib` beside `Flux2App` as `mlx.metallib` (runtime loads it via `@loader_path`).

**App icon + `.app` bundle** (Dock/Finder icon; bare `swift build` binary has no icon):

```bash
bin/package-flux2app.sh
# → build/Flux2App.app (gitignored staging; open /Applications/Flux2App.app after relocate)
```

`CONFIG=Release bin/package-flux2app.sh` matches Project Builder’s compile command. The package script runs metallib + icon steps automatically and stamps version from `Flux2Core.version`.

Source art: `Assets/AppIcon/Flux2App-1024.png`. `bin/build-app-icon.sh` regenerates `AppIcon.icns`.

### Smoke testing in the Tart VM

VM smoke uses the **`circus` CLI** (Circus.app must be running). Set
`CIRCUS_PROFILE=tart-av-dev` (default in `bin/vm-smoke-circus.sh`) to match the
profile picked at Circus launch. Contract:
[docs/CIRCUS-TART-DEV-ENV-API.md](docs/CIRCUS-TART-DEV-ENV-API.md).

Use the **Tart Virtual Mac** guest when UI verification is needed beyond compile checks.

The VM has **Xcode 26.5** and a repo copy at `~/GitHub/flux-2-swift-mix` — it can build natively.

**One-time VM Xcode setup** (after copying Xcode from the host):

```bash
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain   # required for bin/build-mlx-metallib.sh
```

Without the Metal Toolchain component, `xcodebuild` fails with `cannot execute tool 'metal'`.

**One-shot host → VM smoke** (`circus ensure-ready` → `put` → `exec` → `wait` → `get`):

```bash
bin/vm-smoke.sh
# → /tmp/flux2-smoke.png
# Waits for /tmp/flux2-smoke-ready (F2SM_SMOKE_MARKER) with first line `ok`.
# Requires Circus.app running; profile tart-av-dev with flux2-model-cache mount.
```

Fixture: `Tests/Fixtures/VMSmoke/project.json` (self-contained I2I project with embedded reference PNG).

**Full Circus battery** (models + UI + history bundle + CLI I2I + generative fill):

```bash
bin/vm-smoke-full.sh
# → /tmp/flux2-smoke.png, flux2-smoke-history.png, flux2-smoke-i2i.png, flux2-smoke-fill.png
# History fixture: Tests/Fixtures/VMSmokeHistory/VMSmokeHistory.flux2project (v3 + Import step).
# Regenerate fixture when needed: Flux2SmokeFixture export-history-bundle … (requires brew install jpeg-xl).
```

**Launch hooks** (agent / smoke):

| Env var | Purpose |
| --- | --- |
| `F2SM_PROJECT` | Absolute path to a generation project JSON. Opens on launch instead of last-saved project. |
| `F2SM_SMOKE_MARKER` | Optional. When set with `F2SM_PROJECT`, writes a marker file on load: first line `ok` or `error`, then detail. |
| `F2SM_MODELS_DIR` | Optional. Overrides the model cache directory (Flux2App + Flux2CLI). Auto-detected when `/Volumes/My Shared Files/flux2-model-cache` exists. |

```bash
F2SM_PROJECT=/tmp/flux2-smoke/VMSmoke/project.json \
F2SM_SMOKE_MARKER=/tmp/flux2-smoke-ready \
F2SM_MODELS_DIR="/Volumes/My Shared Files/flux2-model-cache" \
./Flux2App &
```

### Models in the Tart VM

**Circus** (Tart Virtual Machine → **Shared Directories** grid) builds `--dir` flags at launch. Default row:

| Include | Full Path | Shortcut |
| --- | --- | --- |
| ✓ | `/Users/drwevans/Library/Caches/models` | `flux2-model-cache` |

Tart flag: `--dir=flux2-model-cache:/Users/drwevans/Library/Caches/models:ro`

Guest path: `/Volumes/My Shared Files/flux2-model-cache` (must keep shortcut **`flux2-model-cache`** unless you also set `F2SM_MODELS_DIR`).

Flux2App auto-detects that guest path. `bin/vm-smoke.sh` requires it and passes `F2SM_MODELS_DIR`.

**Model cache probes** (mount up; lists what weights are visible):

```bash
bin/vm-smoke-models.sh
```

**CLI generate smoke** (Klein 4B I2I when `FLUX.2-klein-4B-klein4b-8bit` exists on the shared cache):

```bash
bin/vm-smoke-generate.sh
# → /tmp/flux2-smoke-i2i.png  (exit 2 = skipped, weights missing)
```

Uses `Tests/Fixtures/VMSmoke/reference.png`, `--model klein-4b`, 4 steps at 512×384. Download Klein weights to the **host** cache first (`flux2 download --model klein-4b`). Dev bf16 will not fit the 64 GB guest.

**CLI generative-fill smoke** (Klein 4B inpaint + Qwen3.5 4-bit VLM when weights exist on the shared cache):

```bash
bin/vm-smoke-generate-fill.sh
# → /tmp/flux2-smoke-fill.png  (exit 2 = skipped, Klein or Qwen3.5 missing)
```

Uses `Tests/Fixtures/VMSmoke/reference.png` + `fill-mask.png`, `flux2 inpaint` with `--enrich-prompt-with-vlm --qwen35-variant 4bit --intent modify`. If Qwen3.5 4-bit is absent, the script tries a host download when `HF_TOKEN` is set; otherwise pre-cache via `flux2 test-qwen35 smoke --variant 4bit --no-think --max-tokens 8` on the **host** (populates the shared cache).

Shares are fixed at **launch** — shut down and **Start** from Circus after changing the list. A VM already running without the share will not see host weights until restart.

**Manual recipe** (same primitives as `bin/vm-smoke.sh`):

```bash
swift build --product Flux2App
bin/build-mlx-metallib.sh
circus ensure-ready --profile tart-av-dev
circus put --host .build/arm64-apple-macosx/debug/Flux2App --guest /tmp/Flux2App
circus put --host .build/arm64-apple-macosx/debug/mlx.metallib --guest /tmp/mlx.metallib
circus exec --timeout 30 -- 'export F2SM_PROJECT=/tmp/.../project.json; cd /tmp && open -n ./Flux2App'
sleep 5
circus exec --timeout 30 -- screencapture -x /tmp/flux2-smoke.png
circus get --guest /tmp/flux2-smoke.png --host /tmp/flux2-smoke.png
```

**In-VM build** (same metallib step required):

```bash
ssh tart-virtual-mac
cd ~/GitHub/flux-2-swift-mix
swift build --product Flux2App
bin/build-mlx-metallib.sh
.build/arm64-apple-macosx/debug/Flux2App &
```

`Flux2App` and `mlx.metallib` must live in the same directory.

Models are not bundled — expect “Not Loaded” on first launch unless weights are cached in the VM’s `~/Library/Caches/models/`.

### VM rule — ask first, then stop, don't thrash

**Ask-first (rule, 2026-06-27):** Before running *any* Circus / Tart VM command — `bin/vm-smoke*.sh`, any `circus …` invocation, or anything else that drives the VM — STOP and tell the user exactly what you intend to run and why, then wait for the go-ahead. This pre-flight confirmation is separate from (and precedes) the "stop, don't thrash" guidance below, which governs what to do once a VM action is already underway and goes wrong.

**If you have trouble in the virtual machine, stop immediately** and report what you observed (connection error, hung command, unexpected UI state, missing tunnel). Do **not**:

- Retry the same SSH/WebDriver command in a loop
- Hammer `open`, `click`, or navigation without new information
- Work around VM failures with repeated host-only attempts while claiming VM verification

Pause so the user can see what's happening (VM off, tunnel down, app not installed in VM, etc.) and decide the next step.

---

## Fork vs upstream features

| Fork (keep) | Upstream (keep after rebase) |
| --- | --- |
| Image Preparation (formatting, Live Area, megapixel budget) | `Flux2Chains` (inpaint, outpaint, subject mask) |
| Edit history (linear document history, bundle persist, sidebar UI) | Small Decoder VAE, profiling CLI |
| `ImageSaveService`, project files | VLM training scoring, mlx 0.31.4 pin |
| `ImageCoordinateMapper`, compositing | |

Dormant `processArea` plumbing in prompt-edit mode is intentional — reserved for future paste-back UI, not barn-door Live Area.

### Flux2App product intent (prompt edit)

The primary workflow is **Image Preparation**: barn doors + megapixel budget + prompt I2I + composite back. Barn doors define a **minimum sufficient scene** (light path, room volume, shadow landing zones) and **megapixel/aspect economics** — not tight selection of edit targets. See [docs/ImagePreparation.md — Operator intent](docs/ImagePreparation.md#operator-intent-how-barn-doors-are-actually-used).

**Generative fill** (rectangle over a blemish, optional Qwen3.5 VLM, RePaint chain) is the in-app surgical path. Full inpaint intent control stays **CLI** (`flux2 inpaint`). See [docs/ImagePreparation.md — Generative fill](docs/ImagePreparation.md#generative-fill).

**Edit history** (sidebar under Mode, ⌃⌘Z document stepping, bundle JXL steps) — agent file map and invariants: [docs/EditHistory-Architecture.md](docs/EditHistory-Architecture.md).

**Do not propose** unless the user asks: Apple Vision integration, polygon mask UI, or FluxForge Studio parity.

---

## Commits

Commits may include `Co-authored-by: Cursor <cursoragent@cursor.com>`. Keep that trailer when amending; do not strip it.
