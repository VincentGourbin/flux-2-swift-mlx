# Agent guide for `flux-2-swift-mix`

Fork of [flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) focused on **single-image, prompt-based editing** in Flux2App. See [docs/ImagePreparation.md](docs/ImagePreparation.md) for the headline workflow.

Single developer, macOS, one machine.

---

## Repos and remotes

| Remote | Repo | Role |
| --- | --- | --- |
| `origin` | `realnotsteve/flux-2-swift-mix` | This fork (standalone GitHub repo) |
| `fork` | `realnotsteve/flux-2-swift-mlx` | GitHub fork used for upstream PRs |
| `upstream` | `VincentGourbin/flux-2-swift-mlx` | Main project |

Feature branch: `mix/v2.4.0`. Upstream PR: [#98](https://github.com/VincentGourbin/flux-2-swift-mlx/pull/98).

After rebasing onto `upstream/main`, this branch carries **both** fork features (Image Preparation, projects, save service) **and** upstream additions (`Flux2Chains`, small-decoder VAE, profiling, etc.). Do not remove upstream features when extending the fork.

---

## Verification

### Build (minimum bar)

```bash
swift package resolve
swift build --product Flux2App
swift build --product Flux2CLI
```

**MLX Metal shaders:** `swift build` alone does **not** compile `mlx.metallib` ([mlx-swift README](https://github.com/ml-explore/mlx-swift#swiftpm)). After every clean build (host or VM), run:

```bash
bin/build-mlx-metallib.sh
```

That `xcodebuild`s mlx-swift’s `Cmlx` scheme and copies `default.metallib` beside `Flux2App` as `mlx.metallib` (runtime loads it via `@loader_path`).

### Smoke testing in the Tart VM

Use the **Tart Virtual Mac** (`tart-virtual-mac` SSH alias; `tart get "Tart Virtual Mac"`) when UI verification is needed beyond compile checks.

The VM has **Xcode 26.5** and a repo copy at `~/GitHub/flux-2-swift-mix` — it can build natively.

**One-time VM Xcode setup** (after copying Xcode from the host):

```bash
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain   # required for bin/build-mlx-metallib.sh
```

Without the Metal Toolchain component, `xcodebuild` fails with `cannot execute tool 'metal'`.

**One-shot host → VM smoke** (build on host, deploy, open fixture project, verify marker, screenshot):

```bash
bin/vm-smoke.sh
# → /tmp/flux2-smoke.png
# Waits for /tmp/flux2-smoke-ready (F2SM_SMOKE_MARKER) with first line `ok`.
```

Fixture: `Tests/Fixtures/VMSmoke/project.json` (self-contained I2I project with embedded reference PNG).

**Launch hooks** (agent / smoke):

| Env var | Purpose |
| --- | --- |
| `F2SM_PROJECT` | Absolute path to a generation project JSON. Opens on launch instead of last-saved project. |
| `F2SM_SMOKE_MARKER` | Optional. When set with `F2SM_PROJECT`, writes a marker file on load: first line `ok` or `error`, then detail. |

```bash
F2SM_PROJECT=/tmp/flux2-smoke/VMSmoke/project.json \
F2SM_SMOKE_MARKER=/tmp/flux2-smoke-ready \
./Flux2App &
```

**Manual recipe:**

```bash
swift build --product Flux2App
bin/build-mlx-metallib.sh
scp .build/arm64-apple-macosx/debug/{Flux2App,mlx.metallib} tart-virtual-mac:/tmp/
ssh tart-virtual-mac 'cd /tmp && ./Flux2App &'
sleep 5
ssh tart-virtual-mac 'screencapture -x /tmp/flux2-smoke.png'
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

### VM rule — stop, don't thrash

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
| `ImageSaveService`, project files | Small Decoder VAE, profiling CLI |
| `ImageCoordinateMapper`, compositing | VLM training scoring, mlx 0.31.4 pin |

Dormant `processArea` plumbing is intentional — reserved for future mask-based inpaint (e.g. Photoshop filter), not barn-door Live Area.

---

## Commits

Commits may include `Co-authored-by: Cursor <cursoragent@cursor.com>`. Keep that trailer when amending; do not strip it.
