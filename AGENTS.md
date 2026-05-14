# flux-2-swift-mlx — AI Agent Instructions

**Audience**: Claude Code, Gemini, Copilot, Cursor, and other AI development assistants.
**Read agent-specific files for tooling specifics**: [CLAUDE.md](CLAUDE.md), [GEMINI.md](GEMINI.md).

This file is the universal source of truth for the project. Agent-specific files extend it; they do not replace it.

---

## 1. Project Overview

`flux-2-swift-mlx` is a native Swift implementation of [Flux.2](https://blackforestlabs.ai/) image-generation models, running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-swift). This is the `intrusive-memory` fork; the original upstream is `VincentGourbin/flux-2-swift-mlx`.

The package ships two libraries and one demo executable:

| Product | Kind | Purpose |
|---|---|---|
| `FluxTextEncoders` | library | Mistral Small 3.2 + Qwen3 text encoders, Pixtral VLM, FLUX.2 embeddings |
| `Flux2Core` | library | Flux.2 image-generation pipeline (T2I, I2I, multi-image, LoRA, training) |
| `Flux2App` | executable | SwiftUI demo macOS app |

The `Flux2CLI` and `FluxEncodersCLI` executables were removed in v3.0.0 (OPERATION FAREWELL EMBRACE) when the project narrowed to a library-only fork. Downstream consumers integrate via Swift Package Manager. See `README.md` for end-user usage; this file is for agents working on the codebase.

---

## 2. Platforms and Toolchain

- **Swift tools**: 6.2 (declared in `Package.swift`)
- **Platforms**: `macOS .v26`, `iOS .v26`
- **Hardware**: Apple Silicon required (arm64; MLX has no x86_64 path)
- **CI runner**: `macos-26` (never older — Swift 6.2 + macOS 26 SDK assumed throughout)
- **iOS Simulator destination** (when needed): `'platform=iOS Simulator,name=iPhone 17,OS=26.1'` — pin an exact OS version, never `OS=latest`.

---

## 3. Repository Layout

```
Package.swift                    # SPM manifest (single source of truth for products/targets)
Sources/
  FluxTextEncoders/              # Library: text encoders + tokenizers
  Flux2Core/                     # Library: image-generation pipeline + training
  Flux2App/                      # SwiftUI demo app
Tests/
  FluxTextEncodersTests/         # Unit tests (Swift Testing) — CI-safe, no GPU/models
  Flux2CoreTests/                # Config-only tests (Swift Testing) — CI-safe
  Flux2GPUTests/                 # GPU + downloaded-model tests — local-only, not in CI
  TestHelpers/                   # Shared test utilities (TestImage, MockFlux2Pipeline)
.github/workflows/
  tests.yml                      # PR test workflow (only workflow in repo)
docs/                            # User-facing guides + screenshots + mission archives
TESTING_REQUIREMENTS.md          # Authoritative testing standard — read before adding tests
```

**Version strings** are embedded at four call sites:

- `Sources/Flux2Core/Flux2Core.swift` — `Flux2Core.version`
- `Sources/Flux2Core/Training/Training.swift` — `Training.version`
- `Sources/FluxTextEncoders/FluxTextEncoders.swift` — `FluxTextEncoders.version`
- `Sources/FluxTextEncoders/FluxTextEncoders.swift` — `MistralVersion.version` (Mistral wrapper)

All four must move together. The shipped tag is the truth; the embedded string must match the tag at release time.

---

## 4. Branching and Release Flow

- `development` is the integration branch. PRs target `development` for cross-cutting work; the periodic release PR goes from `development → main`.
- `main` is release-only. Each release is a squash commit on `main` followed by an annotated `vX.Y.Z` tag on that squash commit.
- `development` is protected (`allow_force_pushes: false` by default). The release skill temporarily lifts that protection to resync `development` to `main` after a squash merge.

The full release procedure lives in the `ship-swift-library` skill. Agents should not invent ad-hoc release flows — invoke the skill or follow it verbatim.

---

## 5. Build and Test Entry Points

The repo ships a `Makefile` that wraps the canonical `xcodebuild` invocations from §6 / `TESTING_REQUIREMENTS.md`. It is the preferred entry point for routine builds and CI-safe tests; agents that need finer-grained control can call `xcodebuild` directly with the same flags.

| Target | Purpose |
|---|---|
| `make build` | Debug build, macOS arm64 |
| `make build-ios` | Build for iOS Simulator (iPhone 17, OS 26.1) |
| `make install` | Debug build + copy CLIs and `mlx-swift_Cmlx.bundle` to `./bin` |
| `make release` | Release build + copy to `./bin` |
| `make test` | Run both CI-required suites (`FluxTextEncodersTests` + `Flux2CoreTests`) |
| `make test-fte` / `make test-core` | Individual CI-safe suites |
| `make test-gpu` | `Flux2GPUTests` — local-only, requires GPU + downloaded weights |
| `make lint` | `swift format -i` over `Sources/` + `Tests/` (no `.swift-format` config — first run rewrites a lot) |
| `make lint-check` | Same scan, non-mutating (CI-friendly) |
| `make resolve` | Resolve SPM dependencies into `.spm` |
| `make clean` | Remove `./bin`, `.spm`, and `Flux2Swift-*` DerivedData |

Run `make help` for the full list. CI (`.github/workflows/tests.yml`) currently still calls `xcodebuild` directly — keep CI and the Makefile in sync when changing flags.

---

## 6. Testing Standard

Authoritative document: [TESTING_REQUIREMENTS.md](TESTING_REQUIREMENTS.md). Summary:

| Target | CI | Local | Requires |
|---|---|---|---|
| `FluxTextEncodersTests` | Yes | Yes | Nothing — no GPU, no downloads |
| `Flux2CoreTests` | Partial (config only) | Yes | Nothing for config; GPU + weights for inference |
| `Flux2GPUTests` | No | Yes | Apple Silicon, ≥16 GB RAM, downloaded model weights |

**Required CI status checks** (configured in `development` and `main` branch protection):

- `Test FluxTextEncoders (macOS)`
- `Test Flux2Core — Config Only (macOS)`

If you rename a CI job, the branch-protection contexts must be updated in the same change — otherwise PRs become unmergeable. Use:

```bash
gh api --method PUT repos/intrusive-memory/flux-2-swift-mlx/branches/<branch>/protection --input <json>
```

The full test suite (including `Flux2GPUTests`) runs locally only. Anything taking longer than ~5 seconds on CI is either doing I/O it shouldn't or belongs in `Flux2GPUTests`.

Tests use **Swift Testing** (`@Test`, `Issue.record`), not XCTest. The migration was completed in v2.7.0 (OPERATION MARCHING RELAY).

---

## 7. Common Agent Tasks

### Adding a test
1. Decide the target by what the test needs: no GPU/weights → `FluxTextEncodersTests` or `Flux2CoreTests`; GPU/weights → `Flux2GPUTests`.
2. Use Swift Testing syntax (`@Test`, `#expect`, `Issue.record`). Don't add XCTest dependencies.
3. Apply `.timeLimit(...)` to anything that could hang.

### Bumping a dependency
1. Edit `Package.swift` — keep all deps on `.upToNextMajor(from: "X.Y.Z")` form. Never use `.exact`, never use `.path()`.
2. Verify the dependency has a published GitHub release/tag matching what you pin.
3. Resolve and test locally before pushing.

### Bumping the project version
1. Update all three `version` strings in source (see §3).
2. Update README install snippets if they reference a tag.
3. Ship via the `ship-swift-library` skill. The version bump rides with the release PR.

### Touching CI workflows
1. The repo has exactly one workflow: `.github/workflows/tests.yml`. There is **no `release.yml`** — see §8.
2. Pin every `uses:` to the latest major (`@v6`, `@v5`, etc.) — older majors trigger Node 16/20 deprecation warnings.
3. Job names are load-bearing — they're referenced by branch-protection contexts. Don't rename without updating protection.

---

## 8. Build Artifact Distribution

This fork **does not produce release binaries**. Only `tests.yml` exists; there is no `release.yml` to build/upload tarballs or zipped CLI/app bundles.

Consumers integrate via SwiftPM:

```swift
.package(url: "https://github.com/intrusive-memory/flux-2-swift-mlx", .upToNextMajor(from: "3.0.0"))
```

The upstream `VincentGourbin/flux-2-swift-mlx` does publish App and CLI binaries. The README links pointed there historically; they were corrected to this fork's release URLs in v2.7.0, but the fork itself does not currently host binary artifacts. With the CLI executables removed in v3.0.0, the fork is now SwiftPM-only.

---

## 9. Documentation Index

- [README.md](README.md) — End-user / library-consumer entry point
- [TESTING_REQUIREMENTS.md](TESTING_REQUIREMENTS.md) — Authoritative testing standard
- [docs/CLI.md](docs/CLI.md) — `flux2` CLI reference
- [docs/TextEncoders.md](docs/TextEncoders.md) — `FluxTextEncoders` library and CLI
- [docs/LoRA.md](docs/LoRA.md) — Loading LoRA adapters
- [docs/examples/TRAINING_GUIDE.md](docs/examples/TRAINING_GUIDE.md) — LoRA training
- [docs/CustomModelIntegration.md](docs/CustomModelIntegration.md) — Adding custom MLX models
- [docs/Flux2App.md](docs/Flux2App.md) — Demo app guide
- [docs/missions/](docs/missions/) — Active mission supervisor state
- [docs/complete/](docs/complete/) — Archived mission briefs and execution plans
- [REQUIREMENTS-instrumentation.md](REQUIREMENTS-instrumentation.md) — Telemetry event surface for flux; reference implementation of the §11 cross-library chokepoint convention

---

## 10. Universal Critical Rules

These apply to every agent. Agent-specific rules live in [CLAUDE.md](CLAUDE.md) / [GEMINI.md](GEMINI.md).

1. **Never commit directly to `main`.** Releases reach `main` only via squash-merged PRs from `development`.
2. **Never delete `development`.** It is long-lived. The release skill resyncs it onto `main` after each release.
3. **Never edit `Package.resolved` by hand.** It is gitignored and regenerated on resolve.
4. **Never add `.path()` dependencies to `Package.swift`.** Every dep must reference a published GitHub release, pinned `.upToNextMajor(from: ...)`.
5. **Keep all three version strings in lockstep** (see §3). A mismatch ships a binary whose self-reported version disagrees with its tag.
6. **Read files before editing.** Don't push speculative diffs.
7. **Don't rename CI job names without updating branch protection contexts.** It deadlocks merges.
8. **Don't add release tarballs by hand to GitHub releases.** The skill explicitly forbids it; if/when a `release.yml` is added, CI owns binary production.
9. **Follow agent-specific instructions** — see [CLAUDE.md](CLAUDE.md) and [GEMINI.md](GEMINI.md).

---

## 11. Telemetry Chokepoint Convention (cross-library)

This section describes the **shared instrumentation pattern** used across the intrusive-memory ML stack (`SwiftTuberia`, `flux-2-swift-mlx`, `Produciesta`, `Vinetas`, and downstream libraries that own diffusion / transformer / audio kernels). Other libraries adopting this pattern should mirror the chokepoint names below with their own short prefix so a single Vinetas-style adapter can route every event without per-library special cases.

**Core principle.** Instrument *boundaries*, not internals. One event when each major phase starts, one when it ends, plus a single side-channel signal when something goes numerically wrong at exit. Per-step / per-block / per-head detail is **deferred until a real failure points the agent at the region**. The default cost of telemetry stays near zero so it ships enabled by default.

### 11.1 Canonical chokepoint catalog

Every ML library in this ecosystem should emit (at minimum) the following boundary categories. Replace `flux` with the library's short prefix.

| Category | Event case (PascalCase) | Adapter sink phase (snake_case) | Memory snapshot? |
|---|---|---|---|
| **Lifecycle — construct** | `pipelineInit` | `<lib>_pipeline_init` | no |
| **Lifecycle — tear down** | `pipelineDispose` | `<lib>_pipeline_dispose` | no |
| **Resource load — complete** | `weightLoadComplete(component:)` | `<lib>_weight_load_complete_<component>` | **yes** |
| **Configuration — runtime configured for one run** | `schedulerConfigured` | `<lib>_scheduler_configured` | no |
| **Single-shot compute — complete** | `<phase>Complete(...stat...)` (e.g. `textEncodeComplete`, `vaeDecodeComplete`) | `<lib>_<phase>_complete[_<subkind>]` | **yes** for the terminal-output phase only |
| **Sustained compute loop — enter** | `<loop>LoopStart(variant:totalSteps:...)` | `<lib>_<loop>_loop_start` | **yes** |
| **Sustained compute loop — exit** | `<loop>LoopEnd(variant:totalSteps:completedSteps:finalStat:...)` | `<lib>_<loop>_loop_end` | **yes** |
| **Side-channel — numerical anomaly** | `numericalAnomaly(phase:kind:stat:)` | `<lib>_anomaly_<kind>` | no |
| **Side-channel — error** | `errorThrown(phase:errorDescription:)` | `<lib>_error_<phase>` | no |
| **Side-channel — cancellation** | `generationCancelled(stepIndex:)` | `<lib>_cancelled` | no |

`flux-2-swift-mlx` uses the prefix **`flux`** and instantiates this catalog as: `pipelineInit`, `pipelineDispose`, `weightLoadComplete`, `textEncodeComplete`, `schedulerConfigured`, `denoiseLoopStart`, `denoiseLoopEnd`, `vaeDecodeComplete`, `numericalAnomaly`, `errorThrown`, `generationCancelled`. The full event surface and emission spec live in [REQUIREMENTS-instrumentation.md](REQUIREMENTS-instrumentation.md).

### 11.2 Naming rules

- **Event case names**: PascalCase noun + lifecycle suffix. Lifecycle suffix is one of: `Init`, `Dispose`, `Configured`, `Start`, `End`, `Complete`. Prefer `Complete` for single-shot operations; reserve `Start`/`End` for sustained loops where you genuinely need both boundaries.
- **Adapter sink phase strings**: lowercase snake_case `<lib>_<noun>_<lifecycle>`. Subkind suffixes via underscore: `<lib>_weight_load_complete_<component>`, `<lib>_anomaly_<kind>`, `<lib>_error_<phase>`.
- **Library prefix**: short, lowercase, no underscores. Canonical assignments: `flux` (this repo), `tuberia` (SwiftTuberia), `produciesta` (Produciesta), `vinetas` (Vinetas host), `secuencia` (SwiftSecuencia), `acervo` (SwiftAcervo).
- **Don't pair `Start` + `Complete`.** Pick one. Durations are carried on the closing event (`durationSeconds`). The only valid `Start`+`End` pairing is for **sustained loops** where the loop body itself is not instrumented and the host needs an explicit memory snapshot at both edges.
- **Don't invent per-step events at this layer.** Per-step / per-block / per-attention-head detail is a *follow-up iteration* triggered by a real anomaly, not part of the baseline surface.

### 11.3 Stat sampling discipline

- Each boundary `*Complete` / `*End` event that carries a tensor stat samples it **exactly once** via the shared `TuberiaTensorStat.sample(...)` (from SwiftTuberia ≥ 0.7.0). Do not define a per-library variant.
- A boundary that samples a stat **must** also feed it through an anomaly check helper: if `hasNaN || hasInf || max.magnitude > TuberiaTensorStat.defaultOutOfRangeThreshold` or `(mean.magnitude < 1e-6 && std < 1e-6)`, emit a `numericalAnomaly` side-channel event alongside the boundary event. One signal, not a per-step stream.
- Total stat samples per request through the entire stack should remain **single-digit** in the happy path. Flux's clean T2I generate samples three (text-encode output, denoise-loop final latent, VAE output). Other libraries should target similar counts.

### 11.4 Side-channel discipline

- **`errorThrown` fires immediately before every `throw`.** No exceptions. If a library has 14 throw sites, it has 14 emission sites. The `phase:` field discriminates them.
- **`numericalAnomaly` fires alongside the boundary event whose stat triggered it**, not in place of it. The boundary still emits; the anomaly is additive context.
- **`generationCancelled` fires at every cancellation check.** `stepIndex` is optional (`Int?`) — pre-loop check sites pass `nil`; in-loop sites pass the current index.

### 11.5 Reporter / setter shape

Every top-level type that owns kernel work exposes:

```swift
public func setTelemetry(_ reporter: (any <Lib>TelemetryReporter)?)
```

backed by an `OSAllocatedUnfairLock<(any <Lib>TelemetryReporter)?>` (because most of these types are `@unchecked Sendable` classes that can't safely hold a plain mutable property). The reporter protocol is:

```swift
public protocol <Lib>TelemetryReporter: Sendable {
    func capture(_ event: <Lib>TelemetryEvent) async
}

public struct Noop<Lib>TelemetryReporter: <Lib>TelemetryReporter {
    public init() {}
    public func capture(_ event: <Lib>TelemetryEvent) async {}
}
```

Hosts (Vinetas) call `setTelemetry` exactly once on the top-level type; that type is responsible for propagating the reporter to every owned subcomponent (text encoders, schedulers, weight loaders, transformers, VAEs, etc.). Subcomponents do not expose setters to the host directly.

### 11.6 Adapter routing (host side, e.g. Vinetas)

Host adapters use an **exhaustive switch with no `default:` arm** so adding an event in any library forces a host-side update. Memory-snapshot routing (per Vinetas `docs/INSTRUMENTATION_PLAN.md` §3.1):

- Route through `captureWithMemorySnapshot`: every `weightLoadComplete`, every `*LoopStart` / `*LoopEnd`, and the terminal-output `*Complete` event (e.g. `vaeDecodeComplete` in flux).
- Route through plain `capture`: everything else (lifecycle, configuration, single-shot non-terminal completes, side-channels).

### 11.7 What goes in REQUIREMENTS-instrumentation.md per library

Each library that adopts this pattern should ship a `REQUIREMENTS-instrumentation.md` in its repo root containing, at minimum:

1. **Design principle** — one paragraph stating "boundaries, not internals" and any deviations.
2. **Public types** — the `<Lib>TelemetryEvent` enum and `<Lib>TelemetryReporter` protocol.
3. **Injection points** — every top-level type that grows a `setTelemetry` setter, plus the propagation rules.
4. **Per-event emission spec** — a table mapping each event to the symbol or callsite where it fires, plus the per-generation count.
5. **Adapter mapping** — the snake_case sink phase string for each case and whether it carries a memory snapshot.
6. **Tests** — at least: boundary-sequence test, noop-overhead test (±2% wall-clock bound), anomaly side-channel test, error-path test, lock-contention test.
7. **Out of scope** — explicit list of per-step / per-block detail the library is *not* instrumenting this iteration.

Flux's [REQUIREMENTS-instrumentation.md](REQUIREMENTS-instrumentation.md) is the reference implementation of this template.

---

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **CLI tools, scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

---

**Last updated**: 2026-05-14 (v3.2.0) — added §11 telemetry chokepoint convention
