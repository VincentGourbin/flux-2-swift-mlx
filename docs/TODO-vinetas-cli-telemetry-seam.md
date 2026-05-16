# RESOLVED — Process-wide Telemetry Seam for CLI Hosts

**RESOLVED in commit**: see `git log --oneline -1` after the implementation commit on `development`.
**Implemented**: 2026-05-15 (OPERATION WIRETAP DARKROOM)

---

# TODO — Process-wide Telemetry Seam for CLI Hosts

**Filed by**: SwiftVinetas — OPERATION WIRETAP DARKROOM (2026-05-15)
**Issue surfaced in**: `SwiftVinetas/Sources/VinetasCLICore/Telemetry/CLITelemetryBootstrap.swift`

## Background

The SwiftVinetas CLI host (`vinetas generate --telemetry …`) wants to install a `Flux2TelemetryReporter` adapter so that Flux2 events are interleaved into the unified JSONL trace alongside Vinetas, PixArt, Tuberia, and Acervo events.

`Flux2WeightLoader.setTelemetry(_:)` (static) at `Sources/Flux2Core/Loading/WeightLoader.swift:23` works from the CLI — weight-load events are captured. ✅

`Flux2Pipeline.setTelemetry(_:)` at `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift:83` is **instance-bound** — the pipeline instance lives privately inside `SwiftVinetas/Sources/SwiftVinetas/Engine/Flux2Engine.swift`. The CLI bootstrap has no reference to that instance, so it cannot install a pipeline-level reporter. Today, **denoise loop, text-encode, VAE-decode, and most pipeline events are not reachable from the CLI**.

## What would unblock the CLI

Any one of the following options would let `CLITelemetryBootstrap.enable(...)` capture full Flux2 pipeline telemetry:

1. **Process-wide reporter shared by all pipelines.** Add `public static var Flux2Telemetry.shared: (any Flux2TelemetryReporter)?` (or equivalent), have every `Flux2Pipeline` instance read it lazily on each emission. The CLI bootstrap then assigns once at startup. Simplest from the CLI's POV; matches the pattern already used by `SwiftAcervo.AcervoManager.shared`.
2. **A static `Flux2Pipeline.setTelemetryForAllInstances(_:)`** that captures a process-wide reporter; existing instances and future ones pick it up. Functionally identical to (1) but lives on the type that owns the events.
3. **Per-instance install API exposed through SwiftVinetas** — would require `Flux2Engine` to grow `public func setFlux2DepReporter(_:)`. Cleaner encapsulation but means every host project has to do the wiring on every instance. Less ergonomic.

Recommendation: option (1) or (2).

## Out of scope for this TODO

- The instance-bound `setTelemetry(_:)` API is fine and should stay — it's the right primitive. The ask is for an additive process-wide layer.
- Don't change the event enum or the reporter protocol — SwiftVinetas already ships an adapter (`Flux2TelemetryCLIAdapter`) conforming to the existing protocol.

## What's already shipped on the SwiftVinetas side

- `Sources/VinetasCLICore/Telemetry/Flux2EventEncoding.swift` — Encodable shim for all 11 cases of `Flux2TelemetryEvent`.
- `Sources/VinetasCLICore/Telemetry/Flux2TelemetryCLIAdapter.swift` — conforms to `Flux2TelemetryReporter`, writes to the shared `TelemetryJSONLSink` with `kind: "flux2"`.

When the process-wide seam lands, the CLI just calls `<NewSeam>.setReporter(adapter)` in one line and the integration test's `kinds ⊇ {flux2}` assertion will start passing for pipeline events too.
