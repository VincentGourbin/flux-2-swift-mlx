// Flux2ProcessWideTelemetryTests.swift
// OPERATION WIRETAP DARKROOM — process-wide Flux2Telemetry seam smoke tests.
// Uses Swift Testing, consistent with the rest of Flux2CoreTests.

import Flux2Core
import TestHelpers
import Testing

/// Smoke-contract tests for the `Flux2Telemetry` process-wide seam.
///
/// Three assertions required by the spec:
///  1. When NO instance reporter is set AND a process-wide reporter is set,
///     emissions reach the process-wide reporter.
///  2. When BOTH an instance reporter and a process-wide reporter are set,
///     the instance reporter wins (instance takes precedence).
///  3. Setting the process-wide reporter to nil restores "no reporter" state.
///
/// ## Test isolation
///
/// `Flux2Telemetry.setReporter` writes a process-global singleton. Every test
/// that installs a reporter uses `defer { Flux2Telemetry.setReporter(nil) }` to
/// clean up regardless of pass/fail, preventing cross-test contamination.
///
/// ## Fire-and-forget pipelineInit race
///
/// `Flux2Pipeline.init` dispatches `pipelineInit` via an unstructured `Task`.
/// After `setReporter` or `setTelemetry` returns, that Task may not yet have
/// delivered the event. Tests that assert on `pipelineDispose` (a direct
/// `await currentTelemetry()?.capture(...)` call from `dispose()`) are
/// deterministic. Tests that need `pipelineInit` sleep 100 ms per the pattern
/// established in `Flux2TelemetryBoundaryEventsTests`.
/// Serialized because every test writes to the process-global `Flux2Telemetry.current`.
/// Running them concurrently (Swift Testing default) would cause cross-test contamination.
@Suite("Flux2 Process-wide Telemetry Seam", .serialized)
struct Flux2ProcessWideTelemetryTests {

  // MARK: - 1. Process-wide reporter receives events when no instance reporter is set

  /// When only a process-wide reporter is installed (no instance-level
  /// `setTelemetry` call), `pipelineDispose` must be delivered to it.
  @Test func processWideReporter_receivesDispose_whenNoInstanceReporterSet() async throws {
    let processReporter = MockFlux2TelemetryReporter()
    Flux2Telemetry.setReporter(processReporter)
    defer { Flux2Telemetry.setReporter(nil) }

    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    // No pipeline.setTelemetry call — only the process-wide reporter is active.

    await pipeline.dispose()

    let events = await processReporter.snapshot()
    let hasDispose = events.contains {
      if case .pipelineDispose = $0 { return true }
      return false
    }
    #expect(hasDispose, "pipelineDispose must be delivered to the process-wide reporter")
  }

  /// pipelineInit (fire-and-forget Task) also reaches the process-wide reporter
  /// when no instance reporter is set. We sleep 100 ms for the Task to settle.
  ///
  /// Note: the pipeline must be kept alive (strong reference) so that the
  /// `[weak self]` capture inside the fire-and-forget init Task does not go
  /// nil before the event is delivered.
  @Test func processWideReporter_receivesInit_whenNoInstanceReporterSet() async throws {
    let processReporter = MockFlux2TelemetryReporter()
    Flux2Telemetry.setReporter(processReporter)
    defer { Flux2Telemetry.setReporter(nil) }

    // Keep a strong reference so the [weak self] capture in the init Task stays alive.
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    // Let the detached pipelineInit Task deliver the event.
    try await Task.sleep(for: .milliseconds(100))

    let events = await processReporter.snapshot()
    let hasInit = events.contains {
      if case .pipelineInit = $0 { return true }
      return false
    }
    // Retain the pipeline through the sleep to prevent premature deallocation.
    withExtendedLifetime(pipeline) {}
    #expect(hasInit, "pipelineInit must be delivered to the process-wide reporter")
  }

  // MARK: - 2. Instance reporter wins when both are set

  /// When an instance reporter is installed via `setTelemetry(_:)` AND a
  /// process-wide reporter is also set, `pipelineDispose` must reach only the
  /// instance reporter. The process-wide reporter must receive nothing.
  @Test func instanceReporter_winsOverProcessWideReporter() async throws {
    let instanceReporter = MockFlux2TelemetryReporter()
    let processReporter = MockFlux2TelemetryReporter()

    Flux2Telemetry.setReporter(processReporter)
    defer { Flux2Telemetry.setReporter(nil) }

    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(instanceReporter)

    await pipeline.dispose()

    let instanceEvents = await instanceReporter.snapshot()
    let processEvents = await processReporter.snapshot()

    let instanceHasDispose = instanceEvents.contains {
      if case .pipelineDispose = $0 { return true }
      return false
    }
    let processHasDispose = processEvents.contains {
      if case .pipelineDispose = $0 { return true }
      return false
    }

    #expect(instanceHasDispose, "instance reporter must receive pipelineDispose")
    #expect(
      !processHasDispose,
      "process-wide reporter must NOT receive pipelineDispose when instance reporter is set")
  }

  // MARK: - 3. setReporter(nil) restores "no reporter" state

  /// After calling `Flux2Telemetry.setReporter(nil)`, a pipeline with no
  /// instance reporter must silently drop events (no reporter receives them).
  @Test func setReporter_nil_restoresNoReporterState() async throws {
    let processReporter = MockFlux2TelemetryReporter()

    Flux2Telemetry.setReporter(processReporter)
    // Immediately clear it.
    Flux2Telemetry.setReporter(nil)

    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    // No instance reporter either.
    await pipeline.dispose()

    // Give the fire-and-forget pipelineInit Task a moment too.
    try await Task.sleep(for: .milliseconds(100))

    let events = await processReporter.snapshot()
    #expect(
      events.isEmpty,
      "no events should reach the reporter after setReporter(nil); got \(events.count)")
    withExtendedLifetime(pipeline) {}
  }

  // MARK: - 4. current accessor reflects the installed reporter

  /// `Flux2Telemetry.current` returns the installed reporter and `nil` after clearing.
  @Test func current_reflectsInstalledReporter() async {
    let reporter = MockFlux2TelemetryReporter()
    defer { Flux2Telemetry.setReporter(nil) }

    #expect(Flux2Telemetry.current == nil, "should be nil before installing")

    Flux2Telemetry.setReporter(reporter)
    #expect(Flux2Telemetry.current != nil, "should be non-nil after installing")

    Flux2Telemetry.setReporter(nil)
    #expect(Flux2Telemetry.current == nil, "should be nil after clearing")
  }
}
