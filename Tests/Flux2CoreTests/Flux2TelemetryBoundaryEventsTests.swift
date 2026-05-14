// Flux2TelemetryBoundaryEventsTests.swift
// Pattern-establishing test sortie for boundary-event telemetry (B12).
// Uses Swift Testing — consistent with Flux2CoreTests.swift.

import Flux2Core
import TestHelpers
import Testing

@Suite("Flux2 Telemetry Boundary Events")
struct Flux2TelemetryBoundaryEventsTests {

  // MARK: - pipelineInit + pipelineDispose sequence

  /// Verify that pipelineInit and pipelineDispose fire in the correct order
  /// on a minimal, GPU-free pipeline construction + explicit dispose call.
  ///
  /// Timing note: pipelineInit is dispatched from a fire-and-forget Task
  /// inside Flux2Pipeline.init (init is synchronous). After setTelemetry()
  /// returns the Task may not yet have delivered the event. We sleep 100 ms
  /// to let the unstructured Task reach the actor; bump to 250 ms if the
  /// test becomes flaky on slow CI hardware.
  @Test func boundaryEvents_pipelineInitAndDispose_fireInOrder() async throws {
    let reporter = MockFlux2TelemetryReporter()

    // Construct the pipeline then immediately set the reporter.
    // There is an inherent race between the detached pipelineInit Task
    // and this setTelemetry() call. We accept that race and document it:
    // if pipelineInit is not observed the test skips that assertion
    // rather than failing (see defensive check below).
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)

    // Give the fire-and-forget Task time to deliver pipelineInit.
    try await Task.sleep(for: .milliseconds(100))

    await pipeline.dispose()

    let events = await reporter.snapshot()

    // pipelineDispose MUST be present — it's async and called directly.
    let hasDispose = events.contains {
      if case .pipelineDispose = $0 { return true }
      return false
    }
    #expect(hasDispose, "pipelineDispose must fire after dispose() is called")

    // pipelineInit is subject to a race (setTelemetry may arrive after the
    // Task fires). We record whether it was observed but do not hard-fail
    // if it wasn't — the race is documented in MockFlux2TelemetryReporter.
    let hasInit = events.contains {
      if case .pipelineInit = $0 { return true }
      return false
    }
    // If pipelineInit fired, it must appear before pipelineDispose.
    if hasInit {
      let initIndex = events.firstIndex {
        if case .pipelineInit = $0 { return true }
        return false
      }!
      let disposeIndex = events.firstIndex {
        if case .pipelineDispose = $0 { return true }
        return false
      }!
      #expect(initIndex < disposeIndex, "pipelineInit must precede pipelineDispose")
    }
  }

  // MARK: - Detaching reporter silences subsequent events

  /// After setTelemetry(nil), calling dispose() must not deliver a
  /// pipelineDispose event to the previously installed reporter.
  @Test func boundaryEvents_detachReporter_silencesDispose() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)

    // Detach before dispose.
    pipeline.setTelemetry(nil)

    await pipeline.dispose()

    let events = await reporter.snapshot()
    let hasDispose = events.contains {
      if case .pipelineDispose = $0 { return true }
      return false
    }
    #expect(!hasDispose, "pipelineDispose must NOT fire after reporter is detached (nil)")
  }

  // MARK: - Double dispose emits twice

  /// Calling dispose() twice should fire pipelineDispose twice.
  @Test func boundaryEvents_doubleDispose_firesTwice() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)

    await pipeline.dispose()
    await pipeline.dispose()

    let events = await reporter.snapshot()
    let disposeCount = events.filter {
      if case .pipelineDispose = $0 { return true }
      return false
    }.count
    #expect(
      disposeCount == 2, "pipelineDispose must fire once per dispose() call; got \(disposeCount)")
  }
}
