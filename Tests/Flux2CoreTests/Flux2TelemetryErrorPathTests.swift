// Flux2TelemetryErrorPathTests.swift
// Sortie B14: verify .errorThrown fires before Flux2Error is thrown.
// Uses Swift Testing — consistent with the other Flux2CoreTests telemetry tests.

import CoreGraphics
import Flux2Core
import TestHelpers
import Testing

@Suite("Flux2 Telemetry Error Paths")
struct Flux2TelemetryErrorPathTests {

  /// Passing an empty images array to generateImageToImage triggers the
  /// validation guard at the top of the method, which emits
  /// `.errorThrown(phase: .invalidConfiguration, ...)` immediately before
  /// throwing `Flux2Error.invalidConfiguration`.
  ///
  /// The emit is a direct `await currentTelemetry()?.capture(...)` (not a
  /// detached Task), so it is guaranteed to complete before the throw escapes
  /// — no sleep is needed between the throw and the snapshot.
  @Test func errorThrown_firesBeforeInvalidConfigurationThrow() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)

    // Allow the detached pipelineInit Task to settle so it does not
    // pollute the captured events list in an unpredictable order.
    // (Best practice from B12 pattern — see MockFlux2TelemetryReporter docs.)
    try await Task.sleep(for: .milliseconds(50))

    // Trigger the validation guard by passing an empty images array.
    // The guard: `guard !images.isEmpty && images.count <= 3 else { ... }`
    // emits .errorThrown(phase: .invalidConfiguration, ...) then throws.
    // No CGImage construction is needed — an empty array suffices.
    var threw = false
    do {
      _ = try await pipeline.generateImageToImage(
        prompt: "test",
        images: [CGImage]()
      )
      Issue.record("Expected Flux2Error.invalidConfiguration to be thrown")
    } catch is Flux2Error {
      threw = true
    }
    #expect(threw, "generateImageToImage with empty images should throw Flux2Error")

    // The errorThrown emit is a direct `await` in the guard body —
    // it completes before the throw unwinds, so no additional sleep is needed.
    let events = await reporter.snapshot()
    let hasErrorThrown = events.contains { event in
      if case .errorThrown(phase: .invalidConfiguration, _) = event { return true }
      return false
    }
    #expect(
      hasErrorThrown,
      "Expected .errorThrown(phase: .invalidConfiguration, ...) in captured events; got: \(events)"
    )
  }
}
