// Flux2TelemetryAnomalyTests.swift
// Sortie B13 — tests for AnomalyCheck.classify(_:) and the numericalAnomaly
// side-channel event. Strategy A: unit-test AnomalyCheck.classify directly
// (CI-safe, no GPU/model-weights required). The NaN-in-denoise-output
// integration assertion (Strategy C) is deferred until fixture latents exist
// in the test target that can run the denoise loop without a live GPU.
//
// Four AnomalyKind cases covered: .nan, .inf, .outOfRange, .zeroLatent.
// One healthy-stat case that asserts nil return.
// Two side-channel event-capture cases via MockFlux2TelemetryReporter.

import TestHelpers
import Testing
import Tuberia

@testable import Flux2Core

// MARK: - Helpers

/// Builds a minimal TuberiaTensorStat without needing a real MLXArray.
/// `shape` and `dtype` are non-semantic for classify(_:) — only the numeric
/// fields and Boolean flags drive the classification logic.
private func makeStat(
  min: Double = 0,
  max: Double = 1,
  mean: Double = 0.5,
  std: Double = 0.1,
  hasNaN: Bool = false,
  hasInf: Bool = false
) -> TuberiaTensorStat {
  TuberiaTensorStat(
    shape: [1, 4, 4, 4],
    dtype: "float32",
    min: min,
    max: max,
    mean: mean,
    std: std,
    hasNaN: hasNaN,
    hasInf: hasInf
  )
}

// MARK: - AnomalyCheck.classify unit tests

@Suite("Flux2 Telemetry Anomaly — AnomalyCheck.classify")
struct Flux2TelemetryAnomalyClassifyTests {

  // MARK: .nan

  /// A stat with hasNaN == true must classify as .nan regardless of other fields.
  @Test func anomalyCheck_detectsNaN() {
    let stat = makeStat(hasNaN: true)
    #expect(AnomalyCheck.classify(stat) == .nan)
  }

  /// NaN takes priority over Inf when both flags are set.
  @Test func anomalyCheck_nan_precedesInf() {
    let stat = makeStat(hasNaN: true, hasInf: true)
    #expect(AnomalyCheck.classify(stat) == .nan)
  }

  // MARK: .inf

  /// A stat with hasInf == true (and no NaN) must classify as .inf.
  @Test func anomalyCheck_detectsInf() {
    let stat = makeStat(hasInf: true)
    #expect(AnomalyCheck.classify(stat) == .inf)
  }

  // MARK: .outOfRange

  /// A stat whose max exceeds defaultOutOfRangeThreshold (1e6) classifies
  /// as .outOfRange. Uses a value just over the threshold.
  @Test func anomalyCheck_detectsOutOfRange() {
    let threshold = TuberiaTensorStat.defaultOutOfRangeThreshold
    let stat = makeStat(max: threshold + 1)
    #expect(AnomalyCheck.classify(stat) == .outOfRange)
  }

  /// A stat whose max is exactly at the threshold boundary (not over) is
  /// not classified as outOfRange (strict inequality in AnomalyCheck).
  @Test func anomalyCheck_exactThreshold_notOutOfRange() {
    let threshold = TuberiaTensorStat.defaultOutOfRangeThreshold
    // max == threshold → not strictly greater, so no anomaly (mean/std are
    // non-zero so zeroLatent doesn't trigger either).
    let stat = makeStat(max: threshold)
    // Should be nil (healthy) — the comparison is strictly >.
    #expect(AnomalyCheck.classify(stat) == nil)
  }

  // MARK: .zeroLatent

  /// A stat with near-zero mean AND near-zero std classifies as .zeroLatent.
  @Test func anomalyCheck_detectsZeroLatent() {
    // Both below 1e-6.
    let stat = makeStat(min: 0, max: 0, mean: 0, std: 0)
    #expect(AnomalyCheck.classify(stat) == .zeroLatent)
  }

  /// A stat with zero mean but non-trivial std is NOT zeroLatent.
  @Test func anomalyCheck_zerMeanNonzeroStd_notZeroLatent() {
    let stat = makeStat(min: -0.5, max: 0.5, mean: 0, std: 0.2)
    #expect(AnomalyCheck.classify(stat) == nil)
  }

  // MARK: Healthy (nil return)

  /// A numerically healthy stat produces nil — no anomaly.
  @Test func anomalyCheck_healthyStat_returnsNil() {
    let stat = makeStat(min: -0.8, max: 0.8, mean: 0.01, std: 0.3)
    #expect(AnomalyCheck.classify(stat) == nil)
  }
}

// MARK: - Side-channel event capture tests

/// Verifies that a manually assembled `numericalAnomaly` event is captured
/// faithfully by MockFlux2TelemetryReporter. This is Strategy B: prove the
/// actor receives anomaly events correctly without driving a GPU pipeline.
@Suite("Flux2 Telemetry Anomaly — Side-channel event capture")
struct Flux2TelemetryAnomalySideChannelTests {

  /// Manually emit a numericalAnomaly(phase: .denoiseLoopEnd, kind: .nan)
  /// event and verify the reporter stores it with the correct fields.
  @Test func sideChannel_nanAnomalyOnDenoiseLoopEnd_capturedCorrectly() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let nanStat = makeStat(hasNaN: true)
    let event = Flux2TelemetryEvent.numericalAnomaly(
      phase: .denoiseLoopEnd,
      kind: .nan,
      stat: nanStat
    )

    await reporter.capture(event)

    // No Task.sleep needed: capture is a direct await, not fire-and-forget.
    let events = await reporter.snapshot()

    #expect(events.count == 1, "Expected exactly 1 event; got \(events.count)")

    guard case .numericalAnomaly(let phase, let kind, let stat) = events.first else {
      Issue.record("Expected numericalAnomaly event; got \(String(describing: events.first))")
      return
    }
    #expect(phase == .denoiseLoopEnd)
    #expect(kind == .nan)
    #expect(stat.hasNaN == true)
  }

  /// Verify that both a denoiseLoopEnd event and a numericalAnomaly side-channel
  /// fire together in the expected order when a NaN stat is encountered.
  /// (Manually assembles both events to mirror what the pipeline would emit.)
  @Test func sideChannel_denoiseLoopEndWithNaN_bothEventsPresent() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let nanStat = makeStat(hasNaN: true)

    // Simulate what the pipeline emits at denoiseLoopEnd when a NaN is found:
    // 1. the primary denoiseLoopEnd event
    // 2. the numericalAnomaly side-channel alongside it
    let loopEndEvent = Flux2TelemetryEvent.denoiseLoopEnd(
      variant: .textToImage,
      totalSteps: 20,
      completedSteps: 20,
      finalLatentStat: nanStat,
      durationSeconds: 1.234
    )
    let anomalyEvent = Flux2TelemetryEvent.numericalAnomaly(
      phase: .denoiseLoopEnd,
      kind: .nan,
      stat: nanStat
    )

    await reporter.capture(loopEndEvent)
    await reporter.capture(anomalyEvent)

    let events = await reporter.snapshot()

    // Assert denoiseLoopEnd is present and precedes numericalAnomaly.
    let loopEndIndex = events.firstIndex {
      if case .denoiseLoopEnd = $0 { return true }
      return false
    }
    let anomalyIndex = events.firstIndex {
      if case .numericalAnomaly = $0 { return true }
      return false
    }

    #expect(loopEndIndex != nil, "denoiseLoopEnd must be present in captured events")
    #expect(anomalyIndex != nil, "numericalAnomaly must be present alongside denoiseLoopEnd")

    if let lei = loopEndIndex, let ai = anomalyIndex {
      #expect(lei < ai, "denoiseLoopEnd must precede numericalAnomaly; got indices \(lei), \(ai)")
    }

    // Verify the denoiseLoopEnd stat carries hasNaN == true.
    if case .denoiseLoopEnd(_, _, _, let finalStat, _) = events[loopEndIndex!] {
      #expect(finalStat.hasNaN == true, "finalLatentStat.hasNaN must be true")
    }

    // Verify the anomaly kind is .nan.
    if case .numericalAnomaly(let phase, let kind, _) = events[anomalyIndex!] {
      #expect(phase == .denoiseLoopEnd)
      #expect(kind == .nan)
    }
  }
}
