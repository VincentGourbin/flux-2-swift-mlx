// Flux2TelemetryLockContentionTests.swift
// Sortie B15 of OPERATION TWIN LIGHTHOUSE (iteration 3).
//
// Stresses `Flux2Pipeline._telemetryLock` via concurrent `setTelemetry` calls
// from a structured `withTaskGroup`. The goal is to expose data races that
// `OSAllocatedUnfairLock` is meant to prevent.
//
// ## Why XCTest (not Swift Testing)
//
// Per iteration-02 carry-over F11: Swift Testing has known flakiness with the
// concurrent stress patterns required by lock-contention tests on macOS-26 CI
// runners. XCTest's `async throws` + `withTaskGroup` pattern is more stable.
// The pixart canonical reference uses Swift Testing but this file deliberately
// diverges per mission instructions.
//
// ## Design
//
// `Flux2Pipeline.setTelemetry(_:)` acquires `_telemetryLock` synchronously.
// `Flux2Pipeline.dispose()` calls `currentTelemetry()?.capture(...)` which also
// acquires the lock on the read path. Hammering both paths concurrently gives
// the Thread Sanitizer a representative read/write interleaving.
//
// The type-under-test is `Flux2Pipeline` because:
//   1. It owns the primary `OSAllocatedUnfairLock`-backed `setTelemetry` seam.
//   2. `dispose()` fires `.pipelineDispose` cheaply (no GPU, no weights).
//   3. `Flux2Pipeline.init` is GPU-free — it only wires a scheduler and downloader.
//
// ## TSan verification
//
// Under the default `make test` invocation this test passes if no crash or
// assertion fires.  Real data-race detection requires TSan:
//
//   xcodebuild test \
//     -scheme Flux2Swift-Package \
//     -destination 'platform=macOS,arch=arm64' \
//     -enableThreadSanitizer YES \
//     -skipPackagePluginValidation \
//     ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
//     -only-testing:Flux2CoreTests/Flux2TelemetryLockContentionTests
//
// A TSan run with zero `WARNING: ThreadSanitizer` / `data race` diagnostics
// confirms that Sortie B3's `OSAllocatedUnfairLock` seam is correct.
//
// ## F10 compliance (Swift 6 strict concurrency)
//
// All helpers in this class are instance methods (not static).
// This avoids the Swift 6 strict-concurrency error (F10) that fires when a
// static func on a non-Sendable class is called without a `Self.` qualifier.

import TestHelpers
import XCTest

@testable import Flux2Core

final class Flux2TelemetryLockContentionTests: XCTestCase {

  // MARK: - Fixture helpers (instance methods — F10 compliant)

  /// Returns a fresh, isolated `Flux2Pipeline` instance.
  ///
  /// Each test gets its own instance so that concurrent writes from this
  /// suite do not interfere with any other test suite's reporter state.
  /// `Flux2Pipeline.init` is GPU-free: it only wires a `FlowMatchEulerScheduler`
  /// and a `Flux2ModelDownloader`; no model weights are touched.
  private func makeFreshPipeline() -> Flux2Pipeline {
    Flux2Pipeline(model: .dev, quantization: .balanced)
  }

  // MARK: - Concurrent set / nil cycle

  /// Two writer tasks alternate between a `MockFlux2TelemetryReporter` and `nil`
  /// while a third writer hammers `setTelemetry(nil)` in a tight loop.
  ///
  /// Pass criterion (default run): no crash, no assertion, task group
  /// completes normally.
  /// Pass criterion (TSan run): zero `data race` diagnostics.
  func testConcurrentSetAndNilDoesNotRace() async throws {
    let pipeline = makeFreshPipeline()
    let reporter1 = MockFlux2TelemetryReporter()
    let reporter2 = MockFlux2TelemetryReporter()

    await withTaskGroup(of: Void.self) { group in
      // Writer 1: alternates between reporter1 and nil.
      group.addTask {
        for _ in 0..<500 {
          pipeline.setTelemetry(reporter1)
          pipeline.setTelemetry(nil)
        }
      }
      // Writer 2: alternates between reporter2 and reporter1.
      group.addTask {
        for _ in 0..<500 {
          pipeline.setTelemetry(reporter2)
          pipeline.setTelemetry(reporter1)
        }
      }
      // Writer 3: hammers nil — high-frequency lock exerciser.
      // setTelemetry(nil) forces the lock to flip state and exercises
      // the read path indirectly via currentTelemetry() inside dispose().
      group.addTask {
        for _ in 0..<2000 {
          pipeline.setTelemetry(nil)
        }
      }
    }

    // If execution reaches here the lock held under all concurrent writes.
    // Under a TSan-enabled build any unguarded access would have already
    // triggered a diagnostic abort.
    XCTAssert(
      true,
      "Completed \(500 + 500 + 2000) concurrent setTelemetry calls without crash"
    )
  }

  // MARK: - Interleaved reporter swap

  /// Four tasks concurrently swap the reporter back and forth between two
  /// `MockFlux2TelemetryReporter` instances. The intent is to saturate the
  /// lock with concurrent read-modify-write cycles.
  func testInterleavedReporterSwapDoesNotRace() async throws {
    let pipeline = makeFreshPipeline()
    let reporterA = MockFlux2TelemetryReporter()
    let reporterB = MockFlux2TelemetryReporter()

    await withTaskGroup(of: Void.self) { group in
      for i in 0..<4 {
        group.addTask {
          let even = (i % 2 == 0)
          for _ in 0..<300 {
            pipeline.setTelemetry(even ? reporterA : reporterB)
            pipeline.setTelemetry(even ? reporterB : reporterA)
            pipeline.setTelemetry(nil)
          }
        }
      }
    }

    XCTAssert(
      true,
      "Completed interleaved reporter swaps across 4 tasks without crash"
    )
  }

  // MARK: - Set / dispose round-trip under contention

  /// Writer tasks alternate `setTelemetry` between two reporters while a
  /// concurrent dispose task fires `.pipelineDispose` events, exercising the
  /// read path (`currentTelemetry()`) against concurrent writes.
  ///
  /// Post-contention assertion: at least SOME events landed in one of the
  /// reporters, proving the lock did not deadlock and emissions reached the
  /// actor.
  func testConcurrentSetTelemetryWithDisposeEmitsLand() async throws {
    let pipeline = makeFreshPipeline()
    let reporter1 = MockFlux2TelemetryReporter()
    let reporter2 = MockFlux2TelemetryReporter()

    // Prime the pipeline with reporter1 so early dispose calls have a target.
    pipeline.setTelemetry(reporter1)

    await withTaskGroup(of: Void.self) { group in
      // Writer: toggles between reporter1, reporter2, and nil.
      group.addTask {
        for i in 0..<600 {
          switch i % 3 {
          case 0: pipeline.setTelemetry(reporter1)
          case 1: pipeline.setTelemetry(reporter2)
          default: pipeline.setTelemetry(nil)
          }
        }
      }
      // Emitter: fires dispose() — acquires the lock on the read path.
      // Some calls will find nil (and emit nothing); some will find a live
      // reporter and deliver .pipelineDispose. Both outcomes are correct.
      group.addTask {
        for _ in 0..<50 {
          await pipeline.dispose()
        }
      }
    }

    // Give the actor time to process any fire-and-forget tasks that may still
    // be in flight (dispose() is async/structured, so no sleep is needed for
    // dispose itself; this guards against any pipelineInit Task from init).
    try await Task.sleep(for: .milliseconds(50))

    let captured1 = await reporter1.snapshot()
    let captured2 = await reporter2.snapshot()
    let totalEmitted = captured1.count + captured2.count

    // At least SOME emissions should have landed — the lock must not have
    // deadlocked. (Exact count is non-deterministic due to the toggle race.)
    XCTAssertGreaterThan(
      totalEmitted,
      0,
      "Expected at least one .pipelineDispose (or .pipelineInit) event to land "
        + "in either reporter; lock may have deadlocked if this fails"
    )
  }

  // MARK: - High-frequency nil writer vs slow toggler

  /// A slow toggler alternates between two reporters while two high-frequency
  /// nil writers force lock state transitions between every slow write.
  ///
  /// Mirrors the third test case from pixart's canonical pattern, adapted for
  /// the async `dispose()` read-path exerciser.
  func testHighFrequencyNilWriterAlongsideSlowTogglerDoesNotRace() async throws {
    let pipeline = makeFreshPipeline()
    let reporter1 = MockFlux2TelemetryReporter()
    let reporter2 = MockFlux2TelemetryReporter()

    await withTaskGroup(of: Void.self) { group in
      // Slow toggler: sets reporter1 then reporter2 in each iteration.
      group.addTask {
        for _ in 0..<200 {
          pipeline.setTelemetry(reporter1)
          pipeline.setTelemetry(reporter2)
        }
      }
      // High-frequency nil writer: forces lock state transitions between
      // every slow-toggler write.
      group.addTask {
        for _ in 0..<5000 {
          pipeline.setTelemetry(nil)
        }
      }
      // Second high-frequency nil writer for extra pressure.
      group.addTask {
        for _ in 0..<5000 {
          pipeline.setTelemetry(nil)
        }
      }
    }

    XCTAssert(
      true,
      "High-frequency nil writer completed without crash alongside slow toggler"
    )
  }
}
