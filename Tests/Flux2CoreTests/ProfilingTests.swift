// ProfilingTests.swift - Tests for profiling integration and Flux2-specific extensions
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
@testable import FluxTextEncoders
import MLXProfiler
import MLX

// MARK: - Typealias Integration Tests

final class ProfilingTypealiasTests: XCTestCase {

    func testFlux2ProfilerIsMLXProfiler() {
        let profiler: Flux2Profiler = Flux2Profiler.shared
        XCTAssertNotNil(profiler)
        XCTAssertTrue(profiler === MLXProfiler.shared)
    }

    func testFluxProfilerIsMLXProfiler() {
        let profiler: FluxProfiler = FluxProfiler.shared
        XCTAssertNotNil(profiler)
        XCTAssertTrue(profiler === MLXProfiler.shared)
    }

    func testBothTypealiasesPointToSameSingleton() {
        XCTAssertTrue(Flux2Profiler.shared === FluxProfiler.shared)
    }
}

// MARK: - Profiler Core API Tests

final class ProfilerCoreAPITests: XCTestCase {

    override func tearDown() {
        let profiler = MLXProfiler.shared
        profiler.activeSession = nil
        profiler.disable()
    }

    func testEnableDisable() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        XCTAssertTrue(profiler.isEnabled)
        profiler.disable()
        XCTAssertFalse(profiler.isEnabled)
    }

    func testTimingRecording() {
        let profiler = Flux2Profiler.shared
        profiler.enable()

        profiler.start("Test Phase")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Test Phase")

        let timings = profiler.getTimings()
        XCTAssertFalse(timings.isEmpty)
        XCTAssertEqual(timings.last?.name, "Test Phase")
        XCTAssertGreaterThan(timings.last?.duration ?? 0, 0.005)
    }

    func testStepRecording() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.setTotalSteps(4)

        profiler.recordStep(duration: 1.0)
        profiler.recordStep(duration: 1.5)
        profiler.recordStep(duration: 2.0)

        let steps = profiler.getStepTimes()
        XCTAssertEqual(steps.count, 3)
        XCTAssertEqual(steps[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(steps[1], 1.5, accuracy: 0.001)
    }

    func testMeasureClosure() {
        let profiler = Flux2Profiler.shared
        profiler.enable()

        let result = profiler.measure("Compute") {
            return 42
        }

        XCTAssertEqual(result, 42)
        let timings = profiler.getTimings()
        XCTAssertTrue(timings.contains(where: { $0.name == "Compute" }))
    }

    func testEnableResetsState() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.start("Phase A")
        profiler.end("Phase A")
        profiler.recordStep(duration: 1.0)
        XCTAssertFalse(profiler.getTimings().isEmpty)

        // Re-enable clears everything
        profiler.enable()
        XCTAssertTrue(profiler.getTimings().isEmpty)
        XCTAssertTrue(profiler.getStepTimes().isEmpty)
    }

    func testDisabledProfilerSkipsRecording() {
        let profiler = Flux2Profiler.shared
        // Enable first to reset, then disable
        profiler.enable()
        profiler.disable()

        profiler.start("Should Not Record")
        profiler.end("Should Not Record")
        profiler.recordStep(duration: 1.0)

        // Timings from before disable should be empty (enable clears them)
        // and new ones should not be recorded
        XCTAssertTrue(profiler.getTimings().isEmpty)
        XCTAssertTrue(profiler.getStepTimes().isEmpty)
    }
}

// MARK: - ProfilingSession Integration Tests

final class ProfilingSessionIntegrationTests: XCTestCase {

    override func tearDown() {
        let profiler = MLXProfiler.shared
        profiler.activeSession = nil
        profiler.disable()
    }

    func testSessionCreationWithMetadata() {
        let session = ProfilingSession()
        session.title = "FLUX.2 TEST"
        session.metadata["model"] = "klein-4b"
        session.metadata["quant"] = "8bit/qint8"
        session.metadata["resolution"] = "512x512"
        session.metadata["steps"] = "4"

        XCTAssertEqual(session.title, "FLUX.2 TEST")
        XCTAssertEqual(session.metadata["model"], "klein-4b")
        XCTAssertEqual(session.metadata["steps"], "4")
    }

    func testSessionPhaseRecording() {
        let session = ProfilingSession()
        session.beginPhase("Load Model", category: .transformerLoad)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Load Model", category: .transformerLoad)

        let events = session.getEvents()
        XCTAssertGreaterThanOrEqual(events.count, 2) // begin + end
        let categories = Set(events.map(\.category))
        XCTAssertTrue(categories.contains(.transformerLoad))
    }

    func testSessionMultiplePhases() {
        let session = ProfilingSession()
        session.beginPhase("Text Encoding", category: .textEncoding)
        session.endPhase("Text Encoding", category: .textEncoding)
        session.beginPhase("Denoising", category: .denoisingLoop)
        session.endPhase("Denoising", category: .denoisingLoop)
        session.beginPhase("VAE Decode", category: .vaeDecode)
        session.endPhase("VAE Decode", category: .vaeDecode)

        let events = session.getEvents()
        let categories = Set(events.map(\.category))
        XCTAssertTrue(categories.contains(.textEncoding))
        XCTAssertTrue(categories.contains(.denoisingLoop))
        XCTAssertTrue(categories.contains(.vaeDecode))
    }

    func testSessionStepRecording() {
        let session = ProfilingSession()
        session.recordStep(index: 1, total: 4, durationUs: 1_000_000)
        session.recordStep(index: 2, total: 4, durationUs: 1_500_000)

        let events = session.getEvents()
        let stepEvents = events.filter { $0.stepIndex != nil }
        XCTAssertEqual(stepEvents.count, 2)
    }

    func testSessionAttachedToProfiler() {
        let profiler = Flux2Profiler.shared
        let session = ProfilingSession()
        session.title = "ATTACHED TEST"

        profiler.enable()
        profiler.activeSession = session

        profiler.start("1. Load Model")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("1. Load Model")

        // Events should be in session
        let events = session.getEvents()
        XCTAssertFalse(events.isEmpty)

        // Report should use session title
        let report = session.generateReport()
        XCTAssertTrue(report.contains("ATTACHED TEST"))
    }

    func testChromeTraceExport() {
        let session = ProfilingSession()
        session.beginPhase("Denoising", category: .denoisingLoop)
        session.endPhase("Denoising", category: .denoisingLoop)

        let data = ChromeTraceExporter.export(session: session)
        XCTAssertGreaterThan(data.count, 0)

        // Valid JSON
        let json = try? JSONSerialization.jsonObject(with: data)
        XCTAssertNotNil(json)

        // Should contain trace events array
        if let dict = json as? [String: Any], let events = dict["traceEvents"] as? [[String: Any]] {
            XCTAssertFalse(events.isEmpty)
        }
    }

    func testChromeTraceComparison() {
        let session1 = ProfilingSession()
        session1.beginPhase("Test", category: .denoisingLoop)
        session1.endPhase("Test", category: .denoisingLoop)

        let session2 = ProfilingSession()
        session2.beginPhase("Test", category: .denoisingLoop)
        session2.endPhase("Test", category: .denoisingLoop)

        let data = ChromeTraceExporter.exportComparison(sessions: [
            (label: "Run 1", session: session1),
            (label: "Run 2", session: session2)
        ])
        XCTAssertGreaterThan(data.count, 0)

        let json = try? JSONSerialization.jsonObject(with: data)
        XCTAssertNotNil(json)
    }
}

// MARK: - generateReport() Extension Tests

final class GenerateReportTests: XCTestCase {

    override func tearDown() {
        let profiler = MLXProfiler.shared
        profiler.activeSession = nil
        profiler.disable()
    }

    func testReportWithEmptyTimings() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        // No phases recorded
        let report = profiler.generateReport()
        XCTAssertEqual(report, "")
    }

    func testReportContainsHeader() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.start("1. Test")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("1. Test")

        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("FLUX.2 PERFORMANCE REPORT"))
        XCTAssertTrue(report.contains("PHASE TIMINGS"))
    }

    func testReportContainsPhaseNames() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.start("1. Load Text Encoder")
        profiler.end("1. Load Text Encoder")
        profiler.start("6. Denoising Loop")
        profiler.end("6. Denoising Loop")

        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("Load Text Encoder"))
        XCTAssertTrue(report.contains("Denoising Loop"))
        XCTAssertTrue(report.contains("TOTAL"))
        XCTAssertTrue(report.contains("100.0%"))
    }

    func testReportWithStepStatistics() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.setTotalSteps(4)
        profiler.start("6. Denoising Loop")
        profiler.recordStep(duration: 5.0)
        profiler.recordStep(duration: 6.0)
        profiler.recordStep(duration: 5.5)
        profiler.recordStep(duration: 7.0)
        profiler.end("6. Denoising Loop")

        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("DENOISING STEP STATISTICS"))
        XCTAssertTrue(report.contains("Steps:              4"))
        XCTAssertTrue(report.contains("Fastest step"))
        XCTAssertTrue(report.contains("Slowest step"))
        XCTAssertTrue(report.contains("Estimated times"))
    }

    func testReportInsightsSection() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.start("1. Small Phase")
        profiler.end("1. Small Phase")
        profiler.start("6. Big Phase")
        Thread.sleep(forTimeInterval: 0.02)
        profiler.end("6. Big Phase")

        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("INSIGHTS"))
        XCTAssertTrue(report.contains("Bottleneck"))
        XCTAssertTrue(report.contains("Big Phase"))
    }

    func testReportMillisecondFormatting() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        // Record a very short phase
        profiler.record("Fast Phase", duration: 0.05) // 50ms
        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("ms"))
    }

    func testReportSecondsFormatting() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.record("Slow Phase", duration: 5.0)
        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("5.00s"))
    }

    func testReportDelegatesToSessionWhenActive() {
        let profiler = Flux2Profiler.shared
        let session = ProfilingSession()
        session.title = "SESSION REPORT"

        profiler.enable()
        profiler.activeSession = session

        profiler.start("1. Test")
        profiler.end("1. Test")

        let report = profiler.generateReport()
        // Should use session's report (with its title), not the simple fallback
        XCTAssertTrue(report.contains("SESSION REPORT"))
    }
}

// MARK: - GenerationMetrics Tests

final class GenerationMetricsTests: XCTestCase {

    func testGenerationMetricsSummary() {
        let metrics = GenerationMetrics(
            tokenizationTime: 0.005,
            prefillTime: 0.050,
            generationTime: 2.0,
            totalTime: 2.055,
            promptTokens: 100,
            generatedTokens: 500,
            prefillTokensPerSecond: 2000.0,
            generationTokensPerSecond: 250.0,
            mlxActiveMemoryMB: 4096.0,
            mlxCacheMemoryMB: 512.0,
            mlxPeakMemoryMB: 5000.0,
            processFootprintMB: 8192.0
        )

        let summary = metrics.summary
        XCTAssertTrue(summary.contains("DETAILED PROFILING"))
        XCTAssertTrue(summary.contains("TOKENIZATION"))
        XCTAssertTrue(summary.contains("PREFILL"))
        XCTAssertTrue(summary.contains("100"))  // promptTokens
        XCTAssertTrue(summary.contains("GENERATION"))
        XCTAssertTrue(summary.contains("500"))  // generatedTokens
        XCTAssertTrue(summary.contains("MLX MEMORY"))
        XCTAssertTrue(summary.contains("PROCESS MEMORY"))
        XCTAssertTrue(summary.contains("TOTAL"))
    }

    func testGenerationMetricsCompactSummary() {
        let metrics = GenerationMetrics(
            tokenizationTime: 0.005,
            prefillTime: 0.050,
            generationTime: 2.0,
            totalTime: 2.055,
            promptTokens: 100,
            generatedTokens: 500,
            prefillTokensPerSecond: 2000.0,
            generationTokensPerSecond: 250.0,
            mlxActiveMemoryMB: 4096.0,
            mlxCacheMemoryMB: 512.0,
            mlxPeakMemoryMB: 5000.0,
            processFootprintMB: 8192.0
        )

        let compact = metrics.compactSummary
        XCTAssertTrue(compact.contains("tok/s"))
        XCTAssertTrue(compact.contains("100 tok"))
        XCTAssertTrue(compact.contains("500 tok"))
        XCTAssertTrue(compact.contains("MB"))
    }

    func testGetMetricsBridgesLLMData() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        profiler.startTokenization()
        profiler.endTokenization(tokenCount: 42)
        profiler.startPrefill()
        profiler.endPrefill()
        profiler.startGeneration()
        profiler.endGeneration(tokenCount: 100)

        let metrics = profiler.getMetrics()
        XCTAssertEqual(metrics.promptTokens, 42)
        XCTAssertEqual(metrics.generatedTokens, 100)
        XCTAssertGreaterThan(metrics.mlxActiveMemoryMB + metrics.mlxCacheMemoryMB + metrics.mlxPeakMemoryMB, 0)

        profiler.disable()
    }
}

// MARK: - MemorySnapshot Tests

final class MemorySnapshotTests: XCTestCase {

    func testMemorySnapshotCurrent() {
        let snapshot = MemorySnapshot.current()
        XCTAssertGreaterThanOrEqual(snapshot.mlxActive, 0)
        XCTAssertGreaterThanOrEqual(snapshot.mlxCache, 0)
        XCTAssertGreaterThan(snapshot.processFootprint, 0)
    }

    func testMemorySnapshotTotal() {
        let snapshot = MemorySnapshot(
            mlxActive: 1000, mlxCache: 500, mlxPeak: 1500,
            processFootprint: 2000, timestamp: Date()
        )
        XCTAssertEqual(snapshot.mlxTotal, 1500)
    }

    func testMemorySnapshotDescription() {
        let snapshot = MemorySnapshot(
            mlxActive: 4 * 1_048_576, mlxCache: 2 * 1_048_576, mlxPeak: 6 * 1_048_576,
            processFootprint: 8 * 1_048_576, timestamp: Date()
        )
        let desc = snapshot.description
        XCTAssertTrue(desc.contains("4MB"))
        XCTAssertTrue(desc.contains("2MB"))
        XCTAssertTrue(desc.contains("6MB"))
    }

    func testMemorySnapshotDelta() {
        let t1 = Date()
        let snap1 = MemorySnapshot(
            mlxActive: 1000, mlxCache: 500, mlxPeak: 1500,
            processFootprint: 2000, timestamp: t1
        )
        let snap2 = MemorySnapshot(
            mlxActive: 3000, mlxCache: 600, mlxPeak: 3000,
            processFootprint: 5000, timestamp: t1.addingTimeInterval(1.0)
        )

        let delta = snap1.delta(to: snap2)
        XCTAssertEqual(delta.mlxActiveDelta, 2000)
        XCTAssertEqual(delta.mlxCacheDelta, 100)
        XCTAssertEqual(delta.mlxPeakDelta, 1500)
        XCTAssertEqual(delta.processFootprintDelta, 3000)
        XCTAssertEqual(delta.duration, 1.0, accuracy: 0.01)
    }

    func testMemoryDeltaDescription() {
        let delta = MemoryDelta(
            mlxActiveDelta: 2 * 1_048_576,
            mlxCacheDelta: 0,
            mlxPeakDelta: 0,
            processFootprintDelta: 0,
            duration: 1.0
        )
        XCTAssertTrue(delta.description.contains("+"))
        XCTAssertTrue(delta.description.contains("2MB"))
    }

    func testMemoryDeltaNegative() {
        let delta = MemoryDelta(
            mlxActiveDelta: -1_048_576,
            mlxCacheDelta: 0,
            mlxPeakDelta: 0,
            processFootprintDelta: 0,
            duration: 0.5
        )
        // Negative delta should not have + prefix
        XCTAssertFalse(delta.description.hasPrefix("MLX: +"))
    }
}

// MARK: - ProfiledStep Tests

final class ProfiledStepTests: XCTestCase {

    func testProfiledStepDelta() {
        let t = Date()
        let step = ProfiledStep(
            name: "Load Model",
            startMemory: MemorySnapshot(mlxActive: 100, mlxCache: 0, mlxPeak: 100, processFootprint: 500, timestamp: t),
            endMemory: MemorySnapshot(mlxActive: 5000, mlxCache: 200, mlxPeak: 5000, processFootprint: 6000, timestamp: t.addingTimeInterval(2.0)),
            duration: 2.0
        )

        XCTAssertEqual(step.delta.mlxActiveDelta, 4900)
        XCTAssertEqual(step.delta.processFootprintDelta, 5500)
    }

    func testProfiledStepDescription() {
        let t = Date()
        let step = ProfiledStep(
            name: "VAE Decode",
            startMemory: MemorySnapshot(mlxActive: 0, mlxCache: 0, mlxPeak: 0, processFootprint: 0, timestamp: t),
            endMemory: MemorySnapshot(mlxActive: 0, mlxCache: 0, mlxPeak: 0, processFootprint: 0, timestamp: t),
            duration: 1.234
        )

        XCTAssertTrue(step.description.contains("VAE Decode"))
        XCTAssertTrue(step.description.contains("1.234"))
    }
}

// MARK: - ProfileSummary Tests

final class ProfileSummaryTests: XCTestCase {

    func testProfileSummaryComputedProperties() {
        let t = Date()
        let snap1 = MemorySnapshot(mlxActive: 1000, mlxCache: 0, mlxPeak: 1000, processFootprint: 0, timestamp: t)
        let snap2 = MemorySnapshot(mlxActive: 5000, mlxCache: 0, mlxPeak: 8000, processFootprint: 0, timestamp: t)

        let step1 = ProfiledStep(name: "Step 1", startMemory: snap1, endMemory: snap2, duration: 1.5)
        let step2 = ProfiledStep(name: "Step 2", startMemory: snap2, endMemory: snap2, duration: 2.5)

        let summary = ProfileSummary(
            deviceInfo: GPU.deviceInfo(),
            initialSnapshot: snap1,
            finalSnapshot: snap2,
            steps: [step1, step2]
        )

        XCTAssertEqual(summary.totalDuration, 4.0, accuracy: 0.001)
        XCTAssertEqual(summary.totalMemoryGrowth, 4000)
        XCTAssertEqual(summary.peakMemoryUsed, 8000)
    }

    func testProfileSummaryDescription() {
        let t = Date()
        let snap = MemorySnapshot(mlxActive: 0, mlxCache: 0, mlxPeak: 0, processFootprint: 0, timestamp: t)
        let step = ProfiledStep(name: "Test", startMemory: snap, endMemory: snap, duration: 1.0)

        let summary = ProfileSummary(
            deviceInfo: GPU.deviceInfo(),
            initialSnapshot: snap,
            finalSnapshot: snap,
            steps: [step]
        )

        let desc = summary.description
        XCTAssertTrue(desc.contains("PROFILING SUMMARY"))
        XCTAssertTrue(desc.contains("Device:"))
        XCTAssertTrue(desc.contains("Test: 1.000s"))
        XCTAssertTrue(desc.contains("Total: 1.000s"))
    }

    func testProfileSummaryEmptySteps() {
        let snap = MemorySnapshot.current()
        let summary = ProfileSummary(
            deviceInfo: GPU.deviceInfo(),
            initialSnapshot: snap,
            finalSnapshot: snap,
            steps: []
        )

        XCTAssertEqual(summary.totalDuration, 0)
        XCTAssertTrue(summary.description.contains("Total: 0.000s"))
    }

    func testProfilerSummaryCompatibility() {
        let profiler = MLXProfiler.shared
        let summary = profiler.summary()

        XCTAssertNotNil(summary.deviceInfo)
        XCTAssertTrue(summary.steps.isEmpty)
        XCTAssertGreaterThanOrEqual(summary.finalSnapshot.processFootprint, 0)
    }
}

// MARK: - withProfiling Tests

final class WithProfilingTests: XCTestCase {

    override func tearDown() {
        MLXProfiler.shared.disable()
    }

    func testWithProfilingEnabled() {
        let (result, metrics) = withProfiling(enabled: true) {
            return 42
        }

        XCTAssertEqual(result, 42)
        XCTAssertNotNil(metrics)
    }

    func testWithProfilingDisabled() {
        let (result, metrics) = withProfiling(enabled: false) {
            return "hello"
        }

        XCTAssertEqual(result, "hello")
        XCTAssertNil(metrics)
    }

    func testWithProfilingRestoresState() {
        let profiler = MLXProfiler.shared
        XCTAssertFalse(profiler.isEnabled)

        _ = withProfiling(enabled: true) { 0 }

        // Should restore to disabled
        XCTAssertFalse(profiler.isEnabled)
    }

    func testWithProfilingCapturesMetrics() {
        let (_, metrics) = withProfiling {
            // Simulate some LLM work
            MLXProfiler.shared.startTokenization()
            MLXProfiler.shared.endTokenization(tokenCount: 10)
            return true
        }

        XCTAssertNotNil(metrics)
        XCTAssertEqual(metrics?.promptTokens, 10)
    }
}

// MARK: - LLM Integration Tests (via FluxProfiler typealias)

final class LLMProfilingIntegrationTests: XCTestCase {

    override func tearDown() {
        MLXProfiler.shared.disable()
    }

    func testLLMMethodsAvailableViaFluxProfiler() {
        let profiler = FluxProfiler.shared
        profiler.enable()

        profiler.startTokenization()
        profiler.endTokenization(tokenCount: 50)
        profiler.startPrefill()
        profiler.endPrefill()
        profiler.startGeneration()
        profiler.addDecodingTime(0.01)
        profiler.endGeneration(tokenCount: 200)

        let metrics = profiler.getLLMMetrics()
        XCTAssertEqual(metrics.promptTokens, 50)
        XCTAssertEqual(metrics.generatedTokens, 200)
        XCTAssertGreaterThan(metrics.decodingTime, 0)
    }

    func testGetMetricsIncludesMemoryData() {
        let profiler = FluxProfiler.shared
        profiler.enable()

        let metrics = profiler.getMetrics()
        // Memory should be captured from system
        XCTAssertGreaterThanOrEqual(metrics.mlxActiveMemoryMB, 0)
        XCTAssertGreaterThan(metrics.processFootprintMB, 0)
    }
}

// MARK: - Category Inference Tests

final class CategoryInferenceTests: XCTestCase {

    func testFlux2PipelinePhaseNames() {
        // Verify that our Flux2 phase naming convention maps to correct categories
        XCTAssertEqual(ProfilingSession.inferCategory("1. Load Text Encoder"), .textEncoderLoad)
        XCTAssertEqual(ProfilingSession.inferCategory("2. Text Encoding"), .textEncoding)
        XCTAssertEqual(ProfilingSession.inferCategory("3. Unload Text Encoder"), .textEncoderUnload)
        XCTAssertEqual(ProfilingSession.inferCategory("4. Load Transformer"), .transformerLoad)
        XCTAssertEqual(ProfilingSession.inferCategory("5. Load VAE"), .vaeLoad)
        XCTAssertEqual(ProfilingSession.inferCategory("6. Denoising Loop"), .denoisingLoop)
        XCTAssertEqual(ProfilingSession.inferCategory("7. VAE Decode"), .vaeDecode)
        XCTAssertEqual(ProfilingSession.inferCategory("8. Post-processing"), .postProcess)
        XCTAssertEqual(ProfilingSession.inferCategory("1b. VLM Interpretation"), .vlmInterpretation)
    }

    func testLLMPhaseNames() {
        XCTAssertEqual(ProfilingSession.inferCategory("Tokenization"), .tokenization)
        XCTAssertEqual(ProfilingSession.inferCategory("Prefill"), .prefill)
        XCTAssertEqual(ProfilingSession.inferCategory("Generation"), .generation)
        XCTAssertEqual(ProfilingSession.inferCategory("Token Decoding"), .decoding)
    }
}
