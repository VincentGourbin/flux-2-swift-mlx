// ProfilingTests.swift - Tests for profiling toolkit
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core

final class ProfilingTests: XCTestCase {

    // MARK: - ProfilingSession Tests

    func testSessionCreation() {
        let session = ProfilingSession(config: .singleRun)
        XCTAssertFalse(session.sessionId.isEmpty)
        XCTAssertTrue(session.systemRAMGB > 0)
        XCTAssertFalse(session.deviceArchitecture.isEmpty)
    }

    func testSessionMetadata() {
        let session = ProfilingSession(config: .singleRun)
        session.modelVariant = "klein-4b"
        session.quantization = "8bit/qint8"
        session.imageSize = "512x512"
        session.steps = 4

        XCTAssertEqual(session.modelVariant, "klein-4b")
        XCTAssertEqual(session.quantization, "8bit/qint8")
        XCTAssertEqual(session.imageSize, "512x512")
        XCTAssertEqual(session.steps, 4)
    }

    func testBeginEndPhaseEvents() {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))

        session.beginPhase("Test Phase", category: .textEncoding)
        // Simulate some work
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Test Phase", category: .textEncoding)

        let events = session.getEvents()
        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0].phase, .begin)
        XCTAssertEqual(events[0].name, "Test Phase")
        XCTAssertEqual(events[0].category, .textEncoding)
        XCTAssertEqual(events[1].phase, .end)
        XCTAssertTrue(events[1].timestampUs > events[0].timestampUs)
    }

    func testCompleteEvent() {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))

        session.recordComplete("Measured Op", category: .denoisingStep, durationUs: 500_000)

        let events = session.getEvents()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].phase, .complete)
        XCTAssertEqual(events[0].durationUs, 500_000)
        XCTAssertEqual(events[0].name, "Measured Op")
    }

    func testDenoisingStepRecording() {
        let config = ProfilingConfig(trackMemory: false, trackPerStepMemory: false)
        let session = ProfilingSession(config: config)

        session.recordDenoisingStep(index: 1, total: 4, durationUs: 3_200_000)
        session.recordDenoisingStep(index: 2, total: 4, durationUs: 3_100_000)
        session.recordDenoisingStep(index: 3, total: 4, durationUs: 3_150_000)
        session.recordDenoisingStep(index: 4, total: 4, durationUs: 3_050_000)

        let events = session.getEvents()
        XCTAssertEqual(events.count, 4)

        for (i, event) in events.enumerated() {
            XCTAssertEqual(event.category, .denoisingStep)
            XCTAssertEqual(event.phase, .complete)
            XCTAssertEqual(event.stepIndex, i + 1)
            XCTAssertEqual(event.totalSteps, 4)
        }
    }

    func testMemoryTracking() {
        let config = ProfilingConfig(trackMemory: true)
        let session = ProfilingSession(config: config)

        session.beginPhase("Memory Test", category: .textEncoding)
        session.endPhase("Memory Test", category: .textEncoding)

        let timeline = session.getMemoryTimeline()
        XCTAssertEqual(timeline.count, 2) // begin + end
        XCTAssertTrue(timeline[0].context.hasPrefix("begin:"))
        XCTAssertTrue(timeline[1].context.hasPrefix("end:"))
    }

    func testMemorySnapshot() {
        let session = ProfilingSession(config: .singleRun)
        session.recordMemorySnapshot(context: "test_point")

        let timeline = session.getMemoryTimeline()
        XCTAssertEqual(timeline.count, 1)
        XCTAssertEqual(timeline[0].context, "test_point")
        XCTAssertTrue(timeline[0].mlxActiveMB >= 0)
        XCTAssertTrue(timeline[0].processFootprintMB > 0)
    }

    // MARK: - Category Inference Tests

    func testCategoryInference() {
        XCTAssertEqual(ProfilingSession.inferCategory("1. Load Text Encoder"), .textEncoderLoad)
        XCTAssertEqual(ProfilingSession.inferCategory("1b. VLM Interpretation"), .vlmInterpretation)
        XCTAssertEqual(ProfilingSession.inferCategory("2. Text Encoding"), .textEncoding)
        XCTAssertEqual(ProfilingSession.inferCategory("3. Unload Text Encoder"), .textEncoderUnload)
        XCTAssertEqual(ProfilingSession.inferCategory("4. Load Transformer"), .transformerLoad)
        XCTAssertEqual(ProfilingSession.inferCategory("5. Load VAE"), .vaeLoad)
        XCTAssertEqual(ProfilingSession.inferCategory("6. Denoising Loop"), .denoisingLoop)
        XCTAssertEqual(ProfilingSession.inferCategory("7. VAE Decode"), .vaeDecode)
        XCTAssertEqual(ProfilingSession.inferCategory("8. Post-processing"), .postProcess)
        XCTAssertEqual(ProfilingSession.inferCategory("Something Else"), .custom)
    }

    // MARK: - Chrome Trace Export Tests

    func testChromeTraceExportFormat() throws {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        session.modelVariant = "klein-4b"
        session.quantization = "8bit/qint8"
        session.imageSize = "512x512"
        session.steps = 4

        session.beginPhase("1. Load Text Encoder", category: .textEncoderLoad)
        session.endPhase("1. Load Text Encoder", category: .textEncoderLoad)
        session.recordDenoisingStep(index: 1, total: 4, durationUs: 3_000_000)

        let data = ChromeTraceExporter.export(session: session)
        XCTAssertTrue(data.count > 0)

        // Verify it's valid JSON
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(json)

        // Verify traceEvents array exists
        let traceEvents = json?["traceEvents"] as? [[String: Any]]
        XCTAssertNotNil(traceEvents)
        XCTAssertTrue(traceEvents!.count > 0)

        // Find metadata events (ph: "M")
        let metadataEvents = traceEvents!.filter { ($0["ph"] as? String) == "M" }
        XCTAssertTrue(metadataEvents.count >= 2) // process_name + thread_names

        // Find process name
        let processNameEvent = metadataEvents.first {
            ($0["name"] as? String) == "process_name"
        }
        XCTAssertNotNil(processNameEvent)

        // Find begin/end events
        let beginEvents = traceEvents!.filter { ($0["ph"] as? String) == "B" }
        let endEvents = traceEvents!.filter { ($0["ph"] as? String) == "E" }
        XCTAssertEqual(beginEvents.count, 1) // Load Text Encoder begin
        XCTAssertEqual(endEvents.count, 1)   // Load Text Encoder end

        // Find complete events (steps)
        let completeEvents = traceEvents!.filter { ($0["ph"] as? String) == "X" }
        XCTAssertEqual(completeEvents.count, 1) // 1 denoising step
    }

    func testChromeTraceThreadLanes() throws {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        session.modelVariant = "test"

        session.beginPhase("Text Encoding", category: .textEncoding)
        session.endPhase("Text Encoding", category: .textEncoding)
        session.beginPhase("Denoising", category: .denoisingLoop)
        session.endPhase("Denoising", category: .denoisingLoop)
        session.beginPhase("VAE", category: .vaeDecode)
        session.endPhase("VAE", category: .vaeDecode)

        let data = ChromeTraceExporter.export(session: session)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let events = json?["traceEvents"] as? [[String: Any]] ?? []

        // Filter non-metadata events
        let nonMeta = events.filter { ($0["ph"] as? String) != "M" && ($0["ph"] as? String) != "i" && ($0["ph"] as? String) != "C" }

        // Check that different categories get different thread IDs
        let tids = Set(nonMeta.compactMap { $0["tid"] as? Int })
        XCTAssertTrue(tids.count >= 3, "Expected at least 3 different thread lanes, got \(tids)")
    }

    func testChromeTraceMemoryCounters() throws {
        let config = ProfilingConfig(trackMemory: true)
        let session = ProfilingSession(config: config)
        session.modelVariant = "test"

        session.beginPhase("Phase 1", category: .textEncoding)
        session.endPhase("Phase 1", category: .textEncoding)

        let data = ChromeTraceExporter.export(session: session)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let events = json?["traceEvents"] as? [[String: Any]] ?? []

        // Should have counter events for memory
        let counterEvents = events.filter { ($0["ph"] as? String) == "C" }
        XCTAssertTrue(counterEvents.count >= 2, "Expected memory counter events")

        // Verify counter args contain memory fields
        if let firstCounter = counterEvents.first, let args = firstCounter["args"] as? [String: Any] {
            XCTAssertNotNil(args["MLX Active (MB)"])
            XCTAssertNotNil(args["Process (MB)"])
        }
    }

    func testComparisonExport() throws {
        let session1 = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        session1.modelVariant = "klein-4b"
        session1.beginPhase("Phase A", category: .textEncoding)
        session1.endPhase("Phase A", category: .textEncoding)

        let session2 = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        session2.modelVariant = "klein-9b"
        session2.beginPhase("Phase A", category: .textEncoding)
        session2.endPhase("Phase A", category: .textEncoding)

        let data = ChromeTraceExporter.exportComparison(sessions: [
            (label: "klein-4b qint8", session: session1),
            (label: "klein-9b qint8", session: session2),
        ])

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let events = json?["traceEvents"] as? [[String: Any]] ?? []

        // Should have events from two different processes (pid 1 and pid 2)
        let pids = Set(events.compactMap { $0["pid"] as? Int })
        XCTAssertEqual(pids, [1, 2], "Expected two process IDs for comparison")
    }

    // MARK: - Benchmark Aggregation Tests

    func testBenchmarkAggregation() {
        // Create 3 mock sessions with known phase durations
        var sessions: [ProfilingSession] = []

        for i in 0..<3 {
            let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))
            session.modelVariant = "test"
            session.steps = 4

            // Simulate phases with slightly different durations
            let baseUs: UInt64 = 1_000_000 // 1 second
            let varUs = UInt64(i) * 100_000 // 0, 0.1s, 0.2s variation

            // Record begin/end pairs
            session.beginPhase("1. Load Text Encoder", category: .textEncoderLoad)
            // Manually add offset to simulate duration
            Thread.sleep(forTimeInterval: Double(baseUs + varUs) / 1_000_000)
            session.endPhase("1. Load Text Encoder", category: .textEncoderLoad)

            // Record some step durations
            session.recordDenoisingStep(index: 1, total: 4, durationUs: 500_000 + varUs)
            session.recordDenoisingStep(index: 2, total: 4, durationUs: 480_000 + varUs)

            sessions.append(session)
        }

        let result = BenchmarkAggregator.aggregate(sessions: sessions, warmupCount: 0)

        XCTAssertEqual(result.measuredRuns, 3)
        XCTAssertEqual(result.warmupRuns, 0)
        XCTAssertTrue(result.phaseStats.count >= 1)
        XCTAssertNotNil(result.stepStats)
        XCTAssertTrue(result.totalStats.meanMs > 0)

        // Verify step stats computed correctly
        if let stepStats = result.stepStats {
            XCTAssertEqual(stepStats.count, 6) // 2 steps * 3 runs
            XCTAssertTrue(stepStats.meanMs > 0)
            XCTAssertTrue(stepStats.minMs <= stepStats.meanMs)
            XCTAssertTrue(stepStats.maxMs >= stepStats.meanMs)
        }
    }

    func testBenchmarkReportGeneration() {
        let result = BenchmarkResult(
            phaseStats: [
                .init(name: "1. Load Text Encoder", meanMs: 5200, stdMs: 100, minMs: 5100, maxMs: 5300, count: 3),
                .init(name: "6. Denoising Loop", meanMs: 12800, stdMs: 200, minMs: 12600, maxMs: 13000, count: 3),
            ],
            stepStats: .init(name: "Step", meanMs: 3200, stdMs: 50, minMs: 3150, maxMs: 3250, count: 12),
            totalStats: .init(name: "TOTAL", meanMs: 18000, stdMs: 300, minMs: 17700, maxMs: 18300, count: 3),
            peakMLXActiveMB: 8500,
            peakProcessMB: 12000,
            warmupRuns: 1,
            measuredRuns: 3,
            modelVariant: "klein-4b",
            quantization: "8bit/qint8",
            imageSize: "512x512",
            steps: 4
        )

        let report = result.generateReport()
        XCTAssertTrue(report.contains("BENCHMARK"))
        XCTAssertTrue(report.contains("klein-4b"))
        XCTAssertTrue(report.contains("Load Text Encoder"))
        XCTAssertTrue(report.contains("Denoising Loop"))
        XCTAssertTrue(report.contains("TOTAL"))
        XCTAssertTrue(report.contains("Measured runs: 3"))
    }

    // MARK: - Report Generation Tests

    func testSessionReport() {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        session.modelVariant = "klein-4b"
        session.quantization = "8bit/qint8"
        session.imageSize = "512x512"
        session.steps = 4

        session.beginPhase("1. Load Text Encoder", category: .textEncoderLoad)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("1. Load Text Encoder", category: .textEncoderLoad)

        session.beginPhase("6. Denoising Loop", category: .denoisingLoop)
        for i in 1...4 {
            session.recordDenoisingStep(index: i, total: 4, durationUs: 100_000)
        }
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("6. Denoising Loop", category: .denoisingLoop)

        let report = session.generateReport()
        XCTAssertTrue(report.contains("PROFILING REPORT"))
        XCTAssertTrue(report.contains("klein-4b"))
        XCTAssertTrue(report.contains("Load Text Encoder"))
        XCTAssertTrue(report.contains("Denoising Loop"))
        XCTAssertTrue(report.contains("DENOISING STEP STATISTICS"))
    }

    // MARK: - ProfilingConfig Tests

    func testDefaultConfigs() {
        let single = ProfilingConfig.singleRun
        XCTAssertTrue(single.trackMemory)
        XCTAssertFalse(single.trackPerStepMemory)
        XCTAssertNil(single.benchmarkRuns)
        XCTAssertTrue(single.exportChromeTrace)

        let benchmark = ProfilingConfig.benchmark(runs: 5, warmup: 2)
        XCTAssertEqual(benchmark.benchmarkRuns, 5)
        XCTAssertEqual(benchmark.warmupRuns, 2)
        XCTAssertFalse(benchmark.exportChromeTrace)

        let detailed = ProfilingConfig.detailed
        XCTAssertTrue(detailed.trackPerStepMemory)
        XCTAssertTrue(detailed.exportChromeTrace)
    }

    // MARK: - Flux2Profiler Session Bridge Tests

    func testProfilerSessionBridge() {
        let profiler = Flux2Profiler.shared
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))

        profiler.enable()
        profiler.activeSession = session
        profiler.setTotalSteps(4)

        profiler.start("Test Phase")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Test Phase")

        profiler.recordStep(duration: 0.5)

        // Give the dispatch queue time to process
        Thread.sleep(forTimeInterval: 0.05)

        let events = session.getEvents()
        // Should have begin + end + complete(step)
        XCTAssertTrue(events.count >= 2, "Expected at least begin+end events, got \(events.count)")

        let beginEvents = events.filter { $0.phase == .begin }
        let endEvents = events.filter { $0.phase == .end }
        XCTAssertEqual(beginEvents.count, 1)
        XCTAssertEqual(endEvents.count, 1)

        // Cleanup
        profiler.activeSession = nil
        profiler.disable()
    }

    // MARK: - Thread Safety

    func testConcurrentAccess() {
        let session = ProfilingSession(config: ProfilingConfig(trackMemory: false))
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // Write from multiple threads simultaneously
        for i in 0..<100 {
            group.enter()
            queue.async {
                session.recordDenoisingStep(index: i, total: 100, durationUs: UInt64(i) * 1000)
                group.leave()
            }
        }

        group.wait()
        let events = session.getEvents()
        XCTAssertEqual(events.count, 100)
    }

    // MARK: - ProfilingCategory Tests

    func testCategoryThreadIds() {
        // Verify each category maps to a lane
        XCTAssertEqual(ProfilingCategory.textEncoding.threadId, 1)
        XCTAssertEqual(ProfilingCategory.textEncoderLoad.threadId, 1)
        XCTAssertEqual(ProfilingCategory.transformerLoad.threadId, 2)
        XCTAssertEqual(ProfilingCategory.denoisingStep.threadId, 2)
        XCTAssertEqual(ProfilingCategory.vaeDecode.threadId, 3)
        XCTAssertEqual(ProfilingCategory.postProcess.threadId, 4)
        XCTAssertEqual(ProfilingCategory.memoryOp.threadId, 5)
        XCTAssertEqual(ProfilingCategory.evalSync.threadId, 6)
    }

    func testCategoryThreadNames() {
        XCTAssertEqual(ProfilingCategory.textEncoding.threadName, "Text Encoding")
        XCTAssertEqual(ProfilingCategory.denoisingLoop.threadName, "Transformer")
        XCTAssertEqual(ProfilingCategory.vaeDecode.threadName, "VAE")
    }
}
