// ProfilingTests.swift - Integration tests for MLXProfiler package
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
import MLXProfiler

// MARK: - Integration Tests

/// Verify that MLXProfiler is correctly re-exported via Flux2Profiler typealias
final class ProfilingIntegrationTests: XCTestCase {

    func testFlux2ProfilerIsMLXProfiler() {
        // Typealias should resolve correctly
        let profiler: Flux2Profiler = Flux2Profiler.shared
        XCTAssertNotNil(profiler)
        XCTAssertTrue(profiler === MLXProfiler.shared)
    }

    func testProfilerEnableDisable() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        XCTAssertTrue(profiler.isEnabled)
        profiler.disable()
        XCTAssertFalse(profiler.isEnabled)
    }

    func testProfilerTimingRecording() {
        let profiler = Flux2Profiler.shared
        profiler.enable()

        profiler.start("Test Phase")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("Test Phase")

        let timings = profiler.getTimings()
        XCTAssertFalse(timings.isEmpty)
        XCTAssertEqual(timings.last?.name, "Test Phase")
        XCTAssertGreaterThan(timings.last?.duration ?? 0, 0.005)

        profiler.disable()
    }

    func testProfilerStepRecording() {
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.setTotalSteps(4)

        profiler.recordStep(duration: 1.0)
        profiler.recordStep(duration: 1.5)

        let steps = profiler.getStepTimes()
        XCTAssertEqual(steps.count, 2)

        profiler.disable()
    }

    func testProfilingSessionCreation() {
        let session = ProfilingSession()
        session.title = "FLUX.2 TEST"
        session.metadata["model"] = "klein-4b"
        session.metadata["resolution"] = "512x512"

        XCTAssertEqual(session.title, "FLUX.2 TEST")
        XCTAssertEqual(session.metadata["model"], "klein-4b")
    }

    func testChromeTraceExport() {
        let session = ProfilingSession()
        session.beginPhase("Test", category: .denoisingLoop)
        session.endPhase("Test", category: .denoisingLoop)

        let data = ChromeTraceExporter.export(session: session)
        XCTAssertGreaterThan(data.count, 0)

        // Should be valid JSON
        let json = try? JSONSerialization.jsonObject(with: data)
        XCTAssertNotNil(json)
    }

    func testGenerateReportExtension() {
        let profiler = Flux2Profiler.shared
        profiler.enable()

        profiler.start("1. Test Phase")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("1. Test Phase")

        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("FLUX.2 PERFORMANCE REPORT"))
        XCTAssertTrue(report.contains("Test Phase"))

        profiler.disable()
    }

    func testSessionBasedReport() {
        let profiler = Flux2Profiler.shared
        let session = ProfilingSession()
        session.title = "FLUX.2 TEST REPORT"

        profiler.enable()
        profiler.activeSession = session

        profiler.start("1. Load Model")
        Thread.sleep(forTimeInterval: 0.01)
        profiler.end("1. Load Model")

        // generateReport() should delegate to session when active
        let report = profiler.generateReport()
        XCTAssertTrue(report.contains("FLUX.2 TEST REPORT"))

        profiler.activeSession = nil
        profiler.disable()
    }
}
