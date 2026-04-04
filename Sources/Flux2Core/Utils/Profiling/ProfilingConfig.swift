// ProfilingConfig.swift - Configuration for profiling sessions
// Copyright 2025 Vincent Gourbin

import Foundation

/// Configuration for what to profile and how
public struct ProfilingConfig: Sendable {
    /// Record memory snapshots at phase transitions
    public var trackMemory: Bool

    /// Record per-step memory during denoising
    public var trackPerStepMemory: Bool

    /// Benchmark mode: number of measured runs (nil = single profiled run)
    public var benchmarkRuns: Int?

    /// Number of warm-up runs before measurement
    public var warmupRuns: Int

    /// Output directory for trace files
    public var outputDirectory: URL?

    /// Whether to export Chrome Trace JSON
    public var exportChromeTrace: Bool

    /// Whether to print summary to console
    public var printSummary: Bool

    public init(
        trackMemory: Bool = true,
        trackPerStepMemory: Bool = false,
        benchmarkRuns: Int? = nil,
        warmupRuns: Int = 1,
        outputDirectory: URL? = nil,
        exportChromeTrace: Bool = true,
        printSummary: Bool = true
    ) {
        self.trackMemory = trackMemory
        self.trackPerStepMemory = trackPerStepMemory
        self.benchmarkRuns = benchmarkRuns
        self.warmupRuns = warmupRuns
        self.outputDirectory = outputDirectory
        self.exportChromeTrace = exportChromeTrace
        self.printSummary = printSummary
    }

    /// Default config for a single profiled run
    public static let singleRun = ProfilingConfig()

    /// Config for benchmarking
    public static func benchmark(runs: Int = 3, warmup: Int = 1) -> ProfilingConfig {
        ProfilingConfig(
            trackMemory: true,
            trackPerStepMemory: false,
            benchmarkRuns: runs,
            warmupRuns: warmup,
            exportChromeTrace: false,
            printSummary: true
        )
    }

    /// Config for detailed profiling with per-step memory
    public static let detailed = ProfilingConfig(
        trackMemory: true,
        trackPerStepMemory: true,
        exportChromeTrace: true,
        printSummary: true
    )
}
