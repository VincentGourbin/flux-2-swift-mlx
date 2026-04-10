// ProfileCommand.swift - CLI profiling and benchmarking commands
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core

// MARK: - Profile Command Group

struct Profile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Profile and benchmark Flux.2 inference pipeline",
        subcommands: [
            ProfileRun.self,
            ProfileBenchmark.self,
            ProfileCompare.self,
        ],
        defaultSubcommand: ProfileRun.self
    )
}

// MARK: - Shared Options

struct ProfileModelOptions: ParsableArguments {
    @Argument(help: "Text prompt for image generation")
    var prompt: String

    @Option(name: .long, help: "Model variant: dev, klein-4b, klein-9b, klein-9b-kv")
    var model: String = "klein-4b"

    @Option(name: .shortAndLong, help: "Number of inference steps")
    var steps: Int?

    @Option(name: .shortAndLong, help: "Guidance scale")
    var guidance: Float?

    @Option(name: .shortAndLong, help: "Image width")
    var width: Int = 512

    @Option(name: .shortAndLong, help: "Image height")
    var height: Int = 512

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .long, help: "Text encoder quantization: bf16, 8bit, 6bit, 4bit")
    var textQuant: String = "8bit"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4")
    var transformerQuant: String = "qint8"

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory")
    var modelsDir: String?

    @Flag(name: .long, help: "Skip saving the generated image")
    var noImage: Bool = false

    func resolveModel() throws -> Flux2Model {
        guard let m = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, klein-9b, or klein-9b-kv")
        }
        return m
    }

    func resolveQuantization() throws -> Flux2QuantizationConfig {
        guard let text = MistralQuantization(rawValue: textQuant) else {
            throw ValidationError("Invalid text quantization: \(textQuant)")
        }
        guard let transformer = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid transformer quantization: \(transformerQuant)")
        }
        return Flux2QuantizationConfig(textEncoder: text, transformer: transformer)
    }
}

// MARK: - Profile Run

struct ProfileRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Single profiled generation with Chrome Trace export"
    )

    @OptionGroup var options: ProfileModelOptions

    @Option(name: .long, help: "Output directory for trace files")
    var outputDir: String = "./profile_results"

    @Flag(name: .long, help: "Record memory at each denoising step")
    var perStepMemory: Bool = false

    @Flag(name: .long, help: "Disable Chrome Trace JSON export")
    var noChromeTrace: Bool = false

    func run() async throws {
        configureModelsDirectory(options.modelsDir)

        let modelVariant = try options.resolveModel()
        let quantConfig = try options.resolveQuantization()
        let token = options.hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        let actualSteps = options.steps ?? modelVariant.defaultSteps
        let actualGuidance = options.guidance ?? modelVariant.defaultGuidance

        // Create output directory
        try FileManager.default.createDirectory(
            atPath: outputDir, withIntermediateDirectories: true
        )

        // Configure profiling session
        let config = ProfilingConfig(
            trackMemory: true,
            trackPerStepMemory: perStepMemory,
            outputDirectory: URL(fileURLWithPath: outputDir),
            exportChromeTrace: !noChromeTrace,
            printSummary: true
        )

        let session = ProfilingSession(config: config)
        session.title = "FLUX.2 PROFILING REPORT"
        session.metadata["model"] = modelVariant.rawValue
        session.metadata["quant"] = "\(options.textQuant)/\(options.transformerQuant)"
        session.metadata["resolution"] = "\(options.width)x\(options.height)"
        session.metadata["steps"] = String(actualSteps)

        // Attach session to profiler
        let profiler = Flux2Profiler.shared
        profiler.enable()
        profiler.activeSession = session

        print("Profiling: \(modelVariant.displayName) \(options.textQuant)/\(options.transformerQuant)")
        print("Image: \(options.width)x\(options.height), \(actualSteps) steps, guidance \(actualGuidance)")
        print()

        // Create pipeline and generate
        let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig, hfToken: token)

        let image = try await pipeline.generateTextToImage(
            prompt: options.prompt,
            height: options.height,
            width: options.width,
            steps: actualSteps,
            guidance: actualGuidance,
            seed: options.seed
        ) { current, total in
            print("\rStep \(current)/\(total)", terminator: "")
            fflush(stdout)
        }
        print()

        // Save image unless skipped
        if !options.noImage {
            let imagePath = "\(outputDir)/profiled_output.png"
            try saveImage(image, to: imagePath)
            print("Image saved to \(imagePath)")
        }

        // Print report
        print(session.generateReport())

        // Export Chrome Trace
        if !noChromeTrace {
            let traceData = ChromeTraceExporter.export(session: session)
            let tracePath = "\(outputDir)/\(modelVariant.rawValue)_\(options.transformerQuant)_trace.json"
            try traceData.write(to: URL(fileURLWithPath: tracePath))
            print("Chrome Trace exported to \(tracePath)")
            print("View in Perfetto: https://ui.perfetto.dev/")
        }

        // Cleanup
        profiler.activeSession = nil
        profiler.disable()
    }
}

// MARK: - Profile Benchmark

struct ProfileBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Statistical benchmarking with warm-up and multiple runs"
    )

    @OptionGroup var options: ProfileModelOptions

    @Option(name: .long, help: "Number of warm-up runs (excluded from stats)")
    var warmup: Int = 1

    @Option(name: .long, help: "Number of measured runs")
    var runs: Int = 3

    @Option(name: .long, help: "Output directory for results")
    var outputDir: String = "./benchmark_results"

    func run() async throws {
        configureModelsDirectory(options.modelsDir)

        let modelVariant = try options.resolveModel()
        let quantConfig = try options.resolveQuantization()
        let token = options.hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        let actualSteps = options.steps ?? modelVariant.defaultSteps
        let actualGuidance = options.guidance ?? modelVariant.defaultGuidance

        // Create output directory
        try FileManager.default.createDirectory(
            atPath: outputDir, withIntermediateDirectories: true
        )

        print("Benchmarking: \(modelVariant.displayName) \(options.textQuant)/\(options.transformerQuant)")
        print("Image: \(options.width)x\(options.height), \(actualSteps) steps")
        print("Warm-up: \(warmup), Measured runs: \(runs)")
        print()

        let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig, hfToken: token)
        let profiler = Flux2Profiler.shared

        let totalRuns = warmup + runs
        var measuredSessions: [ProfilingSession] = []

        for i in 0..<totalRuns {
            let isWarmup = i < warmup
            let runLabel = isWarmup ? "Warm-up \(i + 1)/\(warmup)" : "Run \(i - warmup + 1)/\(runs)"
            print("\(runLabel)...", terminator: " ")
            fflush(stdout)

            // Create session for this run
            let config = ProfilingConfig(trackMemory: true, trackPerStepMemory: false)
            let session = ProfilingSession(config: config)
            session.title = "FLUX.2 PROFILING REPORT"
            session.metadata["model"] = modelVariant.rawValue
            session.metadata["quant"] = "\(options.textQuant)/\(options.transformerQuant)"
            session.metadata["resolution"] = "\(options.width)x\(options.height)"
            session.metadata["steps"] = String(actualSteps)

            profiler.enable()
            profiler.activeSession = session

            // Use fixed seed for reproducibility across runs
            let seed = options.seed ?? 42

            let startTime = CFAbsoluteTimeGetCurrent()
            _ = try await pipeline.generateTextToImage(
                prompt: options.prompt,
                height: options.height,
                width: options.width,
                steps: actualSteps,
                guidance: actualGuidance,
                seed: seed
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            profiler.activeSession = nil
            profiler.disable()

            print(String(format: "%.2fs", elapsed))

            if !isWarmup {
                measuredSessions.append(session)
            }
        }

        print()

        // Aggregate results
        let result = BenchmarkAggregator.aggregate(
            sessions: measuredSessions,
            warmupCount: warmup
        )
        print(result.generateReport())

        // Export comparison trace of all measured runs
        let labeledSessions = measuredSessions.enumerated().map { (i, s) in
            (label: "Run \(i + 1)", session: s)
        }
        let traceData = ChromeTraceExporter.exportComparison(sessions: labeledSessions)
        let tracePath = "\(outputDir)/benchmark_\(modelVariant.rawValue)_\(runs)runs.json"
        try traceData.write(to: URL(fileURLWithPath: tracePath))
        print("Benchmark trace exported to \(tracePath)")
    }
}

// MARK: - Profile Compare

struct ProfileCompare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compare",
        abstract: "Compare performance across model/quantization configurations"
    )

    @Argument(help: "Text prompt for image generation")
    var prompt: String

    @Option(name: .long, help: "Configurations to compare (comma-separated model:quant pairs, e.g. 'klein-4b:qint8,klein-4b:bf16')")
    var configs: String

    @Option(name: .shortAndLong, help: "Number of inference steps")
    var steps: Int?

    @Option(name: .shortAndLong, help: "Image width")
    var width: Int = 512

    @Option(name: .shortAndLong, help: "Image height")
    var height: Int = 512

    @Option(name: .long, help: "Random seed")
    var seed: UInt64 = 42

    @Option(name: .long, help: "Runs per configuration")
    var runs: Int = 1

    @Option(name: .long, help: "Text encoder quantization")
    var textQuant: String = "8bit"

    @Option(name: .long, help: "HuggingFace token")
    var hfToken: String?

    @Option(name: .long, help: "Custom models directory")
    var modelsDir: String?

    @Option(name: .long, help: "Output directory")
    var outputDir: String = "./comparison_results"

    func run() async throws {
        configureModelsDirectory(modelsDir)

        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        // Parse configurations
        let configPairs = configs.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard !configPairs.isEmpty else {
            throw ValidationError("No configurations specified. Use --configs 'klein-4b:qint8,klein-4b:bf16'")
        }

        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        print("Comparing \(configPairs.count) configurations")
        print("Image: \(width)x\(height), seed \(seed)")
        print()

        var labeledSessions: [(label: String, session: ProfilingSession)] = []
        let profiler = Flux2Profiler.shared

        for configStr in configPairs {
            let parts = configStr.split(separator: ":")
            guard parts.count == 2,
                  let modelVariant = Flux2Model(rawValue: String(parts[0])),
                  let transformerQuant = TransformerQuantization(rawValue: String(parts[1]))
            else {
                print("Skipping invalid config: \(configStr) (expected model:quant)")
                continue
            }

            guard let textQuantization = MistralQuantization(rawValue: textQuant) else {
                throw ValidationError("Invalid text quantization: \(textQuant)")
            }

            let quantConfig = Flux2QuantizationConfig(textEncoder: textQuantization, transformer: transformerQuant)
            let actualSteps = steps ?? modelVariant.defaultSteps
            let label = "\(modelVariant.rawValue) \(transformerQuant.rawValue)"

            print("Running: \(label)...")

            for runIdx in 0..<runs {
                let config = ProfilingConfig(trackMemory: true, trackPerStepMemory: false)
                let session = ProfilingSession(config: config)
                session.title = "FLUX.2 PROFILING REPORT"
                session.metadata["model"] = modelVariant.rawValue
                session.metadata["quant"] = "\(textQuant)/\(transformerQuant.rawValue)"
                session.metadata["resolution"] = "\(width)x\(height)"
                session.metadata["steps"] = String(actualSteps)

                profiler.enable()
                profiler.activeSession = session

                let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig, hfToken: token)

                let startTime = CFAbsoluteTimeGetCurrent()
                _ = try await pipeline.generateTextToImage(
                    prompt: prompt,
                    height: height,
                    width: width,
                    steps: actualSteps,
                    guidance: modelVariant.defaultGuidance,
                    seed: seed
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime

                profiler.activeSession = nil
                profiler.disable()

                let runLabel = runs > 1 ? "\(label) (run \(runIdx + 1))" : label
                print("  \(runLabel): \(String(format: "%.2fs", elapsed))")

                labeledSessions.append((label: runLabel, session: session))
            }
        }

        print()

        // Print side-by-side summary
        print("COMPARISON SUMMARY")
        print(String(repeating: "-", count: 60))

        for (label, session) in labeledSessions {
            let events = session.getEvents()
            var totalMs: Double = 0
            var beginTimestamps: [String: UInt64] = [:]

            for event in events {
                if event.phase == .begin {
                    beginTimestamps[event.name] = event.timestampUs
                } else if event.phase == .end, let beginTs = beginTimestamps[event.name] {
                    totalMs += Double(event.timestampUs - beginTs) / 1000.0
                    beginTimestamps.removeValue(forKey: event.name)
                }
            }

            let peakMLX = session.getMemoryTimeline().map(\.mlxActiveMB).max() ?? 0
            let labelPad = label.padding(toLength: 25, withPad: " ", startingAt: 0)
            print("  \(labelPad) \(String(format: "%8.2fs", totalMs / 1000))  Peak: \(String(format: "%.0f", peakMLX))MB")
        }

        print()

        // Export comparison trace
        let traceData = ChromeTraceExporter.exportComparison(sessions: labeledSessions)
        let tracePath = "\(outputDir)/comparison_trace.json"
        try traceData.write(to: URL(fileURLWithPath: tracePath))
        print("Comparison trace exported to \(tracePath)")
        print("View in Perfetto: https://ui.perfetto.dev/")
    }
}
