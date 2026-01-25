/**
 * FluxProfiler.swift
 * Detailed performance profiling for FluxTextEncoders
 *
 * Tracks memory (MLX GPU + system) and time for each step.
 * Uses MLX.GPU for GPU memory and mach task_info for process memory.
 *
 * References:
 * - MLX GPU: https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/GPU.swift
 * - task_vm_info: https://developer.apple.com/forums/thread/105088
 * - phys_footprint: https://gist.github.com/pejalo/671dd2f67e3877b18c38c749742350ca
 */

import Foundation
import MLX

// MARK: - Memory Snapshot

/// Complete memory snapshot combining MLX GPU and system memory
public struct MemorySnapshot: CustomStringConvertible, Sendable {
    /// MLX active memory (in MLXArrays)
    public let mlxActive: Int
    /// MLX cache memory (recyclable)
    public let mlxCache: Int
    /// MLX peak memory since last reset
    public let mlxPeak: Int
    /// Process physical footprint (most accurate for "real" memory)
    public let processFootprint: Int64
    /// Timestamp
    public let timestamp: Date

    public var mlxTotal: Int { mlxActive + mlxCache }

    public var description: String {
        """
        MLX Active: \(formatBytes(mlxActive)) | Cache: \(formatBytes(mlxCache)) | Peak: \(formatBytes(mlxPeak))
        Process: \(formatBytes(Int(processFootprint)))
        """
    }

    /// Compute delta between two snapshots
    public func delta(to other: MemorySnapshot) -> MemoryDelta {
        MemoryDelta(
            mlxActiveDelta: other.mlxActive - mlxActive,
            mlxCacheDelta: other.mlxCache - mlxCache,
            mlxPeakDelta: other.mlxPeak - mlxPeak,
            processFootprintDelta: other.processFootprint - processFootprint,
            duration: other.timestamp.timeIntervalSince(timestamp)
        )
    }
}

/// Delta between two memory snapshots
public struct MemoryDelta: CustomStringConvertible, Sendable {
    public let mlxActiveDelta: Int
    public let mlxCacheDelta: Int
    public let mlxPeakDelta: Int
    public let processFootprintDelta: Int64
    public let duration: TimeInterval

    public var description: String {
        let sign = { (v: Int) -> String in v >= 0 ? "+" : "" }
        let sign64 = { (v: Int64) -> String in v >= 0 ? "+" : "" }
        return """
        MLX: \(sign(mlxActiveDelta))\(formatBytes(mlxActiveDelta)) active, \(sign(mlxCacheDelta))\(formatBytes(mlxCacheDelta)) cache
        Process: \(sign64(processFootprintDelta))\(formatBytes(Int(processFootprintDelta)))
        Duration: \(String(format: "%.3f", duration))s
        """
    }
}

// MARK: - Step Result

/// Result of a profiled step
public struct ProfiledStep: CustomStringConvertible, Sendable {
    public let name: String
    public let startMemory: MemorySnapshot
    public let endMemory: MemorySnapshot
    public let duration: TimeInterval

    public init(name: String, startMemory: MemorySnapshot, endMemory: MemorySnapshot, duration: TimeInterval) {
        self.name = name
        self.startMemory = startMemory
        self.endMemory = endMemory
        self.duration = duration
    }

    public var delta: MemoryDelta {
        startMemory.delta(to: endMemory)
    }

    public var description: String {
        """
        [\(name)] \(String(format: "%.3f", duration))s
          Start: MLX \(formatBytes(startMemory.mlxActive)) | Process \(formatBytes(Int(startMemory.processFootprint)))
          End:   MLX \(formatBytes(endMemory.mlxActive)) | Process \(formatBytes(Int(endMemory.processFootprint)))
          Delta: MLX \(formatDeltaBytes(endMemory.mlxActive - startMemory.mlxActive)) | Process \(formatDeltaBytes(Int(endMemory.processFootprint - startMemory.processFootprint)))
        """
    }
}

// MARK: - Generation Metrics (Legacy format for compact display)

/// Performance metrics for a single generation (compact format)
public struct GenerationMetrics: Sendable {
    public let tokenizationTime: Double
    public let prefillTime: Double
    public let generationTime: Double
    public let totalTime: Double

    public let promptTokens: Int
    public let generatedTokens: Int

    public let prefillTokensPerSecond: Double
    public let generationTokensPerSecond: Double

    // MLX GPU Memory
    public let mlxActiveMemoryMB: Double
    public let mlxCacheMemoryMB: Double
    public let mlxPeakMemoryMB: Double

    // Process Memory
    public let processFootprintMB: Double

    public var summary: String {
        """
        ╔══════════════════════════════════════════════════════════════╗
        ║                    DETAILED PROFILING                        ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ TOKENIZATION                                                 ║
        ║   Time: \(String(format: "%8.2f", tokenizationTime * 1000)) ms                                       ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ PREFILL (\(String(format: "%4d", promptTokens)) tokens)                                       ║
        ║   Time: \(String(format: "%8.2f", prefillTime * 1000)) ms                                       ║
        ║   Speed: \(String(format: "%7.1f", prefillTokensPerSecond)) tok/s                                    ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ GENERATION (\(String(format: "%4d", generatedTokens)) tokens)                                    ║
        ║   Time: \(String(format: "%8.2f", generationTime * 1000)) ms                                       ║
        ║   Speed: \(String(format: "%7.1f", generationTokensPerSecond)) tok/s                                    ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ MLX MEMORY                                                   ║
        ║   Active: \(String(format: "%7.1f", mlxActiveMemoryMB)) MB                                      ║
        ║   Cache:  \(String(format: "%7.1f", mlxCacheMemoryMB)) MB                                      ║
        ║   Peak:   \(String(format: "%7.1f", mlxPeakMemoryMB)) MB                                      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ PROCESS MEMORY                                               ║
        ║   Footprint: \(String(format: "%7.1f", processFootprintMB)) MB                                   ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ TOTAL: \(String(format: "%8.2f", totalTime * 1000)) ms                                       ║
        ╚══════════════════════════════════════════════════════════════╝
        """
    }

    public var compactSummary: String {
        """
        Prefill: \(String(format: "%.0f", prefillTokensPerSecond)) tok/s (\(promptTokens) tok) | \
        Gen: \(String(format: "%.1f", generationTokensPerSecond)) tok/s (\(generatedTokens) tok) | \
        MLX: \(String(format: "%.0f", mlxActiveMemoryMB))MB active, \(String(format: "%.0f", mlxPeakMemoryMB))MB peak | \
        Proc: \(String(format: "%.0f", processFootprintMB))MB
        """
    }
}

// MARK: - Profile Summary

/// Complete profiling summary
public struct ProfileSummary: CustomStringConvertible {
    public let deviceInfo: GPU.DeviceInfo
    public let initialSnapshot: MemorySnapshot
    public let finalSnapshot: MemorySnapshot
    public let steps: [ProfiledStep]

    public var totalDuration: TimeInterval {
        steps.reduce(0) { $0 + $1.duration }
    }

    public var totalMemoryGrowth: Int {
        finalSnapshot.mlxActive - initialSnapshot.mlxActive
    }

    public var peakMemoryUsed: Int {
        finalSnapshot.mlxPeak
    }

    public var description: String {
        var lines: [String] = []

        lines.append("=" .repeated(60))
        lines.append("MISTRAL PROFILING SUMMARY")
        lines.append("=" .repeated(60))

        lines.append("")
        lines.append("Device: \(deviceInfo.architecture)")
        lines.append("System RAM: \(formatBytes(deviceInfo.memorySize))")
        lines.append("Recommended Working Set: \(formatBytes(Int(deviceInfo.maxRecommendedWorkingSetSize)))")

        lines.append("")
        lines.append("-" .repeated(60))
        lines.append("STEPS")
        lines.append("-" .repeated(60))

        for step in steps {
            let memDelta = step.endMemory.mlxActive - step.startMemory.mlxActive
            let procDelta = step.endMemory.processFootprint - step.startMemory.processFootprint
            lines.append(String(format: "%-25s %8.3fs  MLX: %+10s  Proc: %+10s",
                               (step.name as NSString).utf8String!,
                               step.duration,
                               (formatDeltaBytes(memDelta) as NSString).utf8String!,
                               (formatDeltaBytes(Int(procDelta)) as NSString).utf8String!))
        }

        lines.append("")
        lines.append("-" .repeated(60))
        lines.append("TOTALS")
        lines.append("-" .repeated(60))
        lines.append("Total Duration: \(String(format: "%.3f", totalDuration))s")
        lines.append("MLX Peak Memory: \(formatBytes(peakMemoryUsed))")
        lines.append("MLX Final Active: \(formatBytes(finalSnapshot.mlxActive))")
        lines.append("MLX Final Cache: \(formatBytes(finalSnapshot.mlxCache))")
        lines.append("Process Footprint: \(formatBytes(Int(finalSnapshot.processFootprint)))")

        lines.append("")
        lines.append("=" .repeated(60))

        return lines.joined(separator: "\n")
    }
}

// MARK: - Profiler

/// Profiler for tracking generation performance
public final class FluxProfiler: @unchecked Sendable {

    // MARK: - Singleton

    nonisolated(unsafe) public static var shared = FluxProfiler()

    // MARK: - Configuration

    nonisolated(unsafe) public var isEnabled = false

    // MARK: - Internal State

    private var tokenizationStart: CFAbsoluteTime = 0
    private var tokenizationEnd: CFAbsoluteTime = 0

    private var prefillStart: CFAbsoluteTime = 0
    private var prefillEnd: CFAbsoluteTime = 0

    private var generationStart: CFAbsoluteTime = 0
    private var generationEnd: CFAbsoluteTime = 0

    private var promptTokenCount: Int = 0
    private var generatedTokenCount: Int = 0

    private var initialSnapshot: MemorySnapshot?

    /// All recorded steps (for detailed profiling)
    public var steps: [ProfiledStep] = []

    /// Device info
    public let deviceInfo: GPU.DeviceInfo

    private let lock = NSLock()

    // MARK: - Initialization

    public init() {
        self.deviceInfo = GPU.deviceInfo()
    }

    // MARK: - Public API

    /// Take a memory snapshot
    public static func snapshot() -> MemorySnapshot {
        MemorySnapshot(
            mlxActive: Memory.activeMemory,
            mlxCache: Memory.cacheMemory,
            mlxPeak: Memory.peakMemory,
            processFootprint: getProcessMemoryFootprint(),
            timestamp: Date()
        )
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        tokenizationStart = 0
        tokenizationEnd = 0
        prefillStart = 0
        prefillEnd = 0
        generationStart = 0
        generationEnd = 0
        promptTokenCount = 0
        generatedTokenCount = 0
        decodingTime = 0
        steps.removeAll()
        Memory.peakMemory = 0
        initialSnapshot = Self.snapshot()
    }

    // MARK: - Tokenization

    public func startTokenization() {
        guard isEnabled else { return }
        lock.lock()
        tokenizationStart = CFAbsoluteTimeGetCurrent()
        lock.unlock()
    }

    public func endTokenization(tokenCount: Int) {
        guard isEnabled else { return }
        lock.lock()
        tokenizationEnd = CFAbsoluteTimeGetCurrent()
        promptTokenCount = tokenCount
        lock.unlock()
    }

    // MARK: - Prefill

    public func startPrefill() {
        guard isEnabled else { return }
        lock.lock()
        prefillStart = CFAbsoluteTimeGetCurrent()
        lock.unlock()
    }

    public func endPrefill() {
        guard isEnabled else { return }
        lock.lock()
        prefillEnd = CFAbsoluteTimeGetCurrent()
        lock.unlock()
    }

    // MARK: - Generation

    public func startGeneration() {
        guard isEnabled else { return }
        lock.lock()
        generationStart = CFAbsoluteTimeGetCurrent()
        lock.unlock()
    }

    public func endGeneration(tokenCount: Int) {
        guard isEnabled else { return }
        lock.lock()
        generationEnd = CFAbsoluteTimeGetCurrent()
        generatedTokenCount = tokenCount
        lock.unlock()
    }

    // MARK: - Decoding (for tracking decode time separately)

    private var decodingTime: Double = 0

    public func addDecodingTime(_ time: Double) {
        guard isEnabled else { return }
        lock.lock()
        decodingTime += time
        lock.unlock()
    }

    // MARK: - Profiled Steps

    /// Thread-safe helper to append a step
    private func appendStep(_ step: ProfiledStep) {
        lock.lock()
        steps.append(step)
        lock.unlock()
    }

    /// Profile a synchronous step
    public func profile<T>(_ name: String, _ block: () throws -> T) rethrows -> T {
        guard isEnabled else {
            return try block()
        }

        let startMemory = Self.snapshot()
        let startTime = Date()

        let result = try block()

        // Force evaluation of any lazy MLX operations
        eval()

        let endTime = Date()
        let endMemory = Self.snapshot()

        let step = ProfiledStep(
            name: name,
            startMemory: startMemory,
            endMemory: endMemory,
            duration: endTime.timeIntervalSince(startTime)
        )
        appendStep(step)

        return result
    }

    /// Profile an async step
    public func profileAsync<T>(_ name: String, _ block: () async throws -> T) async rethrows -> T {
        guard isEnabled else {
            return try await block()
        }

        let startMemory = Self.snapshot()
        let startTime = Date()

        let result = try await block()

        // Force evaluation of any lazy MLX operations
        eval()

        let endTime = Date()
        let endMemory = Self.snapshot()

        let step = ProfiledStep(
            name: name,
            startMemory: startMemory,
            endMemory: endMemory,
            duration: endTime.timeIntervalSince(startTime)
        )
        appendStep(step)

        return result
    }

    // MARK: - Results

    public func getMetrics() -> GenerationMetrics {
        lock.lock()
        defer { lock.unlock() }

        let tokenizationTime = tokenizationEnd - tokenizationStart
        let prefillTime = prefillEnd - prefillStart
        let generationTime = generationEnd - generationStart
        let totalTime = tokenizationTime + prefillTime + generationTime

        let prefillTPS = prefillTime > 0 ? Double(promptTokenCount) / prefillTime : 0
        let genTPS = generationTime > 0 ? Double(generatedTokenCount) / generationTime : 0

        let snapshot = Self.snapshot()

        return GenerationMetrics(
            tokenizationTime: tokenizationTime,
            prefillTime: prefillTime,
            generationTime: generationTime,
            totalTime: totalTime,
            promptTokens: promptTokenCount,
            generatedTokens: generatedTokenCount,
            prefillTokensPerSecond: prefillTPS,
            generationTokensPerSecond: genTPS,
            mlxActiveMemoryMB: Double(snapshot.mlxActive) / (1024 * 1024),
            mlxCacheMemoryMB: Double(snapshot.mlxCache) / (1024 * 1024),
            mlxPeakMemoryMB: Double(snapshot.mlxPeak) / (1024 * 1024),
            processFootprintMB: Double(snapshot.processFootprint) / (1024 * 1024)
        )
    }

    /// Get detailed summary report
    public func summary() -> ProfileSummary {
        let finalSnapshot = Self.snapshot()
        return ProfileSummary(
            deviceInfo: deviceInfo,
            initialSnapshot: initialSnapshot ?? finalSnapshot,
            finalSnapshot: finalSnapshot,
            steps: steps
        )
    }

    /// Clear MLX cache and take new snapshot
    public func clearCacheAndSnapshot() -> MemorySnapshot {
        Memory.clearCache()
        return Self.snapshot()
    }

    // MARK: - Convenience

    public static func measure<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        guard shared.isEnabled else {
            return try block()
        }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("[\(label)] \(String(format: "%.2f", elapsed * 1000)) ms")
        return result
    }

    public static func measureAsync<T>(_ label: String, _ block: () async throws -> T) async rethrows -> T {
        guard shared.isEnabled else {
            return try await block()
        }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("[\(label)] \(String(format: "%.2f", elapsed * 1000)) ms")
        return result
    }
}

// MARK: - System Memory

/// Get process physical memory footprint using task_vm_info
/// This is the most accurate measure of "real" memory usage
private func getProcessMemoryFootprint() -> Int64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)

    let result = withUnsafeMutablePointer(to: &info) { infoPtr in
        infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
        }
    }

    if result == KERN_SUCCESS {
        return Int64(info.phys_footprint)
    }
    return 0
}

// MARK: - Formatting Helpers

private func formatBytes(_ bytes: Int) -> String {
    let absBytes = abs(bytes)
    if absBytes >= 1024 * 1024 * 1024 {
        return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
    } else if absBytes >= 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    } else if absBytes >= 1024 {
        return String(format: "%.1f KB", Double(bytes) / 1024)
    }
    return "\(bytes) B"
}

private func formatDeltaBytes(_ bytes: Int) -> String {
    let sign = bytes >= 0 ? "+" : ""
    return sign + formatBytes(bytes)
}

private extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}

// MARK: - Global Convenience

public func withProfiling<T>(enabled: Bool = true, _ block: () throws -> T) rethrows -> (result: T, metrics: GenerationMetrics?) {
    let profiler = FluxProfiler.shared
    let wasEnabled = profiler.isEnabled
    profiler.isEnabled = enabled
    profiler.reset()

    let result = try block()

    let metrics = enabled ? profiler.getMetrics() : nil
    profiler.isEnabled = wasEnabled

    return (result, metrics)
}
