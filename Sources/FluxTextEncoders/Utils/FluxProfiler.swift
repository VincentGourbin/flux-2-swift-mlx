// FluxProfiler.swift - Performance Profiling for FluxTextEncoders
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

// MARK: - Profiler (re-exported from MLXProfiler package)

@_exported import MLXProfiler

/// Backward-compatible alias for FluxProfiler
public typealias FluxProfiler = MLXProfiler

// MARK: - Legacy Compatibility

/// Performance metrics for a single LLM generation (legacy format).
/// Wraps LLMMetrics from MLXProfiler with the original API surface.
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

// MARK: - MLXProfiler extension for legacy getMetrics()

extension MLXProfiler {
    /// Get generation metrics in the legacy GenerationMetrics format.
    /// Combines LLM timing data with current memory snapshot.
    public func getMetrics() -> GenerationMetrics {
        let llm = getLLMMetrics()
        let mem = SystemMetrics.mlxMemory()
        let footprint = SystemMetrics.processFootprint()

        return GenerationMetrics(
            tokenizationTime: llm.tokenizationTime,
            prefillTime: llm.prefillTime,
            generationTime: llm.generationTime,
            totalTime: llm.totalTime,
            promptTokens: llm.promptTokens,
            generatedTokens: llm.generatedTokens,
            prefillTokensPerSecond: llm.prefillTokensPerSecond,
            generationTokensPerSecond: llm.generationTokensPerSecond,
            mlxActiveMemoryMB: Double(mem.activeBytes) / (1024 * 1024),
            mlxCacheMemoryMB: Double(mem.cacheBytes) / (1024 * 1024),
            mlxPeakMemoryMB: Double(mem.peakBytes) / (1024 * 1024),
            processFootprintMB: Double(footprint) / (1024 * 1024)
        )
    }
}

// MARK: - Legacy Types (kept for Flux2App compatibility)

/// Memory snapshot for profiling
public struct MemorySnapshot: CustomStringConvertible, Sendable {
    public let mlxActive: Int
    public let mlxCache: Int
    public let mlxPeak: Int
    public let processFootprint: Int64
    public let timestamp: Date

    public var mlxTotal: Int { mlxActive + mlxCache }

    public var description: String {
        "MLX Active: \(mlxActive / 1_048_576)MB | Cache: \(mlxCache / 1_048_576)MB | Peak: \(mlxPeak / 1_048_576)MB"
    }

    public func delta(to other: MemorySnapshot) -> MemoryDelta {
        MemoryDelta(
            mlxActiveDelta: other.mlxActive - mlxActive,
            mlxCacheDelta: other.mlxCache - mlxCache,
            mlxPeakDelta: other.mlxPeak - mlxPeak,
            processFootprintDelta: other.processFootprint - processFootprint,
            duration: other.timestamp.timeIntervalSince(timestamp)
        )
    }

    public static func current() -> MemorySnapshot {
        let mem = SystemMetrics.mlxMemory()
        return MemorySnapshot(
            mlxActive: mem.activeBytes,
            mlxCache: mem.cacheBytes,
            mlxPeak: mem.peakBytes,
            processFootprint: SystemMetrics.processFootprint(),
            timestamp: Date()
        )
    }
}

public struct MemoryDelta: CustomStringConvertible, Sendable {
    public let mlxActiveDelta: Int
    public let mlxCacheDelta: Int
    public let mlxPeakDelta: Int
    public let processFootprintDelta: Int64
    public let duration: TimeInterval

    public var description: String {
        "MLX: \(mlxActiveDelta >= 0 ? "+" : "")\(mlxActiveDelta / 1_048_576)MB active"
    }
}

public struct ProfiledStep: CustomStringConvertible, Sendable {
    public let name: String
    public let startMemory: MemorySnapshot
    public let endMemory: MemorySnapshot
    public let duration: TimeInterval

    public var delta: MemoryDelta { startMemory.delta(to: endMemory) }

    public var description: String {
        "[\(name)] \(String(format: "%.3f", duration))s"
    }
}

public struct ProfileSummary: CustomStringConvertible {
    public let deviceInfo: GPU.DeviceInfo
    public let initialSnapshot: MemorySnapshot
    public let finalSnapshot: MemorySnapshot
    public let steps: [ProfiledStep]

    public var totalDuration: TimeInterval { steps.reduce(0) { $0 + $1.duration } }
    public var totalMemoryGrowth: Int { finalSnapshot.mlxActive - initialSnapshot.mlxActive }
    public var peakMemoryUsed: Int { finalSnapshot.mlxPeak }

    public var description: String {
        var lines = ["PROFILING SUMMARY", "Device: \(deviceInfo.architecture)"]
        for step in steps {
            lines.append("\(step.name): \(String(format: "%.3f", step.duration))s")
        }
        lines.append("Total: \(String(format: "%.3f", totalDuration))s")
        return lines.joined(separator: "\n")
    }
}

// MARK: - MLXProfiler summary() compatibility

extension MLXProfiler {
    /// Legacy summary() method for Flux2App compatibility
    public func summary() -> ProfileSummary {
        let finalSnapshot = MemorySnapshot.current()
        return ProfileSummary(
            deviceInfo: GPU.deviceInfo(),
            initialSnapshot: finalSnapshot,
            finalSnapshot: finalSnapshot,
            steps: []
        )
    }
}

// MARK: - Global Convenience

public func withProfiling<T>(enabled: Bool = true, _ block: () throws -> T) rethrows -> (result: T, metrics: GenerationMetrics?) {
    let profiler = MLXProfiler.shared
    let wasEnabled = profiler.isEnabled
    if enabled { profiler.enable() }

    let result = try block()

    let metrics = enabled ? profiler.getMetrics() : nil
    if !wasEnabled { profiler.disable() }

    return (result, metrics)
}
