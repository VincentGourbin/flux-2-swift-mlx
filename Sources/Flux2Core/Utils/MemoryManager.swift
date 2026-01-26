// MemoryManager.swift - GPU and system memory management
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Memory manager for Flux.2 generation
///
/// Monitors GPU memory usage and helps manage the two-phase
/// pipeline to fit within available RAM.
public final class Flux2MemoryManager: @unchecked Sendable {

    /// Shared instance
    public static let shared = Flux2MemoryManager()

    /// System physical memory in bytes
    public var physicalMemory: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// System physical memory in GB
    public var physicalMemoryGB: Int {
        Int(physicalMemory / 1_073_741_824)
    }

    /// Estimated available memory (rough heuristic)
    public var estimatedAvailableMemoryGB: Int {
        // Reserve some memory for system
        max(0, physicalMemoryGB - 8)
    }

    private init() {}

    // MARK: - Memory Checks

    /// Check if we have enough memory for a configuration
    public func canRun(config: Flux2QuantizationConfig) -> Bool {
        let required = config.estimatedTotalMemoryGB
        return required <= estimatedAvailableMemoryGB
    }

    /// Get recommended configuration for current system
    public func recommendedConfig() -> Flux2QuantizationConfig {
        ModelRegistry.recommendedConfig(forRAMGB: physicalMemoryGB)
    }

    /// Check memory before text encoding phase
    public func checkTextEncodingPhase(config: Flux2QuantizationConfig) -> MemoryCheckResult {
        let required = config.textEncodingPhaseMemoryGB
        let available = estimatedAvailableMemoryGB

        if required > available {
            return .insufficientMemory(
                required: required,
                available: available,
                suggestion: "Use a lower text encoder quantization (4-bit instead of 8-bit)"
            )
        }

        return .ok
    }

    /// Check memory before image generation phase
    public func checkImageGenerationPhase(config: Flux2QuantizationConfig) -> MemoryCheckResult {
        let required = config.imageGenerationPhaseMemoryGB
        let available = estimatedAvailableMemoryGB

        if required > available {
            return .insufficientMemory(
                required: required,
                available: available,
                suggestion: "Use qint8 transformer quantization or reduce image size"
            )
        }

        return .ok
    }

    // MARK: - Memory Cleanup

    /// Clear GPU cache
    /// Call this between phases or periodically during generation
    public func clearCache() {
        MLX.Memory.clearCache()
        Flux2Debug.log("GPU cache cleared")
    }

    /// Suggest garbage collection
    /// Note: Swift's ARC handles this automatically, but we can hint
    public func suggestCleanup() {
        // Force autoreleasepool drain on next opportunity
        autoreleasepool { }
    }

    /// Full memory cleanup (between phases)
    public func fullCleanup() {
        clearCache()
        suggestCleanup()

        Flux2Debug.log("Full memory cleanup performed")
    }

    // MARK: - Memory Monitoring

    /// Get current memory usage summary
    public func memorySummary() -> String {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        if result == KERN_SUCCESS {
            let usedMB = info.resident_size / 1_048_576
            let physicalMB = physicalMemory / 1_048_576

            return """
            Memory Usage:
              Resident: \(usedMB) MB
              Physical: \(physicalMB) MB (\(physicalMemoryGB) GB)
              Estimated Available: ~\(estimatedAvailableMemoryGB) GB
            """
        }

        return "Memory info unavailable"
    }

    /// Log current memory state
    public func logMemoryState() {
        Flux2Debug.log(memorySummary())
    }
}

// MARK: - Memory Check Result

public enum MemoryCheckResult {
    case ok
    case insufficientMemory(required: Int, available: Int, suggestion: String)
    case warning(message: String)

    public var isOk: Bool {
        if case .ok = self { return true }
        return false
    }

    public var message: String {
        switch self {
        case .ok:
            return "Memory check passed"
        case .insufficientMemory(let required, let available, let suggestion):
            return "Insufficient memory: need ~\(required)GB, have ~\(available)GB. \(suggestion)"
        case .warning(let message):
            return "Warning: \(message)"
        }
    }
}

// MARK: - Memory-aware Generation

extension Flux2MemoryManager {

    /// Determine optimal batch size for given image dimensions
    public func optimalBatchSize(
        width: Int,
        height: Int,
        config: Flux2QuantizationConfig
    ) -> Int {
        // For now, always return 1 (single image generation)
        // Batch generation requires more sophisticated memory planning
        return 1
    }

    /// Check if image dimensions are feasible
    public func checkImageSize(width: Int, height: Int) -> MemoryCheckResult {
        let pixels = width * height

        // Very large images need more working memory
        if pixels > 2048 * 2048 {
            return .warning(message: "Large image size may cause memory pressure")
        }

        if pixels > 4096 * 4096 {
            return .insufficientMemory(
                required: 100,
                available: estimatedAvailableMemoryGB,
                suggestion: "Reduce image size to 2048x2048 or smaller"
            )
        }

        return .ok
    }
}
