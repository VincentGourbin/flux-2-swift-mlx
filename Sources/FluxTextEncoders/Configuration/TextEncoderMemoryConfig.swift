/**
 * TextEncoderMemoryConfig.swift
 * Memory optimization configuration for text encoder models
 */

import Foundation

/// Configuration for memory optimization in text encoder models
/// Controls periodic evaluation to prevent computation graph accumulation
public struct TextEncoderMemoryConfig: Sendable, Equatable {
    /// Number of layers between eval() calls (0 = disabled)
    /// Lower values = more frequent eval = lower peak memory but slightly slower
    public var evalFrequency: Int

    /// Whether to clear GPU cache after each eval
    /// Useful for very memory-constrained environments
    public var clearCacheOnEval: Bool

    public init(evalFrequency: Int, clearCacheOnEval: Bool) {
        self.evalFrequency = evalFrequency
        self.clearCacheOnEval = clearCacheOnEval
    }

    // MARK: - Presets

    /// Disabled - no periodic evaluation (fastest, highest memory)
    public static let disabled = TextEncoderMemoryConfig(
        evalFrequency: 0,
        clearCacheOnEval: false
    )

    /// Light optimization - eval every 16 layers
    public static let light = TextEncoderMemoryConfig(
        evalFrequency: 16,
        clearCacheOnEval: false
    )

    /// Moderate optimization - eval every 8 layers (recommended)
    public static let moderate = TextEncoderMemoryConfig(
        evalFrequency: 8,
        clearCacheOnEval: false
    )

    /// Aggressive optimization - eval every 4 layers with cache clearing
    public static let aggressive = TextEncoderMemoryConfig(
        evalFrequency: 4,
        clearCacheOnEval: true
    )

    /// Ultra low memory - eval every 2 layers with cache clearing
    public static let ultraLowMemory = TextEncoderMemoryConfig(
        evalFrequency: 2,
        clearCacheOnEval: true
    )

    /// Auto-select based on available RAM
    public static func recommended(forRAMGB ramGB: Int) -> TextEncoderMemoryConfig {
        switch ramGB {
        case 0..<16:
            return .ultraLowMemory
        case 16..<32:
            return .aggressive
        case 32..<64:
            return .moderate
        case 64..<128:
            return .light
        default:
            return .disabled
        }
    }
}
