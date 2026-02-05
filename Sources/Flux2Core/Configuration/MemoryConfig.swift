// MemoryConfig.swift - Centralized memory management configuration
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Centralized memory configuration for GPU cache management
///
/// Provides dynamic memory management that adapts to:
/// - System RAM availability
/// - Image resolution being generated
/// - Model size (Dev vs Klein)
///
/// On high-RAM systems (64GB+), allows more cache for better performance.
/// On lower-RAM systems, uses conservative limits to prevent OOM.
public struct MemoryConfig {

    // MARK: - Constants

    private static let MB = 1024 * 1024
    private static let GB = 1024 * 1024 * 1024

    // MARK: - Cache Profiles

    /// Memory profile presets - used as hints for dynamic calculation
    public enum CacheProfile: String, CaseIterable, Sendable {
        case conservative  // Minimize memory usage (slower)
        case balanced      // Balance speed and memory
        case performance   // Maximize speed (more memory)
        case auto          // Dynamic based on available RAM

        /// Human-readable description
        public var description: String {
            switch self {
            case .conservative: return "Conservative - Minimize memory (may be slower)"
            case .balanced: return "Balanced - Good tradeoff"
            case .performance: return "Performance - Maximize speed (uses more memory)"
            case .auto: return "Auto - Dynamic based on available RAM"
            }
        }
    }

    // MARK: - System Information

    /// Get system RAM in GB
    public static var systemRAMGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / UInt64(GB))
    }

    /// Estimate available RAM in GB (rough heuristic)
    /// Reserves ~8GB for system + other apps
    public static var estimatedAvailableRAMGB: Int {
        max(8, systemRAMGB - 8)
    }

    /// Calculate what percentage of RAM we can safely use for GPU cache
    /// Higher RAM systems can afford to use more cache
    public static var safeCachePercentage: Double {
        let ram = systemRAMGB
        switch ram {
        case ..<24: return 0.03   // 3% - Very conservative for 16-24GB
        case 24..<48: return 0.05 // 5% - Conservative for 24-48GB
        case 48..<96: return 0.08 // 8% - Moderate for 48-96GB
        case 96..<192: return 0.12 // 12% - Performance for 96-192GB
        default: return 0.15      // 15% - High performance for 192GB+
        }
    }

    // MARK: - Dynamic Cache Calculation

    /// Calculate recommended cache limit based on system RAM
    /// Returns bytes, or nil for unlimited
    public static func recommendedCacheLimit() -> Int? {
        let ram = systemRAMGB

        // Very high RAM systems (128GB+) can use unlimited cache
        if ram >= 128 {
            return nil
        }

        // Calculate based on percentage of RAM
        let cacheBytes = Int(Double(ram * GB) * safeCachePercentage)

        // Clamp to reasonable bounds
        let minCache = 256 * MB   // At least 256 MB
        let maxCache = 8 * GB     // At most 8 GB

        return min(max(cacheBytes, minCache), maxCache)
    }

    /// Calculate cache limit for a specific resolution
    /// Larger images need more cache for intermediate computations
    public static func cacheLimitForResolution(width: Int, height: Int, model: Flux2Model) -> Int {
        let pixels = width * height
        let baseLimit = recommendedCacheLimit() ?? (4 * GB)

        // Scale factor based on resolution
        // 512x512 = 262144 pixels (base)
        // 1024x1024 = 1048576 pixels (4x)
        // 2048x2048 = 4194304 pixels (16x)
        let basePixels = 512 * 512
        let scaleFactor = max(1.0, sqrt(Double(pixels) / Double(basePixels)))

        // Model size factor
        let modelFactor: Double
        switch model {
        case .dev: modelFactor = 1.5      // Dev is larger
        case .klein9B: modelFactor = 1.2  // Klein 9B is medium
        case .klein4B: modelFactor = 1.0  // Klein 4B is base
        }

        let adjustedLimit = Int(Double(baseLimit) * scaleFactor * modelFactor)

        // Clamp to available RAM (leave at least 8GB for system)
        let maxAllowed = (systemRAMGB - 8) * GB
        return min(adjustedLimit, maxAllowed)
    }

    // MARK: - Profile-based Limits (for manual override)

    /// Get cache limit for a specific profile
    public static func cacheLimitForProfile(_ profile: CacheProfile) -> Int? {
        switch profile {
        case .auto:
            return recommendedCacheLimit()
        case .conservative:
            return 512 * MB
        case .balanced:
            return min(2 * GB, (systemRAMGB / 32) * GB)
        case .performance:
            return min(4 * GB, (systemRAMGB / 16) * GB)
        }
    }

    // MARK: - Cache Limit Application

    /// Apply cache limit dynamically based on system RAM
    public static func applyDynamicCacheLimit() {
        if let limit = recommendedCacheLimit() {
            Memory.cacheLimit = limit
            Flux2Debug.log("GPU cache limit: \(limit / MB) MB (dynamic, \(systemRAMGB) GB RAM)")
        } else {
            Flux2Debug.log("GPU cache: unlimited (\(systemRAMGB) GB RAM)")
        }
    }

    /// Apply cache limit for a specific profile
    public static func applyCacheLimit(_ profile: CacheProfile) {
        if let limit = cacheLimitForProfile(profile) {
            Memory.cacheLimit = limit
            Flux2Debug.log("GPU cache limit: \(limit / MB) MB (\(profile.rawValue) profile)")
        } else {
            Flux2Debug.log("GPU cache: unlimited (\(profile.rawValue) profile)")
        }
    }

    /// Apply cache limit with specific byte value
    public static func applyCacheLimit(bytes: Int) {
        Memory.cacheLimit = bytes
        Flux2Debug.log("GPU cache limit: \(bytes / MB) MB")
    }

    /// Clear GPU cache
    public static func clearCache() {
        Memory.clearCache()
    }

    // MARK: - Phase-Specific Limits

    /// Per-phase cache limits for granular control during generation
    public struct PhaseLimits: Sendable {
        public let textEncoding: Int   // Text encoder phase
        public let denoising: Int      // Transformer denoising loop
        public let vaeDecoding: Int    // VAE decode phase

        public init(textEncoding: Int, denoising: Int, vaeDecoding: Int) {
            self.textEncoding = textEncoding
            self.denoising = denoising
            self.vaeDecoding = vaeDecoding
        }

        /// Get dynamic phase limits based on model and system RAM
        /// Automatically scales based on available memory
        public static func forModel(_ model: Flux2Model, profile: CacheProfile) -> PhaseLimits {
            // Use dynamic calculation for auto profile
            if profile == .auto {
                return dynamicLimitsForModel(model)
            }

            // Manual profile overrides
            switch (model, profile) {
            // Dev model (large Mistral encoder + large transformer)
            case (.dev, .conservative):
                return PhaseLimits(textEncoding: 512 * MB, denoising: 1 * GB, vaeDecoding: 512 * MB)
            case (.dev, .balanced):
                return PhaseLimits(textEncoding: 1 * GB, denoising: 2 * GB, vaeDecoding: 1 * GB)
            case (.dev, .performance):
                return PhaseLimits(textEncoding: 2 * GB, denoising: 4 * GB, vaeDecoding: 2 * GB)

            // Klein 4B (smaller, more memory efficient)
            case (.klein4B, .conservative):
                return PhaseLimits(textEncoding: 256 * MB, denoising: 512 * MB, vaeDecoding: 256 * MB)
            case (.klein4B, .balanced):
                return PhaseLimits(textEncoding: 512 * MB, denoising: 1 * GB, vaeDecoding: 512 * MB)
            case (.klein4B, .performance):
                return PhaseLimits(textEncoding: 1 * GB, denoising: 2 * GB, vaeDecoding: 1 * GB)

            // Klein 9B (medium size)
            case (.klein9B, .conservative):
                return PhaseLimits(textEncoding: 512 * MB, denoising: 1 * GB, vaeDecoding: 512 * MB)
            case (.klein9B, .balanced):
                return PhaseLimits(textEncoding: 1 * GB, denoising: 2 * GB, vaeDecoding: 1 * GB)
            case (.klein9B, .performance):
                return PhaseLimits(textEncoding: 2 * GB, denoising: 3 * GB, vaeDecoding: 2 * GB)

            default:
                return dynamicLimitsForModel(model)
            }
        }

        /// Calculate dynamic limits based on available RAM
        private static func dynamicLimitsForModel(_ model: Flux2Model) -> PhaseLimits {
            let baseLimit = recommendedCacheLimit() ?? (4 * GB)

            // Model-specific ratios
            let (textRatio, denoiseRatio, vaeRatio): (Double, Double, Double)
            switch model {
            case .dev:
                // Dev has large text encoder, needs more for text phase
                textRatio = 0.4
                denoiseRatio = 1.0
                vaeRatio = 0.5
            case .klein9B:
                textRatio = 0.3
                denoiseRatio = 0.8
                vaeRatio = 0.4
            case .klein4B:
                textRatio = 0.25
                denoiseRatio = 0.6
                vaeRatio = 0.3
            }

            return PhaseLimits(
                textEncoding: Int(Double(baseLimit) * textRatio),
                denoising: Int(Double(baseLimit) * denoiseRatio),
                vaeDecoding: Int(Double(baseLimit) * vaeRatio)
            )
        }
    }

    // MARK: - Memory Monitoring

    /// Log current memory state
    public static func logMemoryState(context: String = "") {
        let prefix = context.isEmpty ? "" : "[\(context)] "
        let recommended = recommendedCacheLimit()
        let limitStr = recommended.map { "\($0 / MB) MB" } ?? "unlimited"
        Flux2Debug.log("\(prefix)System: \(systemRAMGB) GB RAM, recommended cache: \(limitStr)")
    }

    /// Get a summary of current memory configuration
    public static var configurationSummary: String {
        let recommended = recommendedCacheLimit()
        let limitStr = recommended.map { "\($0 / MB) MB" } ?? "unlimited"
        return """
        Memory Configuration:
          System RAM: \(systemRAMGB) GB
          Safe cache %: \(Int(safeCachePercentage * 100))%
          Recommended cache: \(limitStr)
        """
    }
}
