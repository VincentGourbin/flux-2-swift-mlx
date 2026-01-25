// LoRAConfig.swift - LoRA adapter configuration
// Copyright 2025 Vincent Gourbin

import Foundation

/// Scheduler parameter overrides for specialized LoRAs (e.g., Turbo)
public struct SchedulerOverrides: Sendable, Codable, Equatable {
    /// Custom sigma schedule (replaces computed sigmas)
    /// Used by Turbo LoRAs that require specific noise schedules
    public var customSigmas: [Float]?

    /// Recommended number of inference steps
    public var numSteps: Int?

    /// Recommended guidance scale
    public var guidance: Float?

    /// Recommended strength for I2I
    public var strength: Float?

    public init(
        customSigmas: [Float]? = nil,
        numSteps: Int? = nil,
        guidance: Float? = nil,
        strength: Float? = nil
    ) {
        self.customSigmas = customSigmas
        self.numSteps = numSteps
        self.guidance = guidance
        self.strength = strength
    }

    /// Check if any overrides are set
    public var hasOverrides: Bool {
        customSigmas != nil || numSteps != nil || guidance != nil || strength != nil
    }
}

/// Configuration for a LoRA adapter
public struct LoRAConfig: Sendable, Codable {
    /// Path to the LoRA safetensors file
    public var filePath: String

    /// Scale factor for LoRA weights (typically 0.5 - 1.5)
    public var scale: Float?

    /// Optional activation keyword to prepend to prompt (e.g., "sks")
    public var activationKeyword: String?

    /// Optional scheduler overrides (for Turbo LoRAs, etc.)
    public var schedulerOverrides: SchedulerOverrides?

    /// Unique identifier for this LoRA (derived from filename)
    public var name: String {
        URL(fileURLWithPath: filePath).deletingPathExtension().lastPathComponent
    }

    /// Effective scale (defaults to 1.0 if not specified)
    public var effectiveScale: Float {
        scale ?? 1.0
    }

    /// Initialize LoRA configuration
    /// - Parameters:
    ///   - filePath: Path to the LoRA safetensors file
    ///   - scale: Scale factor for LoRA weights (default: 1.0)
    ///   - activationKeyword: Optional keyword to prepend to prompt
    ///   - schedulerOverrides: Optional scheduler parameter overrides
    public init(
        filePath: String,
        scale: Float? = 1.0,
        activationKeyword: String? = nil,
        schedulerOverrides: SchedulerOverrides? = nil
    ) {
        self.filePath = filePath
        self.scale = scale
        self.activationKeyword = activationKeyword
        self.schedulerOverrides = schedulerOverrides
    }

    /// Load LoRA configuration from a JSON file
    /// - Parameter path: Path to the JSON config file
    /// - Returns: Parsed LoRAConfig
    public static func load(from path: String) throws -> LoRAConfig {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(LoRAConfig.self, from: data)
    }

    /// Save LoRA configuration to a JSON file
    /// - Parameter path: Path to save the JSON config
    public func save(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: URL(fileURLWithPath: path))
    }
}

/// Information about a loaded LoRA
public struct LoRAInfo: Sendable {
    /// Number of layers affected by this LoRA
    public let numLayers: Int

    /// LoRA rank (typically 16)
    public let rank: Int

    /// Total number of parameters
    public let numParameters: Int

    /// Memory usage in MB
    public var memorySizeMB: Float {
        Float(numParameters * 4) / (1024 * 1024)  // F32 = 4 bytes
    }

    /// Target model architecture
    public enum TargetModel: String, Sendable {
        case klein4B = "klein-4b"
        case klein9B = "klein-9b"
        case dev = "dev"
        case unknown = "unknown"
    }

    public let targetModel: TargetModel
}

/// Represents a single LoRA weight pair (A and B matrices)
public struct LoRAWeightPair: @unchecked Sendable {
    /// The A matrix (down projection): [rank, input_dim]
    public let loraA: MLXArrayWrapper

    /// The B matrix (up projection): [output_dim, rank]
    public let loraB: MLXArrayWrapper

    /// The rank of this LoRA pair
    public var rank: Int {
        loraA.shape[0]
    }
}

// Wrapper to make MLXArray Sendable-compatible
import MLX

public struct MLXArrayWrapper: @unchecked Sendable {
    public let array: MLXArray

    public var shape: [Int] {
        array.shape
    }

    public init(_ array: MLXArray) {
        self.array = array
    }
}
