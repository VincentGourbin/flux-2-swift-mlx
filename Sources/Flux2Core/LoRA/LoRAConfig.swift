// LoRAConfig.swift - LoRA adapter configuration
// Copyright 2025 Vincent Gourbin

import Foundation

/// Configuration for a LoRA adapter
public struct LoRAConfig: Sendable {
    /// Path to the LoRA safetensors file
    public let filePath: String

    /// Scale factor for LoRA weights (typically 0.5 - 1.5)
    public var scale: Float

    /// Optional activation keyword to prepend to prompt (e.g., "sks")
    public var activationKeyword: String?

    /// Unique identifier for this LoRA (derived from filename)
    public var name: String {
        URL(fileURLWithPath: filePath).deletingPathExtension().lastPathComponent
    }

    /// Initialize LoRA configuration
    /// - Parameters:
    ///   - filePath: Path to the LoRA safetensors file
    ///   - scale: Scale factor for LoRA weights (default: 1.0)
    ///   - activationKeyword: Optional keyword to prepend to prompt
    public init(filePath: String, scale: Float = 1.0, activationKeyword: String? = nil) {
        self.filePath = filePath
        self.scale = scale
        self.activationKeyword = activationKeyword
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
