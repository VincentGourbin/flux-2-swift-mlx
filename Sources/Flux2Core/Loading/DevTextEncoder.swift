// DevTextEncoder.swift - Text encoding using Mistral for Dev models
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import FluxTextEncoders

/// Wrapper for Mistral text encoding for Flux.2 Dev models
///
/// Uses Mistral Small 3.2 (24B) to extract hidden states from layers [10, 20, 30]
/// producing embeddings with shape: [1, 512, 15360]
///
/// This is similar to KleinTextEncoder but for Dev model which uses Mistral instead of Qwen3.
public class DevTextEncoder: @unchecked Sendable {

    /// Quantization level for Mistral
    public let quantization: MistralQuantization

    /// Whether the model is loaded
    public var isLoaded: Bool { FluxTextEncoders.shared.isModelLoaded }

    /// Maximum sequence length for embeddings
    public let maxSequenceLength: Int = 512

    /// Output embedding dimension (3 layers × 5120 hidden size)
    public let outputDimension: Int = 15360

    public init(quantization: MistralQuantization = .mlx8bit) {
        self.quantization = quantization
    }

    // MARK: - Loading

    /// Load the Mistral model for Dev text encoding
    /// - Parameter modelPath: Path to model directory (or nil to auto-download)
    @MainActor
    public func load(from modelPath: URL? = nil) async throws {
        Flux2Debug.log("Loading Dev text encoder (Mistral, \(quantization.displayName))...")

        if let path = modelPath {
            // Load from local path
            try FluxTextEncoders.shared.loadModel(from: path.path)
        } else {
            // Auto-download based on quantization
            try await FluxTextEncoders.shared.loadModel(
                variant: mistralVariant,
                progress: { progress, message in
                    Flux2Debug.log("Download: \(Int(progress * 100))% - \(message)")
                }
            )
        }

        Flux2Debug.log("Dev text encoder loaded successfully")
    }

    /// Map MistralQuantization to ModelVariant
    private var mistralVariant: ModelVariant {
        switch quantization {
        case .bf16:
            return .bf16
        case .mlx8bit:
            return .mlx8bit
        case .mlx6bit:
            return .mlx6bit
        case .mlx4bit:
            return .mlx4bit
        }
    }

    // MARK: - Encoding

    /// Encode a text prompt to Dev embeddings
    /// - Parameter prompt: Text prompt to encode
    /// - Returns: Embeddings tensor with shape [1, 512, 15360]
    public func encode(_ prompt: String) throws -> MLXArray {
        guard FluxTextEncoders.shared.isModelLoaded else {
            throw Flux2Error.modelNotLoaded("Dev text encoder not loaded")
        }

        Flux2Debug.log("Encoding prompt: \"\(prompt.prefix(50))...\"")

        // Use mflux-compatible embedding extraction (layers 10, 20, 30)
        let embeddings = try FluxTextEncoders.shared.extractMfluxEmbeddings(prompt: prompt)

        // Ensure proper sequence length padding to 512
        let paddedEmbeddings = padToMaxLength(embeddings)

        Flux2Debug.log("Embeddings shape: \(paddedEmbeddings.shape)")

        return paddedEmbeddings
    }

    /// Pad embeddings to max sequence length if needed
    private func padToMaxLength(_ embeddings: MLXArray) -> MLXArray {
        let currentLength = embeddings.shape[1]

        if currentLength >= maxSequenceLength {
            // Truncate if too long
            return embeddings[0..., 0..<maxSequenceLength, 0...]
        } else {
            // Pad with zeros if too short
            let paddingSize = maxSequenceLength - currentLength
            let padding = MLXArray.zeros([embeddings.shape[0], paddingSize, embeddings.shape[2]])
            return concatenated([embeddings, padding], axis: 1)
        }
    }

    // MARK: - Memory Management

    /// Unload the model to free memory
    @MainActor
    public func unload() {
        FluxTextEncoders.shared.unloadModel()

        // Force GPU memory cleanup
        eval([])

        Flux2Debug.log("Dev text encoder unloaded")
    }

    /// Estimated memory usage in GB
    public var estimatedMemoryGB: Int {
        switch quantization {
        case .bf16: return 48
        case .mlx8bit: return 25
        case .mlx6bit: return 19
        case .mlx4bit: return 14
        }
    }
}

// MARK: - Configuration Info

extension DevTextEncoder {

    /// Get information about the loaded model
    public var modelInfo: String {
        guard FluxTextEncoders.shared.isModelLoaded else {
            return "Model not loaded"
        }

        return """
        Dev Text Encoder:
          Model: Mistral Small 3.2 (24B)
          Quantization: \(quantization.displayName)
          Memory: ~\(estimatedMemoryGB)GB
          Output dimension: \(outputDimension)
          Layers extracted: [10, 20, 30]
        """
    }
}
