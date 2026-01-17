// MistralEncoder.swift - Text encoding using MistralCore
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MistralCore

/// Wrapper for MistralCore text encoding for Flux.2
///
/// Uses Mistral Small 3.2 to extract hidden states from layers [10, 20, 30]
/// producing embeddings with shape [1, 512, 15360] for Flux.2 conditioning.
public class Flux2TextEncoder: @unchecked Sendable {

    /// Quantization level
    public let quantization: MistralQuantization

    /// Whether the model is loaded
    public var isLoaded: Bool { MistralCore.shared.isModelLoaded }

    /// Maximum sequence length for embeddings
    public let maxSequenceLength: Int = 512

    public init(quantization: MistralQuantization = .mlx8bit) {
        self.quantization = quantization
    }

    // MARK: - Loading

    /// Load the Mistral model for text encoding
    /// - Parameter modelPath: Path to model directory (or nil to auto-download)
    @MainActor
    public func load(from modelPath: URL? = nil) async throws {
        Flux2Debug.log("Loading Mistral text encoder (\(quantization.displayName))...")

        // Map our quantization to MistralCore's variant
        let variant: ModelVariant
        switch quantization {
        case .bf16:
            variant = .bf16
        case .mlx8bit:
            variant = .mlx8bit
        case .mlx6bit:
            variant = .mlx6bit
        case .mlx4bit:
            variant = .mlx4bit
        }

        // Load model using MistralCore singleton
        if let path = modelPath {
            try MistralCore.shared.loadModel(from: path.path)
        } else {
            try await MistralCore.shared.loadModel(variant: variant) { progress, message in
                Flux2Debug.log("Download: \(Int(progress * 100))% - \(message)")
            }
        }

        Flux2Debug.log("Mistral text encoder loaded successfully")
    }

    // MARK: - Prompt Upsampling

    /// Upsample/enhance a prompt using Mistral's text generation capability
    /// Uses the FLUX.2 T2I upsampling system message to generate more detailed prompts
    /// - Parameter prompt: Original user prompt
    /// - Returns: Enhanced prompt with more visual details
    public func upsamplePrompt(_ prompt: String) throws -> String {
        guard MistralCore.shared.isModelLoaded else {
            throw Flux2Error.modelNotLoaded("Text encoder not loaded")
        }

        Flux2Debug.log("Upsampling prompt: \"\(prompt.prefix(50))...\"")

        // Build messages with FLUX T2I upsampling system message
        let messages = FluxConfig.buildMessages(prompt: prompt, mode: .upsamplingT2I)

        // Generate enhanced prompt using Mistral chat
        let result = try MistralCore.shared.chat(
            messages: messages,
            parameters: GenerateParameters(
                maxTokens: 512,
                temperature: 0.7,
                topP: 0.9
            )
        )

        let enhanced = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        Flux2Debug.log("Enhanced prompt: \"\(enhanced.prefix(100))...\"")

        return enhanced
    }

    // MARK: - Encoding

    /// Encode a text prompt to Flux.2 embeddings
    /// - Parameters:
    ///   - prompt: Text prompt to encode
    ///   - upsample: Whether to enhance the prompt before encoding (default: false)
    /// - Returns: Embeddings tensor [1, 512, 15360]
    public func encode(_ prompt: String, upsample: Bool = false) throws -> MLXArray {
        guard MistralCore.shared.isModelLoaded else {
            throw Flux2Error.modelNotLoaded("Text encoder not loaded")
        }

        // Optionally upsample the prompt
        let finalPrompt: String
        if upsample {
            finalPrompt = try upsamplePrompt(prompt)
        } else {
            finalPrompt = prompt
        }

        Flux2Debug.log("Encoding prompt: \"\(finalPrompt.prefix(50))...\"")

        // Use the FLUX-compatible embedding extraction
        let embeddings = try MistralCore.shared.extractFluxEmbeddings(
            prompt: finalPrompt,
            maxLength: maxSequenceLength
        )

        Flux2Debug.log("Embeddings shape: \(embeddings.shape)")

        return embeddings
    }

    // MARK: - Memory Management

    /// Unload the model to free memory
    @MainActor
    public func unload() {
        MistralCore.shared.unloadModel()

        // Force GPU memory cleanup
        eval([])

        Flux2Debug.log("Text encoder unloaded")
    }

    /// Estimated memory usage in GB
    public var estimatedMemoryGB: Int {
        quantization.estimatedMemoryGB
    }
}

// MARK: - Batch Encoding

extension Flux2TextEncoder {

    /// Encode multiple prompts (for batch generation)
    /// - Parameters:
    ///   - prompts: Array of text prompts
    ///   - upsample: Whether to enhance prompts before encoding (default: false)
    /// - Returns: Stacked embeddings [B, 512, 15360]
    public func encodeBatch(_ prompts: [String], upsample: Bool = false) throws -> MLXArray {
        guard !prompts.isEmpty else {
            throw Flux2Error.invalidConfiguration("Empty prompt list")
        }

        var embeddings: [MLXArray] = []

        for prompt in prompts {
            let emb = try encode(prompt, upsample: upsample)
            embeddings.append(emb)
        }

        // Stack along batch dimension
        return stacked(embeddings, axis: 0).squeezed(axis: 1)
    }
}

// MARK: - Configuration Info

extension Flux2TextEncoder {

    /// Get information about the loaded model
    public var modelInfo: String {
        guard MistralCore.shared.isModelLoaded else {
            return "Model not loaded"
        }

        return """
        Mistral Text Encoder:
          Quantization: \(quantization.displayName)
          Memory: ~\(estimatedMemoryGB)GB
        """
    }
}
