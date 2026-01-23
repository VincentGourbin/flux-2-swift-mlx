// KleinTextEncoder.swift - Text encoding using Qwen3 for Klein models
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import FluxTextEncoders
import CoreGraphics

#if os(macOS)
import AppKit
#endif

/// Wrapper for Qwen3 text encoding for Flux.2 Klein models
///
/// Uses Qwen3 (4B or 8B) to extract hidden states from layers [9, 18, 27]
/// producing embeddings with shape:
/// - Klein 4B: [1, 512, 7680]
/// - Klein 9B: [1, 512, 12288]
public class KleinTextEncoder: @unchecked Sendable {

    /// Klein variant (4B or 9B)
    public let variant: KleinVariant

    /// Quantization level (mapped to Qwen3 variant)
    public let quantization: MistralQuantization

    /// Whether the model is loaded
    public var isLoaded: Bool { FluxTextEncoders.shared.isKleinLoaded }

    /// Maximum sequence length for embeddings
    public let maxSequenceLength: Int = 512

    public init(variant: KleinVariant, quantization: MistralQuantization = .mlx8bit) {
        self.variant = variant
        self.quantization = quantization
    }

    // MARK: - Loading

    /// Load the Qwen3 model for Klein text encoding
    /// - Parameter modelPath: Path to model directory (or nil to auto-download)
    @MainActor
    public func load(from modelPath: URL? = nil) async throws {
        Flux2Debug.log("Loading Klein text encoder (\(variant.displayName), \(quantization.displayName))...")

        if let path = modelPath {
            // Load from local path
            try await FluxTextEncoders.shared.loadKleinModel(variant: variant, from: path.path)
        } else {
            // Determine Qwen3 variant based on Klein variant and quantization
            // Note: Qwen3 package only has 8-bit and 4-bit variants, no bf16
            let qwen3Variant: Qwen3Variant
            switch (variant, quantization) {
            case (.klein4B, .bf16), (.klein4B, .mlx8bit):
                // Use 8-bit for both bf16 and 8bit (bf16 not available for Qwen3)
                qwen3Variant = .qwen3_4B_8bit
            case (.klein4B, .mlx6bit), (.klein4B, .mlx4bit):
                qwen3Variant = .qwen3_4B_4bit
            case (.klein9B, .bf16), (.klein9B, .mlx8bit):
                // Use 8-bit for both bf16 and 8bit (bf16 not available for Qwen3)
                qwen3Variant = .qwen3_8B_8bit
            case (.klein9B, .mlx6bit), (.klein9B, .mlx4bit):
                qwen3Variant = .qwen3_8B_4bit
            }

            try await FluxTextEncoders.shared.loadKleinModel(
                variant: variant,
                qwen3Variant: qwen3Variant
            ) { progress, message in
                Flux2Debug.log("Download: \(Int(progress * 100))% - \(message)")
            }
        }

        Flux2Debug.log("Klein text encoder loaded successfully")
    }

    // MARK: - Prompt Upsampling

    /// Upsample/enhance a prompt using Qwen3's text generation capability
    /// Uses the Klein T2I upsampling system message to generate more detailed prompts
    /// - Parameter prompt: Original user prompt
    /// - Returns: Enhanced prompt with more visual details
    public func upsamplePrompt(_ prompt: String) throws -> String {
        guard FluxTextEncoders.shared.isKleinLoaded else {
            throw Flux2Error.modelNotLoaded("Klein text encoder not loaded")
        }

        Flux2Debug.log("Upsampling prompt: \"\(prompt.prefix(50))...\"")

        // Build messages with Klein T2I upsampling system message
        let messages = KleinConfig.buildMessages(prompt: prompt, mode: .upsamplingT2I)

        // Generate enhanced prompt using Qwen3 chat
        let result = try FluxTextEncoders.shared.chatQwen3(
            messages: messages,
            parameters: .balanced,
            stream: false
        )

        let enhanced = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        Flux2Debug.log("Enhanced prompt: \"\(enhanced.prefix(100))...\"")

        return enhanced
    }

    // MARK: - Encoding

    /// Encode a text prompt to Klein embeddings
    /// - Parameters:
    ///   - prompt: Text prompt to encode
    ///   - upsample: Whether to enhance the prompt before encoding (default: false)
    /// - Returns: Embeddings tensor
    ///           Klein 4B: [1, 512, 7680]
    ///           Klein 9B: [1, 512, 12288]
    public func encode(_ prompt: String, upsample: Bool = false) throws -> MLXArray {
        guard FluxTextEncoders.shared.isKleinLoaded else {
            throw Flux2Error.modelNotLoaded("Klein text encoder not loaded")
        }

        // Optionally upsample the prompt
        let finalPrompt: String
        if upsample {
            finalPrompt = try upsamplePrompt(prompt)
        } else {
            finalPrompt = prompt
        }

        Flux2Debug.log("Encoding prompt: \"\(finalPrompt.prefix(50))...\"")

        // Use the Klein embedding extraction
        let embeddings = try FluxTextEncoders.shared.extractKleinEmbeddings(
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
        FluxTextEncoders.shared.unloadKleinModel()

        // Force GPU memory cleanup
        eval([])

        Flux2Debug.log("Klein text encoder unloaded")
    }

    /// Estimated memory usage in GB
    public var estimatedMemoryGB: Int {
        switch (variant, quantization) {
        case (.klein4B, .bf16): return 10
        case (.klein4B, .mlx8bit): return 5
        case (.klein4B, _): return 3
        case (.klein9B, .bf16): return 18
        case (.klein9B, .mlx8bit): return 10
        case (.klein9B, _): return 6
        }
    }
}

// MARK: - Configuration Info

extension KleinTextEncoder {

    /// Get information about the loaded model
    public var modelInfo: String {
        guard FluxTextEncoders.shared.isKleinLoaded else {
            return "Model not loaded"
        }

        return """
        Klein Text Encoder:
          Variant: \(variant.displayName)
          Quantization: \(quantization.displayName)
          Memory: ~\(estimatedMemoryGB)GB
          Output dimension: \(variant.outputDimension)
        """
    }
}
