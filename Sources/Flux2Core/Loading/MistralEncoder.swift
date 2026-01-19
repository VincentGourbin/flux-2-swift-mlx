// MistralEncoder.swift - Text encoding using MistralCore
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import FluxTextEncoders
import CoreGraphics
#if os(macOS)
import AppKit
#endif

/// Wrapper for MistralCore text encoding for Flux.2
///
/// Uses Mistral Small 3.2 to extract hidden states from layers [10, 20, 30]
/// producing embeddings with shape [1, 512, 15360] for Flux.2 conditioning.
public class Flux2TextEncoder: @unchecked Sendable {

    /// Quantization level
    public let quantization: MistralQuantization

    /// Whether the model is loaded
    public var isLoaded: Bool { FluxTextEncoders.shared.isModelLoaded }

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
            try FluxTextEncoders.shared.loadModel(from: path.path)
        } else {
            try await FluxTextEncoders.shared.loadModel(variant: variant) { progress, message in
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
        guard FluxTextEncoders.shared.isModelLoaded else {
            throw Flux2Error.modelNotLoaded("Text encoder not loaded")
        }

        Flux2Debug.log("Upsampling prompt: \"\(prompt.prefix(50))...\"")

        // Build messages with FLUX T2I upsampling system message
        let messages = FluxConfig.buildMessages(prompt: prompt, mode: .upsamplingT2I)

        // Generate enhanced prompt using Mistral chat
        let result = try FluxTextEncoders.shared.chat(
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

    /// Upsample/enhance a prompt using Mistral VLM's vision capability
    /// Analyzes each reference image and incorporates descriptions into the prompt
    /// - Parameters:
    ///   - prompt: Original user prompt
    ///   - images: Reference images to analyze
    /// - Returns: Enhanced prompt with image descriptions
    @MainActor
    public func upsamplePromptWithImages(_ prompt: String, images: [CGImage]) async throws -> String {
        #if os(macOS)
        guard !images.isEmpty else {
            // Fall back to text-only upsampling if no images
            return try upsamplePrompt(prompt)
        }

        Flux2Debug.log("Upsampling prompt with \(images.count) reference image(s)")

        // Load VLM model if not already loaded
        if !FluxTextEncoders.shared.isVLMLoaded {
            Flux2Debug.log("Loading VLM model for image analysis...")

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

            try await FluxTextEncoders.shared.loadVLMModel(variant: variant) { progress, message in
                Flux2Debug.log("VLM Download: \(Int(progress * 100))% - \(message)")
            }
            Flux2Debug.log("VLM model loaded successfully")
        }

        // Analyze each image
        var imageDescriptions: [String] = []

        for (index, cgImage) in images.enumerated() {
            let imageNumber = index + 1
            Flux2Debug.log("Analyzing image \(imageNumber)/\(images.count)...")

            // Convert CGImage to NSImage for VLM
            let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))

            // Analyze the image with VLM
            let analysisPrompt = "Describe this image in detail. Focus on the main subject, colors, style, and any notable elements."

            let result = try FluxTextEncoders.shared.analyzeImage(
                image: nsImage,
                prompt: analysisPrompt,
                parameters: GenerateParameters(
                    maxTokens: 200,
                    temperature: 0.3,
                    topP: 0.9
                )
            )

            let description = result.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            imageDescriptions.append("Image \(imageNumber): \(description)")
            print("[VLM-Upsample] Image \(imageNumber) description: \(description)")
            fflush(stdout)
        }

        // Build enhanced prompt with image context
        let imageContext = imageDescriptions.joined(separator: "\n")
        let enhancedPrompt = """
        Reference images context:
        \(imageContext)

        User request: \(prompt)

        Generate an image that combines elements from the reference images according to the user's request.
        """

        print("[VLM-Upsample] Enhanced prompt with image context:\n\(enhancedPrompt)")
        fflush(stdout)

        // Use T2I upsampling mode (not I2I) because:
        // - I2I mode is for single-image editing: "convert editing requests into 50-80 word instructions"
        // - T2I mode expands prompts with visual details, which is what we need for multi-image compositing
        let messages = FluxConfig.buildMessages(prompt: enhancedPrompt, mode: .upsamplingT2I)

        let chatResult = try FluxTextEncoders.shared.chat(
            messages: messages,
            parameters: GenerateParameters(
                maxTokens: 512,
                temperature: 0.7,
                topP: 0.9
            )
        )

        let finalPrompt = chatResult.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        print("[VLM-Upsample] Final enhanced prompt:\n\(finalPrompt)")
        fflush(stdout)

        return finalPrompt
        #else
        // Fall back to text-only upsampling on non-macOS platforms
        return try upsamplePrompt(prompt)
        #endif
    }

    // MARK: - Encoding

    /// Encode a text prompt to Flux.2 embeddings
    /// - Parameters:
    ///   - prompt: Text prompt to encode
    ///   - upsample: Whether to enhance the prompt before encoding (default: false)
    /// - Returns: Embeddings tensor [1, 512, 15360]
    public func encode(_ prompt: String, upsample: Bool = false) throws -> MLXArray {
        guard FluxTextEncoders.shared.isModelLoaded else {
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
        let embeddings = try FluxTextEncoders.shared.extractFluxEmbeddings(
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
        FluxTextEncoders.shared.unloadModel()

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
        guard FluxTextEncoders.shared.isModelLoaded else {
            return "Model not loaded"
        }

        return """
        Mistral Text Encoder:
          Quantization: \(quantization.displayName)
          Memory: ~\(estimatedMemoryGB)GB
        """
    }
}
