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

        // Generate enhanced prompt using Mistral chat (stream: false for correct UTF-8)
        let result = try FluxTextEncoders.shared.chat(
            messages: messages,
            parameters: .balanced,
            stream: false
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
            parameters: .balanced,
            stream: false
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

    /// Describe images semantically using VLM for prompt injection (using file paths)
    /// Unlike visual conditioning, this extracts semantic meaning from images
    /// (e.g., "a map showing the Eiffel Tower location" instead of using the image visually)
    /// - Parameters:
    ///   - paths: File paths to images to analyze and describe
    ///   - context: Optional context to guide the description
    /// - Returns: Array of image descriptions
    @MainActor
    public func describeImagePathsForPrompt(_ paths: [String], context: String? = nil) async throws -> [String] {
        #if os(macOS)
        guard !paths.isEmpty else {
            return []
        }

        print("[VLM-DEBUG] Starting VLM interpretation for \(paths.count) image(s)")
        print("[VLM-DEBUG] isModelLoaded=\(FluxTextEncoders.shared.isModelLoaded), isVLMLoaded=\(FluxTextEncoders.shared.isVLMLoaded)")
        fflush(stdout)

        // CRITICAL: Unload any existing text model before loading VLM
        // The text model (MistralForCausalLM) and VLM (MistralVLM) use different architectures
        // Loading VLM on top of text model can cause state corruption
        if FluxTextEncoders.shared.isModelLoaded && !FluxTextEncoders.shared.isVLMLoaded {
            print("[VLM-DEBUG] Unloading text model before VLM load...")
            fflush(stdout)
            FluxTextEncoders.shared.unloadModel()
            // Extra GPU memory clearing and synchronization
            Memory.clearCache()
            eval([])  // Force GPU synchronization
            print("[VLM-DEBUG] GPU cache cleared and synchronized")
            fflush(stdout)
        }

        // Load VLM if not already loaded
        if !FluxTextEncoders.shared.isVLMLoaded {
            print("[VLM-DEBUG] Loading VLM model (8bit)...")
            fflush(stdout)

            try await FluxTextEncoders.shared.loadVLMModel(variant: .mlx8bit) { progress, message in
                print("[VLM-DEBUG] VLM Download: \(Int(progress * 100))% - \(message)")
                fflush(stdout)
            }

            print("[VLM-DEBUG] VLM loaded!")
            fflush(stdout)
        }

        var descriptions: [String] = []

        // Parameters for VLM generation
        let params = GenerateParameters(
            maxTokens: 2048,
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.1,
            repetitionContextSize: 20
        )

        // STEP 1: I2I Upsampling - VLM analyzes each image
        for (index, path) in paths.enumerated() {
            let imageNumber = index + 1
            Flux2Debug.log("Interpreting image \(imageNumber)/\(paths.count): \(path)")

            // I2I upsampling: VLM analyzes the image with the user's context
            let i2iResult = try FluxTextEncoders.shared.analyzeImage(
                path: path,
                prompt: context ?? "",
                systemPrompt: FluxConfig.systemMessage(for: .upsamplingI2I),
                parameters: params
            ) { token in
                return true
            }

            let i2iDescription = i2iResult.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            print("[Step 1 - I2I] Image \(imageNumber) interpretation: \(i2iDescription)")
            fflush(stdout)

            descriptions.append(i2iDescription)
        }

        // STEP 2: T2I Upsampling - Enrich the combined prompt for generation
        // Combine all image interpretations with the user's request
        let combinedContext: String
        if descriptions.count == 1 {
            combinedContext = """
            Based on the image analysis: \(descriptions[0])

            User request: \(context ?? "Generate an image")
            """
        } else {
            let imageContexts = descriptions.enumerated().map { "Image \($0.offset + 1): \($0.element)" }.joined(separator: "\n")
            combinedContext = """
            Based on the image analyses:
            \(imageContexts)

            User request: \(context ?? "Generate an image")
            """
        }

        print("[Step 2 - T2I] Enriching prompt for generation...")
        fflush(stdout)

        // T2I upsampling: Transform the interpretation into a rich generation prompt
        let t2iMessages = FluxConfig.buildMessages(prompt: combinedContext, mode: .upsamplingT2I)
        let t2iResult = try FluxTextEncoders.shared.chat(
            messages: t2iMessages,
            parameters: GenerateParameters(
                maxTokens: 512,
                temperature: 0.15,  // Lower temperature for T2I as per BFL reference
                topP: 0.95,
                repetitionPenalty: 1.1
            ),
            stream: false
        )

        let finalPrompt = t2iResult.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        print("[Step 2 - T2I] Final generation prompt: \(finalPrompt)")
        fflush(stdout)

        // Return the single enriched prompt (not per-image descriptions)
        return [finalPrompt]
        #else
        Flux2Debug.log("VLM interpretation not available on this platform")
        return []
        #endif
    }

    /// Describe images semantically using VLM for prompt injection (CGImage version - deprecated)
    /// Unlike visual conditioning, this extracts semantic meaning from images
    /// (e.g., "a map showing the Eiffel Tower location" instead of using the image visually)
    /// - Parameters:
    ///   - images: Images to analyze and describe
    ///   - context: Optional context to guide the description
    /// - Returns: Array of image descriptions
    @available(*, deprecated, message: "Use describeImagePathsForPrompt instead for better image loading")
    @MainActor
    public func describeImagesForPrompt(_ images: [CGImage], context: String? = nil) async throws -> [String] {
        #if os(macOS)
        guard !images.isEmpty else {
            return []
        }

        Flux2Debug.log("Describing \(images.count) image(s) with VLM for prompt injection")

        // Load VLM model if not already loaded
        if !FluxTextEncoders.shared.isVLMLoaded {
            Flux2Debug.log("Loading VLM model for image interpretation...")

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

        var descriptions: [String] = []

        for (index, cgImage) in images.enumerated() {
            let imageNumber = index + 1
            Flux2Debug.log("Interpreting image \(imageNumber)/\(images.count)...")

            // Convert CGImage to NSImage for VLM
            let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))

            // Use I2I upsampling API as per Flux.2 reference
            let result = try FluxTextEncoders.shared.analyzeImage(
                image: nsImage,
                prompt: context ?? "",
                systemPrompt: FluxConfig.systemMessage(for: .upsamplingI2I),
                parameters: .balanced
            )

            let description = result.text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            descriptions.append(description)
            print("[VLM-Interpret] Image \(imageNumber) description: \(description)")
            fflush(stdout)
        }

        return descriptions
        #else
        Flux2Debug.log("VLM interpretation not available on this platform")
        return []
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
