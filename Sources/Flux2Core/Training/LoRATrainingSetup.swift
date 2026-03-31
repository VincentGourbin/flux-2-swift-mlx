// LoRATrainingSetup.swift - High-level API for setting up LoRA training with VLM evaluation
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import FluxTextEncoders
import CoreGraphics
import ImageIO

// MARK: - Training Setup Result

/// Complete training setup with all parameters ready for training
public struct LoRATrainingSetup: @unchecked Sendable {
    /// Reference image used for VLM evaluation during training
    public let referenceImage: CGImage
    public let referenceImagePath: URL

    /// VLM-generated prompt used for validation image generation
    public let validationPrompt: String

    /// LoRA context (name, description)
    public let context: LoRAContext

    /// Pre-training evaluation (baseline scores)
    public let evaluation: LoRAEvaluation

    /// Recommended training config (ready for training)
    public let recommendation: LoRARecommendation
}

// MARK: - High-Level Setup API

/// High-level API for preparing LoRA training with VLM-supervised evaluation
public class LoRATrainingSetup_API {

    public init() {}

    /// Describe a reference image for use as a validation prompt
    /// Uses the VLM to generate a FLUX.2-compatible description focused on the subject
    /// - Parameters:
    ///   - image: Reference image (CGImage)
    ///   - triggerWord: Trigger word to prepend to the prompt
    /// - Returns: Validation prompt string
    public func describeReferenceForValidation(
        image: CGImage,
        triggerWord: String
    ) throws -> String {
        guard FluxTextEncoders.shared.isQwen35VLMLoaded else {
            throw Flux2Error.modelNotLoaded("Qwen3.5 VLM not loaded")
        }

        let result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
            image: image,
            prompt: "Describe this person's physical appearance for image generation. Focus on: face shape, hair color and style, glasses, clothing, pose, and lighting. Be concise (one paragraph).",
            enableThinking: false,
            maxTokens: 200,
            temperature: 0
        )

        // Prepend trigger word
        return "\(triggerWord), \(result.text)"
    }

    /// Describe a reference image from file path
    public func describeReferenceForValidation(
        path: String,
        triggerWord: String
    ) throws -> String {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Flux2Error.invalidConfiguration("Failed to load image: \(path)")
        }
        return try describeReferenceForValidation(image: image, triggerWord: triggerWord)
    }

    /// Create a complete training setup with VLM evaluation
    /// Chains: reference → describe → evaluate baseline → recommend parameters
    /// - Parameters:
    ///   - referenceImagePath: Path to the reference image for VLM evaluation
    ///   - context: LoRA training context (name + description)
    ///   - model: Target model (klein-4b, klein-9b, dev)
    ///   - datasetPath: Path to training dataset
    ///   - triggerWord: Trigger word for the LoRA
    ///   - quantization: Quantization config for baseline generation
    ///   - seed: Random seed for reproducibility
    ///   - onProgress: Progress callback
    /// - Returns: Complete training setup with all parameters
    public func createEvaluatedTrainingConfig(
        referenceImagePath: String,
        context: LoRAContext,
        model: Flux2Model,
        datasetPath: String,
        triggerWord: String,
        quantization: Flux2QuantizationConfig = .init(textEncoder: .mlx8bit, transformer: .qint8),
        seed: UInt64 = 42,
        hfToken: String? = nil,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> LoRATrainingSetup {

        // Load reference image
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: referenceImagePath) as CFURL, nil),
              let refImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Flux2Error.invalidConfiguration("Failed to load reference image: \(referenceImagePath)")
        }

        // Step 1: Run full evaluation pipeline
        onProgress?("Running LoRA evaluation pipeline...")
        let evaluator = LoRAEvaluator()
        let evaluation = try await evaluator.evaluate(
            referenceImage: refImage,
            context: context,
            model: model,
            quantization: quantization,
            seed: seed,
            hfToken: hfToken,
            onProgress: onProgress
        )

        // Step 2: Generate validation prompt from reference
        onProgress?("Generating validation prompt from reference...")
        if !FluxTextEncoders.shared.isQwen35VLMLoaded {
            let downloader = TextEncoderModelDownloader()
            let vlmPath = try await downloader.downloadQwen35(variant: .qwen35_4B_4bit)
            try await FluxTextEncoders.shared.loadQwen35VLM(from: vlmPath.path)
        }

        let validationPrompt = try describeReferenceForValidation(
            image: refImage,
            triggerWord: triggerWord
        )
        onProgress?("Validation prompt: \"\(validationPrompt.prefix(80))...\"")

        // Unload VLM
        await MainActor.run { FluxTextEncoders.shared.unloadQwen35VLM() }

        return LoRATrainingSetup(
            referenceImage: refImage,
            referenceImagePath: URL(fileURLWithPath: referenceImagePath),
            validationPrompt: validationPrompt,
            context: context,
            evaluation: evaluation,
            recommendation: evaluation.recommendation
        )
    }
}

// MARK: - YAML Generation with VLM Scoring

extension LoRARecommendation {

    /// Export as YAML training config with VLM scoring enabled
    /// Includes validation prompt derived from reference image and VLM scoring configuration
    public func toYAMLWithVLMScoring(
        model: Flux2Model,
        triggerWord: String,
        datasetPath: String = "./dataset",
        validationPrompt: String,
        referenceImagePath: String,
        checkpointEvery: Int = 50
    ) -> String {
        var yaml = toYAML(model: model, triggerWord: triggerWord, datasetPath: datasetPath)

        // Add checkpoints section
        yaml += """

        checkpoints:
          save_every: \(checkpointEvery)
        """

        // Add validation with VLM scoring
        yaml += """

        validation:
          prompts:
            - prompt: "\(validationPrompt.replacingOccurrences(of: "\"", with: "\\\""))"
              apply_trigger: false
              is_512: true
              is_1024: false
          every_n_steps: \(checkpointEvery)
          seed: 42
          steps: 4
          vlm_scoring:
            enabled: true
            reference_images:
              - \(referenceImagePath)
            max_reference_images: 1
            save_best_checkpoint: true
            compare_to_baseline: true
        """

        return yaml
    }
}
