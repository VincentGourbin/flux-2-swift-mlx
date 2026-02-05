// Flux2Pipeline.swift - Main Pipeline for Flux.2 Image Generation
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXRandom
import MLXNN
import CoreGraphics

#if canImport(AppKit)
import AppKit
#endif

/// Generation mode for Flux.2
public enum Flux2GenerationMode: Sendable {
    /// Text-to-Image generation
    case textToImage

    /// Image-to-Image generation with reference images
    /// - Parameters:
    ///   - images: Reference images (1-3). Multiple images are concatenated along sequence dimension
    ///             with unique time-based position IDs for each, allowing the transformer to
    ///             attend to all reference images during generation.
    ///   - strength: Denoising strength (0.0-1.0). 1.0 = full denoising (ignores image),
    ///               0.5 = 50% denoising, 0.1 = minimal changes
    case imageToImage(images: [CGImage], strength: Float)
}

/// Progress callback for generation (currentStep, totalSteps)
public typealias Flux2ProgressCallback = @Sendable (Int, Int) -> Void

/// Checkpoint callback for saving intermediate images (step, image)
public typealias Flux2CheckpointCallback = @Sendable (Int, CGImage) -> Void

/// Result of image generation including the image and metadata
public struct Flux2GenerationResult: Sendable {
    /// The generated image
    public let image: CGImage

    /// The prompt that was actually used for generation
    /// If upsampling was enabled, this contains the enhanced prompt
    public let usedPrompt: String

    /// Whether the prompt was upsampled/enhanced
    public let wasUpsampled: Bool

    /// The original prompt before any enhancement
    public let originalPrompt: String

    public init(image: CGImage, usedPrompt: String, wasUpsampled: Bool, originalPrompt: String) {
        self.image = image
        self.usedPrompt = usedPrompt
        self.wasUpsampled = wasUpsampled
        self.originalPrompt = originalPrompt
    }
}

/// Flux.2 Image Generation Pipeline
///
/// Supports:
/// - Flux.2 Dev (32B) - Mistral text encoder
/// - Flux.2 Klein 4B - Qwen3-4B text encoder (Apache 2.0)
/// - Flux.2 Klein 9B - Qwen3-8B text encoder (Non-commercial)
///
/// Two-phase pipeline for memory efficiency:
/// 1. Text encoding (unloaded after use)
/// 2. Image generation with Transformer + VAE
public class Flux2Pipeline: @unchecked Sendable {
    /// Model variant (dev, klein-4b, klein-9b)
    public let model: Flux2Model

    /// Quantization configuration
    public let quantization: Flux2QuantizationConfig

    /// Memory optimization settings for transformer inference
    public var memoryOptimization: MemoryOptimizationConfig

    /// Text encoder (Mistral - for Dev)
    private var textEncoder: Flux2TextEncoder?

    /// Klein text encoder (Qwen3 - for Klein 4B/9B)
    private var kleinEncoder: KleinTextEncoder?

    /// Diffusion transformer
    private var transformer: Flux2Transformer2DModel?

    /// VAE decoder
    private var vae: AutoencoderKLFlux2?

    /// Scheduler
    private let scheduler: FlowMatchEulerScheduler

    /// Memory manager
    private let memoryManager = Flux2MemoryManager.shared

    /// Model downloader
    private var downloader: Flux2ModelDownloader?

    /// LoRA adapter manager
    private var loraManager: LoRAManager?

    /// Active LoRA scheduler overrides (from loaded LoRA config)
    private var loraSchedulerOverrides: SchedulerOverrides?

    /// Whether models are loaded
    public private(set) var isLoaded: Bool = false

    /// Memory profile for GPU cache management (auto = dynamic based on RAM)
    public var memoryProfile: MemoryConfig.CacheProfile = .auto

    /// Clear cache every N denoising steps (0 = disabled)
    public var clearCacheEveryNSteps: Int = 5

    /// Initialize pipeline
    /// - Parameters:
    ///   - model: Model variant to use (default: .dev)
    ///   - quantization: Quantization settings for each component
    ///   - hfToken: HuggingFace token for gated models
    /// Initialize the Flux.2 pipeline
    /// - Parameters:
    ///   - model: Model variant (dev, klein-4b, klein-9b)
    ///   - quantization: Quantization configuration
    ///   - memoryOptimization: Memory optimization settings (nil = auto-detect based on system RAM)
    ///   - hfToken: HuggingFace token for model downloads
    public init(
        model: Flux2Model = .dev,
        quantization: Flux2QuantizationConfig = .balanced,
        memoryOptimization: MemoryOptimizationConfig? = nil,
        hfToken: String? = nil
    ) {
        self.model = model
        self.quantization = quantization
        self.memoryOptimization = memoryOptimization ?? MemoryOptimizationConfig.recommended(
            forRAMGB: Flux2MemoryManager.shared.physicalMemoryGB
        )
        self.scheduler = FlowMatchEulerScheduler()
        self.downloader = hfToken != nil ? Flux2ModelDownloader(hfToken: hfToken) : Flux2ModelDownloader()
    }

    // MARK: - Model Loading

    /// Load all required models
    /// - Parameter progressCallback: Optional callback for download progress
    public func loadModels(progressCallback: Flux2DownloadProgressCallback? = nil) async throws {
        // Check memory before loading
        let memCheck = memoryManager.checkTextEncodingPhase(config: quantization)
        if !memCheck.isOk {
            Flux2Debug.log("Memory warning: \(memCheck.message)")
        }

        // Download models if needed
        if !hasRequiredModels {
            try await downloadRequiredModels(progress: progressCallback)
        }

        isLoaded = true
        Flux2Debug.log("Pipeline ready for generation")
    }

    /// Download required models
    private func downloadRequiredModels(progress: Flux2DownloadProgressCallback?) async throws {
        guard let downloader = downloader else { return }

        for component in missingModels {
            if case .textEncoder = component {
                // Text encoder is handled separately by Flux2TextEncoder
                continue
            }
            _ = try await downloader.download(component, progress: progress)
        }
    }

    /// Load text encoder for Phase 1
    /// Uses Mistral for Dev, Qwen3 for Klein models
    private func loadTextEncoder() async throws {
        // Check if already loaded
        switch model {
        case .dev:
            guard textEncoder == nil else { return }
        case .klein4B, .klein9B:
            guard kleinEncoder == nil else { return }
        }

        memoryManager.logMemoryState()
        Flux2Debug.log("Loading text encoder for \(model.displayName)...")

        // Map quantization
        let mistralQuant: MistralQuantization
        switch quantization.textEncoder {
        case .bf16:
            mistralQuant = .bf16
        case .mlx8bit:
            mistralQuant = .mlx8bit
        case .mlx6bit:
            mistralQuant = .mlx6bit
        case .mlx4bit:
            mistralQuant = .mlx4bit
        }

        switch model {
        case .dev:
            textEncoder = Flux2TextEncoder(quantization: mistralQuant)
            try await textEncoder!.load()

        case .klein4B:
            kleinEncoder = KleinTextEncoder(variant: .klein4B, quantization: mistralQuant)
            try await kleinEncoder!.load()

        case .klein9B:
            kleinEncoder = KleinTextEncoder(variant: .klein9B, quantization: mistralQuant)
            try await kleinEncoder!.load()
        }

        memoryManager.logMemoryState()
    }

    /// Unload text encoder to free memory for transformer
    @MainActor
    private func unloadTextEncoder() {
        Flux2Debug.log("Unloading text encoder...")

        switch model {
        case .dev:
            textEncoder?.unload()
            textEncoder = nil
        case .klein4B, .klein9B:
            kleinEncoder?.unload()
            kleinEncoder = nil
        }

        memoryManager.fullCleanup()
        memoryManager.logMemoryState()
    }

    /// Load transformer for Phase 2
    private func loadTransformer() async throws {
        guard transformer == nil else { return }

        memoryManager.logMemoryState()
        Flux2Debug.log("Loading transformer for \(model.displayName)...")

        // Get the appropriate transformer variant based on model type and quantization
        let variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization.transformer)

        // Find model path
        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .transformer(variant)) else {
            let downloadCmd: String
            switch model {
            case .dev:
                downloadCmd = "flux2 download --transformer \(quantization.transformer.rawValue)"
            case .klein4B:
                downloadCmd = "flux2 download --model klein-4b"
            case .klein9B:
                downloadCmd = "flux2 download --model klein-9b"
            }
            throw Flux2Error.modelNotLoaded("\(model.displayName) transformer weights not found. Run: \(downloadCmd)")
        }

        // Create model with appropriate config and memory optimization
        transformer = Flux2Transformer2DModel(
            config: model.transformerConfig,
            memoryOptimization: memoryOptimization
        )
        Flux2Debug.log("Memory optimization: \(memoryOptimization)")

        // Load weights
        let weights = try Flux2WeightLoader.loadWeights(from: modelPath)
        try Flux2WeightLoader.applyTransformerWeights(weights, to: transformer!)

        // Merge LoRA weights if any are loaded
        if let loraManager = loraManager, loraManager.count > 0 {
            Flux2WeightLoader.mergeLoRAWeights(from: loraManager, into: transformer!)
            // Free LoRA matrices from memory after fusion (they're now baked into base weights)
            loraManager.clearWeightsAfterFusion()
        }

        // Ensure weights are evaluated
        eval(transformer!.parameters())

        memoryManager.logMemoryState()
        Flux2Debug.log("Transformer loaded successfully")
    }

    // MARK: - LoRA Support

    /// Load a LoRA adapter
    /// - Parameter config: LoRA configuration
    /// - Returns: Information about the loaded LoRA
    @discardableResult
    public func loadLoRA(_ config: LoRAConfig) throws -> LoRAInfo {
        if loraManager == nil {
            loraManager = LoRAManager()
        }

        let info = try loraManager!.loadLoRA(config)

        // Store scheduler overrides if present
        if let overrides = config.schedulerOverrides, overrides.hasOverrides {
            loraSchedulerOverrides = overrides
            Flux2Debug.log("[LoRA] Scheduler overrides detected:")
            if let sigmas = overrides.customSigmas {
                Flux2Debug.log("  - Custom sigmas: \(sigmas.count) values")
            }
            if let steps = overrides.numSteps {
                Flux2Debug.log("  - Recommended steps: \(steps)")
            }
            if let guidance = overrides.guidance {
                Flux2Debug.log("  - Recommended guidance: \(guidance)")
            }
        }

        // Validate compatibility
        if info.targetModel != .unknown {
            let expectedModel: LoRAInfo.TargetModel
            switch model {
            case .dev: expectedModel = .dev
            case .klein4B: expectedModel = .klein4B
            case .klein9B: expectedModel = .klein9B
            }

            if info.targetModel != expectedModel {
                Flux2Debug.log("[LoRA] Warning: LoRA was trained for \(info.targetModel.rawValue), but using \(model.rawValue)")
            }
        }

        // If transformer is already loaded, merge LoRA weights immediately
        if let transformer = transformer {
            Flux2WeightLoader.mergeLoRAWeights(from: loraManager!, into: transformer)
            // Free LoRA matrices from memory after fusion (they're now baked into base weights)
            loraManager!.clearWeightsAfterFusion()
        }

        return info
    }

    /// Unload a LoRA by name
    public func unloadLoRA(name: String) {
        loraManager?.unloadLoRA(name: name)
        // Clear overrides if no more LoRAs are loaded
        if loraManager?.count == 0 {
            loraSchedulerOverrides = nil
        }
    }

    /// Unload all LoRAs
    public func unloadAllLoRAs() {
        loraManager?.unloadAll()
        loraSchedulerOverrides = nil
    }

    /// Whether any LoRAs are loaded
    public var hasLoRA: Bool {
        loraManager?.count ?? 0 > 0
    }

    /// Get active LoRA scheduler overrides (if any)
    /// The CLI can use this to apply recommended settings
    public var activeSchedulerOverrides: SchedulerOverrides? {
        loraSchedulerOverrides
    }

    /// Load VAE for Phase 2
    private func loadVAE() async throws {
        guard vae == nil else { return }

        Flux2Debug.log("Loading VAE...")

        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .vae(.standard)) else {
            throw Flux2Error.modelNotLoaded("VAE weights not found")
        }

        // VAE files are in 'vae' subdirectory
        let vaePath = modelPath.appendingPathComponent("vae")
        let weightsPath = FileManager.default.fileExists(atPath: vaePath.path) ? vaePath : modelPath

        // Create model
        vae = AutoencoderKLFlux2()

        // Load weights
        let weights = try Flux2WeightLoader.loadWeights(from: weightsPath)
        try Flux2WeightLoader.applyVAEWeights(weights, to: vae!)

        // Ensure weights are evaluated
        eval(vae!.parameters())

        Flux2Debug.log("VAE loaded successfully")
    }

    /// Unload transformer to free memory
    private func unloadTransformer() {
        transformer = nil
        memoryManager.clearCache()
    }

    // MARK: - Generation API

    /// Generate image from text prompt
    /// - Parameters:
    ///   - prompt: Text description of the image
    ///   - height: Image height (default 1024)
    ///   - width: Image width (default 1024)
    ///   - steps: Number of denoising steps (default 50)
    ///   - guidance: Guidance scale (default 4.0)
    ///   - seed: Optional random seed
    ///   - upsamplePrompt: Enhance prompt with visual details before encoding (default false)
    ///   - checkpointInterval: Save intermediate image every N steps (nil = disabled)
    ///   - onProgress: Optional progress callback
    ///   - onCheckpoint: Optional callback when checkpoint image is generated
    /// - Returns: Generated image
    public func generateTextToImage(
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int = 1024,
        width: Int = 1024,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> CGImage {
        try await generate(
            mode: .textToImage,
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Generate image with reference images
    /// - Parameters:
    ///   - prompt: Text description
    ///   - images: 1-3 reference images. Multiple images are concatenated along sequence dimension
    ///             with unique time-based position IDs, allowing the transformer to attend to all references.
    ///   - interpretImagePaths: File paths to images to analyze with VLM and inject description into prompt (not used as visual reference)
    ///   - height: Optional height (inferred from first image if nil)
    ///   - width: Optional width (inferred from first image if nil)
    ///   - steps: Number of denoising steps
    ///   - guidance: Guidance scale
    ///   - seed: Optional random seed
    ///   - strength: Denoising strength (0.0-1.0). Default 0.8. Lower = preserve more of original image
    ///   - upsamplePrompt: Enhance prompt with visual details before encoding (default false)
    ///   - checkpointInterval: Save intermediate image every N steps (nil = disabled)
    ///   - onProgress: Optional progress callback
    ///   - onCheckpoint: Optional callback when checkpoint image is generated
    /// - Returns: Generated image
    public func generateImageToImage(
        prompt: String,
        images: [CGImage],
        interpretImagePaths: [String]? = nil,
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        strength: Float = 0.8,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> CGImage {
        guard !images.isEmpty && images.count <= 3 else {
            throw Flux2Error.invalidConfiguration("Provide 1-3 reference images")
        }

        guard strength > 0.0 && strength <= 1.0 else {
            throw Flux2Error.invalidConfiguration("Strength must be between 0.0 and 1.0")
        }

        // Infer dimensions from first image if not provided
        let targetHeight = height ?? images[0].height
        let targetWidth = width ?? images[0].width

        return try await generate(
            mode: .imageToImage(images: images, strength: strength),
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: targetHeight,
            width: targetWidth,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    // MARK: - Generation with Result (includes used prompt)

    /// Generate an image from text with full result including the used prompt
    /// - Returns: Flux2GenerationResult containing image and prompt metadata
    public func generateTextToImageWithResult(
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int = 1024,
        width: Int = 1024,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> Flux2GenerationResult {
        try await generateWithResult(
            mode: .textToImage,
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Generate an image from reference images with full result including the used prompt
    /// - Returns: Flux2GenerationResult containing image and prompt metadata
    public func generateImageToImageWithResult(
        prompt: String,
        images: [CGImage],
        interpretImagePaths: [String]? = nil,
        height: Int? = nil,
        width: Int? = nil,
        steps: Int = 50,
        guidance: Float = 4.0,
        seed: UInt64? = nil,
        strength: Float = 0.8,
        upsamplePrompt: Bool = false,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
    ) async throws -> Flux2GenerationResult {
        guard !images.isEmpty && images.count <= 3 else {
            throw Flux2Error.invalidConfiguration("Provide 1-3 reference images")
        }

        guard strength > 0.0 && strength <= 1.0 else {
            throw Flux2Error.invalidConfiguration("Strength must be between 0.0 and 1.0")
        }

        // Infer dimensions from first image if not provided
        let targetHeight = height ?? images[0].height
        let targetWidth = width ?? images[0].width

        return try await generateWithResult(
            mode: .imageToImage(images: images, strength: strength),
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: targetHeight,
            width: targetWidth,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
    }

    /// Unified generation method (backward compatible - returns just the image)
    public func generate(
        mode: Flux2GenerationMode,
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int,
        width: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        upsamplePrompt: Bool,
        checkpointInterval: Int?,
        onProgress: Flux2ProgressCallback?,
        onCheckpoint: Flux2CheckpointCallback?
    ) async throws -> CGImage {
        let result = try await generateWithResult(
            mode: mode,
            prompt: prompt,
            interpretImagePaths: interpretImagePaths,
            height: height,
            width: width,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint
        )
        return result.image
    }

    /// Unified generation method with full result including used prompt
    public func generateWithResult(
        mode: Flux2GenerationMode,
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int,
        width: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        upsamplePrompt: Bool,
        checkpointInterval: Int?,
        onProgress: Flux2ProgressCallback?,
        onCheckpoint: Flux2CheckpointCallback?
    ) async throws -> Flux2GenerationResult {
        // Validate dimensions
        let (validHeight, validWidth) = LatentUtils.validateDimensions(
            height: height,
            width: width
        )

        // Check image size feasibility
        let sizeCheck = memoryManager.checkImageSize(width: validWidth, height: validHeight)
        if case .insufficientMemory = sizeCheck {
            throw Flux2Error.insufficientMemory(required: 100, available: memoryManager.estimatedAvailableMemoryGB)
        }

        // Set random seed
        if let seed = seed {
            MLXRandom.seed(seed)
        }

        Flux2Debug.log("Starting generation: \(validWidth)x\(validHeight), \(steps) steps, guidance=\(guidance)")

        // Start profiling
        let profiler = Flux2Profiler.shared

        // === PHASE 1: Text Encoding ===
        // Note: Progress will be reported once denoising loop starts with accurate step count
        Flux2Debug.log("=== PHASE 1: Text Encoding ===")

        // MEMORY OPTIMIZATION: Set cache limit for text encoding phase
        let phaseLimits = MemoryConfig.PhaseLimits.forModel(model, profile: memoryProfile)
        MemoryConfig.applyCacheLimit(bytes: phaseLimits.textEncoding)

        profiler.start("1. Load Text Encoder")
        try await loadTextEncoder()
        profiler.end("1. Load Text Encoder")

        // === INTERPRET IMAGES: VLM semantic analysis ===
        // If interpretImagePaths are provided, describe them with VLM and inject into prompt
        // Note: VLM interpretation only available for Dev model (Mistral)
        var enrichedPrompt = prompt
        if let interpretPaths = interpretImagePaths, !interpretPaths.isEmpty {
            switch model {
            case .dev:
                Flux2Debug.log("Interpreting \(interpretPaths.count) image(s) with VLM for prompt injection...")
                profiler.start("1b. VLM Interpretation")

                let descriptions = try await textEncoder!.describeImagePathsForPrompt(interpretPaths, context: prompt)

                if !descriptions.isEmpty {
                    // Build enriched prompt with image descriptions
                    let imageContext = descriptions.enumerated().map { (idx, desc) in
                        "Interpret image \(idx + 1): \(desc)"
                    }.joined(separator: "\n")

                    enrichedPrompt = """
                    \(imageContext)

                    User request: \(prompt)
                    """

                    Flux2Debug.log("Prompt enriched with \(descriptions.count) VLM description(s)")
                    print("[VLM-Interpret] Enriched prompt:\n\(enrichedPrompt)")
                    fflush(stdout)
                }

                profiler.end("1b. VLM Interpretation")

            case .klein4B, .klein9B:
                // Klein + --interpret: load Mistral VLM temporarily to analyze images
                Flux2Debug.log("Klein with --interpret: loading Mistral VLM temporarily to analyze images...")
                profiler.start("1b. VLM Interpretation")

                // Step 1: Unload Qwen3 to free memory for Mistral
                Flux2Debug.log("Unloading Qwen3 to make room for Mistral VLM...")
                await MainActor.run { kleinEncoder?.unload() }
                memoryManager.fullCleanup()

                // Step 2: Load Mistral VLM
                Flux2Debug.log("Loading Mistral VLM for image interpretation...")
                let tempMistralForInterpret = Flux2TextEncoder(quantization: quantization.textEncoder)
                try await tempMistralForInterpret.load()

                // Step 3: VLM interpretation (same logic as Dev)
                let descriptions = try await tempMistralForInterpret.describeImagePathsForPrompt(interpretPaths, context: prompt)

                if !descriptions.isEmpty {
                    let imageContext = descriptions.enumerated().map { (idx, desc) in
                        "Interpret image \(idx + 1): \(desc)"
                    }.joined(separator: "\n")

                    enrichedPrompt = """
                    \(imageContext)

                    User request: \(prompt)
                    """

                    Flux2Debug.log("Prompt enriched with \(descriptions.count) VLM description(s)")
                    print("[VLM-Interpret] Enriched prompt:\n\(enrichedPrompt)")
                    fflush(stdout)
                }

                // Step 4: Unload Mistral
                Flux2Debug.log("Unloading Mistral VLM...")
                await MainActor.run { tempMistralForInterpret.unload() }
                memoryManager.fullCleanup()

                // Step 5: Reload Qwen3 for text encoding
                Flux2Debug.log("Reloading Qwen3 for Klein text encoding...")
                try await kleinEncoder!.load()

                profiler.end("1b. VLM Interpretation")
            }
        }

        profiler.start("2. Text Encoding")
        // Use vision-based upsampling for I2I if enabled (Dev only)
        // Track the final prompt used for generation
        let textEmbeddings: MLXArray
        var finalUsedPrompt: String = enrichedPrompt
        var wasPromptUpsampled: Bool = false

        switch model {
        case .dev:
            if upsamplePrompt, case .imageToImage(let images, _) = mode {
                // Use VLM to analyze reference images and enhance prompt
                Flux2Debug.log("Using vision-based prompt upsampling for I2I with \(images.count) image(s)")
                let enhancedPrompt = try await textEncoder!.upsamplePromptWithImages(enrichedPrompt, images: images)
                finalUsedPrompt = enhancedPrompt
                wasPromptUpsampled = true
                textEmbeddings = try textEncoder!.encode(enhancedPrompt, upsample: false)
            } else {
                let (embeddings, usedPrompt) = try textEncoder!.encodeWithPrompt(enrichedPrompt, upsample: upsamplePrompt)
                textEmbeddings = embeddings
                finalUsedPrompt = usedPrompt
                wasPromptUpsampled = upsamplePrompt && (usedPrompt != enrichedPrompt)
            }

        case .klein4B, .klein9B:
            // Klein I2I with upsampling: load Mistral VLM temporarily to see reference images
            // This matches the official flux2 implementation which loads Mistral for Klein I2I upsampling
            if upsamplePrompt, case .imageToImage(let images, _) = mode {
                Flux2Debug.log("Klein I2I with upsampling: using Mistral VLM to analyze reference images...")

                // Step 1: Unload Qwen3 (already loaded by loadTextEncoder) to free memory for Mistral
                Flux2Debug.log("Unloading Qwen3 to make room for Mistral VLM...")
                await MainActor.run { kleinEncoder?.unload() }
                memoryManager.fullCleanup()

                // Step 2: Load Mistral VLM for vision-aware upsampling
                Flux2Debug.log("Loading Mistral VLM for image analysis...")
                let tempMistralEncoder = Flux2TextEncoder(quantization: quantization.textEncoder)
                try await tempMistralEncoder.load()

                // Step 3: Upsample prompt with images using Mistral VLM
                Flux2Debug.log("Upsampling prompt with \(images.count) reference image(s)...")
                let enhancedPrompt = try await tempMistralEncoder.upsamplePromptWithImages(enrichedPrompt, images: images)
                finalUsedPrompt = enhancedPrompt
                wasPromptUpsampled = true

                // Step 4: Unload Mistral to free memory
                Flux2Debug.log("Unloading Mistral VLM...")
                await MainActor.run { tempMistralEncoder.unload() }
                memoryManager.fullCleanup()

                // Step 5: Reload Qwen3 for text encoding
                Flux2Debug.log("Reloading Qwen3 for Klein text encoding...")
                try await kleinEncoder!.load()

                // Step 6: Encode with Qwen3 (already upsampled, so upsample=false)
                textEmbeddings = try kleinEncoder!.encode(enhancedPrompt, upsample: false)
            } else {
                // Standard Klein encoding (text-only upsampling if enabled)
                let (embeddings, usedPrompt) = try kleinEncoder!.encodeWithPrompt(enrichedPrompt, upsample: upsamplePrompt)
                textEmbeddings = embeddings
                finalUsedPrompt = usedPrompt
                wasPromptUpsampled = upsamplePrompt && (usedPrompt != enrichedPrompt)
            }
        }
        eval(textEmbeddings)
        profiler.end("2. Text Encoding")

        Flux2Debug.log("Text embeddings shape: \(textEmbeddings.shape)")

        // Unload text encoder to free memory
        profiler.start("3. Unload Text Encoder")
        await unloadTextEncoder()
        profiler.end("3. Unload Text Encoder")

        // === PHASE 2: Image Generation ===
        Flux2Debug.log("=== PHASE 2: Image Generation ===")

        // Check memory before loading transformer
        let phase2Check = memoryManager.checkImageGenerationPhase(config: quantization)
        if !phase2Check.isOk {
            Flux2Debug.log("Memory warning: \(phase2Check.message)")
        }

        // Load transformer
        profiler.start("4. Load Transformer")
        try await loadTransformer()
        profiler.end("4. Load Transformer")

        // Load VAE
        profiler.start("5. Load VAE")
        try await loadVAE()
        profiler.end("5. Load VAE")

        // Generate initial latents in PATCHIFIED format [B, 128, H/16, W/16]
        // This is the format expected by the BatchNorm normalization
        var patchifiedLatents: MLXArray

        switch mode {
        case .textToImage:
            patchifiedLatents = LatentUtils.generatePatchifiedLatents(
                height: validHeight,
                width: validWidth,
                seed: seed
            )
            Flux2Debug.log("Generated patchified latents: \(patchifiedLatents.shape)")

        case .imageToImage(let images, let strength):
            _ = strength  // Note: Flux.2 doesn't use strength like SD - it uses conditioning mode

            // === FLUX.2 IMAGE-TO-IMAGE MODE ===
            // Flux.2 uses CONDITIONING mode for all I2I:
            // - Reference images are encoded and concatenated as context
            // - Output starts from random noise (full denoising)
            // - This is different from SD's "traditional" I2I with noise injection
            // Generate random noise for OUTPUT
            patchifiedLatents = LatentUtils.generatePatchifiedLatents(
                height: validHeight,
                width: validWidth,
                seed: seed
            )
            eval(patchifiedLatents)
            Flux2Debug.log("Generated output noise latents: \(patchifiedLatents.shape)")

            // Encode ALL reference images
            let (referenceLatents, referencePositionIds) = try encodeReferenceImages(
                images,
                height: validHeight,
                width: validWidth
            )
            eval(referenceLatents)
            Flux2Debug.log("Encoded \(images.count) reference images: latents \(referenceLatents.shape), posIds \(referencePositionIds.shape)")

            // Pack output latents to sequence format
            var packedOutputLatents = LatentUtils.packPatchifiedToSequence(patchifiedLatents)
            eval(packedOutputLatents)
            let outputSeqLen = packedOutputLatents.shape[1]
            Flux2Debug.log("Output sequence length: \(outputSeqLen)")

            // Generate position IDs for output and text
            let textLength = textEmbeddings.shape[1]
            let (textIds, outputImageIds, _) = LatentUtils.combinePositionIDs(
                textLength: textLength,
                height: validHeight,
                width: validWidth
            )

            // Concatenate reference latents to output latents for transformer input
            // [1, output_seq, 128] + [1, ref_seq, 128] -> [1, total_seq, 128]
            let combinedLatents = concatenated([packedOutputLatents, referenceLatents], axis: 1)
            Flux2Debug.log("Combined latents (output + refs): \(combinedLatents.shape)")
            _ = combinedLatents  // Used for documentation

            // Concatenate position IDs: output IDs + reference IDs
            // Output IDs are [output_seq, 4], reference IDs are [ref_seq, 4]
            let combinedImageIds = concatenated([outputImageIds, referencePositionIds], axis: 0)
            Flux2Debug.log("Combined image IDs: \(combinedImageIds.shape)")

            // Setup scheduler for FULL denoising (no timestep skip in Flux.2 I2I)
            // Check for custom sigmas from LoRA (e.g., Turbo LoRAs)
            if let customSigmas = loraSchedulerOverrides?.customSigmas {
                scheduler.setCustomSigmas(customSigmas)
            } else {
                scheduler.setTimesteps(numInferenceSteps: steps, imageSeqLen: outputSeqLen, strength: 1.0)
            }

            let effectiveSteps = scheduler.sigmas.count - 1
            Flux2Debug.log("Starting I2I denoising loop (\(effectiveSteps) steps)...")

            // Guidance tensor (nil for Klein models which don't use guidance embeddings)
            let guidanceTensor: MLXArray? = model.usesGuidanceEmbeds ? MLXArray([guidance]) : nil

            // MEMORY OPTIMIZATION: Set cache limit for denoising phase
            let phaseLimits = MemoryConfig.PhaseLimits.forModel(model, profile: memoryProfile)
            MemoryConfig.applyCacheLimit(bytes: phaseLimits.denoising)

            profiler.start("6. Denoising Loop")

            for stepIdx in 0..<(scheduler.sigmas.count - 1) {
                let stepStart = Date()

                let sigma = scheduler.sigmas[stepIdx]
                let t = MLXArray([sigma])

                // Concatenate current output latents with reference latents for this step
                let inputLatents = concatenated([packedOutputLatents, referenceLatents], axis: 1)

                // Check transformer is still loaded (may be unloaded during cancellation)
                guard let transformer = transformer else {
                    throw Flux2Error.generationCancelled
                }

                // Run transformer with combined latents
                let noisePred = transformer.callAsFunction(
                    hiddenStates: inputLatents,
                    encoderHiddenStates: textEmbeddings,
                    timestep: t,
                    guidance: guidanceTensor,
                    imgIds: combinedImageIds,
                    txtIds: textIds
                )

                // Extract only the OUTPUT portion of the noise prediction
                // noisePred shape is [1, total_seq, 128], we want first outputSeqLen
                let outputNoisePred = noisePred[0..., 0..<outputSeqLen, 0...]

                // Scheduler step on OUTPUT latents only
                packedOutputLatents = scheduler.step(
                    modelOutput: outputNoisePred,
                    timestep: sigma,
                    sample: packedOutputLatents
                )
                eval(packedOutputLatents)

                // MEMORY OPTIMIZATION: Periodic cache clearing to prevent memory buildup
                if clearCacheEveryNSteps > 0 && (stepIdx + 1) % clearCacheEveryNSteps == 0 {
                    MemoryConfig.clearCache()
                }

                let stepDuration = Date().timeIntervalSince(stepStart)
                profiler.recordStep(duration: stepDuration)

                onProgress?(stepIdx + 1, effectiveSteps)
                Flux2Debug.verbose("Step \(stepIdx + 1)/\(effectiveSteps)")

                // Checkpoint
                if let interval = checkpointInterval,
                   let checkpointCallback = onCheckpoint,
                   (stepIdx + 1) % interval == 0 {
                    var checkpointPatchified = LatentUtils.unpackSequenceToPatchified(
                        packedOutputLatents,
                        height: validHeight,
                        width: validWidth
                    )
                    checkpointPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                        checkpointPatchified,
                        runningMean: vae!.batchNormRunningMean,
                        runningVar: vae!.batchNormRunningVar
                    )
                    let checkpointLatents = LatentUtils.unpatchifyLatents(checkpointPatchified)
                    eval(checkpointLatents)

                    let checkpointDecoded = vae!.decode(checkpointLatents)
                    eval(checkpointDecoded)

                    if let checkpointImage = postprocessVAEOutput(checkpointDecoded) {
                        checkpointCallback(stepIdx + 1, checkpointImage)
                    }
                }

                if stepIdx % 10 == 0 {
                    memoryManager.clearCache()
                }
            }

            profiler.end("6. Denoising Loop")

            // Decode final OUTPUT latents
            profiler.start("7. VAE Decode")
            var finalPatchified = LatentUtils.unpackSequenceToPatchified(
                packedOutputLatents,
                height: validHeight,
                width: validWidth
            )
            finalPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                finalPatchified,
                runningMean: vae!.batchNormRunningMean,
                runningVar: vae!.batchNormRunningVar
            )
            let finalLatents = LatentUtils.unpatchifyLatents(finalPatchified)
            eval(finalLatents)

            let decoded = vae!.decode(finalLatents)
            eval(decoded)
            profiler.end("7. VAE Decode")

            profiler.start("8. Post-processing")
            guard let image = postprocessVAEOutput(decoded) else {
                throw Flux2Error.generationFailed("Failed to convert VAE output to image")
            }
            profiler.end("8. Post-processing")

            if profiler.isEnabled {
                print(profiler.generateReport())
            }

            return Flux2GenerationResult(
                image: image,
                usedPrompt: finalUsedPrompt,
                wasUpsampled: wasPromptUpsampled,
                originalPrompt: prompt
            )
        }

        // === TEXT-TO-IMAGE PATH (I2I returns earlier) ===

        // Pack patchified latents to sequence format for transformer [B, seq_len, 128]
        var packedLatents = LatentUtils.packPatchifiedToSequence(patchifiedLatents)
        eval(packedLatents)

        // Generate position IDs for latents and text
        let textLength = textEmbeddings.shape[1]
        let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(
            textLength: textLength,
            height: validHeight,
            width: validWidth
        )

        // Calculate image sequence length for scheduler mu
        let imageSeqLen = packedLatents.shape[1]
        Flux2Debug.log("Image sequence length: \(imageSeqLen)")

        // Setup scheduler (T2I always uses strength 1.0)
        // Check for custom sigmas from LoRA (e.g., Turbo LoRAs)
        if let customSigmas = loraSchedulerOverrides?.customSigmas {
            scheduler.setCustomSigmas(customSigmas)
        } else {
            scheduler.setTimesteps(numInferenceSteps: steps, imageSeqLen: imageSeqLen, strength: 1.0)
        }

        let effectiveSteps = scheduler.sigmas.count - 1
        Flux2Debug.log("Starting denoising loop (\(effectiveSteps) steps)...")

        // OPTIMIZATION: Create guidance tensor ONCE before the loop
        // Klein models don't use guidance embeddings
        let guidanceTensor: MLXArray? = model.usesGuidanceEmbeds ? MLXArray([guidance]) : nil

        // MEMORY OPTIMIZATION: Set cache limit for denoising phase
        MemoryConfig.applyCacheLimit(bytes: phaseLimits.denoising)

        profiler.start("6. Denoising Loop")

        // Denoising loop - use sigmas (in [0, 1] range) for transformer
        for stepIdx in 0..<(scheduler.sigmas.count - 1) {
            let stepStart = Date()

            let sigma = scheduler.sigmas[stepIdx]
            let t = MLXArray([sigma])

            // Check transformer is still loaded (may be unloaded during cancellation)
            guard let transformer = transformer else {
                throw Flux2Error.generationCancelled
            }

            // Run transformer
            let noisePred = transformer.callAsFunction(
                hiddenStates: packedLatents,
                encoderHiddenStates: textEmbeddings,
                timestep: t,
                guidance: guidanceTensor,
                imgIds: imageIds,
                txtIds: textIds
            )

            // Scheduler step
            packedLatents = scheduler.step(
                modelOutput: noisePred,
                timestep: sigma,
                sample: packedLatents
            )

            // Synchronize GPU
            eval(packedLatents)

            // MEMORY OPTIMIZATION: Periodic cache clearing to prevent memory buildup
            if clearCacheEveryNSteps > 0 && (stepIdx + 1) % clearCacheEveryNSteps == 0 {
                MemoryConfig.clearCache()
            }

            // Record step time
            let stepDuration = Date().timeIntervalSince(stepStart)
            profiler.recordStep(duration: stepDuration)

            // Report progress (using effective steps for I2I)
            onProgress?(stepIdx + 1, effectiveSteps)

            Flux2Debug.verbose("Step \(stepIdx + 1)/\(effectiveSteps)")

            // Generate checkpoint image if requested
            if let interval = checkpointInterval,
               let checkpointCallback = onCheckpoint,
               (stepIdx + 1) % interval == 0 {
                Flux2Debug.verbose("Generating checkpoint at step \(stepIdx + 1)...")

                // Decode current latents to image
                var checkpointPatchified = LatentUtils.unpackSequenceToPatchified(
                    packedLatents,
                    height: validHeight,
                    width: validWidth
                )
                checkpointPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
                    checkpointPatchified,
                    runningMean: vae!.batchNormRunningMean,
                    runningVar: vae!.batchNormRunningVar
                )
                let checkpointLatents = LatentUtils.unpatchifyLatents(checkpointPatchified)
                eval(checkpointLatents)

                let checkpointDecoded = vae!.decode(checkpointLatents)
                eval(checkpointDecoded)
                Flux2Debug.verbose("Checkpoint VAE output shape: \(checkpointDecoded.shape)")

                if let checkpointImage = postprocessVAEOutput(checkpointDecoded) {
                    checkpointCallback(stepIdx + 1, checkpointImage)
                } else {
                    Flux2Debug.log("Warning: Failed to convert checkpoint to image at step \(stepIdx + 1)")
                }
            }

            // Periodic memory cleanup
            if stepIdx % 10 == 0 {
                memoryManager.clearCache()
            }
        }

        profiler.end("6. Denoising Loop")

        Flux2Debug.log("Denoising complete, decoding image...")

        // Unpack sequence latents back to patchified format [B, 128, H/16, W/16]
        var patchifiedFinal = LatentUtils.unpackSequenceToPatchified(
            packedLatents,
            height: validHeight,
            width: validWidth
        )
        Flux2Debug.log("Unpacked patchified latents: \(patchifiedFinal.shape)")

        // CRITICAL: Denormalize patchified latents with VAE BatchNorm AFTER denoising
        // This reverses the normalization applied before the transformer
        Flux2Debug.log("Denormalizing patchified latents with BatchNorm...")
        patchifiedFinal = LatentUtils.denormalizeLatentsWithBatchNorm(
            patchifiedFinal,
            runningMean: vae!.batchNormRunningMean,
            runningVar: vae!.batchNormRunningVar
        )
        eval(patchifiedFinal)

        // Unpatchify to VAE format [B, 32, H/8, W/8]
        let finalLatents = LatentUtils.unpatchifyLatents(patchifiedFinal)
        Flux2Debug.log("Final latents for VAE: \(finalLatents.shape)")
        eval(finalLatents)

        // === PHASE 3: Decode to Image ===
        Flux2Debug.log("=== PHASE 3: VAE Decoding ===")

        // MEMORY OPTIMIZATION: Set cache limit for VAE decoding phase
        MemoryConfig.applyCacheLimit(bytes: phaseLimits.vaeDecoding)

        profiler.start("7. VAE Decode")
        let decoded = vae!.decode(finalLatents)
        eval(decoded)
        profiler.end("7. VAE Decode")

        // Convert to CGImage
        profiler.start("8. Post-processing")
        guard let image = postprocessVAEOutput(decoded) else {
            throw Flux2Error.imageProcessingFailed("Failed to convert output to image")
        }
        profiler.end("8. Post-processing")

        Flux2Debug.log("Generation complete!")
        memoryManager.logMemoryState()

        // Print profiling report if enabled
        if profiler.isEnabled {
            print(profiler.generateReport())
        }

        return Flux2GenerationResult(
            image: image,
            usedPrompt: finalUsedPrompt,
            wasUpsampled: wasPromptUpsampled,
            originalPrompt: prompt
        )
    }

    // MARK: - Private Methods

    /// Encode reference images for image-to-image generation (Flux.2 conditioning mode)
    ///
    /// Following the reference Flux.2 diffusers implementation:
    /// 1. Each image is encoded SEPARATELY through VAE
    /// 2. Each image gets a UNIQUE T-coordinate (T=10, T=20, T=30, etc.)
    /// 3. Latents are concatenated AFTER encoding
    ///
    /// This preserves semantic information:
    /// - The transformer can distinguish between different reference images
    /// - Each reference image has its own position in the "time" dimension
    /// - Images are resized to max 512² pixels each for memory efficiency
    ///
    /// - Parameters:
    ///   - images: Reference images (1-10 supported)
    ///   - height: Target output height
    ///   - width: Target output width
    /// - Returns: Tuple of (latents [1, seq_len, 128], position IDs [seq_len, 4])
    private func encodeReferenceImages(
        _ images: [CGImage],
        height: Int,
        width: Int
    ) throws -> (latents: MLXArray, positionIds: MLXArray) {
        guard let vae = vae else {
            throw Flux2Error.modelNotLoaded("VAE")
        }

        guard !images.isEmpty else {
            throw Flux2Error.invalidConfiguration("No reference images provided")
        }

        Flux2Debug.log("Encoding \(images.count) reference images separately with unique T-coordinates...")

        // === STEP 1: Process each image separately ===
        // Max area per image - matches diffusers pipeline_flux2.py line 892-893
        // Reference uses 1024² for conditioning images (not 768² which is for upsampling)
        let maxImageArea = 1024 * 1024  // ~4096 tokens per image
        let multipleOf = 32  // vae_scale_factor * 2

        var allPackedLatents: [MLXArray] = []
        var latentHeights: [Int] = []
        var latentWidths: [Int] = []

        for (index, image) in images.enumerated() {
            Flux2Debug.log("Processing reference image \(index + 1)/\(images.count): \(image.width)x\(image.height)")

            // Calculate target dimensions for this image
            var targetWidth = image.width
            var targetHeight = image.height
            let pixelCount = targetWidth * targetHeight

            if pixelCount > maxImageArea {
                let scale = sqrt(Double(maxImageArea) / Double(pixelCount))
                targetWidth = Int(Double(targetWidth) * scale)
                targetHeight = Int(Double(targetHeight) * scale)
            }

            // Make dimensions multiples of 32
            targetWidth = (targetWidth / multipleOf) * multipleOf
            targetHeight = (targetHeight / multipleOf) * multipleOf

            // Ensure minimum size
            targetWidth = max(targetWidth, multipleOf)
            targetHeight = max(targetHeight, multipleOf)

            Flux2Debug.log("  -> Resized to \(targetWidth)x\(targetHeight)")

            // Preprocess and encode
            let processed = preprocessImageForVAE(image, targetHeight: targetHeight, targetWidth: targetWidth)

            // Encode with VAE -> [1, 32, H/8, W/8]
            // IMPORTANT: Use samplePosterior=false to get deterministic mean (like diffusers argmax)
            // This preserves color information better than sampling with noise
            let rawLatents = vae.encode(processed, samplePosterior: false)

            // Patchify: [1, 32, H/8, W/8] -> [1, 128, H/16, W/16]
            var patchified = LatentUtils.packLatentsToPatchified(rawLatents)

            // Normalize with BatchNorm (critical for Flux.2)
            patchified = LatentUtils.normalizeLatentsWithBatchNorm(
                patchified,
                runningMean: vae.batchNormRunningMean,
                runningVar: vae.batchNormRunningVar
            )
            eval(patchified)

            // Pack to sequence: [1, 128, H/16, W/16] -> [1, seq_len, 128]
            let packedLatents = LatentUtils.packPatchifiedToSequence(patchified)

            // Remove batch dimension for concatenation: [1, seq_len, 128] -> [seq_len, 128]
            let squeezed = packedLatents.squeezed(axis: 0)
            allPackedLatents.append(squeezed)

            // Track dimensions for position IDs
            let patchifiedH = targetHeight / 16
            let patchifiedW = targetWidth / 16
            latentHeights.append(patchifiedH)
            latentWidths.append(patchifiedW)

            let seqLen = squeezed.shape[0]
            Flux2Debug.log("  -> Encoded to \(seqLen) tokens (T-coordinate will be \(10 + 10 * index))")
        }

        // === STEP 2: Concatenate all latents ===
        // Stack along sequence dimension: [seq1, 128] + [seq2, 128] + ... -> [total_seq, 128]
        let concatenatedLatents = concatenated(allPackedLatents, axis: 0)

        // Add batch dimension back: [total_seq, 128] -> [1, total_seq, 128]
        let finalLatents = concatenatedLatents.expandedDimensions(axis: 0)

        let totalSeqLen = finalLatents.shape[1]
        Flux2Debug.log("Combined reference latents: \(finalLatents.shape) (total \(totalSeqLen) tokens)")

        // === STEP 3: Generate position IDs with unique T-coordinates per image ===
        let positionIds = LatentUtils.generateReferenceImagePositionIDs(
            latentHeights: latentHeights,
            latentWidths: latentWidths,
            scale: 10
        )

        Flux2Debug.log("Position IDs generated: \(positionIds.shape)")
        Flux2Debug.log("Reference encoding complete: \(images.count) images with unique T-coordinates")

        return (latents: finalLatents, positionIds: positionIds)
    }

    /// Preprocess image for VAE encoding
    /// Resizes image to target dimensions using high-quality CoreGraphics interpolation
    private func preprocessImageForVAE(_ image: CGImage, targetHeight: Int, targetWidth: Int) -> MLXArray {
        let sourceWidth = image.width
        let sourceHeight = image.height

        // Resize image using CoreGraphics if needed
        let resizedImage: CGImage
        if sourceWidth != targetWidth || sourceHeight != targetHeight {
            Flux2Debug.log("Resizing image from \(sourceWidth)x\(sourceHeight) to \(targetWidth)x\(targetHeight)")

            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * targetWidth
            var pixelData = [UInt8](repeating: 0, count: targetHeight * bytesPerRow)

            guard let context = CGContext(
                data: &pixelData,
                width: targetWidth,
                height: targetHeight,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                Flux2Debug.log("Failed to create resize context")
                return MLXRandom.normal([1, 3, targetHeight, targetWidth])
            }

            // High quality interpolation
            context.interpolationQuality = .high

            // Draw the image scaled to fit the target dimensions
            context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

            guard let result = context.makeImage() else {
                Flux2Debug.log("Failed to create resized image")
                return MLXRandom.normal([1, 3, targetHeight, targetWidth])
            }

            resizedImage = result
        } else {
            resizedImage = image
        }

        // Now convert to MLXArray
        let width = targetWidth
        let height = targetHeight
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width

        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return MLXRandom.normal([1, 3, targetHeight, targetWidth])
        }

        context.draw(resizedImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert to float array and normalize to [-1, 1]
        var floatData = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * width + x
                let byteIndex = y * bytesPerRow + x * bytesPerPixel

                // RGB channels, normalize to [-1, 1]
                floatData[pixelIndex] = Float(pixelData[byteIndex]) / 127.5 - 1.0
                floatData[height * width + pixelIndex] = Float(pixelData[byteIndex + 1]) / 127.5 - 1.0
                floatData[2 * height * width + pixelIndex] = Float(pixelData[byteIndex + 2]) / 127.5 - 1.0
            }
        }

        // Create MLXArray [1, 3, H, W]
        return MLXArray(floatData).reshaped([1, 3, height, width])
    }

    /// Convert VAE output to CGImage
    /// OPTIMIZED: Uses bulk array extraction instead of per-pixel loop
    private func postprocessVAEOutput(_ tensor: MLXArray) -> CGImage? {
        // tensor shape: [1, 3, H, W]
        let shape = tensor.shape
        guard shape.count == 4, shape[1] == 3 else {
            Flux2Debug.log("Unexpected tensor shape: \(shape)")
            return nil
        }

        let height = shape[2]
        let width = shape[3]

        // Denormalize from [-1, 1] to [0, 255] and convert to UInt8 in MLX
        // This does the conversion on GPU, much faster than CPU loop
        let denormalized = (tensor + 1.0) * 127.5
        let clamped = clip(denormalized, min: 0, max: 255)

        // Convert to [H, W, 3] layout for CGImage and cast to UInt8 on GPU
        let hwc = clamped.squeezed(axis: 0)  // [3, H, W]
            .transposed(axes: [1, 2, 0])      // [H, W, 3]
            .asType(.uint8)                    // Convert to UInt8 on GPU

        // Single eval and bulk extraction - MUCH faster than per-pixel loop
        eval(hwc)
        let pixelData = hwc.asArray(UInt8.self)

        // Create CGImage
        guard let providerRef = CGDataProvider(data: Data(pixelData) as CFData) else {
            return nil
        }

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: providerRef,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

// MARK: - Memory Management

extension Flux2Pipeline {
    /// Estimate memory requirement for current configuration
    public var estimatedMemoryGB: Int {
        quantization.estimatedTotalMemoryGB
    }

    /// Clear all loaded models and free memory
    public func clearAll() async {
        await unloadTextEncoder()
        unloadTransformer()
        vae = nil
        isLoaded = false
        memoryManager.fullCleanup()
    }
}

// MARK: - Model Status

extension Flux2Pipeline {
    /// Check if required models are downloaded
    public var hasRequiredModels: Bool {
        // Check transformer - use the appropriate variant for this model type and quantization
        let transformerVariant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization.transformer)
        let hasTransformer = Flux2ModelDownloader.isDownloaded(.transformer(transformerVariant))

        // Text encoder is handled by MistralCore/FluxTextEncoders, skip check here

        // Check VAE
        let hasVAE = Flux2ModelDownloader.isDownloaded(.vae(.standard))

        return hasTransformer && hasVAE
    }

    /// List missing models
    public var missingModels: [ModelRegistry.ModelComponent] {
        var missing: [ModelRegistry.ModelComponent] = []

        // Use the appropriate transformer variant for this model type
        let transformerVariant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization.transformer)
        if !Flux2ModelDownloader.isDownloaded(.transformer(transformerVariant)) {
            missing.append(.transformer(transformerVariant))
        }

        // Text encoder is handled by MistralCore/FluxTextEncoders

        if !Flux2ModelDownloader.isDownloaded(.vae(.standard)) {
            missing.append(.vae(.standard))
        }

        return missing
    }
}

