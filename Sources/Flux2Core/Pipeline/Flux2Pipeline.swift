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
    ///   - images: Reference images (1-3)
    ///   - strength: Denoising strength (0.0-1.0). 1.0 = full denoising (ignores image),
    ///               0.5 = 50% denoising, 0.1 = minimal changes
    case imageToImage(images: [CGImage], strength: Float)
}

/// Progress callback for generation (currentStep, totalSteps)
public typealias Flux2ProgressCallback = @Sendable (Int, Int) -> Void

/// Checkpoint callback for saving intermediate images (step, image)
public typealias Flux2CheckpointCallback = @Sendable (Int, CGImage) -> Void

/// Flux.2 Image Generation Pipeline
///
/// Two-phase pipeline for memory efficiency:
/// 1. Text encoding with Mistral (unloaded after use)
/// 2. Image generation with Transformer + VAE
public class Flux2Pipeline: @unchecked Sendable {
    /// Quantization configuration
    public let quantization: Flux2QuantizationConfig

    /// Text encoder (Mistral - loaded/unloaded per generation)
    private var textEncoder: Flux2TextEncoder?

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

    /// Whether models are loaded
    public private(set) var isLoaded: Bool = false

    /// Initialize pipeline
    /// - Parameters:
    ///   - quantization: Quantization settings for each component
    ///   - hfToken: HuggingFace token for gated models
    public init(
        quantization: Flux2QuantizationConfig = .balanced,
        hfToken: String? = nil
    ) {
        self.quantization = quantization
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
    private func loadTextEncoder() async throws {
        guard textEncoder == nil else { return }

        memoryManager.logMemoryState()
        Flux2Debug.log("Loading text encoder...")

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

        textEncoder = Flux2TextEncoder(quantization: mistralQuant)
        try await textEncoder!.load()

        memoryManager.logMemoryState()
    }

    /// Unload text encoder to free memory for transformer
    @MainActor
    private func unloadTextEncoder() {
        Flux2Debug.log("Unloading text encoder...")
        textEncoder?.unload()
        textEncoder = nil
        memoryManager.fullCleanup()
        memoryManager.logMemoryState()
    }

    /// Load transformer for Phase 2
    private func loadTransformer() async throws {
        guard transformer == nil else { return }

        memoryManager.logMemoryState()
        Flux2Debug.log("Loading transformer...")

        // Get transformer path
        let variant = ModelRegistry.TransformerVariant(rawValue: quantization.transformer.rawValue)!
        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .transformer(variant)) else {
            throw Flux2Error.modelNotLoaded("Transformer weights not found")
        }

        // Create model
        transformer = Flux2Transformer2DModel()

        // Load weights
        let weights = try Flux2WeightLoader.loadWeights(from: modelPath)
        try Flux2WeightLoader.applyTransformerWeights(weights, to: transformer!)

        // Ensure weights are evaluated
        eval(transformer!.parameters())

        memoryManager.logMemoryState()
        Flux2Debug.log("Transformer loaded successfully")
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
    ///   - images: 1-3 reference images
    ///   - height: Optional height (inferred from images if nil)
    ///   - width: Optional width (inferred from images if nil)
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

    /// Unified generation method
    public func generate(
        mode: Flux2GenerationMode,
        prompt: String,
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
        onProgress?(0, steps)
        Flux2Debug.log("=== PHASE 1: Text Encoding ===")

        profiler.start("1. Load Text Encoder")
        try await loadTextEncoder()
        profiler.end("1. Load Text Encoder")

        profiler.start("2. Text Encoding")
        let textEmbeddings = try textEncoder!.encode(prompt, upsample: upsamplePrompt)
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
        var i2iStrength: Float = 1.0  // Default for T2I (full denoising)

        switch mode {
        case .textToImage:
            patchifiedLatents = LatentUtils.generatePatchifiedLatents(
                height: validHeight,
                width: validWidth,
                seed: seed
            )
            Flux2Debug.log("Generated patchified latents: \(patchifiedLatents.shape)")

        case .imageToImage(let images, let strength):
            i2iStrength = strength
            Flux2Debug.log("I2I mode: strength=\(strength)")

            // Encode reference images
            let rawLatents = try encodeReferenceImages(images, height: validHeight, width: validWidth)
            Flux2Debug.log("Encoded latents shape: \(rawLatents.shape)")

            // Convert from [B, 32, H/8, W/8] to patchified [B, 128, H/16, W/16]
            var encodedPatchified = LatentUtils.packLatentsToPatchified(rawLatents)
            Flux2Debug.log("Patchified encoded latents: \(encodedPatchified.shape)")

            // Normalize encoded patchified latents with BatchNorm
            // This standardizes the encoded latents to have similar distribution as random noise
            Flux2Debug.log("Normalizing patchified latents with BatchNorm...")
            encodedPatchified = LatentUtils.normalizeLatentsWithBatchNorm(
                encodedPatchified,
                runningMean: vae!.batchNormRunningMean,
                runningVar: vae!.batchNormRunningVar
            )
            eval(encodedPatchified)

            // Generate noise with same shape
            let noise = LatentUtils.generatePatchifiedLatents(
                height: validHeight,
                width: validWidth,
                seed: seed
            )

            // Mix encoded latents with noise based on initial sigma
            // For flow matching: noisy = (1 - sigma) * clean + sigma * noise
            // At strength=1.0, sigma=1.0 -> pure noise (like T2I)
            // At strength=0.5, sigma≈0.5 -> 50% noise mixed with image
            let sigma = MLXArray(strength)
            patchifiedLatents = (1 - sigma) * encodedPatchified + sigma * noise
            eval(patchifiedLatents)

            Flux2Debug.log("Mixed latents with noise (sigma=\(strength)): \(patchifiedLatents.shape)")
        }

        // Pack patchified latents to sequence format for transformer [B, seq_len, 128]
        var packedLatents = LatentUtils.packPatchifiedToSequence(patchifiedLatents)
        eval(packedLatents)

        Flux2Debug.log("Packed latents shape: \(packedLatents.shape)")

        // Generate position IDs
        let textLength = textEmbeddings.shape[1]
        let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(
            textLength: textLength,
            height: validHeight,
            width: validWidth
        )

        // Calculate image sequence length for scheduler mu
        let imageSeqLen = packedLatents.shape[1]
        Flux2Debug.log("Image sequence length: \(imageSeqLen)")

        // Setup scheduler with image sequence length and strength for proper mu calculation
        // For I2I, this will skip early timesteps based on strength
        scheduler.setTimesteps(numInferenceSteps: steps, imageSeqLen: imageSeqLen, strength: i2iStrength)

        let effectiveSteps = scheduler.sigmas.count - 1
        Flux2Debug.log("Starting denoising loop (\(effectiveSteps) effective steps)...")

        // OPTIMIZATION: Create guidance tensor ONCE before the loop
        let guidanceTensor = MLXArray([guidance])

        profiler.start("6. Denoising Loop")

        // Denoising loop - use sigmas (in [0, 1] range) for transformer
        for stepIdx in 0..<(scheduler.sigmas.count - 1) {
            let stepStart = Date()

            let sigma = scheduler.sigmas[stepIdx]
            // Create timestep tensor (sigma changes each step)
            let t = MLXArray([sigma])

            // Run transformer
            let noisePred = transformer!.callAsFunction(
                hiddenStates: packedLatents,
                encoderHiddenStates: textEmbeddings,
                timestep: t,
                guidance: guidanceTensor,
                imgIds: imageIds,
                txtIds: textIds
            )

            // Scheduler step (uses sigma internally via stepIndex)
            packedLatents = scheduler.step(
                modelOutput: noisePred,
                timestep: sigma,
                sample: packedLatents
            )

            // Synchronize GPU
            eval(packedLatents)

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

                do {
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
                } catch {
                    Flux2Debug.log("Checkpoint error at step \(stepIdx + 1): \(error)")
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

        return image
    }

    // MARK: - Private Methods

    /// Encode reference images for image-to-image
    private func encodeReferenceImages(
        _ images: [CGImage],
        height: Int,
        width: Int
    ) throws -> MLXArray {
        guard let vae = vae else {
            throw Flux2Error.modelNotLoaded("VAE")
        }

        var allLatents: [MLXArray] = []

        for image in images {
            // Preprocess image
            let processed = preprocessImageForVAE(image, targetHeight: height, targetWidth: width)

            // Encode with VAE
            let latent = vae.encode(processed)
            allLatents.append(latent)
        }

        // If multiple images, average the latents
        if allLatents.count == 1 {
            return allLatents[0]
        } else {
            // Squeeze batch dimension from each latent before stacking
            // Each latent is [1, 32, H/8, W/8], we want to average them
            let squeezedLatents = allLatents.map { $0.squeezed(axis: 0) }  // [32, H/8, W/8]
            let stackedLatents = stacked(squeezedLatents, axis: 0)  // [N, 32, H/8, W/8]
            let averaged = mean(stackedLatents, axis: 0, keepDims: false)  // [32, H/8, W/8]
            return averaged.expandedDimensions(axis: 0)  // [1, 32, H/8, W/8]
        }
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
        // Check transformer
        let transformerVariant = ModelRegistry.TransformerVariant(rawValue: quantization.transformer.rawValue)!
        let hasTransformer = Flux2ModelDownloader.isDownloaded(.transformer(transformerVariant))

        // Text encoder is handled by MistralCore, skip check here

        // Check VAE
        let hasVAE = Flux2ModelDownloader.isDownloaded(.vae(.standard))

        return hasTransformer && hasVAE
    }

    /// List missing models
    public var missingModels: [ModelRegistry.ModelComponent] {
        var missing: [ModelRegistry.ModelComponent] = []

        let transformerVariant = ModelRegistry.TransformerVariant(rawValue: quantization.transformer.rawValue)!
        if !Flux2ModelDownloader.isDownloaded(.transformer(transformerVariant)) {
            missing.append(.transformer(transformerVariant))
        }

        // Text encoder is handled by MistralCore

        if !Flux2ModelDownloader.isDownloaded(.vae(.standard)) {
            missing.append(.vae(.standard))
        }

        return missing
    }
}

