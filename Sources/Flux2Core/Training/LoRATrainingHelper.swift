// LoRATrainingHelper.swift - High-level helper for LoRA training integration
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import CoreGraphics

#if os(macOS)
import AppKit
#endif

/// High-level helper for preparing LoRA training data
///
/// This helper simplifies integration by handling all the complexity of:
/// - Image resizing to valid dimensions (divisible by 16)
/// - VAE encoding with correct normalization
/// - Dimension calculation from latent shapes
/// - Text embedding encoding
/// - Text encoder closure with auto-reload for DOP
///
/// ## Memory-Optimized Usage (Recommended)
///
/// For large models (Klein 4B/9B, Dev), use the memory-optimized flow
/// that loads models sequentially and unloads each after use:
///
/// ```swift
/// let helper = LoRATrainingHelper()
///
/// // 1. Prepare data - VAE and textEncoder are loaded, used, then UNLOADED
/// let (latents, embeddings) = try await helper.prepareTrainingDataMemoryOptimized(
///     images: myImages,
///     vaeLoader: { try await loadVAE() },
///     textEncoderLoader: { try await loadTextEncoder() },
///     triggerWord: "xyz_cat"
/// )
///
/// // 2. Load transformer (now only transformer in memory)
/// let transformer = try await helper.loadTransformerForTraining(modelType: .klein4B)
///
/// // 3. Create lazy text encoder for DOP (loads only when needed)
/// let textEncoderClosure = config.dopEnabled
///     ? helper.createLazyTextEncoderClosure(loader: { try await loadTextEncoder() })
///     : nil
///
/// // 4. Start training
/// // IMPORTANT: Don't pass VAE - it's not used after latent encoding!
/// try await session.start(
///     config: config,
///     modelType: .klein4B,
///     transformer: transformer,
///     cachedLatents: latents,
///     cachedEmbeddings: embeddings,
///     vae: nil,  // VAE is NOT needed!
///     textEncoder: textEncoderClosure
/// )
/// ```
///
/// ## Simple Usage (High Memory)
///
/// For smaller models or systems with plenty of memory:
///
/// ```swift
/// let helper = LoRATrainingHelper()
///
/// // Load all models at once
/// let vae = try await loadVAE()
/// let textEncoder = try await loadTextEncoder()
/// let transformer = try await loadTransformer()
///
/// // Prepare training data
/// let (latents, embeddings) = try await helper.prepareTrainingData(
///     images: myImages,
///     vae: vae,
///     textEncoder: textEncoder,
///     triggerWord: "xyz_cat"
/// )
///
/// // Start training
/// try await session.start(...)
/// ```
public final class LoRATrainingHelper: @unchecked Sendable {

    // MARK: - Initialization

    public init() {}

    // MARK: - Debug Mode

    /// Enable verbose debug logging for training
    ///
    /// Call this before starting training to see detailed logs including:
    /// - Memory cleanup operations
    /// - Model loading steps
    /// - DOP encoder loading
    /// - Timing information for each training step
    ///
    /// Example:
    /// ```swift
    /// let helper = LoRATrainingHelper()
    /// helper.enableDebugMode()  // Call before training
    /// ```
    public func enableDebugMode() {
        Flux2Debug.enableDebugMode()
    }

    /// Disable verbose logging (default mode)
    public func disableDebugMode() {
        Flux2Debug.setNormalMode()
    }

    // MARK: - Training Data Preparation

    /// Input image for training preparation
    public struct TrainingImage: Sendable {
        public let filename: String
        public let image: CGImage
        public let caption: String

        public init(filename: String, image: CGImage, caption: String) {
            self.filename = filename
            self.image = image
            self.caption = caption
        }
    }

    /// Prepare training data from raw images
    ///
    /// This method handles all the complexity of:
    /// - Resizing images to valid dimensions (divisible by 16)
    /// - Encoding with VAE
    /// - Calculating correct dimensions from latent shapes
    /// - Encoding text captions
    ///
    /// - Parameters:
    ///   - images: Array of training images with filenames and captions
    ///   - vae: VAE encoder for latent encoding
    ///   - textEncoder: Text encoder for caption encoding
    ///   - triggerWord: Optional trigger word to prepend to captions
    ///   - progressCallback: Optional callback for progress updates (current, total)
    /// - Returns: Tuple of cached latents and embeddings ready for training
    public func prepareTrainingData(
        images: [TrainingImage],
        vae: AutoencoderKLFlux2,
        textEncoder: TrainingTextEncoder,
        triggerWord: String? = nil,
        progressCallback: ((Int, Int) -> Void)? = nil
    ) async throws -> (latents: [CachedLatentEntry], embeddings: [String: CachedEmbeddingEntry]) {

        var cachedLatents: [CachedLatentEntry] = []
        var cachedEmbeddings: [String: CachedEmbeddingEntry] = [:]

        let total = images.count

        for (index, trainingImage) in images.enumerated() {
            // 1. Resize to valid dimensions (divisible by 16)
            let resizedImage = resizeToValidDimensions(trainingImage.image)

            // 2. Convert to MLXArray and encode with VAE
            let imageArray = cgImageToMLXArray(resizedImage)
            let latent = try encodeImageToLatent(imageArray, vae: vae)

            // 3. Calculate dimensions from latent shape (NOT from original image!)
            // Latent shape after squeeze is [C, H, W]
            // Image dimensions = latent dimensions * 8 (VAE scale factor)
            let imageWidth = latent.shape[2] * 8
            let imageHeight = latent.shape[1] * 8

            cachedLatents.append(CachedLatentEntry(
                filename: trainingImage.filename,
                latent: latent,
                width: imageWidth,
                height: imageHeight
            ))

            // 4. Encode caption (with optional trigger word)
            let fullCaption: String
            if let trigger = triggerWord, !trigger.isEmpty {
                fullCaption = "\(trigger), \(trainingImage.caption)"
            } else {
                fullCaption = trainingImage.caption
            }

            if cachedEmbeddings[fullCaption] == nil {
                // Ensure text encoder is loaded
                if !textEncoder.isLoaded {
                    try await textEncoder.load()
                }
                let embedding = try textEncoder.encodeForTraining(fullCaption)
                cachedEmbeddings[fullCaption] = CachedEmbeddingEntry(
                    caption: fullCaption,
                    embedding: embedding
                )
            }

            progressCallback?(index + 1, total)

            // Clear GPU memory periodically
            if (index + 1) % 10 == 0 {
                MLX.Memory.clearCache()
            }
        }

        return (latents: cachedLatents, embeddings: cachedEmbeddings)
    }

    // MARK: - Text Encoder Closure

    /// Create a text encoder closure that handles auto-reload
    ///
    /// The text encoder may be unloaded during baseline image generation.
    /// This closure automatically reloads it when needed, which is required
    /// for DOP (Differential Output Preservation) to work correctly.
    ///
    /// - Parameter textEncoder: The text encoder to wrap
    /// - Returns: Closure suitable for passing to TrainingSession.start()
    public func createTextEncoderClosure(
        textEncoder: TrainingTextEncoder
    ) -> ((String) async throws -> MLXArray) {
        return { prompt in
            // Reload if unloaded (e.g., after baseline image generation)
            if !textEncoder.isLoaded {
                try await textEncoder.load()
            }
            return try textEncoder.encodeForTraining(prompt)
        }
    }

    /// Create a lazy text encoder closure for memory-optimized training
    ///
    /// This closure loads the text encoder on-demand when DOP needs it.
    /// Use this with `prepareTrainingDataMemoryOptimized()` to minimize memory usage.
    ///
    /// ## Memory-Optimized Training Flow
    /// ```swift
    /// let helper = LoRATrainingHelper()
    ///
    /// // 1. Prepare data with sequential loading (VAE and textEncoder unloaded after)
    /// let (latents, embeddings) = try await helper.prepareTrainingDataMemoryOptimized(
    ///     images: images,
    ///     vaeLoader: { try await loadVAE() },
    ///     textEncoderLoader: { try await loadTextEncoder() },
    ///     triggerWord: "xyz_cat"
    /// )
    ///
    /// // 2. Load transformer (now only transformer is in memory)
    /// let transformer = try await helper.loadTransformerForTraining(modelType: .klein4B)
    ///
    /// // 3. Create lazy text encoder closure for DOP (loads on-demand)
    /// let textEncoderClosure = helper.createLazyTextEncoderClosure(
    ///     loader: { try await loadTextEncoder() }
    /// )
    ///
    /// // 4. Start training - don't pass VAE (it's not used!)
    /// try await session.start(
    ///     config: config,
    ///     modelType: .klein4B,
    ///     transformer: transformer,
    ///     cachedLatents: latents,
    ///     cachedEmbeddings: embeddings,
    ///     vae: nil,  // VAE is NOT needed after latent encoding!
    ///     textEncoder: textEncoderClosure
    /// )
    /// ```
    ///
    /// - Parameter loader: Closure that loads and returns a text encoder
    /// - Returns: Closure suitable for passing to TrainingSession.start()
    public func createLazyTextEncoderClosure(
        loader: @Sendable @escaping () async throws -> TrainingTextEncoder
    ) -> (@Sendable (String) async throws -> MLXArray) {
        // Use actor to safely manage mutable state across Sendable boundary
        let cache = LazyTextEncoderCache(loader: loader)

        return { prompt in
            return try await cache.encode(prompt)
        }
    }

    // MARK: - Image Processing

    /// Resize image to valid dimensions for Flux2 training
    ///
    /// Dimensions must be divisible by 16:
    /// - VAE requires dimensions divisible by 8
    /// - Patchify requires latent dimensions divisible by 2
    /// - Combined: image dimensions must be divisible by 16
    ///
    /// - Parameter image: Original image
    /// - Returns: Resized image with valid dimensions
    public func resizeToValidDimensions(_ image: CGImage) -> CGImage {
        let originalWidth = image.width
        let originalHeight = image.height

        // Round down to nearest multiple of 16
        let validWidth = (originalWidth / 16) * 16
        let validHeight = (originalHeight / 16) * 16

        // Minimum size is 256x256 (16 patches minimum)
        let targetWidth = max(validWidth, 256)
        let targetHeight = max(validHeight, 256)

        // If already valid, return original
        if targetWidth == originalWidth && targetHeight == originalHeight {
            return image
        }

        // Resize
        return resizeImage(image, toWidth: targetWidth, height: targetHeight)
    }

    /// Resize a CGImage to specific dimensions
    private func resizeImage(_ image: CGImage, toWidth width: Int, height: Int) -> CGImage {
        #if os(macOS)
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()!
        #else
        // iOS implementation would go here
        fatalError("iOS not yet supported")
        #endif
    }

    /// Convert CGImage to MLXArray in NCHW format
    private func cgImageToMLXArray(_ image: CGImage) -> MLXArray {
        let width = image.width
        let height = image.height

        // Create bitmap context
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert to Float32 and normalize to [-1, 1]
        var floatData = [Float](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            floatData[i * 3 + 0] = Float(pixelData[i * 4 + 0]) / 127.5 - 1.0  // R
            floatData[i * 3 + 1] = Float(pixelData[i * 4 + 1]) / 127.5 - 1.0  // G
            floatData[i * 3 + 2] = Float(pixelData[i * 4 + 2]) / 127.5 - 1.0  // B
        }

        // Create MLXArray in HWC format then transpose to NCHW
        let hwcArray = MLXArray(floatData, [height, width, 3])
        let chwArray = hwcArray.transposed(2, 0, 1)  // HWC -> CHW
        let nchwArray = chwArray.expandedDimensions(axis: 0)  // CHW -> NCHW

        return nchwArray
    }

    /// Encode image to latent using VAE
    private func encodeImageToLatent(_ image: MLXArray, vae: AutoencoderKLFlux2) throws -> MLXArray {
        // Encode
        var latent = vae.encode(image)

        // Apply Flux2 latent normalization (Ostris formula)
        latent = LatentUtils.normalizeFlux2Latents(latent)

        // Force evaluation
        eval(latent)

        // Remove batch dimension: [1, C, H, W] -> [C, H, W]
        return latent.squeezed(axis: 0)
    }
}

// MARK: - Transformer Loading for Training

extension LoRATrainingHelper {

    /// Load transformer optimized for training
    ///
    /// This method loads the transformer with aggressive memory optimization
    /// settings required for training. Without these settings, the backward
    /// pass can cause memory to explode (100GB+).
    ///
    /// - Parameters:
    ///   - modelType: The model to load
    ///   - weightsPath: Path to model weights
    /// - Returns: Transformer configured for training
    public func loadTransformerForTraining(
        modelType: Flux2Model,
        weightsPath: URL
    ) async throws -> Flux2Transformer2DModel {

        // Get the training variant (base model for Klein)
        let variant = modelType.trainingVariant

        // Create transformer with AGGRESSIVE memory optimization
        // This is CRITICAL for training - without it, memory explodes
        let transformer = Flux2Transformer2DModel(
            config: variant.transformerConfig,
            memoryOptimization: .aggressive  // evalEvery: 4, clearCache: true
        )

        Flux2Debug.log("[LoRATrainingHelper] Loading transformer with aggressive memory optimization")

        // Load weights
        let weights = try Flux2WeightLoader.loadWeights(from: weightsPath)
        try Flux2WeightLoader.applyTransformerWeights(weights, to: transformer)

        // Force evaluation to materialize weights
        eval(transformer.parameters())

        // Clear cache after loading
        MLX.Memory.clearCache()

        Flux2Debug.log("[LoRATrainingHelper] Transformer loaded: \(variant.displayName)")

        return transformer
    }

    /// Load transformer for training with auto-download
    ///
    /// Downloads the model if not present, then loads with training optimizations.
    ///
    /// - Parameters:
    ///   - modelType: The model to load
    ///   - progressCallback: Download progress callback
    /// - Returns: Transformer configured for training
    public func loadTransformerForTraining(
        modelType: Flux2Model,
        progressCallback: (@Sendable (Double, String) -> Void)? = nil
    ) async throws -> Flux2Transformer2DModel {

        // Get the TransformerVariant for training (base model)
        guard let variant = ModelRegistry.TransformerVariant.trainingVariant(for: modelType) else {
            throw LoRATrainingHelperError.failedToEncode("No training variant available for \(modelType)")
        }

        // Check if downloaded, download if needed
        var modelPath = Flux2ModelDownloader.findModelPath(for: .transformer(variant))

        if modelPath == nil {
            Flux2Debug.log("[LoRATrainingHelper] Downloading transformer...")
            let downloader = Flux2ModelDownloader()
            modelPath = try await downloader.download(.transformer(variant), progress: progressCallback)
        }

        guard let path = modelPath else {
            throw LoRATrainingHelperError.failedToEncode("Failed to find or download transformer")
        }

        return try await loadTransformerForTraining(modelType: modelType, weightsPath: path)
    }
}

// MARK: - Memory-Optimized Preparation

extension LoRATrainingHelper {

    /// Memory-optimized training data preparation
    ///
    /// This method minimizes peak memory usage by:
    /// 1. Loading VAE → encoding all images → unloading VAE
    /// 2. Loading text encoder → encoding all captions → unloading text encoder
    /// 3. Clearing GPU cache between phases
    ///
    /// Peak memory is reduced from ~30-40GB to ~15-20GB for Klein 4B.
    ///
    /// - Parameters:
    ///   - images: Array of training images
    ///   - vaeLoader: Closure that loads and returns the VAE
    ///   - textEncoderLoader: Closure that loads and returns the text encoder
    ///   - triggerWord: Optional trigger word
    ///   - progressCallback: Progress callback (phase, current, total)
    /// - Returns: Tuple of cached latents and embeddings
    public func prepareTrainingDataMemoryOptimized(
        images: [TrainingImage],
        vaeLoader: @Sendable @escaping () async throws -> AutoencoderKLFlux2,
        textEncoderLoader: @Sendable @escaping () async throws -> TrainingTextEncoder,
        triggerWord: String? = nil,
        progressCallback: (@Sendable (String, Int, Int) -> Void)? = nil
    ) async throws -> (latents: [CachedLatentEntry], embeddings: [String: CachedEmbeddingEntry]) {

        var cachedLatents: [CachedLatentEntry] = []
        var cachedEmbeddings: [String: CachedEmbeddingEntry] = [:]
        let total = images.count

        // ============================================================
        // PHASE 1: Encode images with VAE (then unload)
        // ============================================================
        progressCallback?("Loading VAE...", 0, total)

        let vae = try await vaeLoader()
        eval(vae.parameters())

        progressCallback?("Encoding latents", 0, total)

        for (index, trainingImage) in images.enumerated() {
            // Resize and encode
            let resizedImage = resizeToValidDimensions(trainingImage.image)
            let imageArray = cgImageToMLXArray(resizedImage)
            let latent = try encodeImageToLatent(imageArray, vae: vae)

            // Calculate dimensions from latent shape
            let imageWidth = latent.shape[2] * 8
            let imageHeight = latent.shape[1] * 8

            cachedLatents.append(CachedLatentEntry(
                filename: trainingImage.filename,
                latent: latent,
                width: imageWidth,
                height: imageHeight
            ))

            progressCallback?("Encoding latents", index + 1, total)

            // Clear cache periodically
            if (index + 1) % 5 == 0 {
                MLX.Memory.clearCache()
            }
        }

        // Unload VAE - critical for memory!
        // VAE parameters go out of scope here, but we force cleanup
        eval([])
        MLX.Memory.clearCache()

        Flux2Debug.log("[LoRATrainingHelper] VAE phase complete, memory released")

        // ============================================================
        // PHASE 2: Encode captions with text encoder (then unload)
        // ============================================================
        progressCallback?("Loading text encoder...", 0, total)

        let textEncoder = try await textEncoderLoader()

        // Collect unique captions
        var uniqueCaptions: [String] = []
        for trainingImage in images {
            let fullCaption: String
            if let trigger = triggerWord, !trigger.isEmpty {
                fullCaption = "\(trigger), \(trainingImage.caption)"
            } else {
                fullCaption = trainingImage.caption
            }
            if cachedEmbeddings[fullCaption] == nil {
                uniqueCaptions.append(fullCaption)
                // Placeholder to mark as "will be encoded"
                cachedEmbeddings[fullCaption] = CachedEmbeddingEntry(
                    caption: fullCaption,
                    embedding: MLXArray([0])  // Temporary placeholder
                )
            }
        }

        progressCallback?("Encoding embeddings", 0, uniqueCaptions.count)

        // Encode each unique caption
        for (index, caption) in uniqueCaptions.enumerated() {
            let embedding = try textEncoder.encodeForTraining(caption)
            cachedEmbeddings[caption] = CachedEmbeddingEntry(
                caption: caption,
                embedding: embedding
            )

            progressCallback?("Encoding embeddings", index + 1, uniqueCaptions.count)

            // Clear cache periodically
            if (index + 1) % 3 == 0 {
                MLX.Memory.clearCache()
            }
        }

        // Unload text encoder
        await textEncoder.unload()
        eval([])
        MLX.Memory.clearCache()

        Flux2Debug.log("[LoRATrainingHelper] Text encoder phase complete, memory released")

        // ============================================================
        // Ready for training - only transformer needs to be loaded
        // ============================================================
        progressCallback?("Ready", total, total)

        return (latents: cachedLatents, embeddings: cachedEmbeddings)
    }

    /// Estimate memory requirements for training
    ///
    /// - Parameters:
    ///   - modelType: The model to train
    ///   - imageCount: Number of training images
    ///   - averageImageSize: Average image dimensions
    /// - Returns: Estimated memory in GB
    public func estimateMemoryGB(
        modelType: Flux2Model,
        imageCount: Int,
        averageImageSize: (width: Int, height: Int) = (512, 512)
    ) -> Float {
        // Base model sizes (approximate)
        let transformerGB: Float
        switch modelType {
        case .klein4B, .klein4BBase:
            transformerGB = 10.0  // bf16
        case .klein9B, .klein9BBase:
            transformerGB = 20.0  // bf16
        case .dev:
            transformerGB = 24.0  // bf16
        }

        // Latent cache size
        let latentChannels: Float = 32
        let latentH = Float(averageImageSize.height) / 8.0
        let latentW = Float(averageImageSize.width) / 8.0
        let latentSize = latentChannels * latentH * latentW * 4  // float32
        let latentCacheGB = Float(imageCount) * latentSize / (1024 * 1024 * 1024)

        // Embedding cache size (512 tokens * hidden dim * 4 bytes)
        let embeddingSize: Float = 512 * 7680 * 4  // Klein
        let embeddingCacheGB = Float(imageCount) * embeddingSize / (1024 * 1024 * 1024)

        // Training overhead (gradients, optimizer state)
        let overheadMultiplier: Float = 2.5  // ~2.5x for training

        return (transformerGB * overheadMultiplier) + latentCacheGB + embeddingCacheGB
    }
}

// MARK: - Convenience Extensions

extension LoRATrainingHelper {

    /// Prepare training data from file URLs
    ///
    /// Convenience method that loads images from disk.
    ///
    /// - Parameters:
    ///   - imageFiles: Array of (fileURL, caption) pairs
    ///   - vae: VAE encoder
    ///   - textEncoder: Text encoder
    ///   - triggerWord: Optional trigger word
    ///   - progressCallback: Optional progress callback
    /// - Returns: Tuple of cached latents and embeddings
    public func prepareTrainingData(
        fromFiles imageFiles: [(url: URL, caption: String)],
        vae: AutoencoderKLFlux2,
        textEncoder: TrainingTextEncoder,
        triggerWord: String? = nil,
        progressCallback: ((Int, Int) -> Void)? = nil
    ) async throws -> (latents: [CachedLatentEntry], embeddings: [String: CachedEmbeddingEntry]) {

        var images: [TrainingImage] = []

        for (url, caption) in imageFiles {
            #if os(macOS)
            guard let nsImage = NSImage(contentsOf: url),
                  let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw LoRATrainingHelperError.failedToLoadImage(url.lastPathComponent)
            }
            #else
            fatalError("iOS not yet supported")
            #endif

            images.append(TrainingImage(
                filename: url.lastPathComponent,
                image: cgImage,
                caption: caption
            ))
        }

        return try await prepareTrainingData(
            images: images,
            vae: vae,
            textEncoder: textEncoder,
            triggerWord: triggerWord,
            progressCallback: progressCallback
        )
    }
}

// MARK: - Lazy Text Encoder Cache

/// Thread-safe cache for lazy text encoder loading
/// Used by createLazyTextEncoderClosure to manage mutable state across Sendable boundaries
/// Note: @unchecked Sendable because MLXArray is not Sendable but we ensure thread-safety
/// by only accessing from async context (single-threaded in practice for training)
private final class LazyTextEncoderCache: @unchecked Sendable {
    private var cachedEncoder: TrainingTextEncoder?
    private let loader: @Sendable () async throws -> TrainingTextEncoder

    init(loader: @Sendable @escaping () async throws -> TrainingTextEncoder) {
        self.loader = loader
    }

    func encode(_ prompt: String) async throws -> MLXArray {
        // Load encoder if not cached or unloaded
        if cachedEncoder == nil || !cachedEncoder!.isLoaded {
            Flux2Debug.log("[LazyTextEncoderCache] Loading text encoder for DOP...")
            cachedEncoder = try await loader()
            if !cachedEncoder!.isLoaded {
                try await cachedEncoder!.load()
            }
        }
        return try cachedEncoder!.encodeForTraining(prompt)
    }
}

// MARK: - Errors

public enum LoRATrainingHelperError: LocalizedError {
    case failedToLoadImage(String)
    case failedToEncode(String)
    case invalidDimensions(width: Int, height: Int)

    public var errorDescription: String? {
        switch self {
        case .failedToLoadImage(let filename):
            return "Failed to load image: \(filename)"
        case .failedToEncode(let reason):
            return "Failed to encode: \(reason)"
        case .invalidDimensions(let width, let height):
            return "Invalid dimensions \(width)x\(height). Must be at least 256x256."
        }
    }
}
