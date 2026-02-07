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
/// ## Usage
/// ```swift
/// let helper = LoRATrainingHelper()
///
/// // Prepare training data from raw images
/// let (latents, embeddings) = try await helper.prepareTrainingData(
///     images: myImages,
///     vae: vae,
///     textEncoder: textEncoder,
///     triggerWord: "xyz_cat"
/// )
///
/// // Get text encoder closure for DOP (handles reload after baseline)
/// let textEncoderClosure = helper.createTextEncoderClosure(textEncoder: textEncoder)
///
/// // Start training
/// try await session.start(
///     config: config,
///     modelType: .klein4B,
///     transformer: transformer,
///     cachedLatents: latents,
///     cachedEmbeddings: embeddings,
///     textEncoder: textEncoderClosure
/// )
/// ```
public final class LoRATrainingHelper: @unchecked Sendable {

    // MARK: - Initialization

    public init() {}

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
