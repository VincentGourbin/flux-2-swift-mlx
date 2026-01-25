/**
 * ImageProcessor.swift
 * Preprocesses images for Mistral Small 3.2 vision encoder (Pixtral)
 */

import Foundation
import MLX
import MLXNN
import Accelerate

#if canImport(AppKit)
import AppKit
#endif

/// Configuration for image preprocessing
public struct ImageProcessorConfig: Codable, Sendable {
    public let imageSize: Int           // Longest edge target size
    public let patchSize: Int           // Vision encoder patch size
    public let imageMean: [Float]       // Normalization mean (RGB)
    public let imageStd: [Float]        // Normalization std (RGB)
    public let rescaleFactor: Float     // Pixel rescaling factor (1/255)

    public static let pixtral = ImageProcessorConfig(
        imageSize: 1540,
        patchSize: 14,
        imageMean: [0.48145466, 0.4578275, 0.40821073],
        imageStd: [0.26862954, 0.26130258, 0.27577711],
        rescaleFactor: 1.0 / 255.0
    )

    public init(
        imageSize: Int,
        patchSize: Int,
        imageMean: [Float],
        imageStd: [Float],
        rescaleFactor: Float
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.rescaleFactor = rescaleFactor
    }
}

/// Image processor for Pixtral vision encoder
public class ImageProcessor {
    public let config: ImageProcessorConfig

    public init(config: ImageProcessorConfig = .pixtral) {
        self.config = config
    }

    /// Preprocess an image for the vision encoder
    /// - Parameter image: Input NSImage
    /// - Returns: MLXArray with shape [1, H, W, 3] (NHWC format for MLX Conv2d)
    public func preprocess(_ image: NSImage) throws -> MLXArray {
        return try preprocess(image, maxSize: config.imageSize)
    }

    /// Preprocess an image for the vision encoder with custom max size
    /// - Parameters:
    ///   - image: Input NSImage
    ///   - maxSize: Maximum size for longest edge (overrides config.imageSize)
    /// - Returns: MLXArray with shape [1, H, W, 3] (NHWC format for MLX Conv2d)
    public func preprocess(_ image: NSImage, maxSize: Int) throws -> MLXArray {
        // Use autoreleasepool to free CoreGraphics memory immediately
        let (pixels, width, height): ([Float], Int, Int) = try autoreleasepool {
            // 1. Resize image maintaining aspect ratio
            let resizedImage = try resizeImage(image, longestEdge: maxSize)

            // 2. Get pixel data as Float array
            return try extractPixels(resizedImage)
        }

        // 3. Convert to MLXArray [H, W, 3]
        var pixelArray = MLXArray(pixels, [height, width, 3])

        // 4. Rescale pixels (÷255)
        pixelArray = pixelArray * config.rescaleFactor

        // 5. Normalize with ImageNet mean/std
        let mean = MLXArray(config.imageMean).reshaped([1, 1, 3])
        let std = MLXArray(config.imageStd).reshaped([1, 1, 3])
        pixelArray = (pixelArray - mean) / std

        // 6. Add batch dimension [1, H, W, 3] (NHWC format for MLX)
        pixelArray = pixelArray.expandedDimensions(axis: 0)

        return pixelArray
    }

    /// Resize image maintaining aspect ratio with longest edge <= target size
    /// Uses pixel dimensions (not point dimensions) to handle Retina displays correctly
    /// NOTE: Only downscales images larger than longestEdge, does NOT upscale smaller images
    /// This matches the Python PixtralImageProcessor behavior
    private func resizeImage(_ image: NSImage, longestEdge: Int) throws -> NSImage {
        // Get actual pixel dimensions from CGImage, not NSImage.size (which is in points)
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw ImageProcessorError.invalidImage
        }

        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)

        // Calculate ratio like Python: max(height/max_height, width/max_width)
        let ratio = max(originalHeight / CGFloat(longestEdge), originalWidth / CGFloat(longestEdge))

        var newWidth: Int
        var newHeight: Int

        if ratio > 1 {
            // Image is larger than longestEdge - downscale
            newHeight = Int(floor(originalHeight / ratio))
            newWidth = Int(floor(originalWidth / ratio))
        } else {
            // Image is smaller - keep original size (don't upscale!)
            newWidth = Int(originalWidth)
            newHeight = Int(originalHeight)
        }

        // Ensure dimensions are divisible by patchSize (14) - matching Python's PixtralImageProcessor
        // Note: The spatialMergeSize alignment is handled by the model's patch merger, not here
        let alignmentFactor = config.patchSize  // 14
        let alignedWidth = ((newWidth + alignmentFactor - 1) / alignmentFactor) * alignmentFactor
        let alignedHeight = ((newHeight + alignmentFactor - 1) / alignmentFactor) * alignmentFactor

        // Create a bitmap context at the exact pixel dimensions we want
        // CRITICAL: Use the image's own colorspace to avoid color profile conversion
        // This matches Python PIL behavior which reads raw pixel values
        let bytesPerPixel = 4
        let bytesPerRow = alignedWidth * bytesPerPixel

        // Use the image's colorspace to avoid any color conversion
        let colorSpace = cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: nil,
            width: alignedWidth,
            height: alignedHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw ImageProcessorError.contextCreationFailed
        }

        // Use high quality interpolation
        context.interpolationQuality = .high

        // Draw the original image scaled to the new size
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: alignedWidth, height: alignedHeight))

        // Create a new CGImage from the context
        guard let resizedCGImage = context.makeImage() else {
            throw ImageProcessorError.contextCreationFailed
        }

        // Convert back to NSImage (with 1:1 point-to-pixel ratio)
        let resizedImage = NSImage(cgImage: resizedCGImage, size: NSSize(width: alignedWidth, height: alignedHeight))

        return resizedImage
    }

    /// Extract RGB pixel data from NSImage
    /// Uses the image's own colorspace to avoid color profile conversion (matches Python PIL)
    /// Optimized with vDSP for vectorized conversion
    private func extractPixels(_ image: NSImage) throws -> (pixels: [Float], width: Int, height: Int) {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw ImageProcessorError.invalidImage
        }

        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow
        let pixelCount = width * height

        var pixelData = [UInt8](repeating: 0, count: totalBytes)

        // CRITICAL: Use image's own colorspace to avoid color profile conversion
        // This matches Python PIL behavior which reads raw pixel values
        let colorSpace = cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw ImageProcessorError.contextCreationFailed
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBA UInt8 to RGB Float using vDSP (vectorized SIMD operations)
        var floatPixels = [Float](repeating: 0, count: pixelCount * 3)

        // Use vDSP_vfltu8 for vectorized UInt8->Float conversion with stride
        // Input: RGBARGBA... (stride 4), Output: RGBRGB... (stride 3)
        pixelData.withUnsafeBufferPointer { srcBuffer in
            floatPixels.withUnsafeMutableBufferPointer { dstBuffer in
                let srcBase = srcBuffer.baseAddress!
                let dstBase = dstBuffer.baseAddress!
                let n = vDSP_Length(pixelCount)

                // R channel: src[0, 4, 8, ...] -> dst[0, 3, 6, ...]
                vDSP_vfltu8(srcBase, 4, dstBase, 3, n)

                // G channel: src[1, 5, 9, ...] -> dst[1, 4, 7, ...]
                vDSP_vfltu8(srcBase + 1, 4, dstBase + 1, 3, n)

                // B channel: src[2, 6, 10, ...] -> dst[2, 5, 8, ...]
                vDSP_vfltu8(srcBase + 2, 4, dstBase + 2, 3, n)
            }
        }

        return (floatPixels, width, height)
    }

    /// Load image from file path
    public func loadImage(from path: String) throws -> NSImage {
        let url = URL(fileURLWithPath: path)
        guard let image = NSImage(contentsOf: url) else {
            throw ImageProcessorError.fileNotFound(path)
        }
        return image
    }

    /// Preprocess image from file path
    public func preprocessFromFile(_ path: String) throws -> MLXArray {
        let image = try loadImage(from: path)
        return try preprocess(image)
    }

    /// Get number of patches for a given image size
    public func getNumPatches(width: Int, height: Int) -> (patchesX: Int, patchesY: Int, total: Int) {
        let patchesX = width / config.patchSize
        let patchesY = height / config.patchSize
        return (patchesX, patchesY, patchesX * patchesY)
    }
}

/// Errors for image processing
public enum ImageProcessorError: LocalizedError {
    case invalidImage
    case contextCreationFailed
    case fileNotFound(String)
    case unsupportedFormat

    public var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid image format"
        case .contextCreationFailed:
            return "Failed to create graphics context"
        case .fileNotFound(let path):
            return "Image file not found: \(path)"
        case .unsupportedFormat:
            return "Unsupported image format"
        }
    }
}
