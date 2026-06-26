/**
 * ImageScaler.swift
 * One honest, deterministic resampler shared by every "resize real pixels" path
 * (HQ-selection resize, the output save bus, chains). Lanczos via Core Image —
 * no detail invention. Lives here so ImageSaveService's private copy and any new
 * consumer share a single implementation (the dedup of ImageSaveService happens
 * in the follow-up pass).
 */

import CoreGraphics
import CoreImage
import Foundation

public enum ImageScaler {
    /// High-quality Lanczos resample of `image` to exactly `size`.
    /// - Throws: `Flux2Error.imageProcessingFailed` if the size is invalid or the
    ///   Core Image filter / render fails.
    public static func lanczos(_ image: CGImage, to size: PixelSize) throws -> CGImage {
        guard size.width > 0, size.height > 0 else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.lanczos: target size must be positive")
        }
        let sourceWidth = CGFloat(image.width)
        let sourceHeight = CGFloat(image.height)
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.lanczos: source image is empty")
        }

        // CILanczosScaleTransform scales uniformly by `scale`, then stretches
        // horizontally by `aspectRatio`. Solve both so the result lands on
        // exactly the requested width and height.
        let scale = CGFloat(size.height) / sourceHeight
        let aspectRatio = (CGFloat(size.width) / sourceWidth) / scale

        guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.lanczos: CILanczosScaleTransform unavailable")
        }
        filter.setValue(CIImage(cgImage: image), forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)

        guard let output = filter.outputImage else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.lanczos: filter produced no output")
        }

        // Render from an explicit rect of the exact target size: Lanczos output
        // extents can be fractional, so cropping the render guarantees integral
        // dimensions.
        let context = CIContext(options: nil)
        let rect = CGRect(x: 0, y: 0, width: CGFloat(size.width), height: CGFloat(size.height))
        guard let result = context.createCGImage(output, from: rect) else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.lanczos: failed to render output")
        }
        return result
    }
}
