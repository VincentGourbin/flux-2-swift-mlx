/**
 * ImageScaler.swift
 * Honest, deterministic resamplers for every "resize real pixels" path — Lanczos
 * and bicubic via Core Image, no detail invention. The single shared home so no
 * caller keeps a private copy; current consumers are the output save bus
 * (ImageSaveService) and the Upsize control.
 */

import CoreGraphics
import CoreImage
import Foundation

public enum ImageScaler {
    /// High-quality Lanczos resample of `image` to exactly `size`.
    /// - Throws: `Flux2Error.imageProcessingFailed` if the size is invalid or the
    ///   Core Image filter / render fails.
    public static func lanczos(_ image: CGImage, to size: PixelSize) throws -> CGImage {
        try scale(image, to: size, filterName: "CILanczosScaleTransform", label: "lanczos")
    }

    /// Bicubic resample of `image` to exactly `size`. Softer than Lanczos; offered
    /// alongside it so the Upsize control can pick the resampler explicitly.
    /// B = 0, C = 0.75 is the standard sharp cubic (the Catmull-Rom family).
    /// - Throws: `Flux2Error.imageProcessingFailed` if the size is invalid or the
    ///   Core Image filter / render fails.
    public static func bicubic(_ image: CGImage, to size: PixelSize) throws -> CGImage {
        try scale(image, to: size, filterName: "CIBicubicScaleTransform", label: "bicubic") { filter in
            filter.setValue(0.0, forKey: "inputB")
            filter.setValue(0.75, forKey: "inputC")
        }
    }

    /// Shared Core Image scale: the only difference between resamplers is the
    /// filter name and any filter-specific parameters supplied by `configure`.
    private static func scale(
        _ image: CGImage,
        to size: PixelSize,
        filterName: String,
        label: String,
        configure: ((CIFilter) -> Void)? = nil
    ) throws -> CGImage {
        guard size.width > 0, size.height > 0 else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.\(label): target size must be positive")
        }
        let sourceWidth = CGFloat(image.width)
        let sourceHeight = CGFloat(image.height)
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.\(label): source image is empty")
        }

        // The scale-transform filters scale uniformly by `scale`, then stretch
        // horizontally by `aspectRatio`. Solve both so the result lands on
        // exactly the requested width and height.
        let scale = CGFloat(size.height) / sourceHeight
        let aspectRatio = (CGFloat(size.width) / sourceWidth) / scale

        guard let filter = CIFilter(name: filterName) else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.\(label): \(filterName) unavailable")
        }
        filter.setValue(CIImage(cgImage: image), forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        configure?(filter)

        guard let output = filter.outputImage else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.\(label): filter produced no output")
        }

        // Render from an explicit rect of the exact target size: scale-transform
        // output extents can be fractional, so cropping the render guarantees
        // integral dimensions.
        let context = CIContext(options: nil)
        let rect = CGRect(x: 0, y: 0, width: CGFloat(size.width), height: CGFloat(size.height))
        guard let result = context.createCGImage(output, from: rect) else {
            throw Flux2Error.imageProcessingFailed("ImageScaler.\(label): failed to render output")
        }
        return result
    }
}
