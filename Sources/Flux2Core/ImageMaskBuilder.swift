import CoreGraphics
import Foundation

/// Builds Flux-compatible inpaint masks from geometric selections.
public enum ImageMaskBuilder {
    /// Grayscale mask for ``Flux2MaskConvention/grayscaleWhiteInpaint``: white inside
    /// `normalizedRect` (inpaint), black outside (keep). Coordinates use a
    /// top-left image origin, matching the SwiftUI preview and ``ImagePreparation``.
    public static func rectangularInpaintMask(
        width: Int,
        height: Int,
        normalizedRect: CGRect
    ) throws -> CGImage {
        guard width > 0, height > 0 else {
            throw Flux2Error.imageProcessingFailed("Mask dimensions must be positive")
        }

        let clamped = ImagePreparation.clampUnitRect(normalizedRect)
        let pixelRect = CGRect(
            x: clamped.minX * CGFloat(width),
            y: clamped.minY * CGFloat(height),
            width: clamped.width * CGFloat(width),
            height: clamped.height * CGFloat(height)
        )
        let inpaintRect = integralPixelRect(pixelRect, imageWidth: width, imageHeight: height)

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw Flux2Error.imageProcessingFailed("Failed to allocate inpaint mask")
        }

        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        context.setFillColor(gray: 1, alpha: 1)
        let drawRect = ImageCoordinateMapper.contextDrawRect(
            forTopLeftRect: inpaintRect,
            canvasHeight: CGFloat(height)
        )
        context.fill(drawRect)

        guard let mask = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to create inpaint mask image")
        }
        return mask
    }

    private static func integralPixelRect(_ rect: CGRect, imageWidth: Int, imageHeight: Int) -> CGRect {
        let imageBounds = CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight)
        let bounded = rect.intersection(imageBounds)

        guard !bounded.isNull, bounded.width > 0, bounded.height > 0 else {
            return CGRect(x: 0, y: 0, width: max(1, min(imageWidth, 1)), height: max(1, min(imageHeight, 1)))
        }

        let minX = floor(bounded.minX)
        let minY = floor(bounded.minY)
        let maxX = ceil(bounded.maxX)
        let maxY = ceil(bounded.maxY)

        return CGRect(
            x: min(max(minX, 0), CGFloat(max(imageWidth - 1, 0))),
            y: min(max(minY, 0), CGFloat(max(imageHeight - 1, 0))),
            width: max(1, min(maxX, CGFloat(imageWidth)) - minX),
            height: max(1, min(maxY, CGFloat(imageHeight)) - minY)
        )
    }
}
