import CoreGraphics
import Foundation

#if canImport(CoreImage)
import CoreImage
#endif

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
        try rasterMask(width: width, height: height) { context, _ in
            let inpaintRect = integralPixelRect(
                pixelRect(from: ImagePreparation.clampUnitRect(normalizedRect), width: width, height: height),
                imageWidth: width,
                imageHeight: height
            )
            context.setFillColor(gray: 1, alpha: 1)
            let drawRect = ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: inpaintRect,
                canvasHeight: CGFloat(height)
            )
            context.fill(drawRect)
        }
    }

    public static func polygonInpaintMask(
        width: Int,
        height: Int,
        normalizedPoints: [CGPoint]
    ) throws -> CGImage {
        guard normalizedPoints.count >= 3 else {
            throw Flux2Error.imageProcessingFailed("A polygon mask needs at least three points.")
        }

        return try rasterMask(width: width, height: height) { context, size in
            var path = CGMutablePath()
            let first = pixelPoint(normalizedPoints[0], width: width, height: height)
            path.move(to: first)
            for point in normalizedPoints.dropFirst() {
                path.addLine(to: pixelPoint(point, width: width, height: height))
            }
            path.closeSubpath()

            context.setFillColor(gray: 1, alpha: 1)
            context.addPath(path)
            context.fillPath()
            _ = size
        }
    }

    /// Compose layered primitives, then apply automatic edge feathering.
    public static func buildInpaintMask(
        definition: InpaintMaskDefinition,
        image: CGImage,
        legacyRectangle: CGRect? = nil,
        visionMasks: [UUID: CGImage] = [:]
    ) throws -> CGImage {
        var layers = definition.layers
        if layers.isEmpty, let legacyRectangle {
            layers = [
                InpaintMaskLayer(
                    combineMode: .add,
                    primitive: .rectangle(.init(legacyRectangle))
                )
            ]
        }

        guard !layers.isEmpty else {
            throw Flux2Error.invalidConfiguration("Draw a fill region before generating.")
        }

        let width = image.width
        let height = image.height
        var combined: [UInt8]?

        for (index, layer) in layers.enumerated() {
            let raster = try rasterLayer(
                layer.primitive,
                layerID: layer.id,
                width: width,
                height: height,
                sourceImage: image,
                visionMasks: visionMasks
            )

            if index == 0 {
                combined = raster
                continue
            }

            guard var base = combined else { continue }
            switch layer.combineMode {
            case .add:
                base = zip(base, raster).map { max($0, $1) }
            case .clip:
                base = zip(base, raster).map { min($0, $1) }
            }
            combined = base
        }

        guard let combined else {
            throw Flux2Error.imageProcessingFailed("Failed to compose inpaint mask.")
        }

        let hardMask = try grayImage(from: combined, width: width, height: height)
        return try featherAutomatically(hardMask)
    }

    public static func automaticFeatherRadiusPixels(width: Int, height: Int) -> CGFloat {
        let shorter = CGFloat(min(max(width, 1), max(height, 1)))
        return min(12, max(2, shorter / 200))
    }

    // MARK: - Private helpers

    private static func rasterLayer(
        _ primitive: InpaintMaskPrimitive,
        layerID: UUID,
        width: Int,
        height: Int,
        sourceImage: CGImage,
        visionMasks: [UUID: CGImage]
    ) throws -> [UInt8] {
        let maskImage: CGImage
        switch primitive {
        case .rectangle(let rect):
            maskImage = try rectangularInpaintMask(
                width: width,
                height: height,
                normalizedRect: rect.cgRect
            )
        case .polygon(let points):
            maskImage = try polygonInpaintMask(
                width: width,
                height: height,
                normalizedPoints: points.map(\.cgPoint)
            )
        case .visionSubject:
            guard let visionMask = visionMasks[layerID] else {
                throw Flux2Error.invalidConfiguration("Vision subject mask is not available.")
            }
            guard visionMask.width == width, visionMask.height == height else {
                throw Flux2Error.imageProcessingFailed("Vision mask dimensions must match the source image.")
            }
            maskImage = visionMask
        }

        return try grayBytes(from: maskImage)
    }

    private static func rasterMask(
        width: Int,
        height: Int,
        draw: (CGContext, CGSize) -> Void
    ) throws -> CGImage {
        guard width > 0, height > 0 else {
            throw Flux2Error.imageProcessingFailed("Mask dimensions must be positive")
        }

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
        draw(context, CGSize(width: width, height: height))

        guard let mask = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to create inpaint mask image")
        }
        return mask
    }

    private static func grayBytes(from image: CGImage) throws -> [UInt8] {
        let width = image.width
        let height = image.height
        let count = width * height
        var bytes = [UInt8](repeating: 0, count: count)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw Flux2Error.imageProcessingFailed("Failed to read mask bytes")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return bytes
    }

    private static func grayImage(from bytes: [UInt8], width: Int, height: Int) throws -> CGImage {
        var mutable = bytes
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &mutable,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ), let image = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to create grayscale mask image")
        }
        return image
    }

    private static func featherAutomatically(_ mask: CGImage) throws -> CGImage {
        #if canImport(CoreImage)
        let radius = automaticFeatherRadiusPixels(width: mask.width, height: mask.height)
        guard radius > 0 else { return mask }
        let input = CIImage(cgImage: mask)
        guard let filter = CIFilter(name: "CIGaussianBlur") else { return mask }
        filter.setValue(input, forKey: kCIInputImageKey)
        filter.setValue(radius, forKey: kCIInputRadiusKey)
        guard let output = filter.outputImage else { return mask }
        let context = CIContext(options: [.cacheIntermediates: false])
        let extent = CGRect(x: 0, y: 0, width: mask.width, height: mask.height)
        guard let blurred = context.createCGImage(output, from: extent) else { return mask }
        return blurred
        #else
        return mask
        #endif
    }

    private static func pixelRect(from normalizedRect: CGRect, width: Int, height: Int) -> CGRect {
        CGRect(
            x: normalizedRect.minX * CGFloat(width),
            y: normalizedRect.minY * CGFloat(height),
            width: normalizedRect.width * CGFloat(width),
            height: normalizedRect.height * CGFloat(height)
        )
    }

    private static func pixelPoint(_ normalizedPoint: CGPoint, width: Int, height: Int) -> CGPoint {
        CGPoint(
            x: normalizedPoint.x * CGFloat(width),
            y: normalizedPoint.y * CGFloat(height)
        )
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
