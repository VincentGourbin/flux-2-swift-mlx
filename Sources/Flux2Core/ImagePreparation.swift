import CoreGraphics
import Foundation

public enum ImagePreparation {
    public static let generationSizeMultiple = 32
    public static let referenceConditioningArea = 1024 * 1024

    public static func clampUnitRect(_ rect: CGRect) -> CGRect {
        let minSize: CGFloat = 0.01
        let width = min(max(rect.width, minSize), 1)
        let height = min(max(rect.height, minSize), 1)
        let x = min(max(rect.minX, 0), 1 - width)
        let y = min(max(rect.minY, 0), 1 - height)
        return CGRect(x: x, y: y, width: width, height: height)
    }

    public static func snapToMultiple(_ value: Int, multiple: Int) -> Int {
        guard multiple > 0 else { return value }
        return max(multiple, ((value + multiple - 1) / multiple) * multiple)
    }

    public static func referenceMatchedSize(
        width: Int,
        height: Int,
        maxArea: Int = referenceConditioningArea,
        multiple: Int = generationSizeMultiple
    ) -> (width: Int, height: Int) {
        var w = Double(max(1, width))
        var h = Double(max(1, height))
        let area = w * h
        if area > Double(maxArea) {
            let scale = (Double(maxArea) / area).squareRoot()
            w *= scale
            h *= scale
        }
        let flooredWidth = max(multiple, (Int(w) / multiple) * multiple)
        let flooredHeight = max(multiple, (Int(h) / multiple) * multiple)
        return (flooredWidth, flooredHeight)
    }

    public static func conditioningPixelBudget(for megapixelBudget: Double) -> Int {
        let clamped = min(max(megapixelBudget, ImagePreparationSettings.minMegapixelBudget), ImagePreparationSettings.maxMegapixelBudget)
        return Int((clamped * 1_000_000).rounded())
    }

    public static func budgetFilledSize(
        sourceAspect: Double,
        settings: ImagePreparationSettings
    ) -> (width: Int, height: Int) {
        let targetAspect: Double
        switch settings.sizingFavor {
        case .original:
            targetAspect = sourceAspect
        case .horizontal:
            targetAspect = max(sourceAspect, 4.0 / 3.0)
        case .vertical:
            targetAspect = min(sourceAspect, 3.0 / 4.0)
        }

        let scale = min(max(settings.preparationScale, 0.1), 1.0)
        let pixelBudget = Double(conditioningPixelBudget(for: settings.megapixelBudget)) * scale * scale
        let rawHeight = (pixelBudget / max(targetAspect, 0.0001)).squareRoot()
        let rawWidth = rawHeight * targetAspect

        return (
            snapToMultiple(Int(rawWidth.rounded()), multiple: settings.pixelAlignment),
            snapToMultiple(Int(rawHeight.rounded()), multiple: settings.pixelAlignment)
        )
    }

    public static func generationSize(
        referenceImage: CGImage,
        settings: ImagePreparationSettings
    ) -> (width: Int, height: Int) {
        var settings = settings
        settings.clampValues()
        let sourceRect = integralPixelRect(from: settings.contextArea, in: referenceImage)
        let sourceAspect = Double(sourceRect.width) / Double(sourceRect.height)
        let size = budgetFilledSize(sourceAspect: sourceAspect, settings: settings)
        return referenceMatchedSize(
            width: size.width,
            height: size.height,
            maxArea: conditioningPixelBudget(for: settings.megapixelBudget),
            multiple: settings.pixelAlignment
        )
    }

    public static func prepare(
        referenceImages: [CGImage],
        settings: ImagePreparationSettings
    ) throws -> PreparedImageToImageInput {
        guard let original = referenceImages.first else {
            throw Flux2Error.invalidConfiguration("Add a reference image before generating")
        }

        var settings = settings
        settings.clampValues()

        let contextRect = integralPixelRect(from: settings.contextArea, in: original)
        let processRect = integralProcessRect(in: original, contextRect: contextRect, processArea: settings.processArea)
        let targetSize = generationSize(referenceImage: original, settings: settings)

        let contextImage = try cropImage(original, to: contextRect)
        let transform = preparationTransform(
            sourceWidth: contextImage.width,
            sourceHeight: contextImage.height,
            targetWidth: targetSize.width,
            targetHeight: targetSize.height,
            method: settings.sizingMethod
        )
        let preparedFirstImage = try renderImage(
            contextImage,
            targetWidth: targetSize.width,
            targetHeight: targetSize.height,
            transform: transform
        )

        let remainingImages = referenceImages.dropFirst()
        let plan = ImageCompositionPlan(
            originalImage: original,
            contextRect: contextRect,
            processRect: processRect,
            transform: transform
        )

        return PreparedImageToImageInput(
            images: [preparedFirstImage] + remainingImages,
            width: targetSize.width,
            height: targetSize.height,
            compositionPlan: settings.compositeBack ? plan : nil
        )
    }

    public static func composite(
        _ generatedImage: CGImage,
        using plan: ImageCompositionPlan
    ) throws -> CGImage {
        let processInContext = plan.processRect.offsetBy(dx: -plan.contextRect.minX, dy: -plan.contextRect.minY)
        let mappedProcessRect = plan.transform.canvasRect(forSourceRect: processInContext)
        let canvasBounds = CGRect(x: 0, y: 0, width: generatedImage.width, height: generatedImage.height)
        let visibleCanvasRect = mappedProcessRect.intersection(canvasBounds)

        guard !visibleCanvasRect.isNull, visibleCanvasRect.width > 0, visibleCanvasRect.height > 0 else {
            throw Flux2Error.imageProcessingFailed("Process area falls outside the generated canvas")
        }

        let generatedCropRect = integralPixelRect(visibleCanvasRect, imageWidth: generatedImage.width, imageHeight: generatedImage.height)
        let generatedPatch = try cropImage(generatedImage, to: generatedCropRect)
        let visibleSourceRect = plan.transform.sourceRect(forCanvasRect: generatedCropRect)
        let destinationRect = integralPixelRect(
            visibleSourceRect.offsetBy(dx: plan.contextRect.minX, dy: plan.contextRect.minY),
            imageWidth: plan.originalImage.width,
            imageHeight: plan.originalImage.height
        )

        guard let context = makeImageContext(width: plan.originalImage.width, height: plan.originalImage.height) else {
            throw Flux2Error.imageProcessingFailed("Failed to create composition context")
        }

        context.interpolationQuality = .high
        context.draw(
            plan.originalImage,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: CGRect(x: 0, y: 0, width: plan.originalImage.width, height: plan.originalImage.height),
                canvasHeight: CGFloat(plan.originalImage.height)
            )
        )
        context.draw(
            generatedPatch,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: destinationRect,
                canvasHeight: CGFloat(plan.originalImage.height)
            )
        )

        guard let compositedImage = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to composite generated patch")
        }

        return compositedImage
    }

    // MARK: - Private helpers

    private static func preparationTransform(
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        method: ImageSizingMethod
    ) -> ImagePreparationTransform {
        let xScale = CGFloat(targetWidth) / CGFloat(max(sourceWidth, 1))
        let yScale = CGFloat(targetHeight) / CGFloat(max(sourceHeight, 1))
        let scale = method == .crop ? max(xScale, yScale) : min(xScale, yScale)
        let drawnWidth = CGFloat(sourceWidth) * scale
        let drawnHeight = CGFloat(sourceHeight) * scale

        return ImagePreparationTransform(
            targetWidth: targetWidth,
            targetHeight: targetHeight,
            scale: scale,
            offsetX: (CGFloat(targetWidth) - drawnWidth) / 2,
            offsetY: (CGFloat(targetHeight) - drawnHeight) / 2
        )
    }

    private static func renderImage(
        _ image: CGImage,
        targetWidth: Int,
        targetHeight: Int,
        transform: ImagePreparationTransform
    ) throws -> CGImage {
        guard let context = makeImageContext(width: targetWidth, height: targetHeight) else {
            throw Flux2Error.imageProcessingFailed("Failed to create prepared image context")
        }

        context.interpolationQuality = .high
        let drawRect = CGRect(
            x: transform.offsetX,
            y: transform.offsetY,
            width: CGFloat(image.width) * transform.scale,
            height: CGFloat(image.height) * transform.scale
        )
        context.draw(
            image,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: drawRect,
                canvasHeight: CGFloat(targetHeight)
            )
        )

        guard let renderedImage = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to render prepared image")
        }

        return renderedImage
    }

    private static func cropImage(_ image: CGImage, to rect: CGRect) throws -> CGImage {
        let cropRect = integralPixelRect(rect, imageWidth: image.width, imageHeight: image.height)

        if let cropped = image.cropping(to: cropRect) {
            return cropped
        }

        guard let context = makeImageContext(width: Int(cropRect.width), height: Int(cropRect.height)) else {
            throw Flux2Error.imageProcessingFailed("Failed to create crop context")
        }

        let sourceDrawRect = CGRect(
            x: -cropRect.minX,
            y: -cropRect.minY,
            width: CGFloat(image.width),
            height: CGFloat(image.height)
        )
        context.draw(
            image,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: sourceDrawRect,
                canvasHeight: cropRect.height
            )
        )

        guard let croppedImage = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to crop image")
        }

        return croppedImage
    }

    private static func integralProcessRect(
        in image: CGImage,
        contextRect: CGRect,
        processArea: CGRect?
    ) -> CGRect {
        guard let processArea else {
            return contextRect
        }

        let rawProcessRect = pixelRect(from: clampUnitRect(processArea), in: image)
        let clampedProcessRect = rawProcessRect.intersection(contextRect)

        guard !clampedProcessRect.isNull, clampedProcessRect.width > 0, clampedProcessRect.height > 0 else {
            return contextRect
        }

        return integralPixelRect(clampedProcessRect, imageWidth: image.width, imageHeight: image.height)
    }

    private static func pixelRect(from normalizedRect: CGRect, in image: CGImage) -> CGRect {
        CGRect(
            x: normalizedRect.minX * CGFloat(image.width),
            y: normalizedRect.minY * CGFloat(image.height),
            width: normalizedRect.width * CGFloat(image.width),
            height: normalizedRect.height * CGFloat(image.height)
        )
    }

    private static func integralPixelRect(from normalizedRect: CGRect, in image: CGImage) -> CGRect {
        integralPixelRect(
            pixelRect(from: clampUnitRect(normalizedRect), in: image),
            imageWidth: image.width,
            imageHeight: image.height
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

    private static func makeImageContext(width: Int, height: Int) -> CGContext? {
        CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    }
}
