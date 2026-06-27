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

    /// Round `value` DOWN to `multiple` (floor), never below `multiple` itself.
    /// The single home for the floor-to-alignment step — `ScalingPolicy.snapDown`
    /// and the reference-render sizing both route through here so they can't drift.
    public static func floorToMultiple(_ value: Int, multiple: Int) -> Int {
        guard multiple > 0 else { return value }
        return max(multiple, (value / multiple) * multiple)
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
        let flooredWidth = floorToMultiple(Int(w), multiple: multiple)
        let flooredHeight = floorToMultiple(Int(h), multiple: multiple)
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

    /// Apply Image Formatting (Favour + crop/pad) to the full reference frame at an
    /// explicit canvas size. Used for aligned A/B preview and variant saves.
    public static func formatToCanvas(
        referenceImage: CGImage,
        settings: ImagePreparationSettings,
        targetWidth: Int,
        targetHeight: Int
    ) throws -> CGImage {
        var settings = settings
        settings.clampValues()
        settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)

        let contextRect = integralPixelRect(from: settings.contextArea, in: referenceImage)
        let contextImage = try cropImage(referenceImage, to: contextRect)
        let transform = preparationTransform(
            sourceWidth: contextImage.width,
            sourceHeight: contextImage.height,
            targetWidth: targetWidth,
            targetHeight: targetHeight,
            method: settings.sizingMethod
        )
        return try renderImage(
            contextImage,
            targetWidth: targetWidth,
            targetHeight: targetHeight,
            transform: transform
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

        // Conditioning fidelity: never UP-sample the reference the VAE encodes.
        // Up-sampling a low-quality JPEG (Core Graphics .high) smears its 8x8
        // block edges into fuzzy gradients that FLUX.2 reads as real structure —
        // out of distribution, since super-resolution priors are trained on
        // native degraded JPEGs, not interpolated ones. Render the reference at
        // native scale (clamped so the render never exceeds 1.0) and let the
        // model perform the enlargement generatively. The output canvas and the
        // composite mapping stay at targetSize via `transform`.
        let referenceSize = referenceRenderSize(
            contextWidth: contextImage.width,
            contextHeight: contextImage.height,
            targetSize: targetSize,
            outputScale: transform.scale,
            alignment: settings.pixelAlignment
        )
        let referenceTransform = preparationTransform(
            sourceWidth: contextImage.width,
            sourceHeight: contextImage.height,
            targetWidth: referenceSize.width,
            targetHeight: referenceSize.height,
            method: settings.sizingMethod
        )
        let preparedFirstImage = try renderImage(
            contextImage,
            targetWidth: referenceSize.width,
            targetHeight: referenceSize.height,
            transform: referenceTransform
        )

        let preparedAdditionalImages = try referenceImages.dropFirst().map { image in
            try formatFullFrameReference(image, settings: settings)
        }
        // Fix 2: a full-frame edit (process area covers the whole original) has no
        // surrounding pixels to preserve, so there's nothing to composite back —
        // pasting the patch into the original would only down-sample the budget
        // canvas to the source resolution. Skip the plan so a full-frame
        // enlarge/rebuild outputs at the budget size. Partial / Live-Area edits
        // still paste back into the full-resolution original.
        let fullFrame = isFullFrame(processRect: processRect, original: original)
        let plan = ImageCompositionPlan(
            originalImage: original,
            contextRect: contextRect,
            processRect: processRect,
            transform: transform
        )

        return PreparedImageToImageInput(
            images: [preparedFirstImage] + preparedAdditionalImages,
            width: targetSize.width,
            height: targetSize.height,
            compositionPlan: (settings.compositeBack && !fullFrame) ? plan : nil
        )
    }

    /// Apply Image Formatting to a full-frame reference (Favour, Method, scale, megapixel budget).
    /// Used for additional conditioning images that do not use Live Area.
    public static func formatFullFrameReference(
        _ referenceImage: CGImage,
        settings: ImagePreparationSettings
    ) throws -> CGImage {
        var settings = settings
        settings.clampValues()
        settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)

        let targetSize = generationSize(referenceImage: referenceImage, settings: settings)
        return try formatToCanvas(
            referenceImage: referenceImage,
            settings: settings,
            targetWidth: targetSize.width,
            targetHeight: targetSize.height
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

    /// Size for the conditioning reference render. When the output canvas would
    /// up-sample the source (`outputScale > 1`), cap the reference at the source's
    /// native pixels (clamped per-dimension and floored to `alignment`) so the VAE
    /// encodes real pixels, never interpolation-smeared ones. When the source is
    /// already at/above budget, the reference matches the output size as before.
    private static func referenceRenderSize(
        contextWidth: Int,
        contextHeight: Int,
        targetSize: (width: Int, height: Int),
        outputScale: CGFloat,
        alignment: Int
    ) -> (width: Int, height: Int) {
        guard outputScale > 1 else { return targetSize }
        let rawWidth = min(Int(CGFloat(targetSize.width) / outputScale), contextWidth)
        let rawHeight = min(Int(CGFloat(targetSize.height) / outputScale), contextHeight)
        let flooredWidth = floorToMultiple(rawWidth, multiple: alignment)
        let flooredHeight = floorToMultiple(rawHeight, multiple: alignment)
        return (flooredWidth, flooredHeight)
    }

    /// Whether the resolved `processRect` covers the whole original — a full-frame
    /// edit with no surrounding pixels to preserve. Drives the Fix 2 composite skip.
    static func isFullFrame(processRect: CGRect, original: CGImage) -> Bool {
        processRect.minX <= 0
            && processRect.minY <= 0
            && processRect.width >= CGFloat(original.width)
            && processRect.height >= CGFloat(original.height)
    }

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

    /// Crop a reference image to a normalized top-left unit rectangle.
    public static func cropReferenceImage(_ image: CGImage, normalizedRect: CGRect) throws -> CGImage {
        let rect = integralPixelRect(from: normalizedRect, in: image)
        return try cropImage(image, to: rect)
    }

    /// Default Qwen context frame for generative fill: selection bounds plus padding.
    public static func autoVLMContextArea(
        maskLayers: [InpaintMaskLayer],
        processArea: CGRect?,
        draftPolygonPoints: [CGPoint] = []
    ) -> CGRect {
        paddedVLMContextArea(
            from: selectionBounds(
                maskLayers: maskLayers,
                processArea: processArea,
                draftPolygonPoints: draftPolygonPoints
            ),
            paddingFraction: 0.75
        )
    }

    /// Tightest effective Qwen context frame: selection bounds with minimum size only.
    public static func minimumVLMContextArea(
        maskLayers: [InpaintMaskLayer],
        processArea: CGRect?,
        draftPolygonPoints: [CGPoint] = []
    ) -> CGRect {
        paddedVLMContextArea(
            from: selectionBounds(
                maskLayers: maskLayers,
                processArea: processArea,
                draftPolygonPoints: draftPolygonPoints
            ),
            paddingFraction: 0
        )
    }

    /// Interpolate Qwen context between minimum, auto, and full-frame using a -1…1 slider (0 = auto).
    public static func fillVLMContextArea(
        maskLayers: [InpaintMaskLayer],
        processArea: CGRect?,
        draftPolygonPoints: [CGPoint] = [],
        scale: CGFloat
    ) -> CGRect {
        let auto = autoVLMContextArea(
            maskLayers: maskLayers,
            processArea: processArea,
            draftPolygonPoints: draftPolygonPoints
        )
        let clampedScale = min(max(scale, -1), 1)
        if abs(clampedScale) < 0.0001 {
            return auto
        }
        if clampedScale < 0 {
            let minimum = minimumVLMContextArea(
                maskLayers: maskLayers,
                processArea: processArea,
                draftPolygonPoints: draftPolygonPoints
            )
            return lerpUnitRect(from: auto, to: minimum, t: -clampedScale)
        }
        let fullFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
        return lerpUnitRect(from: auto, to: fullFrame, t: clampedScale)
    }

    private static func selectionBounds(
        maskLayers: [InpaintMaskLayer],
        processArea: CGRect?,
        draftPolygonPoints: [CGPoint]
    ) -> CGRect? {
        var bbox: CGRect?
        func union(_ rect: CGRect) {
            guard rect.width > 0, rect.height > 0 else { return }
            if let existing = bbox {
                bbox = existing.union(rect)
            } else {
                bbox = rect
            }
        }

        for layer in maskLayers {
            switch layer.primitive {
            case .rectangle(let rect):
                union(rect.cgRect)
            case .polygon(let points):
                guard let first = points.first else { continue }
                var minX = first.x
                var maxX = first.x
                var minY = first.y
                var maxY = first.y
                for point in points {
                    minX = min(minX, point.x)
                    maxX = max(maxX, point.x)
                    minY = min(minY, point.y)
                    maxY = max(maxY, point.y)
                }
                union(CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY))
            case .visionSubject(let selection):
                union(selection.boundingRect)
            }
        }

        if let processArea {
            union(processArea)
        }

        for point in draftPolygonPoints {
            union(CGRect(x: point.x - 0.005, y: point.y - 0.005, width: 0.01, height: 0.01))
        }

        return bbox
    }

    private static func paddedVLMContextArea(from bbox: CGRect?, paddingFraction: CGFloat) -> CGRect {
        guard var bounds = bbox else {
            return CGRect(x: 0, y: 0, width: 1, height: 1)
        }

        if paddingFraction > 0 {
            let padX = bounds.width * paddingFraction
            let padY = bounds.height * paddingFraction
            bounds = bounds.insetBy(dx: -padX, dy: -padY)
        }

        let minSize: CGFloat = 0.15
        if bounds.width < minSize {
            let extra = (minSize - bounds.width) / 2
            bounds.origin.x -= extra
            bounds.size.width = minSize
        }
        if bounds.height < minSize {
            let extra = (minSize - bounds.height) / 2
            bounds.origin.y -= extra
            bounds.size.height = minSize
        }

        return clampUnitRect(bounds)
    }

    private static func lerpUnitRect(from start: CGRect, to end: CGRect, t: CGFloat) -> CGRect {
        let clamped = min(max(t, 0), 1)
        return clampUnitRect(CGRect(
            x: start.minX + (end.minX - start.minX) * clamped,
            y: start.minY + (end.minY - start.minY) * clamped,
            width: start.width + (end.width - start.width) * clamped,
            height: start.height + (end.height - start.height) * clamped
        ))
    }
}
