// ImagePreparation.swift - Barn-door Live Area formatting, sizing, and paste-back composite for I2I
// Copyright 2025 Vincent Gourbin
//
// Originally contributed by Bill Evans (@realnotsteve) in PR #99; this file
// carries the Image Preparation core (formatting, Live Area, megapixel budget,
// composite). See docs/ImagePreparation.md.

import CoreGraphics
import Foundation

public enum ImagePreparation {
    public static let generationSizeMultiple = 32
    public static let referenceConditioningArea = 1024 * 1024

    public static func clampUnitRect(_ rect: CGRect) -> CGRect {
        // A NaN axis (a caller-supplied "nan" numeric string) survives
        // min/max unclamped — Swift's max(nan, x) returns nan when nan is the
        // *first* argument — and CGRect.intersection then silently discards
        // that axis instead of erroring. Replace with the full-frame default
        // before clamping rather than propagate NaN further.
        guard rect.minX.isFinite, rect.minY.isFinite, rect.width.isFinite, rect.height.isFinite else {
            return CGRect(x: 0, y: 0, width: 1, height: 1)
        }
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
    /// The single home for the floor-to-alignment step — the reference-render
    /// sizing below routes through here so it can't drift.
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
        // .rounded(), not truncation: a scale computation landing just under
        // an exact integer (e.g. 999.9999999997) would otherwise drop a whole
        // alignment step before flooring, undershooting the requested budget
        // by up to `multiple` pixels on either axis for no reason.
        let flooredWidth = floorToMultiple(Int(w.rounded()), multiple: multiple)
        let flooredHeight = floorToMultiple(Int(h.rounded()), multiple: multiple)
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

        let (contextRect, processRect) = resolvedRects(settings: settings, image: original)
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

        // Never upsample: same rationale as the primary reference (see
        // referenceRenderSize's doc on prepare()) — a secondary reference
        // smaller than the budget/aspect target was previously rendered
        // straight to the full target size, smearing an interpolated
        // reference into the VAE. Render at native scale instead when the
        // source is smaller, and let the model enlarge generatively.
        let outputScale = CGFloat(targetSize.width) / CGFloat(max(1, referenceImage.width))
        let renderSize = referenceRenderSize(
            contextWidth: referenceImage.width,
            contextHeight: referenceImage.height,
            targetSize: targetSize,
            outputScale: outputScale,
            alignment: settings.pixelAlignment
        )
        return try formatToCanvas(
            referenceImage: referenceImage,
            settings: settings,
            targetWidth: renderSize.width,
            targetHeight: renderSize.height
        )
    }

    /// The region of the generated canvas that actually lands back in the
    /// original image, or throws the same error `composite()` would.
    ///
    /// Depends only on `plan` and the canvas size the plan was built for
    /// (`plan.transform.targetWidth/targetHeight`) — not on any generated
    /// pixels — so it can validate a plan is composable *before* running
    /// generation, instead of discarding a completed (possibly multi-minute)
    /// generation if `--process-area`/`--live-area` turn out to map to an
    /// empty canvas rect.
    public static func validateComposable(_ plan: ImageCompositionPlan) throws {
        _ = try visibleCanvasRect(for: plan, canvasWidth: plan.transform.targetWidth, canvasHeight: plan.transform.targetHeight)
    }

    private static func visibleCanvasRect(for plan: ImageCompositionPlan, canvasWidth: Int, canvasHeight: Int) throws -> CGRect {
        let processInContext = plan.processRect.offsetBy(dx: -plan.contextRect.minX, dy: -plan.contextRect.minY)
        let mappedProcessRect = plan.transform.canvasRect(forSourceRect: processInContext)
        let canvasBounds = CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight)
        let visibleCanvasRect = mappedProcessRect.intersection(canvasBounds)

        guard !visibleCanvasRect.isNull, visibleCanvasRect.width > 0, visibleCanvasRect.height > 0 else {
            throw Flux2Error.imageProcessingFailed("Process area falls outside the generated canvas")
        }
        return visibleCanvasRect
    }

    public static func composite(
        _ generatedImage: CGImage,
        using plan: ImageCompositionPlan
    ) throws -> CGImage {
        // The plan's transform (scale/offset) was computed for a specific
        // canvas size during prepare(); compositing against an image of a
        // different size would silently map the patch using a coordinate
        // space it wasn't built for, rather than failing loudly.
        guard generatedImage.width == plan.transform.targetWidth, generatedImage.height == plan.transform.targetHeight else {
            throw Flux2Error.imageProcessingFailed(
                "Generated image (\(generatedImage.width)x\(generatedImage.height)) does not match the composition plan's canvas (\(plan.transform.targetWidth)x\(plan.transform.targetHeight))")
        }
        // Known ~1-2px edge tolerance: generatedCropRect is rounded outward in
        // canvas space, then mapped back through the inverse transform and
        // rounded outward again for destinationRect below. Two independent
        // outward-rounding steps across a scale factor can each add a
        // fractional pixel, so the pasted patch can overshoot the intended
        // Live Area by a pixel or two rather than landing pixel-exact.
        let visibleCanvasRect = try visibleCanvasRect(for: plan, canvasWidth: generatedImage.width, canvasHeight: generatedImage.height)
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

    /// Whether `settings` resolves to a full-frame edit on `image`: the process
    /// area covers the whole image. Resolves the process rect through the exact
    /// same path `prepare()` uses, so this and `prepare()`'s own full-frame check
    /// (which additionally skips composite-back for a full-frame edit — Fix 2)
    /// can never disagree about the rect-cover geometry.
    ///
    /// This is *not* the same question as "will `prepare()` return a composite
    /// plan": that also depends on `settings.compositeBack` (`--no-composite`),
    /// which this predicate does not consult. A caller deciding "will the result
    /// paste back into the original" needs
    /// `settings.compositeBack && !isFullFrame(settings:image:)`.
    public static func isFullFrame(settings: ImagePreparationSettings, image: CGImage) -> Bool {
        isFullFrame(processRect: resolvedRects(settings: settings, image: image).process, original: image)
    }

    /// The integral context + process rects for `settings` on `image` — the single
    /// derivation shared by `prepare()` and `isFullFrame(settings:image:)` so the
    /// composite-back decision and the public predicate can never disagree.
    private static func resolvedRects(
        settings: ImagePreparationSettings,
        image: CGImage
    ) -> (context: CGRect, process: CGRect) {
        var settings = settings
        settings.clampValues()
        let contextRect = integralPixelRect(from: settings.contextArea, in: image)
        let processRect = integralProcessRect(in: image, contextRect: contextRect, processArea: settings.processArea)
        return (contextRect, processRect)
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

        guard let context = makeImageContext(
            width: integralDimension(cropRect.width),
            height: integralDimension(cropRect.height)
        ) else {
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
                canvasHeight: CGFloat(integralDimension(cropRect.height))
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
            // `min(imageWidth, 1)` always evaluates to 1 for any real image —
            // fall back to the full image bounds, not a degenerate 1x1 rect.
            return CGRect(x: 0, y: 0, width: max(1, imageWidth), height: max(1, imageHeight))
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

    private static func integralDimension(_ value: CGFloat) -> Int {
        max(1, Int(value.rounded()))
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
}
