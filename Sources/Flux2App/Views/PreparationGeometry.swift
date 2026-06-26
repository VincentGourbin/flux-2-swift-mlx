/**
 * PreparationGeometry.swift
 * Pure coordinate math for the Image Preparation preview: mapping between the
 * reference image's normalized [0,1] space and the on-screen `imageRect`, plus
 * aspect-fit, pixel snapping, and outpaint-canvas layout. Extracted from
 * `ImagePreparationPreview` so the transforms are a single source of truth and
 * unit-testable (this is where selection/overlay coordinate bugs hide).
 *
 * Everything here is a pure function of the image's pixel dimensions and the
 * caller-supplied rects — no view, view-model, or SwiftUI state.
 */

import CoreGraphics
import Flux2Core

struct PreparationGeometry: Equatable {
    let imageWidth: Int
    let imageHeight: Int
    /// Display backing scale used for pixel-snapping overlays. Defaults to 2
    /// (Retina) so the type is usable without a screen (e.g. in tests).
    var backingScale: CGFloat = 2

    // MARK: - Aspect fit

    /// Letterbox `content` (given by its pixel size) centered inside `size`.
    private static func aspectFit(contentWidth: CGFloat, contentHeight: CGFloat, in size: CGSize) -> CGRect {
        guard contentWidth > 0, contentHeight > 0, size.width > 0, size.height > 0 else {
            return .zero
        }

        let contentAspect = contentWidth / contentHeight
        let viewAspect = size.width / size.height
        let width: CGFloat
        let height: CGFloat

        if contentAspect > viewAspect {
            width = size.width
            height = width / contentAspect
        } else {
            height = size.height
            width = height * contentAspect
        }

        return CGRect(
            x: (size.width - width) / 2,
            y: (size.height - height) / 2,
            width: width,
            height: height
        )
    }

    /// The reference image fitted (letterboxed) inside `size`.
    func fittedImageRect(in size: CGSize) -> CGRect {
        Self.aspectFit(contentWidth: CGFloat(imageWidth), contentHeight: CGFloat(imageHeight), in: size)
    }

    /// The expanded outpaint canvas (image + padding) fitted inside `size`.
    func fittedCanvasRect(in size: CGSize, padding: OutpaintPadding) -> CGRect {
        Self.aspectFit(
            contentWidth: CGFloat(imageWidth + padding.left + padding.right),
            contentHeight: CGFloat(imageHeight + padding.top + padding.bottom),
            in: size
        )
    }

    /// Where the original image sits inside a fitted outpaint `canvasRect`.
    func imageRectInsideCanvas(canvasRect: CGRect, padding: OutpaintPadding) -> CGRect {
        let canvasWidth = CGFloat(imageWidth + padding.left + padding.right)
        let canvasHeight = CGFloat(imageHeight + padding.top + padding.bottom)
        guard canvasWidth > 0, canvasHeight > 0 else { return canvasRect }

        let scale = canvasRect.width / canvasWidth
        return CGRect(
            x: canvasRect.minX + CGFloat(padding.left) * scale,
            y: canvasRect.minY + CGFloat(padding.top) * scale,
            width: CGFloat(imageWidth) * scale,
            height: CGFloat(imageHeight) * scale
        )
    }

    /// View points per image pixel for the fitted `imageRect`.
    func imagePixelScale(imageRect: CGRect) -> CGFloat {
        guard imageWidth > 0 else { return 1 }
        return imageRect.width / CGFloat(imageWidth)
    }

    // MARK: - Normalized <-> view

    /// A normalized rect projected into `imageRect`, snapped to whole device pixels.
    func displayRect(for normalized: CGRect, in imageRect: CGRect) -> CGRect {
        pixelAligned(
            CGRect(
                x: imageRect.minX + normalized.minX * imageRect.width,
                y: imageRect.minY + normalized.minY * imageRect.height,
                width: normalized.width * imageRect.width,
                height: normalized.height * imageRect.height
            )
        )
    }

    /// A normalized point projected into `imageRect`.
    func displayPoint(_ normalized: CGPoint, in imageRect: CGRect) -> CGPoint {
        CGPoint(
            x: imageRect.minX + normalized.x * imageRect.width,
            y: imageRect.minY + normalized.y * imageRect.height
        )
    }

    /// A view-space location mapped back to normalized [0,1] within `imageRect` (clamped).
    func normalizedPoint(for location: CGPoint, in imageRect: CGRect) -> CGPoint {
        CGPoint(
            x: min(max((location.x - imageRect.minX) / imageRect.width, 0), 1),
            y: min(max((location.y - imageRect.minY) / imageRect.height, 0), 1)
        )
    }

    /// Floor/ceil to whole device pixels so stacked overlays don't leave 1px gaps.
    func pixelAligned(_ rect: CGRect) -> CGRect {
        let scale = backingScale
        let minX = (rect.minX * scale).rounded(.down) / scale
        let minY = (rect.minY * scale).rounded(.down) / scale
        let maxX = (rect.maxX * scale).rounded(.up) / scale
        let maxY = (rect.maxY * scale).rounded(.up) / scale
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    // MARK: - Snapping / limits (normalized space, 16px grid)

    /// Smallest normalized barn-door step: 16 image pixels.
    var minimumContextWidth: CGFloat { CGFloat(16) / CGFloat(max(imageWidth, 16)) }
    var minimumContextHeight: CGFloat { CGFloat(16) / CGFloat(max(imageHeight, 16)) }

    func snapX(_ x: CGFloat) -> CGFloat { snap(x, pixels: imageWidth) }
    func snapY(_ y: CGFloat) -> CGFloat { snap(y, pixels: imageHeight) }

    func snap(_ value: CGFloat, pixels: Int) -> CGFloat {
        let step = CGFloat(16) / CGFloat(max(pixels, 16))
        return min(max((value / step).rounded() * step, 0), 1)
    }
}
