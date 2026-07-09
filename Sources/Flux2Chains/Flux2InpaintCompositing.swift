// Flux2InpaintCompositing.swift — crop-and-stitch + pixel compositing helpers
// Copyright 2025 Vincent Gourbin
//
// Host-side equivalents of diffusers' `padding_mask_crop` (get_crop_region)
// and `apply_overlay`: find the masked region, crop around it so the full
// token budget goes to the edit, then paste the generated crop back onto the
// untouched original in pixel space using the soft mask as per-pixel alpha.
//
// All rects use IMAGE space with a top-left origin — the convention of
// `CGImage.cropping(to:)`.

import Foundation
import CoreGraphics
import Flux2Core

enum Flux2InpaintCompositing {

    /// Maximum scan resolution (longest side) used to locate the mask's
    /// bounding box. The bbox is mapped back conservatively (rounded outward)
    /// so sub-scan-pixel precision loss is absorbed by the crop padding.
    private static let bboxScanMaxSide = 512

    // MARK: - Mask bounding box

    /// Bounding box of the inpaint region (`weight > threshold`) in image
    /// space (top-left origin), or `nil` when the mask is empty.
    static func maskBoundingBox(
        _ mask: CGImage,
        convention: Flux2MaskConvention,
        imageWidth: Int,
        imageHeight: Int,
        threshold: Float = 8.0 / 255.0
    ) -> CGRect? {
        let scale = min(1.0, Double(bboxScanMaxSide) / Double(max(mask.width, mask.height)))
        let scanW = max(1, Int(Double(mask.width) * scale))
        let scanH = max(1, Int(Double(mask.height) * scale))

        guard let weights = renderInpaintWeights(mask, width: scanW, height: scanH, convention: convention) else {
            return nil
        }

        var minX = scanW, minY = scanH, maxX = -1, maxY = -1
        for y in 0..<scanH {
            for x in 0..<scanW {
                if weights[y * scanW + x] > threshold {
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }
        guard maxX >= 0 else { return nil }

        // Map scan pixels back to image space, rounding outward by one scan
        // pixel so the true mask edge is always inside the bbox.
        let sx = Double(imageWidth) / Double(scanW)
        let sy = Double(imageHeight) / Double(scanH)
        let x0 = max(0, Int(Double(minX - 1) * sx))
        let y0 = max(0, Int(Double(minY - 1) * sy))
        let x1 = min(imageWidth, Int(Double(maxX + 2) * sx))
        let y1 = min(imageHeight, Int(Double(maxY + 2) * sy))
        guard x1 > x0, y1 > y0 else { return nil }
        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    // MARK: - Crop region (diffusers get_crop_region)

    /// Expand `bbox` by `padding` pixels on every side, then grow it to match
    /// the image's aspect ratio (so the crop maps cleanly onto the working
    /// resolution), shifting/clamping to stay inside the image.
    static func expandCropRegion(
        bbox: CGRect,
        padding: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> CGRect {
        var x0 = Int(bbox.minX) - padding
        var y0 = Int(bbox.minY) - padding
        var x1 = Int(bbox.maxX) + padding
        var y1 = Int(bbox.maxY) + padding

        // Grow the short dimension to the image's aspect ratio, centred.
        let targetAspect = Double(imageWidth) / Double(imageHeight)
        let w = x1 - x0
        let h = y1 - y0
        let cropAspect = Double(w) / Double(h)
        if cropAspect > targetAspect {
            let desiredH = Int((Double(w) / targetAspect).rounded(.up))
            let grow = desiredH - h
            y0 -= grow / 2
            y1 += grow - grow / 2
        } else {
            let desiredW = Int((Double(h) * targetAspect).rounded(.up))
            let grow = desiredW - w
            x0 -= grow / 2
            x1 += grow - grow / 2
        }

        // Shift back inside the image, then clamp (crop caps at image size).
        if x0 < 0 { x1 -= x0; x0 = 0 }
        if y0 < 0 { y1 -= y0; y0 = 0 }
        if x1 > imageWidth { x0 -= (x1 - imageWidth); x1 = imageWidth; x0 = max(0, x0) }
        if y1 > imageHeight { y0 -= (y1 - imageHeight); y1 = imageHeight; y0 = max(0, y0) }

        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    // MARK: - Pixel composite (diffusers apply_overlay)

    /// Paste `generated` (any resolution — it is resized to `cropRect`'s size)
    /// onto `original` at `cropRect`, using `maskCrop` (the mask restricted to
    /// the same region) as per-pixel blend weight:
    /// `out = original·(1-m) + generated·m`. Soft mask values give a seamless
    /// transition; pixels with `m == 0` are bit-identical to the original.
    ///
    /// Assumes an **opaque** original (photos). Sources with alpha < 255 are
    /// flattened: the buffer is premultiplied RGBA and the output forces
    /// alpha = 255, so semi-transparent regions come out darkened rather than
    /// bit-exact.
    static func composite(
        original: CGImage,
        generated: CGImage,
        cropRect: CGRect,
        maskCrop: CGImage,
        convention: Flux2MaskConvention
    ) -> CGImage? {
        let fullW = original.width
        let fullH = original.height
        let cropX = Int(cropRect.minX)
        let cropY = Int(cropRect.minY)
        let cropW = Int(cropRect.width)
        let cropH = Int(cropRect.height)
        guard cropW > 0, cropH > 0,
              cropX >= 0, cropY >= 0,
              cropX + cropW <= fullW, cropY + cropH <= fullH else { return nil }

        guard var base = renderRGBA(original, width: fullW, height: fullH),
              let gen = renderRGBA(generated, width: cropW, height: cropH),
              let weights = renderInpaintWeights(maskCrop, width: cropW, height: cropH, convention: convention)
        else { return nil }

        for y in 0..<cropH {
            let baseRow = (cropY + y) * fullW
            let genRow = y * cropW
            for x in 0..<cropW {
                let m = weights[genRow + x]
                if m <= 0 { continue }
                let b = (baseRow + cropX + x) * 4
                let g = (genRow + x) * 4
                if m >= 1 {
                    base[b] = gen[g]
                    base[b + 1] = gen[g + 1]
                    base[b + 2] = gen[g + 2]
                } else {
                    let inv = 1 - m
                    base[b] = UInt8(max(0, min(255, Float(base[b]) * inv + Float(gen[g]) * m)))
                    base[b + 1] = UInt8(max(0, min(255, Float(base[b + 1]) * inv + Float(gen[g + 1]) * m)))
                    base[b + 2] = UInt8(max(0, min(255, Float(base[b + 2]) * inv + Float(gen[g + 2]) * m)))
                }
            }
        }

        return makeImage(fromRGBA: &base, width: fullW, height: fullH)
    }

    // MARK: - Rasterisation primitives

    /// Render any CGImage into a top-left-indexed RGBA8 buffer at the given size.
    private static func renderRGBA(_ image: CGImage, width: Int, height: Int) -> [UInt8]? {
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return ok ? buf : nil
    }

    /// Render a mask into inpaint weights (`1.0` = repaint, `0.0` = keep) at
    /// the given size, honouring the convention. Mirrors
    /// `Flux2Pipeline.packMaskForLatentBlending`'s reading rules, but at an
    /// arbitrary resolution for pixel-space use.
    static func renderInpaintWeights(
        _ mask: CGImage,
        width: Int,
        height: Int,
        convention: Flux2MaskConvention
    ) -> [Float]? {
        switch convention {
        case .grayscaleWhiteInpaint:
            var buf = [UInt8](repeating: 0, count: width * height)
            let ok = buf.withUnsafeMutableBytes { raw -> Bool in
                guard let context = CGContext(
                    data: raw.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue
                ) else { return false }
                context.interpolationQuality = .high
                context.draw(mask, in: CGRect(x: 0, y: 0, width: width, height: height))
                return true
            }
            guard ok else { return nil }
            return buf.map { Float($0) / 255.0 }
        case .alphaTransparentInpaint:
            guard let rgba = renderRGBA(mask, width: width, height: height) else { return nil }
            return (0..<(width * height)).map { 1.0 - Float(rgba[$0 * 4 + 3]) / 255.0 }
        }
    }

    private static func makeImage(fromRGBA buf: inout [UInt8], width: Int, height: Int) -> CGImage? {
        // Force full opacity so downstream consumers see an opaque photo.
        for i in 0..<(width * height) {
            buf[i * 4 + 3] = 255
        }
        return buf.withUnsafeMutableBytes { raw -> CGImage? in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return nil }
            return context.makeImage()
        }
    }
}
