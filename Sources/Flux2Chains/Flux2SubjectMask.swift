// Flux2SubjectMask.swift — Auto-segmentation mask helper for .changeScene workflows
// Copyright 2025 Vincent Gourbin
//
// Why this exists:
// The `.changeScene` inpaint workflow ("keep subject, change scene around")
// is *extremely* sensitive to mask quality. A loose hand-drawn blob lets
// the subject's anatomy (paw tips, tail, ears) extend into the soft mask
// ramp, where RePaint partially regenerates them — FLUX then invents a
// second-anatomy chimera to rationalise the half-cat shape it sees. No
// amount of prompt engineering fixes that.
//
// What does fix it: a pixel-accurate subject silhouette, slightly
// dilated, with a short edge ramp. Vision's
// `VNGenerateForegroundInstanceMaskRequest` (macOS 14+) produces exactly
// that for free — no extra weights, no extra dependency, instant on
// Apple Silicon.

import Foundation
import Accelerate
import CoreGraphics
import CoreImage
import Vision

/// Build inpainting masks from auto-segmented subjects.
public enum Flux2SubjectMask {

    public enum Error: Swift.Error, CustomStringConvertible, Sendable {
        case noSubjectDetected
        case maskRenderFailed
        case visionFailure(String)
        case unsupportedPixelFormat(UInt32)

        public var description: String {
            switch self {
            case .noSubjectDetected:
                return "Vision could not detect a foreground subject in the image."
            case .maskRenderFailed:
                return "Failed to render the segmentation mask to a CGImage."
            case .visionFailure(let msg):
                return "Vision request failed: \(msg)"
            case .unsupportedPixelFormat(let fmt):
                return "Unsupported CVPixelBuffer format from Vision: 0x\(String(format: "%08X", fmt)). Vision may have changed its mask output format; please file a bug."
            }
        }
    }

    // MARK: - Shared resources
    //
    // CIContexts compile Metal pipelines on first use (~50–200 ms one-time
    // cost). Caching one process-wide amortises that across calls, which
    // matters for hosts that invoke `makeChangeSceneMask` repeatedly
    // (e.g. FluxForge Studio live preview). `cacheIntermediates: false`
    // prevents the context from holding onto multi-MB intermediate
    // CIImages between calls — relevant when input masks are 20+ MP.
    // CIContext is thread-safe per Apple docs, so static let is fine.
    private static let sharedCIContext: CIContext = {
        CIContext(options: [.cacheIntermediates: false])
    }()

    // Note on Vision request caching: `VNGenerateForegroundInstanceMaskRequest`
    // stores its results as mutable state on the request instance. Caching
    // one statically would let concurrent calls clobber each other's
    // `results`. The request itself is cheap to instantiate (no Metal
    // compilation), so we create it per call and stay concurrency-safe.

    /// Generate a Flux2-ready grayscale mask suitable for the
    /// `.changeScene` inpaint workflow: BLACK over the subject (keep)
    /// and WHITE everywhere else (inpaint). Compatible with
    /// `Flux2MaskConvention.grayscaleWhiteInpaint`.
    ///
    /// Pipeline:
    /// 1. (Optional) downscale the source to `targetMaxPixels` so all
    ///    subsequent work happens at the smaller resolution. Use this
    ///    when the caller knows the chain will downscale anyway — saves
    ///    a 24× factor on a 24-MP iPhone shot vs the chain's default
    ///    1 MP working size.
    /// 2. Run `VNGenerateForegroundInstanceMaskRequest` on the working
    ///    image. We OR-combine all detected instances so multi-part
    ///    subjects (e.g. an animal split into head + body) all get kept.
    /// 3. Vision's float mask (1.0 = subject) is inverted into Flux
    ///    convention (0 = subject = keep), vectorised via Accelerate.
    /// 4. Optional `CIGaussianBlur` for a soft edge — keeps RePaint from
    ///    leaving a visible seam without losing anatomy.
    ///
    /// - Parameters:
    ///   - image: The source image to segment.
    ///   - edgeSoftnessPixels: Width of the Gaussian transition on the
    ///     subject boundary, in pixels of the *working* image (after
    ///     downscaling). Default 4. Set to 0 for a bit-hard edge.
    ///   - targetMaxPixels: When set, the source is downscaled with
    ///     Lanczos so the total pixel count is at most this many before
    ///     Vision runs. The returned mask is at the downscaled size, so
    ///     callers that pair the mask back with the unscaled source
    ///     must resize themselves. `nil` (default) keeps the source as
    ///     is — the right choice for ad-hoc inspection (CLI
    ///     `mask-subject`). Pass `1024 * 1024` from hosts that will
    ///     route the mask straight to `Flux2MaskedInpaintingChain`.
    /// - Returns: A grayscale CGImage at the *working* dimensions
    ///   (source dims when `targetMaxPixels` is nil, else clamped).
    /// - Throws: ``Error/noSubjectDetected`` if Vision finds nothing,
    ///   ``Error/visionFailure`` if the request throws,
    ///   ``Error/unsupportedPixelFormat`` if Vision returns a format we
    ///   don't know how to decode, or ``Error/maskRenderFailed`` if we
    ///   couldn't compose the final CGImage.
    @available(macOS 14.0, *)
    public static func makeChangeSceneMask(
        from image: CGImage,
        edgeSoftnessPixels: Int = 4,
        targetMaxPixels: Int? = nil
    ) throws -> CGImage {
        // Step 1: optional downscale. Doing it BEFORE Vision means the
        // model runs on fewer pixels AND the returned mask is small
        // enough to skip the identity-scale draw in renderInvertedMask.
        let workingImage = downscaledIfNeeded(image, targetMaxPixels: targetMaxPixels) ?? image

        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: workingImage, options: [:])

        do {
            try handler.perform([request])
        } catch {
            throw Error.visionFailure(String(describing: error))
        }

        guard let observation = request.results?.first, !observation.allInstances.isEmpty else {
            throw Error.noSubjectDetected
        }

        let pixelBuffer: CVPixelBuffer
        do {
            pixelBuffer = try observation.generateScaledMaskForImage(
                forInstances: observation.allInstances,
                from: handler
            )
        } catch {
            throw Error.visionFailure("Failed to scale subject mask: \(error)")
        }

        return try renderInvertedMask(
            from: pixelBuffer,
            targetWidth: workingImage.width,
            targetHeight: workingImage.height,
            edgeSoftnessPixels: edgeSoftnessPixels
        )
    }

    // MARK: - Internals

    /// Render a CVPixelBuffer (Vision's float mask) into a
    /// Flux-convention grayscale CGImage. The hot path (Vision returns
    /// a buffer already at `targetWidth×targetHeight`, which it does
    /// for `generateScaledMaskForImage(from:)`) avoids any
    /// CoreGraphics draw — we go straight from inverted bytes to
    /// CGImage. The fallback path (sizes differ, e.g. unexpected
    /// EXIF orientation) does one rescaling draw.
    @available(macOS 14.0, *)
    private static func renderInvertedMask(
        from pixelBuffer: CVPixelBuffer,
        targetWidth: Int,
        targetHeight: Int,
        edgeSoftnessPixels: Int
    ) throws -> CGImage {
        let pbW = CVPixelBufferGetWidth(pixelBuffer)
        let pbH = CVPixelBufferGetHeight(pixelBuffer)
        let pbFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw Error.maskRenderFailed
        }
        let srcRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let count = pbW * pbH

        // 1) Decode the Vision mask into a Flux-convention UInt8 buffer
        //    (subject=0 black=keep, background=255 white=inpaint).
        //    Vectorised via Accelerate for the float path, which is what
        //    `generateScaledMaskForImage` actually returns today.
        var inverted = [UInt8](repeating: 0, count: count)
        switch pbFormat {
        case kCVPixelFormatType_OneComponent32Float:
            // Read float mask values into a contiguous buffer, then run
            // three vDSP passes: clamp to [0,1], compute (255 - 255*v),
            // and saturating-cast to UInt8.
            var floats = [Float](repeating: 0, count: count)
            if srcRowBytes == pbW * MemoryLayout<Float>.size {
                // Tightly packed: one memcpy.
                floats.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!, srcBase, count * MemoryLayout<Float>.size)
                }
            } else {
                // Row padding: copy row-by-row.
                floats.withUnsafeMutableBufferPointer { dst in
                    for y in 0..<pbH {
                        let row = srcBase.advanced(by: y * srcRowBytes)
                        memcpy(
                            dst.baseAddress!.advanced(by: y * pbW),
                            row,
                            pbW * MemoryLayout<Float>.size
                        )
                    }
                }
            }
            var lo: Float = 0
            var hi: Float = 1
            floats.withUnsafeMutableBufferPointer { fb in
                vDSP_vclip(fb.baseAddress!, 1, &lo, &hi, fb.baseAddress!, 1, vDSP_Length(count))
            }
            // (1 - v) * 255  =  255 + (-255) * v  → vDSP_vsmsa
            var negScale: Float = -255
            var bias: Float = 255
            floats.withUnsafeMutableBufferPointer { fb in
                vDSP_vsmsa(fb.baseAddress!, 1, &negScale, &bias, fb.baseAddress!, 1, vDSP_Length(count))
            }
            inverted.withUnsafeMutableBufferPointer { dst in
                floats.withUnsafeBufferPointer { src in
                    vDSP_vfixu8(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(count))
                }
            }

        case kCVPixelFormatType_OneComponent8:
            // Defensive path: kept in case Vision ever switches back.
            // Scalar loop is fine for byte inversion; vImage would be
            // marginal here.
            for y in 0..<pbH {
                let row = srcBase.advanced(by: y * srcRowBytes)
                    .assumingMemoryBound(to: UInt8.self)
                for x in 0..<pbW {
                    inverted[y * pbW + x] = 255 &- row[x]
                }
            }

        default:
            throw Error.unsupportedPixelFormat(pbFormat)
        }

        // 2) Build the final CGImage. Hot path: the Vision mask is
        //    already at the target dimensions (Vision returns at the
        //    handler's image size), so we skip the intermediate
        //    CGContext draw and create the CGImage directly from
        //    `inverted`. The fallback handles the rare case where
        //    Vision returns at a different size (e.g. EXIF orientation
        //    surprises) by scaling via a CGContext draw.
        let cs = CGColorSpaceCreateDeviceGray()
        let hardImage: CGImage
        if pbW == targetWidth && pbH == targetHeight {
            guard let ctx = CGContext(
                data: &inverted,
                width: pbW, height: pbH,
                bitsPerComponent: 8,
                bytesPerRow: pbW,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ), let image = ctx.makeImage() else {
                throw Error.maskRenderFailed
            }
            hardImage = image
        } else {
            guard let srcCtx = CGContext(
                data: &inverted,
                width: pbW, height: pbH,
                bitsPerComponent: 8,
                bytesPerRow: pbW,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ), let srcImage = srcCtx.makeImage() else {
                throw Error.maskRenderFailed
            }
            var dst = [UInt8](repeating: 255, count: targetWidth * targetHeight)
            guard let dstCtx = CGContext(
                data: &dst,
                width: targetWidth, height: targetHeight,
                bitsPerComponent: 8,
                bytesPerRow: targetWidth,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else {
                throw Error.maskRenderFailed
            }
            dstCtx.interpolationQuality = .high
            dstCtx.draw(srcImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
            guard let image = dstCtx.makeImage() else {
                throw Error.maskRenderFailed
            }
            hardImage = image
        }

        if edgeSoftnessPixels > 0,
           let blurred = applyGaussianBlur(hardImage, radius: edgeSoftnessPixels) {
            return blurred
        }
        return hardImage
    }

    /// Soften the mask edge with a small Gaussian via the shared
    /// Core Image context. Falls back to the hard mask if anything
    /// goes wrong — better a slight visible seam than a broken mask.
    private static func applyGaussianBlur(_ image: CGImage, radius: Int) -> CGImage? {
        let ci = CIImage(cgImage: image)
        guard let filter = CIFilter(name: "CIGaussianBlur") else { return nil }
        filter.setValue(ci, forKey: kCIInputImageKey)
        filter.setValue(Double(radius), forKey: kCIInputRadiusKey)
        guard let output = filter.outputImage else { return nil }
        // CIGaussianBlur expands the extent by the blur radius on every
        // side; clamp back to the original extent so the mask keeps the
        // input's dimensions.
        return sharedCIContext.createCGImage(output, from: ci.extent)
    }

    /// Downscale `image` so its total pixel count is ≤ `targetMaxPixels`,
    /// preserving aspect ratio. Returns nil when no downscaling is
    /// needed (input already fits, or `targetMaxPixels` is nil). Uses
    /// `CILanczosScaleTransform` for high-quality, deterministic
    /// resampling.
    ///
    /// `internal` so unit tests can verify the no-op vs scaling
    /// boundaries without needing a real subject image (Vision-free
    /// path).
    internal static func downscaledIfNeeded(_ image: CGImage, targetMaxPixels: Int?) -> CGImage? {
        guard let target = targetMaxPixels else { return nil }
        let srcPixels = image.width * image.height
        guard srcPixels > target else { return nil }
        let scale = (Double(target) / Double(srcPixels)).squareRoot()
        let ci = CIImage(cgImage: image)
        guard let filter = CIFilter(name: "CILanczosScaleTransform") else { return nil }
        filter.setValue(ci, forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(1.0, forKey: kCIInputAspectRatioKey)
        guard let output = filter.outputImage else { return nil }
        return sharedCIContext.createCGImage(output, from: output.extent)
    }
}
