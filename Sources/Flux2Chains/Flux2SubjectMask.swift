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
import CoreGraphics
import CoreImage
import Vision

/// Build inpainting masks from auto-segmented subjects.
public enum Flux2SubjectMask {

    public enum Error: Swift.Error, CustomStringConvertible, Sendable {
        case noSubjectDetected
        case maskRenderFailed
        case visionFailure(String)

        public var description: String {
            switch self {
            case .noSubjectDetected:
                return "Vision could not detect a foreground subject in the image."
            case .maskRenderFailed:
                return "Failed to render the segmentation mask to a CGImage."
            case .visionFailure(let msg):
                return "Vision request failed: \(msg)"
            }
        }
    }

    /// Generate a Flux2-ready grayscale mask suitable for the
    /// `.changeScene` inpaint workflow: BLACK over the subject (keep)
    /// and WHITE everywhere else (inpaint). Compatible with
    /// `Flux2MaskConvention.grayscaleWhiteInpaint`.
    ///
    /// Pipeline:
    /// 1. Run `VNGenerateForegroundInstanceMaskRequest` on `image` —
    ///    returns a per-instance segmentation. We OR-combine the
    ///    instances so every subject the model finds is kept.
    /// 2. The Vision mask is WHITE on the subject; we **invert** it to
    ///    Flux's BLACK-keep convention.
    /// 3. Apply a small Gaussian blur (`edgeSoftnessPixels`, default
    ///    4 px) on the inverted mask so the boundary isn't bit-hard —
    ///    keeps RePaint from leaving a visible seam, while staying
    ///    tight enough that the subject's anatomy doesn't get
    ///    repainted.
    ///
    /// - Parameters:
    ///   - image: The source image to segment.
    ///   - edgeSoftnessPixels: Width of the Gaussian transition on the
    ///     subject boundary, in pixels of the source. Default 4. Set to
    ///     0 for a bit-hard edge (visible seam likely).
    /// - Returns: A grayscale CGImage at the same dimensions as the
    ///   input, ready to pass to `Flux2MaskedInpaintingChain` with the
    ///   default `.grayscaleWhiteInpaint` mask convention.
    /// - Throws: ``Error/noSubjectDetected`` if Vision finds nothing,
    ///   ``Error/visionFailure`` if the request throws, or
    ///   ``Error/maskRenderFailed`` if we couldn't compose the result.
    @available(macOS 14.0, *)
    public static func makeChangeSceneMask(
        from image: CGImage,
        edgeSoftnessPixels: Int = 4
    ) throws -> CGImage {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])

        do {
            try handler.perform([request])
        } catch {
            throw Error.visionFailure(String(describing: error))
        }

        guard let observation = request.results?.first, !observation.allInstances.isEmpty else {
            throw Error.noSubjectDetected
        }

        // Combine all detected instances into a single mask (some
        // animals split into multiple instances).
        let pixelBuffer: CVPixelBuffer
        do {
            pixelBuffer = try observation.generateScaledMaskForImage(
                forInstances: observation.allInstances,
                from: handler
            )
        } catch {
            throw Error.visionFailure("Failed to scale subject mask: \(error)")
        }

        // The pixel buffer is single-channel float, white = subject.
        // Convert to grayscale UInt8 in Flux convention (black = keep =
        // subject ⇒ INVERT) at the same dimensions as the input.
        guard let mask = renderInvertedMask(
            from: pixelBuffer,
            targetWidth: image.width,
            targetHeight: image.height,
            edgeSoftnessPixels: edgeSoftnessPixels
        ) else {
            throw Error.maskRenderFailed
        }
        return mask
    }

    // MARK: - Internals

    /// Render a CVPixelBuffer (kCVPixelFormatType_OneComponent8 from
    /// Vision) into a Flux-convention grayscale CGImage: subject → 0
    /// (black, keep), background → 255 (white, inpaint), with an
    /// optional Gaussian-style softening on the boundary.
    private static func renderInvertedMask(
        from pixelBuffer: CVPixelBuffer,
        targetWidth: Int,
        targetHeight: Int,
        edgeSoftnessPixels: Int
    ) -> CGImage? {
        let pbW = CVPixelBufferGetWidth(pixelBuffer)
        let pbH = CVPixelBufferGetHeight(pixelBuffer)
        let pbFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let srcRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // 1) Read the Vision mask into a UInt8 grayscale buffer in
        //    Flux convention (subject = black, background = white). The
        //    format depends on the Vision API:
        //    - `generateMaskForInstances(...)` returns
        //      `kCVPixelFormatType_OneComponent8` (subject = 255).
        //    - `generateScaledMaskForImage(...)` returns
        //      `kCVPixelFormatType_OneComponent32Float` (subject = 1.0).
        //    Treating the float buffer as UInt8 reads 4-byte floats as
        //    4 separate bytes ⇒ random-looking mask; that's the bug we
        //    saw the first time.
        var inverted = [UInt8](repeating: 0, count: pbW * pbH)
        switch pbFormat {
        case kCVPixelFormatType_OneComponent32Float:
            for y in 0..<pbH {
                let row = srcBase.advanced(by: y * srcRowBytes)
                    .assumingMemoryBound(to: Float32.self)
                for x in 0..<pbW {
                    let v = max(0, min(1, row[x]))  // clamp [0, 1]
                    inverted[y * pbW + x] = UInt8((1.0 - v) * 255.0)
                }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<pbH {
                let row = srcBase.advanced(by: y * srcRowBytes)
                    .assumingMemoryBound(to: UInt8.self)
                for x in 0..<pbW {
                    inverted[y * pbW + x] = 255 &- row[x]
                }
            }
        default:
            return nil  // unknown format ⇒ caller bubbles maskRenderFailed
        }
        let cs = CGColorSpaceCreateDeviceGray()
        guard let srcCtx = CGContext(
            data: &inverted,
            width: pbW, height: pbH,
            bitsPerComponent: 8,
            bytesPerRow: pbW,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ), let srcImage = srcCtx.makeImage() else { return nil }

        // 3) Draw into the target-sized grayscale context. Core
        //    Graphics' bilinear sampling already softens the boundary
        //    a bit when the source is larger or smaller; we add an
        //    explicit small Gaussian-like blur next if requested.
        var dst = [UInt8](repeating: 255, count: targetWidth * targetHeight)
        guard let dstCtx = CGContext(
            data: &dst,
            width: targetWidth, height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        dstCtx.interpolationQuality = .high
        dstCtx.draw(srcImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        guard let hardImage = CGContext(
            data: &dst,
            width: targetWidth, height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )?.makeImage() else { return nil }

        guard edgeSoftnessPixels > 0 else { return hardImage }
        return applyGaussianBlur(hardImage, radius: edgeSoftnessPixels) ?? hardImage
    }

    /// Soften the mask edge with a small Gaussian via Core Image
    /// (CIGaussianBlur). Falls back to the hard mask if anything goes
    /// wrong — better a slight visible seam than a black-broken mask.
    private static func applyGaussianBlur(_ image: CGImage, radius: Int) -> CGImage? {
        let ci = CIImage(cgImage: image)
        let filter = CIFilter(name: "CIGaussianBlur")
        filter?.setValue(ci, forKey: kCIInputImageKey)
        filter?.setValue(Double(radius), forKey: kCIInputRadiusKey)
        guard let output = filter?.outputImage else { return nil }
        // Clamp to the original extent — CIGaussianBlur expands the
        // image by the blur radius on every side; we want the same
        // dimensions back.
        let originalExtent = ci.extent
        let context = CIContext(options: nil)
        return context.createCGImage(output, from: originalExtent)
    }
}
