// Flux2Pipeline+ChainHelpers.swift — Public hooks for Flux2Chains
// Copyright 2025 Vincent Gourbin
//
// Minimal access surface for chain extensions living in the `Flux2Chains`
// library. Anything more invasive (custom denoising loops, etc.) belongs in
// `Flux2Pipeline.swift` proper.

import Foundation
import MLX
import CoreGraphics

extension Flux2Pipeline {
    /// Encode a `CGImage` to packed-sequence latents `[1, (h/16)*(w/16), 128]`,
    /// already normalised with the VAE BatchNorm statistics so the tensor is
    /// directly comparable with the denoiser's working latents.
    ///
    /// Used by chains that need access to the original-image latent (e.g.
    /// `Flux2MaskedInpaintingChain` for RePaint-style per-step blending).
    /// Loads the VAE on demand and keeps it resident — the subsequent
    /// `generateWithResult` call will reuse it.
    ///
    /// - Parameters:
    ///   - image: Source image.
    ///   - targetHeight: Target output height in pixels (must be a multiple of 32).
    ///   - targetWidth: Target output width in pixels (must be a multiple of 32).
    ///   - samplePosterior: When `true`, sample from the VAE posterior; when
    ///     `false`, use the deterministic mean (default — matches the existing
    ///     I2I path).
    /// - Returns: Packed-sequence latents ready to be blended with the
    ///   denoising state inside a `Flux2StepHook`.
    public func encodeImageToPackedSequence(
        _ image: CGImage,
        targetHeight: Int,
        targetWidth: Int,
        samplePosterior: Bool = false
    ) async throws -> MLXArray {
        try await ensureVAELoaded()
        guard let vae = vaeForChains else {
            throw Flux2Error.modelNotLoaded("VAE")
        }

        let preprocessed = preprocessImageForVAEPublic(
            image,
            targetHeight: targetHeight,
            targetWidth: targetWidth
        )
        let rawLatents = vae.encode(preprocessed, samplePosterior: samplePosterior)
        var patchified = LatentUtils.packLatentsToPatchified(rawLatents)
        patchified = LatentUtils.normalizeLatentsWithBatchNorm(
            patchified,
            runningMean: vae.batchNormRunningMean,
            runningVar: vae.batchNormRunningVar
        )
        let packed = LatentUtils.packPatchifiedToSequence(patchified)
        eval(packed)
        return packed
    }

    /// Clamp `(width, height)` to the nearest lower multiple of 32 within
    /// `maxPixels` pixels — the layout the FLUX.2 transformer expects.
    ///
    /// - Parameters:
    ///   - width: Requested output width.
    ///   - height: Requested output height.
    ///   - maxPixels: Maximum total pixel count (default 1024² ≈ 1 048 576).
    public static func resolveChainDimensions(
        width: Int,
        height: Int,
        maxPixels: Int = 1024 * 1024
    ) -> (height: Int, width: Int) {
        let multipleOf = 32
        var w = width
        var h = height
        let pixelCount = w * h
        if pixelCount > maxPixels {
            let scale = (Double(maxPixels) / Double(pixelCount)).squareRoot()
            w = Int(Double(w) * scale)
            h = Int(Double(h) * scale)
        }
        w = max(multipleOf, (w / multipleOf) * multipleOf)
        h = max(multipleOf, (h / multipleOf) * multipleOf)
        return (height: h, width: w)
    }

    /// Rasterise a grayscale mask `CGImage` into a packed-sequence-aligned
    /// `[1, seq, 1]` MLX array suitable for broadcasting against `[1, seq, 128]`
    /// latents.
    ///
    /// `white (1.0)` means **inpaint** (the model is allowed to write to this
    /// area), `black (0.0)` means **keep** (the original-image latent is forced
    /// back at every step). Soft values in `[0, 1]` are preserved — callers
    /// that want hard mask edges should pre-binarise.
    public static func packMaskForLatentBlending(
        _ mask: CGImage,
        targetHeight: Int,
        targetWidth: Int
    ) -> MLXArray {
        let latentH = targetHeight / 16
        let latentW = targetWidth / 16
        let bytesPerRow = latentW
        var pixels = [UInt8](repeating: 0, count: latentH * latentW)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &pixels,
            width: latentW,
            height: latentH,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return MLXArray.zeros([1, latentH * latentW, 1])
        }
        context.interpolationQuality = .high
        context.draw(mask, in: CGRect(x: 0, y: 0, width: latentW, height: latentH))

        let floats = pixels.map { Float($0) / 255.0 }
        return MLXArray(floats).reshaped([1, latentH * latentW, 1])
    }
}
