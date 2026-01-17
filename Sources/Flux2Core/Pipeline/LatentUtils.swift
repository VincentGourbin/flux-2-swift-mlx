// LatentUtils.swift - Latent space utilities for Flux.2
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXRandom

/// Utilities for working with Flux.2 latent space
public enum LatentUtils {

    // MARK: - Latent Generation

    /// Generate random initial noise for text-to-image in PATCHIFIED format
    /// This generates latents directly in the format expected for BatchNorm normalization
    /// - Parameters:
    ///   - batchSize: Number of images to generate
    ///   - height: Image height
    ///   - width: Image width
    ///   - latentChannels: Number of latent channels (32 for Flux.2)
    ///   - patchSize: Patch size (2 for Flux.2)
    ///   - seed: Optional random seed
    /// - Returns: Random patchified latent noise [B, 128, H/16, W/16] for Flux.2
    public static func generatePatchifiedLatents(
        batchSize: Int = 1,
        height: Int,
        width: Int,
        latentChannels: Int = 32,
        patchSize: Int = 2,
        seed: UInt64? = nil
    ) -> MLXArray {
        // Set seed if provided
        if let seed = seed {
            MLXRandom.seed(seed)
        }

        // Patchified dimensions: channels = 32 * 2 * 2 = 128, spatial = original / 16
        let patchifiedChannels = latentChannels * patchSize * patchSize  // 128
        let patchifiedHeight = height / (8 * patchSize)   // H/16
        let patchifiedWidth = width / (8 * patchSize)     // W/16

        return MLXRandom.normal([batchSize, patchifiedChannels, patchifiedHeight, patchifiedWidth])
    }

    /// Generate random initial noise for text-to-image (legacy, non-patchified)
    /// - Parameters:
    ///   - batchSize: Number of images to generate
    ///   - height: Image height (will be divided by 8 for latent size)
    ///   - width: Image width (will be divided by 8 for latent size)
    ///   - latentChannels: Number of latent channels (32 for Flux.2)
    ///   - seed: Optional random seed
    /// - Returns: Random latent noise [B, C, H/8, W/8]
    public static func generateInitialLatents(
        batchSize: Int = 1,
        height: Int,
        width: Int,
        latentChannels: Int = 32,
        seed: UInt64? = nil
    ) -> MLXArray {
        // Set seed if provided
        if let seed = seed {
            MLXRandom.seed(seed)
        }

        let latentHeight = height / 8
        let latentWidth = width / 8

        return MLXRandom.normal([batchSize, latentChannels, latentHeight, latentWidth])
    }

    // MARK: - Packing/Unpacking for Patchified Format

    /// Pack patchified latents to sequence format for transformer
    /// Converts [B, 128, H/16, W/16] to [B, H/16*W/16, 128]
    /// - Parameter patchified: Patchified latents [B, 128, H/16, W/16]
    /// - Returns: Sequence latents [B, seq_len, 128]
    public static func packPatchifiedToSequence(_ patchified: MLXArray) -> MLXArray {
        let shape = patchified.shape
        let B = shape[0]
        let C = shape[1]  // 128
        let H = shape[2]
        let W = shape[3]

        // Permute from [B, C, H, W] to [B, H, W, C] then reshape to [B, H*W, C]
        let transposed = patchified.transposed(0, 2, 3, 1)  // [B, H, W, C]
        return transposed.reshaped([B, H * W, C])  // [B, seq_len, 128]
    }

    /// Unpack sequence latents back to patchified format
    /// Converts [B, seq_len, 128] to [B, 128, H/16, W/16]
    /// - Parameters:
    ///   - sequence: Sequence latents [B, seq_len, 128]
    ///   - height: Original image height
    ///   - width: Original image width
    /// - Returns: Patchified latents [B, 128, H/16, W/16]
    public static func unpackSequenceToPatchified(
        _ sequence: MLXArray,
        height: Int,
        width: Int
    ) -> MLXArray {
        let shape = sequence.shape
        let B = shape[0]
        let C = shape[2]  // 128

        let patchifiedH = height / 16
        let patchifiedW = width / 16

        // Reshape from [B, seq_len, C] to [B, H, W, C] then permute to [B, C, H, W]
        let reshaped = sequence.reshaped([B, patchifiedH, patchifiedW, C])
        return reshaped.transposed(0, 3, 1, 2)  // [B, C, H, W]
    }

    /// Unpatchify latents from patchified format to VAE format
    /// Converts [B, 128, H/16, W/16] to [B, 32, H/8, W/8]
    /// - Parameters:
    ///   - patchified: Patchified latents [B, 128, H/16, W/16]
    ///   - latentChannels: Number of base latent channels (32)
    ///   - patchSize: Patch size (2)
    /// - Returns: VAE-ready latents [B, 32, H/8, W/8]
    public static func unpatchifyLatents(
        _ patchified: MLXArray,
        latentChannels: Int = 32,
        patchSize: Int = 2
    ) -> MLXArray {
        let shape = patchified.shape
        let B = shape[0]
        let patchifiedC = shape[1]  // 128 = 32 * 2 * 2
        let patchifiedH = shape[2]
        let patchifiedW = shape[3]

        // Output dimensions
        let outH = patchifiedH * patchSize
        let outW = patchifiedW * patchSize

        // Reshape from [B, C*p*p, H, W] to [B, C, p, p, H, W]
        var unpacked = patchified.reshaped([B, latentChannels, patchSize, patchSize, patchifiedH, patchifiedW])

        // Permute to [B, C, H, p, W, p]
        unpacked = unpacked.transposed(0, 1, 4, 2, 5, 3)

        // Reshape to [B, C, H*p, W*p]
        return unpacked.reshaped([B, latentChannels, outH, outW])
    }

    // MARK: - Legacy Packing/Unpacking (from non-patchified format)

    /// Pack latents for transformer input (legacy)
    /// Converts [B, C, H, W] to [B, (H/p)*(W/p), C*p*p]
    /// - Parameters:
    ///   - latents: Latent tensor [B, 32, H, W]
    ///   - patchSize: Patch size (default 2 for Flux.2)
    /// - Returns: Packed latents for transformer
    public static func packLatents(
        _ latents: MLXArray,
        patchSize: Int = 2
    ) -> MLXArray {
        let shape = latents.shape
        let B = shape[0]
        let C = shape[1]
        let H = shape[2]
        let W = shape[3]

        // Number of patches in each dimension
        let numPatchesH = H / patchSize
        let numPatchesW = W / patchSize

        // Reshape: [B, C, H, W] -> [B, C, nH, pH, nW, pW]
        var packed = latents.reshaped([B, C, numPatchesH, patchSize, numPatchesW, patchSize])

        // Permute to group patches: [B, nH, nW, C, pH, pW]
        packed = packed.transposed(0, 2, 4, 1, 3, 5)

        // Flatten to [B, nH*nW, C*pH*pW]
        let numPatches = numPatchesH * numPatchesW
        let patchDim = C * patchSize * patchSize  // 32 * 2 * 2 = 128

        return packed.reshaped([B, numPatches, patchDim])
    }

    /// Unpack latents from transformer output (legacy)
    /// Converts [B, (H/p)*(W/p), C*p*p] back to [B, C, H, W]
    /// - Parameters:
    ///   - packed: Packed latents from transformer
    ///   - height: Original image height
    ///   - width: Original image width
    ///   - latentChannels: Number of latent channels (32)
    ///   - patchSize: Patch size (2)
    /// - Returns: Unpacked latents [B, C, H/8, W/8]
    public static func unpackLatents(
        _ packed: MLXArray,
        height: Int,
        width: Int,
        latentChannels: Int = 32,
        patchSize: Int = 2
    ) -> MLXArray {
        let shape = packed.shape
        let B = shape[0]

        let latentH = height / 8
        let latentW = width / 8
        let numPatchesH = latentH / patchSize
        let numPatchesW = latentW / patchSize

        // Reshape from [B, nH*nW, C*pH*pW] to [B, nH, nW, C, pH, pW]
        var unpacked = packed.reshaped([B, numPatchesH, numPatchesW, latentChannels, patchSize, patchSize])

        // Permute back to [B, C, nH, pH, nW, pW]
        unpacked = unpacked.transposed(0, 3, 1, 4, 2, 5)

        // Reshape to [B, C, H, W]
        return unpacked.reshaped([B, latentChannels, latentH, latentW])
    }

    // MARK: - Position IDs

    /// Generate position IDs for image latents
    /// - Parameters:
    ///   - height: Image height
    ///   - width: Image width
    ///   - patchSize: Patch size (2)
    /// - Returns: Position IDs [numPatches, 4] for (T, H, W, L) encoding
    public static func generateImagePositionIDs(
        height: Int,
        width: Int,
        patchSize: Int = 2
    ) -> MLXArray {
        let latentH = height / 8
        let latentW = width / 8
        let numPatchesH = latentH / patchSize
        let numPatchesW = latentW / patchSize
        let numPatches = numPatchesH * numPatchesW

        var positions: [Int32] = []

        for h in 0..<numPatchesH {
            for w in 0..<numPatchesW {
                // Position encoding: [T=0, H, W, L=0]
                // T (temporal) and L (layer) are 0 for images
                positions.append(contentsOf: [0, Int32(h), Int32(w), 0])
            }
        }

        return MLXArray(positions).reshaped([numPatches, 4])
    }

    /// Generate position IDs for text sequence
    /// - Parameter length: Text sequence length
    /// - Returns: Position IDs [length, 4]
    public static func generateTextPositionIDs(length: Int) -> MLXArray {
        var positions: [Int32] = []

        for l in 0..<length {
            // Text uses L dimension for position
            positions.append(contentsOf: [0, 0, 0, Int32(l)])
        }

        return MLXArray(positions).reshaped([length, 4])
    }

    /// Combine text and image position IDs
    public static func combinePositionIDs(
        textLength: Int,
        height: Int,
        width: Int,
        patchSize: Int = 2
    ) -> (textIds: MLXArray, imageIds: MLXArray, combinedIds: MLXArray) {
        let textIds = generateTextPositionIDs(length: textLength)
        let imageIds = generateImagePositionIDs(height: height, width: width, patchSize: patchSize)
        let combinedIds = concatenated([textIds, imageIds], axis: 0)

        return (textIds: textIds, imageIds: imageIds, combinedIds: combinedIds)
    }

    // MARK: - Image Size Validation

    /// Validate and adjust image dimensions
    /// - Parameters:
    ///   - height: Requested height
    ///   - width: Requested width
    ///   - patchSize: Patch size (2)
    /// - Returns: Adjusted (height, width) divisible by required factor
    public static func validateDimensions(
        height: Int,
        width: Int,
        patchSize: Int = 2
    ) -> (height: Int, width: Int) {
        // Must be divisible by 8 (VAE) * patchSize = 16
        let factor = 8 * patchSize

        let adjustedHeight = ((height + factor - 1) / factor) * factor
        let adjustedWidth = ((width + factor - 1) / factor) * factor

        return (height: adjustedHeight, width: adjustedWidth)
    }

    /// Get latent dimensions for given image size
    public static func getLatentDimensions(
        height: Int,
        width: Int
    ) -> (latentH: Int, latentW: Int, numPatches: Int) {
        let latentH = height / 8
        let latentW = width / 8
        let numPatches = (latentH / 2) * (latentW / 2)  // With patch size 2

        return (latentH: latentH, latentW: latentW, numPatches: numPatches)
    }
}

// MARK: - Latent Scaling

extension LatentUtils {
    /// Scale latents by VAE scaling factor
    public static func scaleLatents(_ latents: MLXArray, scalingFactor: Float = 0.18215) -> MLXArray {
        latents * scalingFactor
    }

    /// Unscale latents for VAE decoding
    public static func unscaleLatents(_ latents: MLXArray, scalingFactor: Float = 0.18215) -> MLXArray {
        latents / scalingFactor
    }
}

// MARK: - BatchNorm Normalization for Flux.2

extension LatentUtils {
    /// Normalize latents using BatchNorm statistics BEFORE transformer
    /// This is critical for Flux.2 - latents must be normalized before denoising
    /// - Parameters:
    ///   - latents: Input latents [B, C, H, W] in NCHW format
    ///   - runningMean: BatchNorm running mean [C]
    ///   - runningVar: BatchNorm running variance [C]
    ///   - eps: Epsilon for numerical stability (Flux.2 uses 1e-4)
    /// - Returns: Normalized latents
    public static func normalizeLatentsWithBatchNorm(
        _ latents: MLXArray,
        runningMean: MLXArray,
        runningVar: MLXArray,
        eps: Float = 1e-4  // Flux.2 batch_norm_eps = 0.0001
    ) -> MLXArray {
        // Reshape stats for NCHW broadcast: [C] -> [1, C, 1, 1]
        let C = runningMean.shape[0]
        let mean = runningMean.reshaped([1, C, 1, 1])
        let std = sqrt(runningVar.reshaped([1, C, 1, 1]) + eps)

        // Normalize: (x - mean) / std
        return (latents - mean) / std
    }

    /// Denormalize latents using BatchNorm statistics AFTER denoising, BEFORE VAE decode
    /// This reverses the normalization applied before the transformer
    /// - Parameters:
    ///   - latents: Denoised latents [B, C, H, W] in NCHW format
    ///   - runningMean: BatchNorm running mean [C]
    ///   - runningVar: BatchNorm running variance [C]
    ///   - eps: Epsilon for numerical stability (Flux.2 uses 1e-4)
    /// - Returns: Denormalized latents ready for VAE decode
    public static func denormalizeLatentsWithBatchNorm(
        _ latents: MLXArray,
        runningMean: MLXArray,
        runningVar: MLXArray,
        eps: Float = 1e-4  // Flux.2 batch_norm_eps = 0.0001
    ) -> MLXArray {
        // Reshape stats for NCHW broadcast: [C] -> [1, C, 1, 1]
        let C = runningMean.shape[0]
        let mean = runningMean.reshaped([1, C, 1, 1])
        let std = sqrt(runningVar.reshaped([1, C, 1, 1]) + eps)

        // Denormalize: x * std + mean
        return latents * std + mean
    }
}
