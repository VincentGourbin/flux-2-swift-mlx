/**
 * Qwen35ImageProcessor.swift
 * Image preprocessing for Qwen3.5 vision encoder
 *
 * Converts CGImage to MLXArray with:
 * - Resize to dimensions divisible by patch_size (16)
 * - Normalize to [-1, 1] using mean=0.5, std=0.5
 * - Output: [1, H, W, 3] (NHWC for MLX Conv2d)
 */

import Foundation
import MLX
import CoreGraphics

#if canImport(AppKit)
import AppKit
#endif

public class Qwen35ImageProcessor {

    /// Patch size must divide image dimensions
    public let patchSize: Int
    /// Spatial merge size (output tokens reduced by merge²)
    public let spatialMergeSize: Int
    /// Min/max image dimensions
    public let minPixels: Int
    public let maxPixels: Int

    /// Maximum total number of pixels (controls memory and quality)
    public let maxTotalPixels: Int

    public init(patchSize: Int = 16, spatialMergeSize: Int = 2, minPixels: Int = 256, maxPixels: Int = 1280, maxTotalPixels: Int = 1003520) {
        self.patchSize = patchSize
        self.spatialMergeSize = spatialMergeSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.maxTotalPixels = maxTotalPixels  // ~1004x1000 or 768x1308 etc.
    }

    /// Preprocess a CGImage for the vision encoder
    /// Uses smart_resize: preserves aspect ratio, ensures dimensions divisible by factor,
    /// and limits total pixel count (not just max side)
    /// - Returns: MLXArray [1, H, W, 3] normalized to [-1, 1]
    public func preprocess(_ image: CGImage) -> MLXArray {
        let factor = patchSize * spatialMergeSize  // 32
        var targetH = image.height
        var targetW = image.width

        // Step 1: Scale down if total pixels exceed budget (preserving aspect ratio)
        let totalPixels = targetH * targetW
        if totalPixels > maxTotalPixels {
            let scale = sqrt(Float(maxTotalPixels) / Float(totalPixels))
            targetH = Int(Float(targetH) * scale)
            targetW = Int(Float(targetW) * scale)
        }

        // Step 2: Ensure max side doesn't exceed maxPixels
        let maxSide = max(targetH, targetW)
        if maxSide > maxPixels {
            let scale = Float(maxPixels) / Float(maxSide)
            targetH = Int(Float(targetH) * scale)
            targetW = Int(Float(targetW) * scale)
        }

        // Step 3: Ensure min side meets minimum
        let minSide = min(targetH, targetW)
        if minSide < minPixels {
            let scale = Float(minPixels) / Float(minSide)
            targetH = Int(Float(targetH) * scale)
            targetW = Int(Float(targetW) * scale)
        }

        // Step 4: Round to nearest multiple of factor
        targetH = max(factor, ((targetH + factor / 2) / factor) * factor)
        targetW = max(factor, ((targetW + factor / 2) / factor) * factor)

        // Draw into bitmap context at target size
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = targetW * 4
        guard let context = CGContext(
            data: nil,
            width: targetW,
            height: targetH,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            // Fallback: return zeros
            return MLXArray.zeros([1, targetH, targetW, 3])
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))

        guard let data = context.data else {
            return MLXArray.zeros([1, targetH, targetW, 3])
        }

        let pixelCount = targetH * targetW
        let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)

        // Extract RGB, normalize to [-1, 1]: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
        var floats = [Float](repeating: 0, count: pixelCount * 3)
        for i in 0..<pixelCount {
            let offset = i * 4
            floats[i * 3 + 0] = Float(ptr[offset + 0]) / 127.5 - 1.0   // R
            floats[i * 3 + 1] = Float(ptr[offset + 1]) / 127.5 - 1.0   // G
            floats[i * 3 + 2] = Float(ptr[offset + 2]) / 127.5 - 1.0   // B
        }

        // [H, W, 3] → [1, H, W, 3]
        return MLXArray(floats).reshaped([1, targetH, targetW, 3])
    }

    /// Get the preprocessed dimensions for an image
    public func preprocessedSize(for image: CGImage) -> (height: Int, width: Int) {
        let factor = patchSize * spatialMergeSize
        var h = image.height
        var w = image.width

        let maxSide = max(h, w)
        if maxSide > maxPixels {
            let scale = Float(maxPixels) / Float(maxSide)
            h = Int(Float(h) * scale)
            w = Int(Float(w) * scale)
        }

        let minSide = min(h, w)
        if minSide < minPixels {
            let scale = Float(minPixels) / Float(minSide)
            h = Int(Float(h) * scale)
            w = Int(Float(w) * scale)
        }

        h = max(factor, ((h + factor / 2) / factor) * factor)
        w = max(factor, ((w + factor / 2) / factor) * factor)

        return (h, w)
    }
}
