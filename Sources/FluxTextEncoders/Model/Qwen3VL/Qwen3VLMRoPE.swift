/**
 * Qwen3VLMRoPE.swift
 * Multi-dimensional Rotary Position Embedding for Qwen3-VL
 *
 * MRoPE splits head_dim into 3 sections [temporal, height, width],
 * each receiving separate position IDs. Uses interleaved rotation
 * pattern (mrope_interleaved=true in Qwen3-VL config).
 *
 * For text-only: temporal = sequential [0..seq-1], height = width = 0
 * The height/width sections get cos(0)=1, sin(0)=0, effectively no rotation.
 */

import Foundation
import MLX
import MLXNN

// MARK: - Multi-dimensional RoPE

/// Multi-dimensional Rotary Position Embedding for Qwen3-VL
/// Uses interleaved (Llama-style) rotation: pairs are adjacent (d0,d1), (d2,d3), ...
public class Qwen3VLMRoPE: Module {
    /// Section sizes [temporal, height, width], e.g. [24, 20, 20]
    /// Each section gets 2x its size in actual dimensions
    let sectionSizes: [Int]
    let base: Float

    /// Precomputed inverse frequencies per section
    /// Shape per section: [section_size]
    private let invFreqs: [MLXArray]

    public init(sectionSizes: [Int] = [24, 20, 20], base: Float = 1_000_000.0) {
        self.sectionSizes = sectionSizes
        self.base = base

        // freq_i = 1 / (theta ^ (2i / dim)) where dim = section_size * 2
        self.invFreqs = sectionSizes.map { sectionSize in
            let dim = Float(sectionSize * 2)
            let indices = MLXArray(stride(from: Float(0), to: Float(sectionSize), by: 1))
            return 1.0 / MLX.pow(MLXArray(base), (2.0 * indices) / dim)
        }

        super.init()
    }

    /// Apply MRoPE to queries or keys
    /// - Parameters:
    ///   - x: Input tensor [batch, heads, seq, head_dim]
    ///   - positionIds: Position IDs [3, seq] for [temporal, height, width]
    /// - Returns: Rotated tensor with same shape
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> MLXArray {
        let headDim = x.dim(3)

        // Split x into 3 sections along head_dim
        // sectionSizes = [24, 20, 20] → actual dims = [48, 40, 40] = 128
        var offset = 0
        var rotatedSections: [MLXArray] = []

        for (i, sectionSize) in sectionSizes.enumerated() {
            let actualDim = sectionSize * 2
            // Extract section: [batch, heads, seq, actualDim]
            let section = x[.ellipsis, offset..<(offset + actualDim)]

            // Get position IDs for this section: [seq]
            let posIds = positionIds[i]

            // Compute angles: [seq, sectionSize]
            let angles = MLX.outer(posIds.asType(.float32), invFreqs[i])

            // For interleaved pattern (Llama-style):
            // cos/sin shape needs to be [1, 1, seq, actualDim] where each freq is repeated for the pair
            // angles[j] applies to pair (x[2j], x[2j+1])
            // So we need to repeat each angle value twice: [a0, a0, a1, a1, a2, a2, ...]
            let anglesRepeated = MLX.repeat(angles, count: 2, axis: -1)
                .reshaped([1, 1, angles.dim(0), actualDim])

            let cos = MLX.cos(anglesRepeated)
            let sin = MLX.sin(anglesRepeated)

            // Interleaved rotation: for pairs (x0,x1), (x2,x3), ...
            // rotated = x * cos + rotate_interleaved(x) * sin
            // where rotate_interleaved swaps adjacent pairs: [-x1, x0, -x3, x2, ...]
            let rotated = section * cos + rotateInterleaved(section) * sin

            rotatedSections.append(rotated)
            offset += actualDim
        }

        // Handle any remaining dimensions
        if offset < headDim {
            rotatedSections.append(x[.ellipsis, offset..<headDim])
        }

        return concatenated(rotatedSections, axis: -1)
    }

    /// Interleaved rotation (Llama-style): swap adjacent pairs with negation
    /// [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
    private func rotateInterleaved(_ x: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        // Use stride slicing to extract even/odd elements
        // x_even = x[..., 0::2], x_odd = x[..., 1::2]
        let xEven = x[.ellipsis, .stride(from: 0, to: dim, by: 2)]   // [..., dim/2]
        let xOdd = x[.ellipsis, .stride(from: 1, to: dim, by: 2)]    // [..., dim/2]

        // Interleave [-x_odd, x_even] back into [..., dim]
        // Stack pairs: [..., dim/2, 2] then reshape to [..., dim]
        let negOdd = -xOdd
        let pairs = stacked([negOdd, xEven], axis: -1)  // [..., dim/2, 2]

        return pairs.reshaped(x.shape)
    }

    /// Generate text-only position IDs
    /// For pure text: temporal = [offset, offset+1, ..., offset+seq-1], height = width = [0, 0, ..., 0]
    /// - Parameters:
    ///   - seqLen: Sequence length
    ///   - offset: Starting position offset (for KV-cached generation)
    /// - Returns: Position IDs tensor [3, seqLen]
    public static func textOnlyPositionIds(seqLen: Int, offset: Int = 0) -> MLXArray {
        let temporal = MLXArray(stride(from: Int32(offset), to: Int32(offset + seqLen), by: 1))
        let zeros = MLXArray.zeros([seqLen], dtype: .int32)
        return stacked([temporal, zeros, zeros])  // [3, seqLen]
    }
}
