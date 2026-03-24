/**
 * Qwen35MRoPE.swift
 * Partial Multi-dimensional Rotary Position Embedding for Qwen3.5
 *
 * Key differences from Qwen3-VL MRoPE:
 * - partial_rotary_factor = 0.25 → only 64 of 256 head_dim dimensions are rotated
 * - Sections [11, 11, 10] = 32 base dims → 64 actual (cos+sin pairs)
 * - Interleaved MRoPE: temporal/height/width frequencies are interleaved, not concatenated
 * - Non-rotated dimensions (64..256) pass through unchanged
 *
 * The interleaved pattern assigns frequencies to 3 dimensions cyclically:
 * freq[0]=temporal, freq[1]=height, freq[2]=width, freq[3]=temporal, ...
 * Then apply_interleaved_mrope selects the correct position_ids for each frequency.
 */

import Foundation
import MLX
import MLXNN

public class Qwen35MRoPE: Module {
    let rotaryDim: Int          // 64 (25% of 256)
    let base: Float
    let mropeSections: [Int]    // [11, 11, 10]

    /// Precomputed inverse frequencies: [rotaryDim / 2] = [32]
    private let invFreq: MLXArray

    public init(rotaryDim: Int = 64, base: Float = 10_000_000.0, mropeSections: [Int] = [11, 11, 10]) {
        self.rotaryDim = rotaryDim
        self.base = base
        self.mropeSections = mropeSections

        // inv_freq = 1 / (theta ^ (2i / dim)) for i in [0, dim/2)
        let halfDim = rotaryDim / 2  // 32
        let indices = MLXArray(stride(from: Float(0), to: Float(halfDim), by: 1))
        self.invFreq = 1.0 / MLX.pow(MLXArray(base), (2.0 * indices) / Float(rotaryDim))

        super.init()
    }

    /// Compute cos/sin embeddings with interleaved MRoPE
    /// - Parameters:
    ///   - positionIds: [3, batch, seqLen] for [temporal, height, width]
    /// - Returns: (cos, sin) each [batch, seqLen, rotaryDim]
    public func computeCosSin(positionIds: MLXArray) -> (MLXArray, MLXArray) {
        // positionIds: [3, B, S]
        let numPositionDims = 3
        let B = positionIds.dim(1)
        let S = positionIds.dim(2)
        let halfDim = rotaryDim / 2  // 32

        // Compute freqs for all 3 position dimensions: [3, B, halfDim, S]
        // inv_freq: [halfDim], position_ids: [3, B, S]
        let invFreqExpanded = invFreq.reshaped([1, 1, halfDim, 1])  // [1, 1, halfDim, 1]
        let posExpanded = positionIds.reshaped([numPositionDims, B, 1, S]).asType(.float32)  // [3, B, 1, S]

        // freqs[dim, b, freq_idx, seq] = inv_freq[freq_idx] * pos[dim, b, seq]
        let freqs = invFreqExpanded * posExpanded  // [3, B, halfDim, S]
        // Transpose to [3, B, S, halfDim]
        let freqsTransposed = freqs.transposed(0, 1, 3, 2)

        // Apply interleaved MRoPE:
        // Start with temporal freqs, then overwrite height/width at their interleaved positions
        var result = freqsTransposed[0]  // [B, S, halfDim] — start with temporal

        // Sections [11, 11, 10]: freq indices are assigned cyclically
        // freq[i] gets dim (i % 3): 0=temporal, 1=height, 2=width
        // For dim 1 (height): indices 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31 (11 values, step 3 from offset 1)
        // For dim 2 (width):  indices 2, 5, 8, 11, 14, 17, 20, 23, 26, 29     (10 values, step 3 from offset 2)
        for dim in 1..<numPositionDims {
            let sectionLen = mropeSections[dim]
            let dimFreqs = freqsTransposed[dim]  // [B, S, halfDim]
            // Interleaved indices: offset=dim, step=3, count=sectionLen
            for i in 0..<sectionLen {
                let idx = dim + i * 3
                if idx < halfDim {
                    // result[:, :, idx] = dimFreqs[:, :, idx]
                    let slice = dimFreqs[0..., 0..., idx..<(idx+1)]
                    result = setSlice(result, at: idx, with: slice)
                }
            }
        }

        // Duplicate for cos/sin: [B, S, halfDim] → [B, S, rotaryDim]
        let fullFreqs = concatenated([result, result], axis: -1)
        let cos = MLX.cos(fullFreqs)
        let sin = MLX.sin(fullFreqs)

        return (cos, sin)
    }

    /// Apply partial MRoPE to queries or keys
    /// Only rotates the first `rotaryDim` dimensions, passes the rest through
    /// - Parameters:
    ///   - x: [batch, heads, seq, head_dim]
    ///   - cos: [batch, seq, rotaryDim]
    ///   - sin: [batch, seq, rotaryDim]
    /// - Returns: Rotated tensor, same shape
    public static func applyPartialRotary(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        rotaryDim: Int
    ) -> MLXArray {
        let headDim = x.dim(-1)

        // Split into rotated and pass-through portions
        let xRot = x[.ellipsis, 0..<rotaryDim]       // [B, H, S, rotaryDim]
        let xPass = x[.ellipsis, rotaryDim..<headDim]  // [B, H, S, headDim-rotaryDim]

        // Reshape cos/sin for broadcasting: [B, 1, S, rotaryDim]
        let cosB = cos.expandedDimensions(axis: 1)
        let sinB = sin.expandedDimensions(axis: 1)

        // Apply rotation (non-interleaved / GPT-NeoX style: split halves)
        let rotated = xRot * cosB + rotateHalf(xRot) * sinB

        return concatenated([rotated, xPass], axis: -1)
    }

    /// Generate text-only position IDs [3, 1, seqLen]
    public static func textOnlyPositionIds(seqLen: Int, offset: Int = 0) -> MLXArray {
        let temporal = MLXArray(stride(from: Int32(offset), to: Int32(offset + seqLen), by: 1)).reshaped([1, seqLen])
        let zeros = MLXArray.zeros([1, seqLen], dtype: .int32)
        return stacked([temporal, zeros, zeros])  // [3, 1, seqLen]
    }

    // MARK: - Private

    /// Rotate half (GPT-NeoX style): [-x2, x1]
    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0..<half]
        let x2 = x[.ellipsis, half..<(half * 2)]
        return concatenated([-x2, x1], axis: -1)
    }

    /// Set a single index slice along the last dimension
    private func setSlice(_ tensor: MLXArray, at index: Int, with value: MLXArray) -> MLXArray {
        // Build result by concatenating [before, value, after]
        let lastDim = tensor.dim(-1)
        if index == 0 {
            let after = tensor[0..., 0..., 1..<lastDim]
            return concatenated([value, after], axis: -1)
        } else if index == lastDim - 1 {
            let before = tensor[0..., 0..., 0..<index]
            return concatenated([before, value], axis: -1)
        } else {
            let before = tensor[0..., 0..., 0..<index]
            let after = tensor[0..., 0..., (index+1)..<lastDim]
            return concatenated([before, value, after], axis: -1)
        }
    }
}
