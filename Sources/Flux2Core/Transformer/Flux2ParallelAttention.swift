// Flux2ParallelAttention.swift - Fused Attention+FFN for Single-Stream Blocks
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MLXFast

/// Parallel Self-Attention for Single-Stream Blocks
///
/// This module fuses the QKV projection with MLP gate/up projections
/// for more efficient computation. Used in the 48 single-stream blocks.
///
/// Architecture:
/// - Single projection: dim -> (3*innerDim + 2*mlpHidden) for Q,K,V,gate,up
/// - Self-attention on concatenated text+image
/// - Output projection: (innerDim + mlpHidden) -> dim
public class Flux2ParallelSelfAttention: Module, @unchecked Sendable {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let innerDim: Int
    let mlpHiddenDim: Int

    // Fused input projection: Q, K, V, MLP gate, MLP up
    // @ModuleInfo allows update(modules:) to replace with LoRA
    @ModuleInfo var toQkvMlp: Linear

    // QK normalization
    let normQ: RMSNorm
    let normK: RMSNorm

    // Fused output projection: attention output + MLP output -> dim
    @ModuleInfo var toOut: Linear

    /// Initialize Parallel Self-Attention
    /// - Parameters:
    ///   - dim: Model dimension (6144)
    ///   - numHeads: Number of attention heads (48)
    ///   - headDim: Dimension per head (128)
    ///   - mlpRatio: MLP hidden dimension ratio (default 4.0)
    public init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        mlpRatio: Float = 3.0
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = headDim
        self.innerDim = numHeads * headDim
        self.mlpHiddenDim = Int(Float(dim) * mlpRatio)

        // Fused projection dimensions:
        // Q: innerDim, K: innerDim, V: innerDim, gate: mlpHidden, up: mlpHidden
        let totalProjDim = innerDim * 3 + mlpHiddenDim * 2
        self._toQkvMlp.wrappedValue = Linear(dim, totalProjDim, bias: false)

        // QK normalization
        self.normQ = RMSNorm(dim: headDim)
        self.normK = RMSNorm(dim: headDim)

        // Output projection: attention (innerDim) + mlp (mlpHidden) -> dim (no bias)
        self._toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
    }

    /// Forward pass
    /// - Parameters:
    ///   - hiddenStates: Combined image+text hidden states [B, S, dim]
    ///   - rotaryEmb: Optional RoPE embeddings
    /// - Returns: Updated hidden states [B, S, dim]
    public func callAsFunction(
        hiddenStates: MLXArray,
        rotaryEmb: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]

        // Fused projection
        let projected = toQkvMlp(hiddenStates)

        // Split into Q, K, V, gate, up
        var q = projected[0..., 0..., 0..<innerDim]
        var k = projected[0..., 0..., innerDim..<(innerDim * 2)]
        let v = projected[0..., 0..., (innerDim * 2)..<(innerDim * 3)]
        let mlpGate = projected[0..., 0..., (innerDim * 3)..<(innerDim * 3 + mlpHiddenDim)]
        let mlpUp = projected[0..., 0..., (innerDim * 3 + mlpHiddenDim)...]

        // Reshape for multi-head attention: [B, S, H*D] -> [B, H, S, D]
        q = reshapeForAttention(q, batchSize: batchSize, seqLen: seqLen)
        k = reshapeForAttention(k, batchSize: batchSize, seqLen: seqLen)
        let vReshaped = reshapeForAttention(v, batchSize: batchSize, seqLen: seqLen)

        // Apply QK normalization
        q = normQ(q)
        k = normK(k)

        // Apply RoPE if provided
        if let rope = rotaryEmb {
            (q, k) = applyRoPE(q: q, k: k, cos: rope.cos, sin: rope.sin)
        }

        // Compute attention
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: vReshaped,
            scale: Float(1.0 / sqrt(Float(headDim))),
            mask: nil
        )

        // Reshape attention output: [B, H, S, D] -> [B, S, H*D]
        let attnOut = reshapeFromAttention(attnOutput, batchSize: batchSize, seqLen: seqLen)

        // Compute MLP output: SwiGLU(gate, up)
        let mlpOut = silu(mlpGate) * mlpUp

        // Concatenate attention and MLP outputs
        let combined = concatenated([attnOut, mlpOut], axis: -1)

        // Final projection
        return toOut(combined)
    }

    // MARK: - Helper Functions

    private func reshapeForAttention(_ x: MLXArray, batchSize: Int, seqLen: Int) -> MLXArray {
        x.reshaped([batchSize, seqLen, numHeads, headDim])
            .transposed(0, 2, 1, 3)
    }

    private func reshapeFromAttention(_ x: MLXArray, batchSize: Int, seqLen: Int) -> MLXArray {
        x.transposed(0, 2, 1, 3)
            .reshaped([batchSize, seqLen, numHeads * headDim])
    }

    private func applyRoPE(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cosExpanded = cos.expandedDimensions(axes: [0, 1])
        let sinExpanded = sin.expandedDimensions(axes: [0, 1])

        let qRotated = rotateHalf(q)
        let kRotated = rotateHalf(k)

        let qOut = q * cosExpanded + qRotated * sinExpanded
        let kOut = k * cosExpanded + kRotated * sinExpanded

        return (qOut, kOut)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        // x shape: [B, H, S, D]
        // Diffusers approach: reshape to [B, H, S, D/2, 2], then [-imag, real]
        let batchSize = x.shape[0]
        let numHeads = x.shape[1]
        let seqLen = x.shape[2]
        let dim = x.shape[3]
        let halfDim = dim / 2

        // Reshape to [B, H, S, D/2, 2]
        let xReshaped = x.reshaped([batchSize, numHeads, seqLen, halfDim, 2])

        // Get real and imag parts (consecutive pairs)
        let xReal = xReshaped[0..., 0..., 0..., 0..., 0]  // [B, H, S, D/2]
        let xImag = xReshaped[0..., 0..., 0..., 0..., 1]  // [B, H, S, D/2]

        // Create rotated: stack [-imag, real] and flatten
        let xRotatedStacked = stacked([-xImag, xReal], axis: -1)  // [B, H, S, D/2, 2]
        return xRotatedStacked.reshaped([batchSize, numHeads, seqLen, dim])  // [B, H, S, D]
    }
}

/// Alternative implementation with separate projections
/// (for compatibility with some weight formats)
public class Flux2ParallelSelfAttentionSplit: Module, @unchecked Sendable {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let innerDim: Int
    let mlpHiddenDim: Int

    // Separate projections (var for LoRA injection)
    var toQ: Linear
    var toK: Linear
    var toV: Linear
    var mlpGate: Linear
    var mlpUp: Linear

    // QK normalization
    let normQ: RMSNorm
    let normK: RMSNorm

    // Separate output projections (var for LoRA injection)
    var toAttnOut: Linear
    var mlpDown: Linear

    public init(
        dim: Int,
        numHeads: Int,
        headDim: Int,
        mlpRatio: Float = 3.0
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = headDim
        self.innerDim = numHeads * headDim
        self.mlpHiddenDim = Int(Float(dim) * mlpRatio)

        self.toQ = Linear(dim, innerDim, bias: false)
        self.toK = Linear(dim, innerDim, bias: false)
        self.toV = Linear(dim, innerDim, bias: false)
        self.mlpGate = Linear(dim, mlpHiddenDim, bias: false)
        self.mlpUp = Linear(dim, mlpHiddenDim, bias: false)

        self.normQ = RMSNorm(dim: headDim)
        self.normK = RMSNorm(dim: headDim)

        self.toAttnOut = Linear(innerDim, dim, bias: false)
        self.mlpDown = Linear(mlpHiddenDim, dim, bias: false)
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        rotaryEmb: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]

        // Separate projections
        var q = toQ(hiddenStates)
        var k = toK(hiddenStates)
        let v = toV(hiddenStates)
        let gate = mlpGate(hiddenStates)
        let up = mlpUp(hiddenStates)

        // Reshape for attention
        q = q.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vReshaped = v.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        // QK norm
        q = normQ(q)
        k = normK(k)

        // RoPE
        if let rope = rotaryEmb {
            let cosExpanded = rope.cos.expandedDimensions(axes: [0, 1])
            let sinExpanded = rope.sin.expandedDimensions(axes: [0, 1])

            let qRot = rotateHalf(q)
            let kRot = rotateHalf(k)

            q = q * cosExpanded + qRot * sinExpanded
            k = k * cosExpanded + kRot * sinExpanded
        }

        // Attention
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: vReshaped,
            scale: Float(1.0 / sqrt(Float(headDim))),
            mask: nil
        )

        // Reshape back
        let attnReshaped = attnOut.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, innerDim])

        // MLP
        let mlpOut = silu(gate) * up

        // Separate output projections and add
        return toAttnOut(attnReshaped) + mlpDown(mlpOut)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        // x shape: [B, H, S, D]
        // Diffusers approach: reshape to [B, H, S, D/2, 2], then [-imag, real]
        let batchSize = x.shape[0]
        let numHeads = x.shape[1]
        let seqLen = x.shape[2]
        let dim = x.shape[3]
        let halfDim = dim / 2

        // Reshape to [B, H, S, D/2, 2]
        let xReshaped = x.reshaped([batchSize, numHeads, seqLen, halfDim, 2])

        // Get real and imag parts (consecutive pairs)
        let xReal = xReshaped[0..., 0..., 0..., 0..., 0]  // [B, H, S, D/2]
        let xImag = xReshaped[0..., 0..., 0..., 0..., 1]  // [B, H, S, D/2]

        // Create rotated: stack [-imag, real] and flatten
        let xRotatedStacked = stacked([-xImag, xReal], axis: -1)  // [B, H, S, D/2, 2]
        return xRotatedStacked.reshaped([batchSize, numHeads, seqLen, dim])  // [B, H, S, D]
    }
}
