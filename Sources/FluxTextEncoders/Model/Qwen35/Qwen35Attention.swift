/**
 * Qwen35Attention.swift
 * Full GQA Attention for Qwen3.5 (used at every 4th layer)
 *
 * Key features:
 * - 16 Q heads, 4 KV heads, head_dim=256 (GQA with 4x expansion)
 * - Output gate: q_proj outputs 2x → split into [q, gate], output = o_proj(attn * sigmoid(gate))
 * - Partial MRoPE: only 64 of 256 dims are rotated (25%)
 * - q_norm, k_norm before RoPE (same as Qwen3)
 */

import Foundation
import MLX
import MLXNN

public class Qwen35Attention: Module {
    let config: Qwen35TextConfig
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rotaryDim: Int

    // q_proj outputs 2x head_dim (for q + gate)
    @ModuleInfo public var q_proj: Linear
    @ModuleInfo public var k_proj: Linear
    @ModuleInfo public var v_proj: Linear
    @ModuleInfo public var o_proj: Linear

    @ModuleInfo public var q_norm: RMSNorm
    @ModuleInfo public var k_norm: RMSNorm

    public var rope: Qwen35MRoPE

    public init(config: Qwen35TextConfig) {
        self.config = config
        self.numHeads = config.numAttentionHeads       // 16
        self.numKVHeads = config.numKeyValueHeads       // 4
        self.headDim = config.headDim                   // 256
        self.scale = pow(Float(config.headDim), -0.5)
        self.rotaryDim = config.rotaryDim               // 64

        // Q projection: hidden → numHeads * headDim * 2 (for q + gate)
        self._q_proj = ModuleInfo(wrappedValue: Linear(
            config.hiddenSize, numHeads * headDim * 2, bias: config.attentionBias))
        self._k_proj = ModuleInfo(wrappedValue: Linear(
            config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        self._v_proj = ModuleInfo(wrappedValue: Linear(
            config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        self._o_proj = ModuleInfo(wrappedValue: Linear(
            numHeads * headDim, config.hiddenSize, bias: config.attentionBias))

        self._q_norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps))
        self._k_norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps))

        self.rope = Qwen35MRoPE(
            rotaryDim: config.rotaryDim,
            base: config.ropeTheta,
            mropeSections: config.mropeSectionSizes
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        positionIds: MLXArray,
        cache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        // Q projection → split into [q, gate]
        let qProjOut = q_proj(x)  // [B, S, numHeads * headDim * 2]
        let qAndGate = qProjOut.reshaped([B, S, numHeads, headDim * 2])
        let (queriesRaw, gateRaw) = (
            qAndGate[0..., 0..., 0..., 0..<headDim],
            qAndGate[0..., 0..., 0..., headDim..<(headDim * 2)]
        )
        let gate = gateRaw.reshaped([B, S, numHeads * headDim])  // [B, S, numHeads * headDim]

        // K, V projections
        let keysRaw = k_proj(x)
        let valuesRaw = v_proj(x)

        // Reshape to [B, S, H, D]
        var queries = queriesRaw  // already [B, S, numHeads, headDim]
        var keys = keysRaw.reshaped([B, S, numKVHeads, headDim])
        var values = valuesRaw.reshaped([B, S, numKVHeads, headDim])

        // Q/K normalization before RoPE
        queries = q_norm(queries)
        keys = k_norm(keys)

        // Transpose to [B, H, S, D]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Compute cos/sin for partial MRoPE
        let (cos, sin) = rope.computeCosSin(positionIds: positionIds)

        // Apply partial rotary (only first 64 of 256 dims)
        queries = Qwen35MRoPE.applyPartialRotary(queries, cos: cos, sin: sin, rotaryDim: rotaryDim)
        keys = Qwen35MRoPE.applyPartialRotary(keys, cos: cos, sin: sin, rotaryDim: rotaryDim)

        // Update KV cache
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // GQA: expand KV heads
        let repeatFactor = numHeads / numKVHeads
        if repeatFactor > 1 {
            let kvSeqLen = keys.dim(2)
            keys = keys.expandedDimensions(axis: 2)
            keys = MLX.broadcast(keys, to: [B, numKVHeads, repeatFactor, kvSeqLen, headDim])
            keys = keys.reshaped([B, numHeads, kvSeqLen, headDim])

            values = values.expandedDimensions(axis: 2)
            values = MLX.broadcast(values, to: [B, numKVHeads, repeatFactor, kvSeqLen, headDim])
            values = values.reshaped([B, numHeads, kvSeqLen, headDim])
        }

        // Scaled dot-product attention
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape: [B, H, S, D] → [B, S, H*D]
        let output = attnOutput.transposed(0, 2, 1, 3).reshaped([B, S, numHeads * headDim])

        // Apply output gate: output * sigmoid(gate)
        let gatedOutput = output * sigmoid(gate)

        return o_proj(gatedOutput)
    }
}
