/**
 * Qwen3VLAttention.swift
 * Multi-Head Attention with GQA and MRoPE for Qwen3-VL
 *
 * Key differences from Qwen3Attention:
 * - head_dim=128 (explicit) vs 80 (computed from hidden_size/num_heads)
 * - Q/K/V projections are larger: 2560→4096 for Q, 2560→1024 for K/V
 * - MRoPE (multi-dimensional) instead of standard 1D RoPE
 * - Takes position_ids parameter for MRoPE
 */

import Foundation
import MLX
import MLXNN

// MARK: - Qwen3-VL Attention

/// Qwen3-VL Attention with GQA and multi-dimensional RoPE
public class Qwen3VLAttention: Module {
    let config: Qwen3VLTextConfig
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo public var q_proj: Linear
    @ModuleInfo public var k_proj: Linear
    @ModuleInfo public var v_proj: Linear
    @ModuleInfo public var o_proj: Linear

    // Qwen3-specific: RMSNorm on Q and K (applied per-head, BEFORE RoPE)
    @ModuleInfo public var q_norm: RMSNorm
    @ModuleInfo public var k_norm: RMSNorm

    public var rope: Qwen3VLMRoPE

    public init(config: Qwen3VLTextConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim  // 128 (explicit, NOT hidden_size / num_heads)
        self.scale = 1.0 / sqrt(Float(config.headDim))

        // Projections — note: numHeads * headDim = 32 * 128 = 4096 ≠ hiddenSize (2560)
        self._q_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numHeads * config.headDim, bias: config.attentionBias))
        self._k_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * config.headDim, bias: config.attentionBias))
        self._v_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * config.headDim, bias: config.attentionBias))
        self._o_proj = ModuleInfo(wrappedValue: Linear(numHeads * config.headDim, hiddenSize, bias: config.attentionBias))

        // Q and K normalization (RMSNorm at head_dim level, same as Qwen3)
        self._q_norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps))
        self._k_norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps))

        // MRoPE instead of standard RoPE
        self.rope = Qwen3VLMRoPE(sectionSizes: config.mropeSectionSizes, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        positionIds: MLXArray,
        cache: KVCache? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]

        // Project Q, K, V
        var queries = q_proj(hiddenStates)
        var keys = k_proj(hiddenStates)
        var values = v_proj(hiddenStates)

        // Reshape for multi-head attention
        queries = queries.reshaped([batchSize, seqLen, numHeads, headDim])
        keys = keys.reshaped([batchSize, seqLen, numKVHeads, headDim])
        values = values.reshaped([batchSize, seqLen, numKVHeads, headDim])

        // CRITICAL: Qwen3 applies RMSNorm to Q and K BEFORE RoPE
        queries = q_norm(queries)
        keys = k_norm(keys)

        // Transpose to [batch, heads, seq, head_dim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply MRoPE (multi-dimensional rotary position embedding)
        queries = rope(queries, positionIds: positionIds)
        keys = rope(keys, positionIds: positionIds)

        // Update KV cache if provided
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // GQA: expand KV heads to match Q heads
        let repeatFactor = numHeads / numKVHeads
        if repeatFactor > 1 {
            let kvSeqLen = keys.dim(2)
            keys = keys.expandedDimensions(axis: 2)
            keys = MLX.broadcast(keys, to: [batchSize, numKVHeads, repeatFactor, kvSeqLen, headDim])
            keys = keys.reshaped([batchSize, numHeads, kvSeqLen, headDim])

            values = values.expandedDimensions(axis: 2)
            values = MLX.broadcast(values, to: [batchSize, numKVHeads, repeatFactor, kvSeqLen, headDim])
            values = values.reshaped([batchSize, numHeads, kvSeqLen, headDim])
        }

        // Scaled dot-product attention
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
        let outputTransposed = output.transposed(0, 2, 1, 3)
        let outputReshaped = outputTransposed.reshaped([batchSize, seqLen, numHeads * headDim])

        return o_proj(outputReshaped)
    }
}
