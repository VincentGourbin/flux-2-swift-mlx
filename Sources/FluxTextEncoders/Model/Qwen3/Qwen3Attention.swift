/**
 * Qwen3Attention.swift
 * Multi-Head Attention with Grouped Query Attention (GQA) and RoPE for Qwen3
 *
 * Similar to Mistral but without Llama-4 attention scaling
 */

import Foundation
import MLX
import MLXNN

// MARK: - Qwen3 RoPE

/// Rotary Position Embedding for Qwen3
public class Qwen3RoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    public init(dimensions: Int, traditional: Bool = false, base: Float = 1_000_000.0, scale: Float = 1.0) {
        self.dimensions = dimensions
        self.traditional = traditional
        self.base = base
        self.scale = scale
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Use MLXFast.RoPE directly without reshaping (fix for mlx-swift 0.30.2)
        // PR #316: reshaping before RoPE causes incorrect behavior with non-zero offsets
        return MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
            offset: offset)
    }
}

// MARK: - Qwen3 Attention

/// Qwen3 Attention with Grouped Query Attention (GQA)
/// Key differences from Mistral:
/// - No Llama-4 attention scaling
/// - head_dim varies by model size (80 for 4B, 128 for 8B)
/// - CRITICAL: Qwen3 has q_norm and k_norm (RMSNorm) applied BEFORE RoPE
public class Qwen3Attention: Module {
    let config: Qwen3TextConfig
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

    public var rope: Qwen3RoPE

    public init(config: Qwen3TextConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        // Projections
        // Q projection: hidden_size -> num_heads * head_dim
        self._q_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numHeads * headDim, bias: config.attentionBias))
        // K projection: hidden_size -> num_kv_heads * head_dim
        self._k_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        // V projection: hidden_size -> num_kv_heads * head_dim
        self._v_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        // O projection: num_heads * head_dim -> hidden_size
        self._o_proj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, hiddenSize, bias: config.attentionBias))

        // Qwen3-specific: Q and K normalization (RMSNorm at head_dim level)
        self._q_norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps))
        self._k_norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps))

        // RoPE - using standard rope theta of 1M
        self.rope = Qwen3RoPE(dimensions: headDim, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
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

        // CRITICAL: Qwen3 applies RMSNorm to Q and K BEFORE RoPE (after reshape, before transpose)
        // This is the key difference from other models like Mistral/Llama
        queries = q_norm(queries)
        keys = k_norm(keys)

        // Transpose to [batch, heads, seq, head_dim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE (after normalization)
        let offset = cache?.length ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Note: Qwen3 does NOT use Llama-4 attention scaling (unlike Mistral)
        // This is a key difference from MistralAttention

        // Update KV cache if provided
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // GQA: Use broadcasting instead of explicit repeat when possible
        // Shape: queries [B, num_heads, S, D], keys/values [B, num_kv_heads, S, D]
        let repeatFactor = numHeads / numKVHeads

        // For GQA, expand KV using efficient broadcast-based expansion
        if repeatFactor > 1 {
            let kvSeqLen = keys.dim(2)
            keys = keys.expandedDimensions(axis: 2)
            keys = MLX.broadcast(keys, to: [batchSize, numKVHeads, repeatFactor, kvSeqLen, headDim])
            keys = keys.reshaped([batchSize, numHeads, kvSeqLen, headDim])

            values = values.expandedDimensions(axis: 2)
            values = MLX.broadcast(values, to: [batchSize, numKVHeads, repeatFactor, kvSeqLen, headDim])
            values = values.reshaped([batchSize, numHeads, kvSeqLen, headDim])
        }

        // Scaled dot-product attention using MLXFast
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let outputTransposed = output.transposed(0, 2, 1, 3)
        let outputReshaped = outputTransposed.reshaped([batchSize, seqLen, numHeads * headDim])

        return o_proj(outputReshaped)
    }
}
