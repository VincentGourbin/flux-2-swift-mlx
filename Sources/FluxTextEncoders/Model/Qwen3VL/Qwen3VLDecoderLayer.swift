/**
 * Qwen3VLDecoderLayer.swift
 * Single decoder layer for Qwen3-VL transformer (language component)
 *
 * Same structure as Qwen3DecoderLayer but uses Qwen3VLAttention (MRoPE)
 * and accepts position_ids for multi-dimensional position encoding.
 *
 * Note: DeepStack cross-attention layers [5, 11, 17] are NOT implemented
 * in this Phase 1 (text-only evaluation).
 */

import Foundation
import MLX
import MLXNN

/// Single Qwen3-VL decoder layer
public class Qwen3VLDecoderLayer: Module {
    let config: Qwen3VLTextConfig

    public var self_attn: Qwen3VLAttention
    public var mlp: Qwen3VLMLP
    public var input_layernorm: RMSNorm
    public var post_attention_layernorm: RMSNorm

    public init(config: Qwen3VLTextConfig) {
        self.config = config

        self.self_attn = Qwen3VLAttention(config: config)
        self.mlp = Qwen3VLMLP(config: config)
        self.input_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.post_attention_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        positionIds: MLXArray,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Self-attention with residual
        let residual = hiddenStates
        let normalizedHidden = input_layernorm(hiddenStates)
        let attnOutput = self_attn(normalizedHidden, mask: mask, positionIds: positionIds, cache: cache)
        var hidden = residual + attnOutput

        // MLP with residual
        let residual2 = hidden
        let normalizedHidden2 = post_attention_layernorm(hidden)
        let mlpOutput = mlp(normalizedHidden2)
        hidden = residual2 + mlpOutput

        return hidden
    }
}
