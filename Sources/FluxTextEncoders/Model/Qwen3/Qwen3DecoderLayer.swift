/**
 * Qwen3DecoderLayer.swift
 * Single decoder layer for Qwen3 transformer
 */

import Foundation
import MLX
import MLXNN

/// Single Qwen3 decoder layer
/// Same structure as Mistral decoder layer
public class Qwen3DecoderLayer: Module {
    let config: Qwen3TextConfig

    public var self_attn: Qwen3Attention
    public var mlp: Qwen3MLP
    public var input_layernorm: RMSNorm
    public var post_attention_layernorm: RMSNorm

    public init(config: Qwen3TextConfig) {
        self.config = config

        self.self_attn = Qwen3Attention(config: config)
        self.mlp = Qwen3MLP(config: config)
        self.input_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.post_attention_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Self-attention with residual
        let residual = hiddenStates
        let normalizedHidden = input_layernorm(hiddenStates)
        let attnOutput = self_attn(normalizedHidden, mask: mask, cache: cache)
        var hidden = residual + attnOutput

        // MLP with residual
        let residual2 = hidden
        let normalizedHidden2 = post_attention_layernorm(hidden)
        let mlpOutput = mlp(normalizedHidden2)
        hidden = residual2 + mlpOutput

        return hidden
    }
}
