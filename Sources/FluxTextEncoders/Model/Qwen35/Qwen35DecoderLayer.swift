/**
 * Qwen35DecoderLayer.swift
 * Hybrid decoder layer: either linear attention (Gated DeltaNet) or full attention (GQA)
 */

import Foundation
import MLX
import MLXNN

public class Qwen35DecoderLayer: Module {
    public let isLinear: Bool

    public var linear_attn: Qwen35GatedDeltaNet?
    public var self_attn: Qwen35Attention?
    public var mlp: Qwen35MLP
    public var input_layernorm: RMSNorm
    public var post_attention_layernorm: RMSNorm

    public init(config: Qwen35TextConfig, layerIndex: Int) {
        self.isLinear = config.isLinearLayer(layerIndex)

        if isLinear {
            self.linear_attn = Qwen35GatedDeltaNet(config: config)
            self.self_attn = nil
        } else {
            self.self_attn = Qwen35Attention(config: config)
            self.linear_attn = nil
        }

        self.mlp = Qwen35MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self.input_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.post_attention_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        positionIds: MLXArray? = nil,
        kvCache: KVCache? = nil,
        deltaCache: inout Qwen35GatedDeltaNet.DeltaNetCache
    ) -> MLXArray {
        let normed = input_layernorm(x)

        let attnOutput: MLXArray
        if isLinear {
            attnOutput = linear_attn!(normed, mask: mask, cache: &deltaCache)
        } else {
            attnOutput = self_attn!(normed, mask: mask, positionIds: positionIds!, cache: kvCache)
        }

        let h = x + attnOutput
        let out = h + mlp(post_attention_layernorm(h))
        return out
    }
}
