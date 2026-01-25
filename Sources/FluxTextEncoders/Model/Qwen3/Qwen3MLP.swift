/**
 * Qwen3MLP.swift
 * Feed-Forward Network with SwiGLU activation for Qwen3
 *
 * Same architecture as Mistral MLP (SwiGLU)
 */

import Foundation
import MLX
import MLXNN

/// Qwen3 MLP with SwiGLU activation (gate * silu(up))
/// Same structure as Mistral MLP
public class Qwen3MLP: Module {
    let config: Qwen3TextConfig

    @ModuleInfo public var gate_proj: Linear
    @ModuleInfo public var up_proj: Linear
    @ModuleInfo public var down_proj: Linear

    public init(config: Qwen3TextConfig) {
        self.config = config

        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        // SwiGLU: gate_proj and up_proj are separate
        // No bias in Qwen3 MLP (attentionBias controls all)
        self._gate_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false))
        self._up_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false))
        self._down_proj = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: false))

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gate_proj(x))
        let up = up_proj(x)
        return down_proj(gate * up)
    }
}
