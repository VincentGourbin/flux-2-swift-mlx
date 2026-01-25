/**
 * MistralMLP.swift
 * Feed-Forward Network with SwiGLU activation for Mistral
 */

import Foundation
import MLX
import MLXNN

/// Mistral MLP with SwiGLU activation (gate * silu(up))
public class MistralMLP: Module {
    let config: MistralTextConfig

    @ModuleInfo public var gate_proj: Linear
    @ModuleInfo public var up_proj: Linear
    @ModuleInfo public var down_proj: Linear

    public init(config: MistralTextConfig) {
        self.config = config

        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        // SwiGLU: gate_proj and up_proj are separate
        self._gate_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: config.mlpBias))
        self._up_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: config.mlpBias))
        self._down_proj = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: config.mlpBias))

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gate_proj(x))
        let up = up_proj(x)
        return down_proj(gate * up)
    }
}
