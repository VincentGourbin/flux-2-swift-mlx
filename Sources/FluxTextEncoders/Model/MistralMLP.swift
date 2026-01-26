/**
 * MistralMLP.swift
 * Feed-Forward Network with SwiGLU activation for Mistral
 * Optimized with kernel fusion for the gating operation
 */

import Foundation
import MLX
import MLXNN

/// Mistral MLP with SwiGLU activation (gate * silu(up))
/// Optimized with kernel fusion for better performance
public class MistralMLP: Module, @unchecked Sendable {
    let config: MistralTextConfig

    @ModuleInfo public var gate_proj: Linear
    @ModuleInfo public var up_proj: Linear
    @ModuleInfo public var down_proj: Linear

    /// Compiled gating function for kernel fusion
    nonisolated(unsafe) private static let compiledGate: (MLXArray, MLXArray) -> MLXArray = compile { gate, up in
        silu(gate) * up
    }

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
        // SwiGLU: down(silu(gate(x)) * up(x)) - compiled for kernel fusion
        let gateOut = gate_proj(x)
        let upOut = up_proj(x)
        return down_proj(Self.compiledGate(gateOut, upOut))
    }
}
