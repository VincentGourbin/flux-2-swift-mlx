/**
 * Qwen3MLP.swift
 * Feed-Forward Network with SwiGLU activation for Qwen3
 * Optimized with kernel fusion for the gating operation
 *
 * Same architecture as Mistral MLP (SwiGLU)
 */

import Foundation
import MLX
import MLXNN

/// Qwen3 MLP with SwiGLU activation (gate * silu(up))
/// Same structure as Mistral MLP, optimized with kernel fusion
public class Qwen3MLP: Module, @unchecked Sendable {
    let config: Qwen3TextConfig

    @ModuleInfo public var gate_proj: Linear
    @ModuleInfo public var up_proj: Linear
    @ModuleInfo public var down_proj: Linear

    /// Compiled gating function for kernel fusion
    nonisolated(unsafe) private static let compiledGate: (MLXArray, MLXArray) -> MLXArray = compile { gate, up in
        silu(gate) * up
    }

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
        // SwiGLU: down(silu(gate(x)) * up(x)) - compiled for kernel fusion
        let gateOut = gate_proj(x)
        let upOut = up_proj(x)
        return down_proj(Self.compiledGate(gateOut, upOut))
    }
}
