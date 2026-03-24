/**
 * Qwen3VLMLP.swift
 * Feed-Forward Network with SwiGLU activation for Qwen3-VL
 *
 * Same architecture as Qwen3MLP but uses Qwen3VLTextConfig
 * (intermediate_size=9728 vs 9216 for Qwen3-4B)
 */

import Foundation
import MLX
import MLXNN

/// Qwen3-VL MLP with SwiGLU activation
public class Qwen3VLMLP: Module, @unchecked Sendable {
    let config: Qwen3VLTextConfig

    @ModuleInfo public var gate_proj: Linear
    @ModuleInfo public var up_proj: Linear
    @ModuleInfo public var down_proj: Linear

    /// Compiled gating function for kernel fusion
    nonisolated(unsafe) private static let compiledGate: (MLXArray, MLXArray) -> MLXArray = compile { gate, up in
        silu(gate) * up
    }

    public init(config: Qwen3VLTextConfig) {
        self.config = config

        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gate_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false))
        self._up_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false))
        self._down_proj = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: false))

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateOut = gate_proj(x)
        let upOut = up_proj(x)
        return down_proj(Self.compiledGate(gateOut, upOut))
    }
}
