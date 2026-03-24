/**
 * Qwen35MLP.swift
 * SwiGLU Feed-Forward Network for Qwen3.5
 */

import Foundation
import MLX
import MLXNN

public class Qwen35MLP: Module, @unchecked Sendable {
    @ModuleInfo public var gate_proj: Linear
    @ModuleInfo public var up_proj: Linear
    @ModuleInfo public var down_proj: Linear

    nonisolated(unsafe) private static let compiledGate: (MLXArray, MLXArray) -> MLXArray = compile { gate, up in
        silu(gate) * up
    }

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gate_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false))
        self._up_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: false))
        self._down_proj = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: false))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(Self.compiledGate(gate_proj(x), up_proj(x)))
    }
}
