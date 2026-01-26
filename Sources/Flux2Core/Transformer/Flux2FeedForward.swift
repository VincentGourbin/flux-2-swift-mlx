// Flux2FeedForward.swift - SwiGLU FeedForward Network
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// GEGLU activation: GELU-gated Linear Unit
/// Used in some transformer variants
/// Optimized with kernel fusion for the gating operation
public class GEGLU: Module, @unchecked Sendable {
    let proj: Linear
    let dim: Int
    let innerDim: Int

    /// Compiled gating function for kernel fusion
    nonisolated(unsafe) private static let compiledGate: (MLXArray, MLXArray) -> MLXArray = compile { gate, value in
        gelu(gate) * value
    }

    public init(dim: Int, innerDim: Int, bias: Bool = false) {
        self.dim = dim
        self.innerDim = innerDim
        // Projects to 2x inner_dim for gate and value
        self.proj = Linear(dim, innerDim * 2, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(x)
        // Split into gate and value
        let chunks = split(projected, parts: 2, axis: -1)
        let gate = chunks[0]
        let value = chunks[1]
        // GEGLU: gelu(gate) * value - compiled for kernel fusion
        return Self.compiledGate(gate, value)
    }
}

/// SwiGLU activation: Swish-gated Linear Unit
/// Used in Flux.2 transformer feedforward blocks
/// Optimized with kernel fusion for the gating operation
public class SwiGLU: Module, @unchecked Sendable {
    let proj: Linear
    let dim: Int
    let innerDim: Int

    /// Compiled gating function for kernel fusion
    nonisolated(unsafe) private static let compiledGate: (MLXArray, MLXArray) -> MLXArray = compile { gate, value in
        silu(gate) * value
    }

    public init(dim: Int, innerDim: Int, bias: Bool = false) {
        self.dim = dim
        self.innerDim = innerDim
        // Projects to 2x inner_dim for gate and value
        self.proj = Linear(dim, innerDim * 2, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(x)
        // Split into gate and value
        let chunks = split(projected, parts: 2, axis: -1)
        let gate = chunks[0]
        let value = chunks[1]
        // SwiGLU: silu(gate) * value (silu = swish) - compiled for kernel fusion
        return Self.compiledGate(gate, value)
    }
}

/// FeedForward network for Flux.2 transformer blocks
///
/// Architecture: Linear -> SwiGLU -> Linear
/// - Input: [B, S, dim]
/// - Output: [B, S, dim]
public class Flux2FeedForward: Module, @unchecked Sendable {
    let dim: Int
    let innerDim: Int
    let activation: SwiGLU
    let linearOut: Linear

    /// Initialize FeedForward block
    /// - Parameters:
    ///   - dim: Model dimension (6144 for Flux.2)
    ///   - innerDim: Inner dimension (typically 4 * dim for standard FFN)
    ///   - bias: Whether to use bias in linear layers (default false to match checkpoint)
    public init(
        dim: Int,
        innerDim: Int? = nil,
        bias: Bool = false
    ) {
        self.dim = dim
        // Default inner_dim is 4 * dim, but Flux.2 may use different ratio
        self.innerDim = innerDim ?? (dim * 4)

        // SwiGLU combines input projection and gating
        self.activation = SwiGLU(dim: dim, innerDim: self.innerDim, bias: bias)

        // Output projection
        self.linearOut = Linear(self.innerDim, dim, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: projects to 2*innerDim, applies silu gate, outputs innerDim
        var hidden = activation(x)
        // Project back to dim
        hidden = linearOut(hidden)
        return hidden
    }
}

/// FeedForward variant with separate gate and up projections
/// (Alternative implementation matching some model architectures)
public class Flux2FeedForwardSplit: Module, @unchecked Sendable {
    let dim: Int
    let innerDim: Int
    let linearGate: Linear
    let linearUp: Linear
    let linearDown: Linear

    public init(
        dim: Int,
        innerDim: Int? = nil,
        bias: Bool = false
    ) {
        self.dim = dim
        self.innerDim = innerDim ?? (dim * 4)

        self.linearGate = Linear(dim, self.innerDim, bias: bias)
        self.linearUp = Linear(dim, self.innerDim, bias: bias)
        self.linearDown = Linear(self.innerDim, dim, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU with separate projections
        let gate = silu(linearGate(x))
        let up = linearUp(x)
        let hidden = gate * up
        return linearDown(hidden)
    }
}

/// Context-specific FeedForward for text encoder hidden states
/// Used in double-stream blocks for processing text separately
public class Flux2ContextFeedForward: Module, @unchecked Sendable {
    let feedForward: Flux2FeedForward

    public init(dim: Int, innerDim: Int? = nil, bias: Bool = false) {
        self.feedForward = Flux2FeedForward(dim: dim, innerDim: innerDim, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        feedForward(x)
    }
}
