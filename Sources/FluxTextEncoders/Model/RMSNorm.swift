/**
 * RMSNorm.swift
 * Root Mean Square Layer Normalization for Mistral
 */

import Foundation
import MLX
import MLXNN

/// RMS Normalization layer used in Mistral models
public class RMSNorm: Module, UnaryLayer {
    var weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Manual RMSNorm implementation
        let variance = mean(x * x, axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + MLXArray([eps]))
        return weight * normalized
    }
}
