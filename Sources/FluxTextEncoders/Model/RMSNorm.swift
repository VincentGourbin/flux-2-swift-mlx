/**
 * RMSNorm.swift
 * Root Mean Square Layer Normalization for Mistral
 * Note: MLXFast.rmsNorm() causes type promotion issues with scaled_dot_product_attention,
 * so we use the manual implementation for now.
 */

import Foundation
import MLX
import MLXNN

/// RMS Normalization layer used in Mistral and Qwen3 models
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
        // Note: MLXFast.rmsNorm() causes bfloat16 type promotion issues with scaled_dot_product_attention
        // Note: Mixed precision (converting to Float32 and back) also causes type promotion issues
        let variance = mean(x * x, axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + MLXArray([eps]))
        return weight * normalized
    }
}
