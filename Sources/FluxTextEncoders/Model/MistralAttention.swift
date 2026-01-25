/**
 * MistralAttention.swift
 * Multi-Head Attention with Grouped Query Attention (GQA) and RoPE
 */

import Foundation
import MLX
import MLXNN

// MARK: - Llama-4 Attention Scaling

/// Llama-4 attention scaling for long sequences
/// Formula: 1 + beta * log(1 + floor(position / max_position_embeddings))
/// This helps the model handle very long sequences by scaling queries based on position
func getLlama4AttentionScale(
    start: Int,
    stop: Int,
    beta: Float,
    maxPositionEmbeddings: Int
) -> MLXArray {
    // Create position array: [start, start+1, ..., stop-1]
    let positions = MLXArray(Array(start..<stop).map { Float($0) })
    let maxPos = Float(maxPositionEmbeddings)

    // scaling = 1 + beta * log(1 + floor(pos / max_pos))
    let floored = MLX.floor(positions / maxPos)
    let scaling = 1.0 + beta * MLX.log(1.0 + floored)

    // Reshape to [seq_len, 1] for broadcasting
    return scaling.reshaped([stop - start, 1])
}

// MARK: - Scaled Dot-Product Attention

/// Manual scaled dot-product attention implementation
func scaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXArray? = nil
) -> MLXArray {
    // queries, keys, values: [batch, heads, seq, head_dim]
    // Compute attention scores: Q @ K^T
    let scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * MLXArray([scale])

    // Apply mask if provided
    var maskedScores = scores
    if let mask = mask {
        maskedScores = scores + mask
    }

    // Softmax over last dimension
    let weights = softmax(maskedScores, axis: -1)

    // Apply attention to values
    return matmul(weights, values)
}

/// Quantization configuration for KV cache
public struct KVCacheQuantizationConfig: Sendable {
    /// Number of bits for quantization (4 or 8)
    public let bits: Int
    /// Group size for quantization
    public let groupSize: Int
    /// Threshold in tokens after which to quantize the cache
    public let quantizeThreshold: Int

    /// 8-bit quantization (less aggressive, better quality)
    public static let bits8 = KVCacheQuantizationConfig(bits: 8, groupSize: 64, quantizeThreshold: 1024)

    /// 4-bit quantization (more aggressive, 75% memory reduction)
    public static let bits4 = KVCacheQuantizationConfig(bits: 4, groupSize: 64, quantizeThreshold: 2048)

    /// No quantization (default)
    public static let none = KVCacheQuantizationConfig(bits: 0, groupSize: 0, quantizeThreshold: Int.max)

    public init(bits: Int, groupSize: Int, quantizeThreshold: Int) {
        self.bits = bits
        self.groupSize = groupSize
        self.quantizeThreshold = quantizeThreshold
    }
}

/// Optimized KV Cache with chunk pre-allocation for efficient generation
/// Pre-allocates memory in 256-token chunks to reduce memory fragmentation
/// Pattern from mlx-swift-lm: avoids repeated concatenation allocations
/// Supports optional quantization for long sequences (>1K tokens)
public class KVCache {
    /// Pre-allocated key buffer (full precision or dequantized)
    private var keysBuffer: MLXArray?
    /// Pre-allocated value buffer (full precision or dequantized)
    private var valuesBuffer: MLXArray?
    /// Current number of tokens stored in cache
    private var currentLength: Int = 0
    /// Total allocated capacity (in tokens)
    private var allocatedCapacity: Int = 0

    // MARK: - Quantization State

    /// Quantized keys storage
    private var quantizedKeys: MLXArray?
    private var keysScales: MLXArray?
    private var keysBiases: MLXArray?

    /// Quantized values storage
    private var quantizedValues: MLXArray?
    private var valuesScales: MLXArray?
    private var valuesBiases: MLXArray?

    /// Whether the cache is currently quantized
    private var isQuantized: Bool = false

    /// Quantization configuration
    public var quantizationConfig: KVCacheQuantizationConfig = .none

    /// Chunk size for pre-allocation (reduces allocation frequency)
    /// 256 tokens = good balance between memory waste and allocation frequency
    public static let chunkSize: Int = 256

    public init(quantizationConfig: KVCacheQuantizationConfig = .none) {
        self.quantizationConfig = quantizationConfig
    }

    /// Update cache with new keys and values using optimized chunked allocation
    /// Uses in-place writes when possible, only allocates when capacity exceeded
    /// Automatically quantizes when threshold is reached
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let newTokens = newKeys.dim(2)
        let requiredCapacity = currentLength + newTokens

        // If quantized, dequantize first for the update
        if isQuantized {
            dequantizeCache()
        }

        // Check if we need to expand the buffer
        if keysBuffer == nil || requiredCapacity > allocatedCapacity {
            expandBuffer(newKeys: newKeys, newValues: newValues, requiredCapacity: requiredCapacity)
        } else {
            // Append new data to existing buffer
            let validKeys = keysBuffer![0..., 0..., 0..<currentLength, 0...]
            let validValues = valuesBuffer![0..., 0..., 0..<currentLength, 0...]

            keysBuffer = concatenated([validKeys, newKeys], axis: 2)
            valuesBuffer = concatenated([validValues, newValues], axis: 2)
        }

        currentLength = requiredCapacity

        // Check if we should quantize (threshold reached and not already quantized)
        if !isQuantized &&
           quantizationConfig.bits > 0 &&
           currentLength >= quantizationConfig.quantizeThreshold {
            quantizeCache()
        }

        // Return only the valid portion of the cache
        let validKeys = keysBuffer![0..., 0..., 0..<currentLength, 0...]
        let validValues = valuesBuffer![0..., 0..., 0..<currentLength, 0...]

        return (validKeys, validValues)
    }

    /// Expand buffer to accommodate more tokens
    private func expandBuffer(newKeys: MLXArray, newValues: MLXArray, requiredCapacity: Int) {
        if let existingKeys = keysBuffer, let existingValues = valuesBuffer, currentLength > 0 {
            let validKeys = existingKeys[0..., 0..., 0..<currentLength, 0...]
            let validValues = existingValues[0..., 0..., 0..<currentLength, 0...]

            keysBuffer = concatenated([validKeys, newKeys], axis: 2)
            valuesBuffer = concatenated([validValues, newValues], axis: 2)
        } else {
            keysBuffer = newKeys
            valuesBuffer = newValues
        }

        allocatedCapacity = keysBuffer!.dim(2)
    }

    // MARK: - Quantization Methods

    /// Quantize the cache to reduce memory usage
    /// Call this after reaching a certain sequence length for long-context generation
    public func quantizeCache() {
        guard let keys = keysBuffer, let values = valuesBuffer, currentLength > 0 else { return }
        guard !isQuantized else { return }
        guard quantizationConfig.bits > 0 else { return }

        // Get valid portion
        let validKeys = keys[0..., 0..., 0..<currentLength, 0...]
        let validValues = values[0..., 0..., 0..<currentLength, 0...]

        // Reshape from [B, H, S, D] to [B*H*S, D] for quantization
        let shape = validKeys.shape
        let batchSize = shape[0]
        let numHeads = shape[1]
        let seqLen = shape[2]
        let headDim = shape[3]

        // Ensure headDim is divisible by groupSize (pad if needed)
        let groupSize = quantizationConfig.groupSize
        let paddedHeadDim = ((headDim + groupSize - 1) / groupSize) * groupSize

        if headDim == paddedHeadDim && headDim % 32 == 0 {
            // Reshape to 2D: [B*H*S, D]
            let keysFlat = validKeys.reshaped([batchSize * numHeads * seqLen, headDim])
            let valuesFlat = validValues.reshaped([batchSize * numHeads * seqLen, headDim])

            // Quantize
            let (qk, sk, bk) = MLX.quantized(keysFlat, groupSize: groupSize, bits: quantizationConfig.bits)
            let (qv, sv, bv) = MLX.quantized(valuesFlat, groupSize: groupSize, bits: quantizationConfig.bits)

            // Store quantized data
            quantizedKeys = qk
            keysScales = sk
            keysBiases = bk

            quantizedValues = qv
            valuesScales = sv
            valuesBiases = bv

            isQuantized = true

            // Clear full-precision buffers to save memory
            keysBuffer = nil
            valuesBuffer = nil
        }
        // If dimensions don't work for quantization, keep full precision
    }

    /// Dequantize the cache back to full precision
    private func dequantizeCache() {
        guard isQuantized else { return }
        guard let qk = quantizedKeys, let sk = keysScales,
              let qv = quantizedValues, let sv = valuesScales else { return }

        // Dequantize
        let keysFlat = MLX.dequantized(
            qk, scales: sk, biases: keysBiases,
            groupSize: quantizationConfig.groupSize,
            bits: quantizationConfig.bits
        )
        let valuesFlat = MLX.dequantized(
            qv, scales: sv, biases: valuesBiases,
            groupSize: quantizationConfig.groupSize,
            bits: quantizationConfig.bits
        )

        // Store dequantized data back to buffers
        keysBuffer = keysFlat
        valuesBuffer = valuesFlat

        // Clear quantized data
        quantizedKeys = nil
        keysScales = nil
        keysBiases = nil
        quantizedValues = nil
        valuesScales = nil
        valuesBiases = nil

        isQuantized = false
    }

    /// Whether the cache is currently in quantized state
    public var quantized: Bool {
        return isQuantized
    }

    /// Current length of cached sequence (number of valid tokens)
    public var length: Int {
        return currentLength
    }

    /// Offset for position encoding (alias for length)
    public var offset: Int {
        return currentLength
    }

    /// Total allocated capacity
    public var capacity: Int {
        return allocatedCapacity
    }

    /// Estimated memory usage in bytes
    public var estimatedMemoryBytes: Int {
        if isQuantized {
            // Quantized: bits per element
            let elementsPerArray = currentLength * (quantizedKeys?.dim(1) ?? 128)
            let bitsPerElement = quantizationConfig.bits
            return (elementsPerArray * bitsPerElement / 8) * 2  // keys + values
        } else {
            // Full precision: 4 bytes per float32
            guard let keys = keysBuffer else { return 0 }
            let elementsPerArray = keys.shape.reduce(1, *)
            return elementsPerArray * 4 * 2  // keys + values, float32
        }
    }

    /// Clear the cache to free memory
    public func clear() {
        keysBuffer = nil
        valuesBuffer = nil
        quantizedKeys = nil
        keysScales = nil
        keysBiases = nil
        quantizedValues = nil
        valuesScales = nil
        valuesBiases = nil
        currentLength = 0
        allocatedCapacity = 0
        isQuantized = false
    }

    /// Trim cache to release unused memory
    public func trim() {
        if isQuantized {
            // Can't easily trim quantized cache
            return
        }

        guard let keys = keysBuffer, let values = valuesBuffer, currentLength > 0 else {
            clear()
            return
        }

        if currentLength < allocatedCapacity {
            keysBuffer = keys[0..., 0..., 0..<currentLength, 0...]
            valuesBuffer = values[0..., 0..., 0..<currentLength, 0...]
            allocatedCapacity = currentLength
        }
    }
}

/// Rotary Position Embedding - wraps MLXFast.RoPE for optimal performance
public class MistralRoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    public init(dimensions: Int, traditional: Bool = false, base: Float = 10000.0, scale: Float = 1.0) {
        self.dimensions = dimensions
        self.traditional = traditional
        self.base = base
        self.scale = scale
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Use MLXFast.RoPE directly without reshaping (fix for mlx-swift 0.30.2)
        // PR #316: reshaping before RoPE causes incorrect behavior with non-zero offsets
        return MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
            offset: offset)
    }
}

/// Mistral Attention with Grouped Query Attention (GQA)
public class MistralAttention: Module {
    let config: MistralTextConfig
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo public var q_proj: Linear
    @ModuleInfo public var k_proj: Linear
    @ModuleInfo public var v_proj: Linear
    @ModuleInfo public var o_proj: Linear
    public var rope: MistralRoPE

    public init(config: MistralTextConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        // Projections
        self._q_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numHeads * headDim, bias: config.attentionBias))
        self._k_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        self._v_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        self._o_proj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, hiddenSize, bias: config.attentionBias))

        // RoPE - using MLXFast.RoPE for numerical consistency with Python
        self.rope = MistralRoPE(dimensions: headDim, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]

        // Project Q, K, V
        var queries = q_proj(hiddenStates)
        var keys = k_proj(hiddenStates)
        var values = v_proj(hiddenStates)

        // Reshape for multi-head attention
        queries = queries.reshaped([batchSize, seqLen, numHeads, headDim])
        keys = keys.reshaped([batchSize, seqLen, numKVHeads, headDim])
        values = values.reshaped([batchSize, seqLen, numKVHeads, headDim])

        // Transpose to [batch, heads, seq, head_dim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE
        let offset = cache?.length ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Apply Llama-4 attention scaling to queries (CRITICAL for Ministral3!)
        // This scales queries based on position to handle long sequences
        let attnScale = getLlama4AttentionScale(
            start: offset,
            stop: offset + seqLen,
            beta: config.llama4ScalingBeta,
            maxPositionEmbeddings: config.originalMaxPositionEmbeddings
        )
        // Reshape for broadcasting: [seq_len, 1] -> [1, 1, seq_len, 1]
        queries = queries * attnScale.reshaped([1, 1, seqLen, 1])

        // Update KV cache if provided
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // GQA: Use broadcasting instead of explicit repeat when possible
        // MLXFast.scaledDotProductAttention supports GQA broadcasting natively
        // when num_kv_heads divides num_heads evenly
        // Shape: queries [B, num_heads, S, D], keys/values [B, num_kv_heads, S, D]
        let repeatFactor = numHeads / numKVHeads

        // For GQA, we need to expand KV for the attention computation
        // Use efficient tile-based expansion that's more memory efficient than repeat
        if repeatFactor > 1 {
            // Reshape KV from [B, kv_heads, S, D] to [B, kv_heads, 1, S, D]
            // Then broadcast to [B, kv_heads, repeat, S, D]
            // Finally reshape to [B, num_heads, S, D]
            let kvSeqLen = keys.dim(2)
            keys = keys.expandedDimensions(axis: 2)
            keys = MLX.broadcast(keys, to: [batchSize, numKVHeads, repeatFactor, kvSeqLen, headDim])
            keys = keys.reshaped([batchSize, numHeads, kvSeqLen, headDim])

            values = values.expandedDimensions(axis: 2)
            values = MLX.broadcast(values, to: [batchSize, numKVHeads, repeatFactor, kvSeqLen, headDim])
            values = values.reshaped([batchSize, numHeads, kvSeqLen, headDim])
        }

        // Scaled dot-product attention using MLXFast for optimal performance
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let outputTransposed = output.transposed(0, 2, 1, 3)
        let outputReshaped = outputTransposed.reshaped([batchSize, seqLen, numHeads * headDim])

        return o_proj(outputReshaped)
    }
}
