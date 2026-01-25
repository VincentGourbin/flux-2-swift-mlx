/**
 * VisionEncoder.swift
 * Pixtral Vision Encoder for Mistral Small 3.2
 * 24-layer transformer with 2D RoPE positional encoding
 */

import Foundation
import MLX
import MLXNN
import MLXFast

/// Debug flag - set VLM_DEBUG=1 to enable verbose logging
private let visionDebug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

/// Configuration for the Pixtral vision encoder
public struct PixtralVisionConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let imageSize: Int
    public let numChannels: Int
    public let hiddenAct: String
    public let ropeTheta: Float
    public let attentionDropout: Float
    public let normEps: Float

    public static let mistralSmall = PixtralVisionConfig(
        hiddenSize: 1024,
        intermediateSize: 4096,
        numHiddenLayers: 24,
        numAttentionHeads: 16,
        headDim: 64,
        patchSize: 14,
        imageSize: 1540,
        numChannels: 3,
        hiddenAct: "silu",
        ropeTheta: 10000.0,
        attentionDropout: 0.0,
        normEps: 1e-5
    )

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        headDim: Int,
        patchSize: Int,
        imageSize: Int,
        numChannels: Int,
        hiddenAct: String,
        ropeTheta: Float,
        attentionDropout: Float,
        normEps: Float = 1e-5
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.headDim = headDim
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.numChannels = numChannels
        self.hiddenAct = hiddenAct
        self.ropeTheta = ropeTheta
        self.attentionDropout = attentionDropout
        self.normEps = normEps
    }
}

// MARK: - 2D RoPE for Vision

/// Pixtral 2D Rotary Position Embedding
/// Matches Python's PixtralRotaryEmbedding which uses different frequencies for H and W
/// The embedding is precomputed for all positions up to maxPatchesPerSide
fileprivate class PixtralRoPE {
    let dim: Int
    let invFreq: MLXArray  // [maxPatches^2, dim]

    init(headDim: Int, ropeTheta: Float, maxPatchesPerSide: Int) {
        let dim = headDim  // 64 - use local variable first
        self.dim = dim

        // Create base frequencies: shape [32]
        // freqs = 1.0 / (theta ** (arange(0, dim, 2) / dim))
        let freqSeq = stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) }
        let freqs = freqSeq.map { 1.0 / pow(ropeTheta, $0) }
        let freqsArray = MLXArray(freqs)  // [32]

        // Split into even and odd indexed frequencies
        // freqs_h uses even indices [0, 2, 4, ...] -> freqs[::2]
        // freqs_w uses odd indices [1, 3, 5, ...] -> freqs[1::2]
        let freqsH = freqsArray[stride(from: 0, to: freqs.count, by: 2)]  // [16]
        let freqsW = freqsArray[stride(from: 1, to: freqs.count, by: 2)]  // [16]

        // Create position grids
        let h = MLXArray(Array(0..<maxPatchesPerSide).map { Float($0) })  // [110]
        let w = MLXArray(Array(0..<maxPatchesPerSide).map { Float($0) })  // [110]

        // Compute outer products
        // freqs_h = outer(h, freqs_h) -> [110, 16]
        // freqs_w = outer(w, freqs_w) -> [110, 16]
        let freqsHOuter = h.expandedDimensions(axis: 1) * freqsH  // [110, 16]
        let freqsWOuter = w.expandedDimensions(axis: 1) * freqsW  // [110, 16]

        // Tile to create 2D grid
        // freqs_h[:, None, :] tiled to [110, 110, 16]
        // freqs_w[None, :, :] tiled to [110, 110, 16]
        let freqsHTiled = MLX.tiled(freqsHOuter.expandedDimensions(axis: 1), repetitions: [1, maxPatchesPerSide, 1])
        let freqsWTiled = MLX.tiled(freqsWOuter.expandedDimensions(axis: 0), repetitions: [maxPatchesPerSide, 1, 1])

        // Concatenate and reshape
        // inv_freq = concatenate([freqs_h, freqs_w], axis=-1) -> [110, 110, 32]
        // inv_freq = inv_freq.reshape(-1, dim // 2) -> [12100, 32]
        let combined = MLX.concatenated([freqsHTiled, freqsWTiled], axis: -1)  // [110, 110, 32]
        let reshapedCombined = combined.reshaped([maxPatchesPerSide * maxPatchesPerSide, dim / 2])  // [12100, 32]

        // Duplicate to get full dim
        // inv_freq = concatenate((inv_freq, inv_freq), axis=-1) -> [12100, 64]
        self.invFreq = MLX.concatenated([reshapedCombined, reshapedCombined], axis: -1)  // [12100, 64]
    }

    /// Get cos and sin for given position IDs
    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        // positionIds shape: [seq_len]
        // inv_freq[position_ids] -> [seq_len, dim]
        let freqs = invFreq[positionIds]  // [seq_len, 64]
        let cos = MLX.cos(freqs).asType(x.dtype)  // [seq_len, 64]
        let sin = MLX.sin(freqs).asType(x.dtype)  // [seq_len, 64]
        return (cos, sin)
    }
}

/// Rotate half implementation matching Python's rotate_half
/// x[..., :half] -> x1, x[..., half:] -> x2
/// return concatenate([-x2, x1], axis=-1)
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    // Split along the last dimension into two equal halves
    let parts = x.split(parts: 2, axis: -1)
    let x1 = parts[0]
    let x2 = parts[1]
    return MLX.concatenated([-x2, x1], axis: -1)
}

/// Apply rotary position embedding to Q and K
/// Matches Python's apply_rotary_pos_emb
private func applyRotaryPosEmb(
    _ q: MLXArray,
    _ k: MLXArray,
    cos: MLXArray,
    sin: MLXArray,
    unsqueezeDim: Int = 1
) -> (MLXArray, MLXArray) {
    // cos/sin shape: [seq_len, dim]
    // Expand to [1, seq_len, dim] for unsqueezeDim=1 or [seq_len, 1, dim] for unsqueezeDim=0
    let cosExpanded = cos.expandedDimensions(axis: unsqueezeDim)
    let sinExpanded = sin.expandedDimensions(axis: unsqueezeDim)

    // q_embed = q * cos + rotate_half(q) * sin
    let qEmbed = q * cosExpanded + rotateHalf(q) * sinExpanded
    let kEmbed = k * cosExpanded + rotateHalf(k) * sinExpanded

    return (qEmbed, kEmbed)
}

/// Generate position IDs using meshgrid like Python's position_ids_in_meshgrid
/// GPU-based implementation (replaces O(height * width) CPU nested loop)
private func positionIdsInMeshgrid(height: Int, width: Int, maxWidth: Int) -> MLXArray {
    // h_grid, v_grid = meshgrid(arange(height), arange(width), indexing="ij")
    // ids = h_grid * max_width + v_grid

    // Row indices: [height, 1]
    let h = MLXArray(Array(0..<height).map { Int32($0) }).expandedDimensions(axis: 1)
    // Column indices: [1, width]
    let w = MLXArray(Array(0..<width).map { Int32($0) }).expandedDimensions(axis: 0)

    // Broadcasting: h * maxWidth + w -> [height, width]
    let positions = h * Int32(maxWidth) + w

    // Flatten to [height * width]
    return positions.flattened()
}

// MARK: - Pixtral Attention

/// Multi-head attention with 2D RoPE for vision
/// Matches Python's mlx_vlm.models.pixtral.vision.Attention
public class PixtralAttention: Module, UnaryLayer {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(config: PixtralVisionConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)  // head_dim ** -0.5

        let hiddenSize = config.hiddenSize

        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("Use forwardWithPositions instead")
    }

    /// Forward with position embeddings
    /// - Parameters:
    ///   - queries: Input tensor [B, L, D]
    ///   - keys: Input tensor [B, S, D]
    ///   - values: Input tensor [B, S, D]
    ///   - positionEmbeddings: (cos, sin) each of shape [seq_len, head_dim]
    ///   - mask: Optional attention mask [B, 1, L, S]
    public func forward(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil
    ) -> MLXArray {
        // Project Q, K, V
        var q = qProj(queries)
        var k = kProj(keys)
        let v = vProj(values)

        let (batchSize, seqLen, _) = (q.shape[0], q.shape[1], q.shape[2])
        let (_, srcLen, _) = (k.shape[0], k.shape[1], k.shape[2])

        // Reshape: [B, L, D] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        q = q.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        k = k.reshaped([batchSize, srcLen, numHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        let vT = v.reshaped([batchSize, srcLen, numHeads, headDim]).transposed(axes: [0, 2, 1, 3])

        // Apply rotary position embedding
        // Note: Python uses unsqueeze_dim=0, meaning cos/sin are expanded at axis 0
        (q, k) = applyRotaryPosEmb(q, k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin, unsqueezeDim: 0)

        // Scaled dot-product attention: [B, H, L, D] @ [B, H, D, S] -> [B, H, L, S]
        var attnWeights = MLX.matmul(q, k.transposed(axes: [0, 1, 3, 2])) * scale

        // Apply mask if provided
        if let m = mask {
            attnWeights = attnWeights + m
        }

        attnWeights = softmax(attnWeights, axis: -1)

        // [B, H, L, S] @ [B, H, S, D] -> [B, H, L, D]
        var output = MLX.matmul(attnWeights, vT)

        // Reshape back: [B, H, L, D] -> [B, L, H, D] -> [B, L, H*D]
        output = output.transposed(axes: [0, 2, 1, 3]).reshaped([batchSize, seqLen, numHeads * headDim])

        return oProj(output)
    }

    // Legacy method for backward compatibility
    public func forwardWithPositions(_ x: MLXArray, positions: (row: MLXArray, col: MLXArray)?) -> MLXArray {
        fatalError("Use forward(queries:keys:values:positionEmbeddings:mask:) instead")
    }
}

// MARK: - Pixtral MLP

/// SiLU-gated MLP for vision encoder
public class PixtralMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(config: PixtralVisionConfig) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SiLU-gated: down(silu(gate(x)) * up(x))
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

// MARK: - Pixtral Transformer Layer

/// Single transformer layer for vision encoder
public class PixtralTransformerLayer: Module, UnaryLayer {
    @ModuleInfo var attention: PixtralAttention
    @ModuleInfo(key: "feed_forward") var feedForward: PixtralMLP
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    public init(config: PixtralVisionConfig) {
        self._attention.wrappedValue = PixtralAttention(config: config)
        self._feedForward.wrappedValue = PixtralMLP(config: config)
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("Use forward(x:positionEmbeddings:mask:) instead")
    }

    /// Forward with position embeddings (new API matching Python)
    public func forward(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil
    ) -> MLXArray {
        // Pre-norm architecture
        let normed = attentionNorm(x)
        let attnOut = attention.forward(
            queries: normed,
            keys: normed,
            values: normed,
            positionEmbeddings: positionEmbeddings,
            mask: mask
        )
        var h = x + attnOut
        h = h + feedForward(ffnNorm(h))
        return h
    }
}

// MARK: - Vision Model (Wrapper)

/// Wrapper class to match the weight key structure: vision_tower.vision_model.*
public class VisionModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: VisionEncoder

    public init(config: PixtralVisionConfig = .mistralSmall) {
        self._visionModel.wrappedValue = VisionEncoder(config: config)
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        return visionModel(pixelValues)
    }

    public func getOutputInfo(imageHeight: Int, imageWidth: Int) -> (numPatches: Int, patchesH: Int, patchesW: Int) {
        return visionModel.getOutputInfo(imageHeight: imageHeight, imageWidth: imageWidth)
    }
}

// MARK: - Vision Transformer Wrapper

/// Wrapper class to match the safetensors key structure: transformer.layers.*
public class VisionTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [PixtralTransformerLayer]

    public init(config: PixtralVisionConfig) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            PixtralTransformerLayer(config: config)
        }
    }
}

// MARK: - Vision Encoder

/// Pixtral Vision Encoder - 24 layer transformer
public class VisionEncoder: Module {
    let config: PixtralVisionConfig
    fileprivate let rope: PixtralRoPE

    @ModuleInfo(key: "patch_conv") var patchConv: Conv2d
    @ModuleInfo(key: "ln_pre") var lnPre: RMSNorm
    @ModuleInfo(key: "transformer") var transformer: VisionTransformer

    public init(config: PixtralVisionConfig = .mistralSmall) {
        self.config = config

        // Create RoPE with max patches per side
        let maxPatchesPerSide = config.imageSize / config.patchSize  // 1540/14 = 110
        self.rope = PixtralRoPE(
            headDim: config.headDim,
            ropeTheta: config.ropeTheta,
            maxPatchesPerSide: maxPatchesPerSide
        )

        // Patch embedding via convolution
        // MLX Conv2d expects input in NHWC format: [batch, H, W, channels]
        // Output: [batch, H/patch, W/patch, hidden_size]
        let patchSize = config.patchSize
        self._patchConv.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init([patchSize, patchSize]),
            stride: .init([patchSize, patchSize]),
            bias: false
        )

        self._lnPre.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)

        // Create transformer with layers
        self._transformer.wrappedValue = VisionTransformer(config: config)
    }

    /// Forward pass through vision encoder
    /// - Parameter pixelValues: Image tensor [batch, H, W, 3] (NHWC format for MLX)
    /// - Returns: Encoded patches [batch, num_patches, hidden_size]
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        // 1. Patch embedding via convolution
        // MLX Conv2d expects NHWC: [batch, H, W, 3]
        // Output: [batch, H/patch, W/patch, hidden_size]
        var x = patchConv(pixelValues)

        let batchSize = x.shape[0]
        let patchH = x.shape[1]
        let patchW = x.shape[2]
        let hiddenSize = x.shape[3]
        let numPatches = patchH * patchW

        // Debug: After patch_conv
        if visionDebug {
            print("[VisionEnc] After patch_conv: shape=\(x.shape), mean=\(MLX.mean(x).item(Float.self)), std=\(MLX.std(x).item(Float.self))")
            fflush(stdout)
        }

        // 2. Flatten patches: [batch, H/p, W/p, hidden_size] -> [batch, num_patches, hidden_size]
        x = x.reshaped([batchSize, numPatches, hiddenSize])

        // 3. Pre-layer norm
        x = lnPre(x)

        // Debug: After ln_pre
        if visionDebug {
            print("[VisionEnc] After ln_pre: shape=\(x.shape), mean=\(MLX.mean(x).item(Float.self)), std=\(MLX.std(x).item(Float.self))")
            fflush(stdout)
        }

        // 4. Generate position IDs and get position embeddings
        let maxWidth = config.imageSize / config.patchSize
        let positionIds = positionIdsInMeshgrid(height: patchH, width: patchW, maxWidth: maxWidth)
        let positionEmbeddings = rope(x, positionIds: positionIds)

        // 5. Generate attention mask (optional, for single image it's all zeros)
        // For multi-image scenarios, we'd need a block diagonal mask
        // For now, no mask needed for single image
        let mask: MLXArray? = nil

        // 6. Pass through transformer layers
        for (i, layer) in transformer.layers.enumerated() {
            x = layer.forward(x, positionEmbeddings: positionEmbeddings, mask: mask)
            // Log every 5th layer and the last few to match Python trace
            if visionDebug && (i % 5 == 0 || i >= 21) {
                let std = MLX.std(x).item(Float.self)
                let maxAbs = MLX.max(MLX.abs(x)).item(Float.self)
                print("[VisionEnc] Layer \(String(format: "%2d", i)): std=\(String(format: "%.4f", std)), max_abs=\(String(format: "%.4f", maxAbs))")
                fflush(stdout)
            }
        }

        return x
    }

    /// Get output shape info for downstream processing
    public func getOutputInfo(imageHeight: Int, imageWidth: Int) -> (numPatches: Int, patchesH: Int, patchesW: Int) {
        let patchesH = imageHeight / config.patchSize
        let patchesW = imageWidth / config.patchSize
        return (patchesH * patchesW, patchesH, patchesW)
    }
}
