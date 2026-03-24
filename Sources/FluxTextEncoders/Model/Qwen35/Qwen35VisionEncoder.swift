/**
 * Qwen35VisionEncoder.swift
 * Vision Transformer encoder for Qwen3.5 VLM
 *
 * 24-layer ViT with:
 * - Patch embedding via manual extraction + Linear (Conv3d weights reshaped)
 * - Learned + rotary 2D position embeddings
 * - Standard attention (combined QKV) with GELU MLP
 * - Spatial merger (2x2): norm(x) → reshape → fc1 → GELU → fc2
 *
 * Single-image mode: image is duplicated to 2 temporal frames to match
 * the Conv3d temporal_patch_size=2 kernel.
 */

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Vision Attention

public class Qwen35VisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(self.headDim), -0.5)

        self._qkv = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize * 3, bias: true))
        self._proj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true))
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rotaryPosEmb: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        let qkvOut = qkv(x).reshaped([B, S, 3, numHeads, headDim])
        var q = qkvOut[0..., 0..., 0, 0..., 0...]
        var k = qkvOut[0..., 0..., 1, 0..., 0...]
        let v = qkvOut[0..., 0..., 2, 0..., 0...]

        if let rotEmb = rotaryPosEmb {
            q = applyRotaryPosEmb(q, freqs: rotEmb)
            k = applyRotaryPosEmb(k, freqs: rotEmb)
        }

        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT, scale: scale, mask: nil)

        let output = attnOut.transposed(0, 2, 1, 3).reshaped([B, S, numHeads * headDim])
        return proj(output)
    }

    private func applyRotaryPosEmb(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let cos = MLX.cos(freqs)
        let sin = MLX.sin(freqs)
        let cosB = cos.reshaped([1, cos.dim(0), 1, cos.dim(1)])
        let sinB = sin.reshaped([1, sin.dim(0), 1, sin.dim(1)])

        let halfDim = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0..<halfDim]
        let x2 = x[.ellipsis, halfDim..<(halfDim * 2)]
        let rot1 = x1 * cosB - x2 * sinB
        let rot2 = x1 * sinB + x2 * cosB
        return concatenated([rot1, rot2], axis: -1)
    }
}

// MARK: - Vision MLP

public class Qwen35VisionMLP: Module {
    @ModuleInfo var linear_fc1: Linear
    @ModuleInfo var linear_fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._linear_fc1 = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: true))
        self._linear_fc2 = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: true))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear_fc2(gelu(linear_fc1(x)))
    }
}

// MARK: - Vision Block

public class Qwen35VisionBlock: Module {
    var attn: Qwen35VisionAttention
    var mlp: Qwen35VisionMLP
    var norm1: LayerNorm
    var norm2: LayerNorm

    init(config: Qwen35VisionConfig) {
        self.attn = Qwen35VisionAttention(hiddenSize: config.hiddenSize, numHeads: config.numHeads)
        self.mlp = Qwen35VisionMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self.norm1 = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self.norm2 = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, rotaryPosEmb: MLXArray? = nil) -> MLXArray {
        var h = x + attn(norm1(x), rotaryPosEmb: rotaryPosEmb)
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - Patch Embedding (Conv3d)

/// Patch embedding using native Conv3d
/// For single images: duplicate to 2 temporal frames, then Conv3d(kernel=[2,16,16], stride=[2,16,16])
/// Input: [B, H, W, C] → [B, 2, H, W, C] → Conv3d → [B, 1, gridH, gridW, 1024] → [B, numPatches, 1024]
public class Qwen35PatchEmbed: Module {
    let patchSize: Int
    let temporalPatchSize: Int
    let hiddenSize: Int

    @ModuleInfo public var proj: Conv3d

    init(config: Qwen35VisionConfig) {
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.hiddenSize = config.hiddenSize

        // Conv3d(inC=3, outC=1024, kernel=[2,16,16], stride=[2,16,16])
        // Weight shape: [1024, 2, 16, 16, 3] (MLXNN channels-last format)
        let ks: IntOrTriple = [temporalPatchSize, patchSize, patchSize]
        self._proj = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: config.inChannels,
            outputChannels: config.hiddenSize,
            kernelSize: ks,
            stride: ks,
            padding: 0,
            bias: true
        ))
        super.init()
    }

    /// Input: [B, H, W, C] (NHWC single image)
    /// Output: [B, gridH*gridW, hiddenSize] in spatial-merge order
    func callAsFunction(_ pixelValues: MLXArray, mergeSize: Int) -> MLXArray {
        let B = pixelValues.dim(0)
        let H = pixelValues.dim(1)
        let W = pixelValues.dim(2)

        // Duplicate to 2 temporal frames: [B, H, W, C] → [B, 2, H, W, C] (NDHWC)
        let frame = pixelValues.expandedDimensions(axis: 1)
        let temporal = concatenated([frame, frame], axis: 1)

        // Conv3d: [B, 2, H, W, 3] → [B, 1, gridH, gridW, 1024]
        var h = proj(temporal)

        let gridH = H / patchSize
        let gridW = W / patchSize

        // Reorder to spatial-merge order to match Python preprocessing:
        // [B, 1, gridH, gridW, D] → [B, gridH/m, m, gridW/m, m, D]
        // → permute(0, 1, 3, 2, 4, 5) → [B, gridH/m, gridW/m, m, m, D]
        // → flatten → [B, numPatches, D]
        h = h.squeezed(axis: 1)  // [B, gridH, gridW, D]
        h = h.reshaped([B, gridH / mergeSize, mergeSize, gridW / mergeSize, mergeSize, hiddenSize])
        h = h.transposed(0, 1, 3, 2, 4, 5)
        h = h.reshaped([B, gridH * gridW, hiddenSize])

        return h
    }
}

// MARK: - Patch Merger

/// Spatial merge: norm → reshape 2x2 blocks → fc1 → GELU → fc2
/// norm uses hiddenSize (1024), applied BEFORE spatial merge
/// Input: [B, numPatches, hiddenSize=1024]
/// Output: [B, numPatches/4, outHiddenSize=2560]
public class Qwen35PatchMerger: Module {
    let spatialMergeSize: Int
    let mergedDim: Int  // hiddenSize * merge² = 4096

    var norm: LayerNorm  // dim=1024 (applied before merge)
    @ModuleInfo var linear_fc1: Linear  // 4096 → 4096
    @ModuleInfo var linear_fc2: Linear  // 4096 → 2560

    init(config: Qwen35VisionConfig) {
        self.spatialMergeSize = config.spatialMergeSize
        self.mergedDim = config.hiddenSize * spatialMergeSize * spatialMergeSize

        // Norm on hiddenSize (1024), NOT mergedDim — matches model weights
        self.norm = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self._linear_fc1 = ModuleInfo(wrappedValue: Linear(mergedDim, mergedDim, bias: true))
        self._linear_fc2 = ModuleInfo(wrappedValue: Linear(mergedDim, config.outHiddenSize, bias: true))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // 1. Normalize BEFORE merge (on dim=1024)
        // Patches are already in merge order: [B, numPatches, D] where consecutive
        // groups of merge² patches belong to the same 2×2 spatial block
        var h = norm(x)

        // 2. Group consecutive merge² patches: [B, numPatches, D] → [B, numPatches/4, D*4]
        let B = h.dim(0)
        let numPatches = h.dim(1)
        let numMergedPatches = numPatches / (spatialMergeSize * spatialMergeSize)
        h = h.reshaped([B, numMergedPatches, mergedDim])

        // 3. Project: GELU(fc1) → fc2
        h = gelu(linear_fc1(h))
        h = linear_fc2(h)

        return h
    }
}

// MARK: - Vision Encoder

public class Qwen35VisionEncoder: Module {
    let config: Qwen35VisionConfig

    // Patch embedding (Conv3d emulated via Linear)
    public var patch_embed: Qwen35PatchEmbed

    // Learned position embeddings — loaded manually (not a Module parameter)
    public var posEmbedStorage: MLXArray

    // Transformer blocks
    public var blocks: [Qwen35VisionBlock]

    // Spatial merger (NO final norm — not in model weights)
    public var merger: Qwen35PatchMerger

    let rotaryDim: Int

    public init(config: Qwen35VisionConfig) {
        self.config = config

        self.patch_embed = Qwen35PatchEmbed(config: config)
        self.posEmbedStorage = MLXArray.zeros([config.numPositionEmbeddings, config.hiddenSize])

        self.blocks = (0..<config.depth).map { _ in
            Qwen35VisionBlock(config: config)
        }

        self.merger = Qwen35PatchMerger(config: config)
        self.rotaryDim = config.hiddenSize / config.numHeads

        super.init()
    }

    /// Encode an image to vision embeddings
    /// - Parameter pixelValues: [B, H, W, C] (NHWC format, normalized to [-1,1])
    /// - Returns: [B, numMergedPatches, outHiddenSize=2560]
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let imgH = pixelValues.dim(1)
        let imgW = pixelValues.dim(2)
        let gridH = imgH / config.patchSize
        let gridW = imgW / config.patchSize
        let numPatches = gridH * gridW

        let mergeSize = config.spatialMergeSize

        // 1. Patch embedding (output in spatial-merge order)
        var hidden = patch_embed(pixelValues, mergeSize: mergeSize)

        // 2. Add interpolated 2D position embeddings (also in merge order)
        let posEmb = interpolatePositionEmbeddings(gridH: gridH, gridW: gridW)
        eval(posEmb)
        print("[Swift-ViT] posEmb: \(posEmb.shape), mean=\(MLX.mean(posEmb).item(Float.self)), norm=\(MLX.sqrt(MLX.mean(posEmb*posEmb)).item(Float.self))")
        print("[Swift-ViT] posEmb first5: \(posEmb.reshaped([-1])[0..<5].asArray(Float.self))")
        hidden = hidden + posEmb.reshaped([1, -1, config.hiddenSize])

        // 3. Rotary 2D position embeddings (in merge order)
        let rotaryEmb = compute2DRotaryEmb(gridH: gridH, gridW: gridW, mergeSize: mergeSize)
        eval(rotaryEmb)
        print("[Swift-ViT] rotary: \(rotaryEmb.shape), mean=\(MLX.mean(rotaryEmb).item(Float.self))")
        print("[Swift-ViT] rotary patch[0] first8: \(rotaryEmb[0][0..<8].asArray(Float.self))")
        print("[Swift-ViT] rotary patch[1] full: \(rotaryEmb[1].asArray(Float.self))")
        print("[Swift-ViT] rotary patch[2] first8: \(rotaryEmb[2][0..<8].asArray(Float.self))")

        // 4. Transformer blocks
        for (i, block) in blocks.enumerated() {
            hidden = block(hidden, rotaryPosEmb: rotaryEmb)
            if i == 0 {
                eval(hidden)
                print("[Swift-ViT] after block 0: mean=\(MLX.mean(hidden).item(Float.self)), norm=\(MLX.sqrt(MLX.mean(hidden*hidden)).item(Float.self))")
            }
        }

        // 5. Spatial merger (patches already in merge order)
        hidden = merger(hidden)

        return hidden
    }

    public func numOutputTokens(imageHeight: Int, imageWidth: Int) -> Int {
        let gridH = imageHeight / config.patchSize
        let gridW = imageWidth / config.patchSize
        return (gridH / config.spatialMergeSize) * (gridW / config.spatialMergeSize)
    }

    // MARK: - Position Embedding Interpolation

    /// Bilinear interpolation of learned position embeddings from the 48×48 grid
    /// to the actual patch grid (gridH × gridW), then reorder to match spatial merge pattern.
    ///
    /// posEmbedStorage: [numGridPerSide², hiddenSize] = [2304, 1024] for a 48×48 grid
    /// Output: [gridH*gridW, hiddenSize] reordered for merger's 2×2 grouping
    private func interpolatePositionEmbeddings(gridH: Int, gridW: Int) -> MLXArray {
        let numGridPerSide = Int(sqrt(Float(config.numPositionEmbeddings)))  // 48
        let mergeSize = config.spatialMergeSize  // 2

        // Map target grid positions to source grid via linspace
        // h_idxs = linspace(0, 47, gridH), w_idxs = linspace(0, 47, gridW)
        let hIdxs: MLXArray
        let wIdxs: MLXArray
        if gridH == 1 {
            hIdxs = MLXArray([Float(0)])
        } else {
            hIdxs = MLXArray(stride(from: Float(0), through: Float(numGridPerSide - 1),
                                     by: Float(numGridPerSide - 1) / Float(gridH - 1)))
        }
        if gridW == 1 {
            wIdxs = MLXArray([Float(0)])
        } else {
            wIdxs = MLXArray(stride(from: Float(0), through: Float(numGridPerSide - 1),
                                     by: Float(numGridPerSide - 1) / Float(gridW - 1)))
        }

        // Floor/ceil indices for bilinear interpolation
        let hFloor = hIdxs.asType(.int32)
        let wFloor = wIdxs.asType(.int32)
        let hCeil = MLX.minimum(hFloor + 1, MLXArray(Int32(numGridPerSide - 1)))
        let wCeil = MLX.minimum(wFloor + 1, MLXArray(Int32(numGridPerSide - 1)))

        // Fractional parts
        let dh = hIdxs - hFloor.asType(.float32)  // [gridH]
        let dw = wIdxs - wFloor.asType(.float32)  // [gridW]

        // Compute 4 corner indices into the flat posEmbedStorage [numGridPerSide², hidden]
        // index = h * numGridPerSide + w
        let ngs = Int32(numGridPerSide)
        let baseH = hFloor * ngs       // [gridH]
        let baseHCeil = hCeil * ngs    // [gridH]

        // 4 corner index grids: [gridH, gridW] flattened to [gridH*gridW]
        let idx00 = (baseH.reshaped([gridH, 1]) + wFloor.reshaped([1, gridW])).reshaped([-1])
        let idx01 = (baseH.reshaped([gridH, 1]) + wCeil.reshaped([1, gridW])).reshaped([-1])
        let idx10 = (baseHCeil.reshaped([gridH, 1]) + wFloor.reshaped([1, gridW])).reshaped([-1])
        let idx11 = (baseHCeil.reshaped([gridH, 1]) + wCeil.reshaped([1, gridW])).reshaped([-1])

        // 4 corner weights: [gridH*gridW]
        let w00 = ((1 - dh).reshaped([gridH, 1]) * (1 - dw).reshaped([1, gridW])).reshaped([-1])
        let w01 = ((1 - dh).reshaped([gridH, 1]) * dw.reshaped([1, gridW])).reshaped([-1])
        let w10 = (dh.reshaped([gridH, 1]) * (1 - dw).reshaped([1, gridW])).reshaped([-1])
        let w11 = (dh.reshaped([gridH, 1]) * dw.reshaped([1, gridW])).reshaped([-1])

        // Gather position embeddings at 4 corners and weighted sum
        let emb00 = posEmbedStorage[idx00] * w00.reshaped([-1, 1])  // [gridH*gridW, hidden]
        let emb01 = posEmbedStorage[idx01] * w01.reshaped([-1, 1])
        let emb10 = posEmbedStorage[idx10] * w10.reshaped([-1, 1])
        let emb11 = posEmbedStorage[idx11] * w11.reshaped([-1, 1])

        var posEmb = emb00 + emb01 + emb10 + emb11  // [gridH*gridW, hidden]

        // Reorder to spatial-merge order (matching patch_embed output)
        let featureDim = config.hiddenSize
        posEmb = posEmb.reshaped([gridH, gridW, featureDim])
        posEmb = posEmb.reshaped([gridH / mergeSize, mergeSize, gridW / mergeSize, mergeSize, featureDim])
        posEmb = posEmb.transposed(0, 2, 1, 3, 4)
        posEmb = posEmb.reshaped([-1, featureDim])

        return posEmb
    }

    // MARK: - 2D Rotary Embeddings

    private func compute2DRotaryEmb(gridH: Int, gridW: Int, mergeSize: Int) -> MLXArray {
        // Python VisionRotaryEmbedding: dim=32 (= head_dim/2 = 64/2), theta=10000
        // inv_freq = 1/(theta^(arange(0, dim, 2)/dim)) → 16 values
        // freq_table = outer(positions, inv_freq) → [maxHW, 16]
        // Then h_emb = freq_table[h_positions], w_emb = freq_table[w_positions]
        // Final: concat([h_emb, w_emb]) → [numPatches, 32]
        let dim = rotaryDim / 2  // 32 (same as Python VisionRotaryEmbedding.dim)
        let numFreqs = dim / 2   // 16
        let indices = MLXArray(stride(from: Float(0), to: Float(dim), by: 2))  // [0, 2, 4, ..., 30]
        let invFreq = 1.0 / MLX.pow(MLXArray(Float(10000.0)), indices / Float(dim))  // [16]

        // Build freq_table: [max(gridH, gridW), 16]
        let maxHW = max(gridH, gridW)
        let positions = MLXArray(stride(from: Float(0), to: Float(maxHW), by: 1))
        let freqTable = MLX.outer(positions, invFreq)  // [maxHW, 16]

        // Build position indices in merge order
        // (same as Python rot_pos_emb: block_rows/cols + intra offsets)
        let mergedH = gridH / mergeSize
        let mergedW = gridW / mergeSize
        let numPatches = gridH * gridW

        // For each merge block (bh, bw) and intra position (ih, iw):
        // row = bh * mergeSize + ih, col = bw * mergeSize + iw
        // Order: iterate (bh, bw, ih, iw)
        var hIndices = [Int32]()
        var wIndices = [Int32]()
        hIndices.reserveCapacity(numPatches)
        wIndices.reserveCapacity(numPatches)

        for bh in 0..<mergedH {
            for bw in 0..<mergedW {
                for ih in 0..<mergeSize {
                    for iw in 0..<mergeSize {
                        hIndices.append(Int32(bh * mergeSize + ih))
                        wIndices.append(Int32(bw * mergeSize + iw))
                    }
                }
            }
        }

        let hIdx = MLXArray(hIndices)
        let wIdx = MLXArray(wIndices)

        // Lookup: h_emb = freqTable[hIdx], w_emb = freqTable[wIdx]
        let hEmb = freqTable[hIdx]  // [numPatches, 16]
        let wEmb = freqTable[wIdx]  // [numPatches, 16]

        // Concat [h_emb, w_emb] → [numPatches, 32]
        return concatenated([hEmb, wEmb], axis: -1)
    }
}
