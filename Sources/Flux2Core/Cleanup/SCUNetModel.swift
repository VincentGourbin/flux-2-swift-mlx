// SCUNetModel.swift — SCUNet (practical blind denoise / JPEG-artifact removal) in MLX.
//
// Direct translation of the Python MLX reference verified in the lab (pixel-identical
// to the PyTorch `scunet_color_real_psnr` checkpoint, ~78 dB PSNR). Everything runs
// channels-last (NHWC), which is MLX-native and exactly what the window attention wants.
//
// Property names are chosen so the flattened MLXNN parameter keys match the exported
// safetensors 1:1 (e.g. `m_down1.0.trans_block.msa.relative_position_params`), so the
// weights load by exact name with no remapping. The Conv2d / ConvTransposed2d weights in
// the exported file are already in MLX OHWI layout, so no transpose happens here.
//
// swiftlint:disable identifier_name type_name

import Foundation
import MLX
import MLXNN

/// Exact (erf) GELU as a no-parameter module, so it can occupy the activation slot in an
/// `mlp` sequence and keep the surrounding Linear layers at indices 0 and 2 (matching the
/// PyTorch `nn.Sequential(Linear, GELU, Linear)` key layout).
final class GELUModule: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(x) }
}

/// Window multi-head self-attention (regular `W` or shifted `SW`).
final class WMSA: Module, UnaryLayer {
    let embedding_layer: Linear
    let linear: Linear
    let relative_position_params: MLXArray

    // Non-parameter constants (Swift arrays / scalars are not discovered by MLXNN).
    private let nHeads: Int
    private let headDim: Int
    private let windowSize: Int
    private let outDim: Int
    private let shifted: Bool
    private let scale: Float
    private let tokens: Int          // P = windowSize^2
    private let relIdx: [Int32]      // (P*P,) gather index into flattened rel-pos table
    private let rowBlock: [Float]    // (P*P,) additive mask for the last window-row
    private let colBlock: [Float]    // (P*P,) additive mask for the last window-col

    init(inputDim: Int, outputDim: Int, headDim: Int, windowSize: Int, shifted: Bool) {
        self.nHeads = inputDim / headDim
        self.headDim = headDim
        self.windowSize = windowSize
        self.outDim = outputDim
        self.shifted = shifted
        self.scale = pow(Float(headDim), -0.5)
        let ws = windowSize
        let P = ws * ws
        self.tokens = P

        self.embedding_layer = Linear(inputDim, 3 * inputDim, bias: true)
        self.linear = Linear(inputDim, outputDim, bias: true)
        self.relative_position_params = MLXArray.zeros([inputDim / headDim, 2 * ws - 1, 2 * ws - 1])

        // Relative-position gather index: for token a=(ai,aj), b=(bi,bj),
        // idx = (ai-bi+ws-1)*(2ws-1) + (aj-bj+ws-1).
        var idx = [Int32](repeating: 0, count: P * P)
        let span = 2 * ws - 1
        for a in 0..<P {
            let ai = a / ws, aj = a % ws
            for b in 0..<P {
                let bi = b / ws, bj = b % ws
                idx[a * P + b] = Int32((ai - bi + ws - 1) * span + (aj - bj + ws - 1))
            }
        }
        self.relIdx = idx

        // Shifted-window masks (additive: 0 keep, -inf block). Boundary at s = ws/2.
        let s = ws - ws / 2
        var row = [Float](repeating: 0, count: P * P)
        var col = [Float](repeating: 0, count: P * P)
        let neg = -Float.infinity
        for a in 0..<P {
            let ai = a / ws, aj = a % ws
            for b in 0..<P {
                let bi = b / ws, bj = b % ws
                if (ai < s) != (bi < s) { row[a * P + b] = neg }
                if (aj < s) != (bj < s) { col[a * P + b] = neg }
            }
        }
        self.rowBlock = row
        self.colBlock = col

        super.init()
    }

    private func relativeBias() -> MLXArray {
        let span = 2 * windowSize - 1
        let flat = relative_position_params.reshaped([nHeads, span * span])
        let gathered = flat.take(MLXArray(relIdx), axis: 1)        // (nHeads, P*P)
        return gathered.reshaped([nHeads, tokens, tokens])         // (nHeads, P, P)
    }

    /// Additive shifted-window mask of shape (nH*nW, P, P) built from the row/col blocks.
    private func shiftMask(nH: Int, nW: Int) -> MLXArray {
        let P = tokens
        let row = MLXArray(rowBlock).reshaped([1, 1, P, P])
        let col = MLXArray(colBlock).reshaped([1, 1, P, P])
        let rowPart: MLXArray
        if nH == 1 {
            rowPart = row
        } else {
            rowPart = concatenated([MLXArray.zeros([nH - 1, 1, P, P]), row], axis: 0)  // (nH,1,P,P)
        }
        let colPart: MLXArray
        if nW == 1 {
            colPart = col
        } else {
            colPart = concatenated([MLXArray.zeros([1, nW - 1, P, P]), col], axis: 1)  // (1,nW,P,P)
        }
        let grid = rowPart + colPart                                                    // (nH,nW,P,P)
        return grid.reshaped([nH * nW, P, P])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let ws = windowSize
        var x = input
        if shifted {
            x = roll(x, shift: -(ws / 2), axis: 1)
            x = roll(x, shift: -(ws / 2), axis: 2)
        }
        let b = x.shape[0], H = x.shape[1], W = x.shape[2], C = x.shape[3]
        let nH = H / ws, nW = W / ws, P = tokens, NW = nH * nW

        x = x.reshaped([b, nH, ws, nW, ws, C])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, NW, P, C])

        var qkv = embedding_layer(x)                                  // (b,NW,P,3C)
        qkv = qkv.reshaped([b, NW, P, 3, nHeads, headDim])
            .transposed(3, 0, 1, 2, 4, 5)                             // (3,b,NW,P,nHeads,hd)
        let q = qkv[0].transposed(0, 1, 3, 2, 4)                      // (b,NW,nHeads,P,hd)
        let k = qkv[1].transposed(0, 1, 3, 2, 4)
        let v = qkv[2].transposed(0, 1, 3, 2, 4)

        var sim = matmul(q, k.transposed(0, 1, 2, 4, 3)) * scale      // (b,NW,nHeads,P,P)
        sim = sim + relativeBias().expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        if shifted {
            sim = sim + shiftMask(nH: nH, nW: nW).reshaped([1, NW, 1, P, P])
        }
        let probs = softmax(sim, axis: -1)
        var out = matmul(probs, v)                                    // (b,NW,nHeads,P,hd)
        out = out.transposed(0, 1, 3, 2, 4).reshaped([b, NW, P, nHeads * headDim])
        out = linear(out)                                             // (b,NW,P,outDim)
        out = out.reshaped([b, nH, nW, ws, ws, outDim])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, H, W, outDim])
        if shifted {
            out = roll(out, shift: ws / 2, axis: 1)
            out = roll(out, shift: ws / 2, axis: 2)
        }
        return out
    }
}

/// Transformer block: LN -> WMSA -> residual, LN -> MLP -> residual (channels-last).
final class Block: Module, UnaryLayer {
    let ln1: LayerNorm
    let msa: WMSA
    let ln2: LayerNorm
    let mlp: [Module]   // [Linear, GELUModule, Linear] -> keys mlp.0 / mlp.2

    init(dim: Int, headDim: Int, windowSize: Int, shifted: Bool) {
        self.ln1 = LayerNorm(dimensions: dim)
        self.msa = WMSA(inputDim: dim, outputDim: dim, headDim: headDim, windowSize: windowSize, shifted: shifted)
        self.ln2 = LayerNorm(dimensions: dim)
        self.mlp = [Linear(dim, 4 * dim), GELUModule(), Linear(4 * dim, dim)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + msa(ln1(x))
        var f = ln2(h)
        for layer in mlp { f = (layer as! UnaryLayer)(f) }
        h = h + f
        return h
    }
}

/// Conv + Transformer block: a conv branch and a window-attention branch on split channels.
final class ConvTransBlock: Module, UnaryLayer {
    let conv1_1: Conv2d
    let conv1_2: Conv2d
    let conv_block: [Module]   // [Conv2d, ReLU, Conv2d] -> keys conv_block.0 / conv_block.2
    let trans_block: Block
    private let convDim: Int

    init(convDim: Int, transDim: Int, headDim: Int, windowSize: Int, shifted: Bool) {
        self.convDim = convDim
        let total = convDim + transDim
        self.conv1_1 = Conv2d(inputChannels: total, outputChannels: total, kernelSize: 1, stride: 1, padding: 0, bias: true)
        self.conv1_2 = Conv2d(inputChannels: total, outputChannels: total, kernelSize: 1, stride: 1, padding: 0, bias: true)
        self.conv_block = [
            Conv2d(inputChannels: convDim, outputChannels: convDim, kernelSize: 3, stride: 1, padding: 1, bias: false),
            ReLU(),
            Conv2d(inputChannels: convDim, outputChannels: convDim, kernelSize: 3, stride: 1, padding: 1, bias: false),
        ]
        self.trans_block = Block(dim: transDim, headDim: headDim, windowSize: windowSize, shifted: shifted)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1_1(x)
        let convX0 = y[0..., 0..., 0..., 0..<convDim]
        let transX0 = y[0..., 0..., 0..., convDim...]
        var convX = convX0
        for layer in conv_block { convX = (layer as! UnaryLayer)(convX) }
        convX = convX + convX0
        let transX = trans_block(transX0)
        let res = conv1_2(concatenated([convX, transX], axis: -1))
        return x + res
    }
}

/// SCUNet encoder/decoder. `config` is the per-stage block count (default [4]*7), `dim`=64.
public final class SCUNetModel: Module, UnaryLayer {
    let m_head: [Module]
    let m_down1: [Module]
    let m_down2: [Module]
    let m_down3: [Module]
    let m_body: [Module]
    let m_up3: [Module]
    let m_up2: [Module]
    let m_up1: [Module]
    let m_tail: [Module]

    public init(inChannels: Int = 3, config: [Int] = [4, 4, 4, 4, 4, 4, 4], dim: Int = 64) {
        let hd = 32, ws = 8
        func ctbs(_ cd: Int, _ td: Int, _ count: Int) -> [Module] {
            (0..<count).map { ConvTransBlock(convDim: cd, transDim: td, headDim: hd, windowSize: ws, shifted: $0 % 2 != 0) }
        }
        func down(_ inC: Int, _ outC: Int) -> Conv2d {
            Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 2, stride: 2, padding: 0, bias: false)
        }
        func up(_ inC: Int, _ outC: Int) -> ConvTransposed2d {
            ConvTransposed2d(inputChannels: inC, outputChannels: outC, kernelSize: 2, stride: 2, padding: 0, bias: false)
        }

        self.m_head = [Conv2d(inputChannels: inChannels, outputChannels: dim, kernelSize: 3, stride: 1, padding: 1, bias: false)]
        self.m_down1 = ctbs(dim / 2, dim / 2, config[0]) + [down(dim, 2 * dim)]
        self.m_down2 = ctbs(dim, dim, config[1]) + [down(2 * dim, 4 * dim)]
        self.m_down3 = ctbs(2 * dim, 2 * dim, config[2]) + [down(4 * dim, 8 * dim)]
        self.m_body = ctbs(4 * dim, 4 * dim, config[3])
        self.m_up3 = [up(8 * dim, 4 * dim)] + ctbs(2 * dim, 2 * dim, config[4])
        self.m_up2 = [up(4 * dim, 2 * dim)] + ctbs(dim, dim, config[5])
        self.m_up1 = [up(2 * dim, dim)] + ctbs(dim / 2, dim / 2, config[6])
        self.m_tail = [Conv2d(inputChannels: dim, outputChannels: inChannels, kernelSize: 3, stride: 1, padding: 1, bias: false)]
        super.init()
    }

    private func runSeq(_ layers: [Module], _ x: MLXArray) -> MLXArray {
        var y = x
        for layer in layers { y = (layer as! UnaryLayer)(y) }
        return y
    }

    /// Input/output are NHWC float in [0,1]; spatial dims must be multiples of 64.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = runSeq(m_head, x)
        let x2 = runSeq(m_down1, x1)
        let x3 = runSeq(m_down2, x2)
        let x4 = runSeq(m_down3, x3)
        var y = runSeq(m_body, x4)
        y = runSeq(m_up3, y + x4)
        y = runSeq(m_up2, y + x3)
        y = runSeq(m_up1, y + x2)
        y = runSeq(m_tail, y + x1)
        return y
    }

    /// Load weights from a safetensors file whose keys already match the flattened
    /// MLXNN parameter names (and are in MLX OHWI conv layout). Returns load diagnostics.
    @discardableResult
    public func loadWeights(fromSafetensors url: URL) throws -> (matched: Int, missing: [String], unused: [String]) {
        let weights = try loadArrays(url: url)
        let flat = parameters().flattened()
        let modelKeys = Set(flat.map { $0.0 })
        var updates: [String: MLXArray] = [:]
        var missing: [String] = []
        for (key, _) in flat {
            if let value = weights[key] {
                updates[key] = value
            } else {
                missing.append(key)
            }
        }
        let unused = weights.keys.filter { !modelKeys.contains($0) }.sorted()
        update(parameters: ModuleParameters.unflattened(updates))
        eval(parameters())
        return (updates.count, missing.sorted(), unused)
    }
}
