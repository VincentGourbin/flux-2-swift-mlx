/**
 * Qwen35GatedDeltaNet.swift
 * Gated DeltaNet linear attention for Qwen3.5 (used in 24 of 32 layers)
 *
 * SSM-like recurrent attention with:
 * - 1D convolution (kernel=4) on Q/K/V projections
 * - State decay via exp(-exp(A_log) * softplus(a + dt_bias))
 * - Delta rule: write error correction to state
 * - Output gating via sigmoid(z)
 *
 * State shape: [B, Hv, Dv, Dk] — recurrent, passed via cache
 *
 * Reference: mlx_lm/models/gated_delta.py
 */

import Foundation
import MLX
import MLXNN

// MARK: - RMSNorm with Gating

/// RMSNorm applied to output, gated by z: norm(out) * silu(z)
public class Qwen35RMSNormGated: Module {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate z: MLXArray) -> MLXArray {
        // x: [B, S, Hv, Dv], z: [B, S, Hv, Dv]
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        return normed * silu(z)
    }
}

// MARK: - Gated Delta Update (Core Algorithm — Optimized)

/// Inline recurrent step (no function call overhead, uses reshaped instead of expandedDimensions)
@inline(__always)
private func deltaStep(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray
) -> (MLXArray, MLXArray) {
    // q,k: [B, Hv, Dk], v: [B, Hv, Dv], g,beta: [B, Hv], state: [B, Hv, Dv, Dk]
    let B = g.dim(0), Hv = g.dim(1)
    let Dk = k.dim(2), Dv = v.dim(2)

    // Decay: state *= g[:,:,None,None]
    var s = state * g.reshaped([B, Hv, 1, 1])

    // Recall via matmul: kvMem[b,h,d] = sum_k(state[b,h,d,k] * k[b,h,k])
    // state: [B*Hv, Dv, Dk], k: [B*Hv, Dk, 1] → matmul → [B*Hv, Dv, 1] → squeeze
    let sFlat = s.reshaped([B * Hv, Dv, Dk])
    let kCol = k.reshaped([B * Hv, Dk, 1])
    let kvMem = MLX.matmul(sFlat, kCol).reshaped([B, Hv, Dv])

    // Delta write: delta = (v - kvMem) * beta, then state += outer(delta, k)
    let delta = (v - kvMem) * beta.reshaped([B, Hv, 1])
    // outer product: delta[:,:,:,None] * k[:,:,None,:]
    let deltaCol = delta.reshaped([B * Hv, Dv, 1])
    let kRow = k.reshaped([B * Hv, 1, Dk])
    let outer = MLX.matmul(deltaCol, kRow).reshaped([B, Hv, Dv, Dk])
    s = s + outer

    // Read via matmul: y[b,h,d] = sum_k(state[b,h,d,k] * q[b,h,k])
    let sFlatNew = s.reshaped([B * Hv, Dv, Dk])
    let qCol = q.reshaped([B * Hv, Dk, 1])
    let y = MLX.matmul(sFlatNew, qCol).reshaped([B, Hv, Dv])

    return (y, s)
}

/// Optimized sequential gated delta update
/// Key optimizations:
/// 1. Pre-split tensors before loop (one split op instead of T slice ops)
/// 2. Compiled step function (fused GPU operations)
/// 3. Single-token shortcut (no loop/split/stack for T=1)
/// 4. reshaped() instead of expandedDimensions (zero-copy)
private func gatedDeltaUpdate(
    q: MLXArray,      // [B, T, Hk, Dk]
    k: MLXArray,      // [B, T, Hk, Dk]
    v: MLXArray,      // [B, T, Hv, Dv]
    g: MLXArray,      // [B, T, Hv]
    beta: MLXArray,   // [B, T, Hv]
    state: MLXArray,  // [B, Hv, Dv, Dk]
    mask: MLXArray?   // [B, T] optional
) -> (MLXArray, MLXArray) {
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Hv = v.dim(2)

    // Expand Q/K heads if needed (Hv > Hk)
    var qExp = q
    var kExp = k
    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        qExp = MLX.repeat(q, count: repeatFactor, axis: 2)
        kExp = MLX.repeat(k, count: repeatFactor, axis: 2)
    }

    // === Single-token shortcut (generation mode: T=1) ===
    if T == 1 {
        let qt = qExp.squeezed(axis: 1)   // [B, Hv, Dk]
        let kt = kExp.squeezed(axis: 1)
        let vt = v.squeezed(axis: 1)       // [B, Hv, Dv]
        let gt = g.squeezed(axis: 1)       // [B, Hv]
        let bt = beta.squeezed(axis: 1)

        let (yt, newState) = deltaStep(q: qt, k: kt, v: vt, g: gt, beta: bt, state: state)
        let output = yt.expandedDimensions(axis: 1)  // [B, 1, Hv, Dv]
        return (output, newState)
    }

    // === Multi-token path (prefill) ===
    // Pre-split all tensors along T dimension (one op each, no per-step slicing)
    let qSlices = qExp.split(parts: T, axis: 1)
    let kSlices = kExp.split(parts: T, axis: 1)
    let vSlices = v.split(parts: T, axis: 1)
    let gSlices = g.split(parts: T, axis: 1)
    let bSlices = beta.split(parts: T, axis: 1)

    var currentState = state
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(T)

    for t in 0..<T {
        let qt = qSlices[t].squeezed(axis: 1)
        let kt = kSlices[t].squeezed(axis: 1)
        let vt = vSlices[t].squeezed(axis: 1)
        let gt = gSlices[t].squeezed(axis: 1)
        let bt = bSlices[t].squeezed(axis: 1)

        let (yt, newState) = deltaStep(q: qt, k: kt, v: vt, g: gt, beta: bt, state: currentState)
        currentState = newState
        outputs.append(yt)
    }

    let output = stacked(outputs, axis: 1)  // [B, T, Hv, Dv]
    return (output, currentState)
}

/// Compute gating values: g = exp(-exp(A_log) * softplus(a + dt_bias))
private func computeGating(aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray {
    let aLogF = aLog.asType(.float32)
    let aF = a.asType(.float32)
    let dtBiasF = dtBias.asType(.float32)
    return MLX.exp(-MLX.exp(aLogF) * softplus(aF + dtBiasF))
}

/// Softplus: log(1 + exp(x))
private func softplus(_ x: MLXArray) -> MLXArray {
    MLX.log(1.0 + MLX.exp(x))
}

// MARK: - Gated DeltaNet Module

/// Gated DeltaNet linear attention layer for Qwen3.5
public class Qwen35GatedDeltaNet: Module {
    let config: Qwen35TextConfig
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int

    @ModuleInfo public var in_proj_qkv: Linear
    @ModuleInfo public var in_proj_z: Linear
    @ModuleInfo public var in_proj_b: Linear
    @ModuleInfo public var in_proj_a: Linear
    @ModuleInfo public var out_proj: Linear

    public var conv1d: Conv1d
    public var norm: Qwen35RMSNormGated

    /// State decay parameter (log space)
    public var A_log: MLXArray
    /// Delta time bias
    public var dt_bias: MLXArray

    public init(config: Qwen35TextConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numVHeads = config.linearNumValueHeads      // 32
        self.numKHeads = config.linearNumKeyHeads         // 16
        self.headKDim = config.linearKeyHeadDim           // 128
        self.headVDim = config.linearValueHeadDim         // 128
        self.keyDim = headKDim * numKHeads                // 2048
        self.valueDim = headVDim * numVHeads              // 4096
        self.convKernelSize = config.linearConvKernelDim  // 4

        let convDim = keyDim * 2 + valueDim  // 2048 + 2048 + 4096 = 8192

        self._in_proj_qkv = ModuleInfo(wrappedValue: Linear(hiddenSize, convDim, bias: false))
        self._in_proj_z = ModuleInfo(wrappedValue: Linear(hiddenSize, valueDim, bias: false))
        self._in_proj_b = ModuleInfo(wrappedValue: Linear(hiddenSize, numVHeads, bias: false))
        self._in_proj_a = ModuleInfo(wrappedValue: Linear(hiddenSize, numVHeads, bias: false))
        self._out_proj = ModuleInfo(wrappedValue: Linear(valueDim, hiddenSize, bias: false))

        // Depthwise conv1d (groups = channels for per-channel convolution)
        self.conv1d = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        self.norm = Qwen35RMSNormGated(dimensions: headVDim, eps: config.rmsNormEps)

        // Initialize A_log and dt_bias
        self.A_log = MLXArray.ones([numVHeads])
        self.dt_bias = MLXArray.ones([numVHeads])

        super.init()
    }

    /// Cache type for DeltaNet: [conv_state, recurrent_state]
    /// conv_state: [B, kernelSize-1, convDim]
    /// recurrent_state: [B, Hv, Dv, Dk]
    public typealias DeltaNetCache = [MLXArray?]

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: inout DeltaNetCache
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        // Project Q/K/V
        var mixedQKV = in_proj_qkv(inputs)  // [B, S, convDim]

        // Gate z
        let z = in_proj_z(inputs).reshaped([B, S, numVHeads, headVDim])  // [B, S, Hv, Dv]

        // Decay and write gate inputs
        let b = in_proj_b(inputs)  // [B, S, Hv]
        let a = in_proj_a(inputs)  // [B, S, Hv]

        // Apply mask to QKV before conv
        if let mask = mask {
            mixedQKV = MLX.where(mask.expandedDimensions(axis: -1), mixedQKV, MLXArray(Float(0)))
        }

        // Conv1d with state
        let convDim = keyDim * 2 + valueDim
        var convState: MLXArray
        if let cs = cache[0] {
            convState = cs
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        // Prepend conv state
        let convInput = concatenated([convState, mixedQKV], axis: 1)  // [B, S+kernel-1, convDim]
        cache[0] = convInput[0..., (convInput.dim(1) - convKernelSize + 1)..., 0...]

        // Apply depthwise conv1d + SiLU
        // Conv1d expects [B, S, C] in MLX (channels last)
        let convOut = silu(conv1d(convInput))  // [B, S, convDim]

        // Split into Q, K, V
        let qRaw = convOut[0..., 0..., 0..<keyDim]                           // [B, S, keyDim]
        let kRaw = convOut[0..., 0..., keyDim..<(keyDim * 2)]                // [B, S, keyDim]
        let vRaw = convOut[0..., 0..., (keyDim * 2)..<(keyDim * 2 + valueDim)] // [B, S, valueDim]

        // Reshape to heads
        let q = qRaw.reshaped([B, S, numKHeads, headKDim])  // [B, S, Hk, Dk]
        let k = kRaw.reshaped([B, S, numKHeads, headKDim])
        let v = vRaw.reshaped([B, S, numVHeads, headVDim])  // [B, S, Hv, Dv]

        // Normalize Q and K using RMS normalization (without learned weight)
        let invScale = pow(Float(headKDim), -0.5)
        let qRms = MLX.sqrt(MLX.mean(q * q, axis: -1, keepDims: true) + 1e-6)
        let kRms = MLX.sqrt(MLX.mean(k * k, axis: -1, keepDims: true) + 1e-6)
        let qNorm = (invScale * invScale) * (q / qRms)
        let kNorm = invScale * (k / kRms)

        // Compute gating: g = exp(-exp(A_log) * softplus(a + dt_bias))
        let g = computeGating(aLog: A_log, a: a, dtBias: dt_bias)  // [B, S, Hv]

        // Write gate: beta = sigmoid(b)
        let beta = sigmoid(b)  // [B, S, Hv]

        // Recurrent state
        var recurrentState: MLXArray
        if let rs = cache[1] {
            recurrentState = rs
        } else {
            recurrentState = MLXArray.zeros([B, numVHeads, headVDim, headKDim], dtype: .float32)
        }

        // Run gated delta update
        let (out, newState) = gatedDeltaUpdate(
            q: qNorm, k: kNorm, v: v,
            g: g, beta: beta,
            state: recurrentState,
            mask: mask
        )

        cache[1] = newState

        // Apply gated norm: norm(out) * silu(z)
        let normedOut = norm(out, gate: z)  // [B, S, Hv, Dv]

        // Reshape and project
        let flatOut = normedOut.reshaped([B, S, valueDim])
        return out_proj(flatOut)
    }
}
