/**
 * MultiModalProjector.swift
 * Projects vision features to language model embedding space
 * Includes patch merging (2x2 spatial merge) and linear projections
 */

import Foundation
import MLX
import MLXNN

/// Debug flag - set VLM_DEBUG=1 to enable verbose logging
private let projDebug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

/// Configuration for the multimodal projector
public struct MultiModalProjectorConfig: Codable, Sendable {
    public let visionHiddenSize: Int      // Input from vision encoder (1024)
    public let textHiddenSize: Int        // Output for text model (5120)
    public let spatialMergeSize: Int      // 2x2 patch merging
    public let projectorHiddenAct: String // GELU activation
    public let normEps: Float

    public static let mistralSmall = MultiModalProjectorConfig(
        visionHiddenSize: 1024,
        textHiddenSize: 5120,
        spatialMergeSize: 2,
        projectorHiddenAct: "gelu",
        normEps: 1e-5
    )

    public init(
        visionHiddenSize: Int,
        textHiddenSize: Int,
        spatialMergeSize: Int,
        projectorHiddenAct: String,
        normEps: Float = 1e-5
    ) {
        self.visionHiddenSize = visionHiddenSize
        self.textHiddenSize = textHiddenSize
        self.spatialMergeSize = spatialMergeSize
        self.projectorHiddenAct = projectorHiddenAct
        self.normEps = normEps
    }
}

// MARK: - Patch Merger

/// Merges spatial patches (2x2) into single patches
/// This reduces the number of vision tokens by factor of spatialMergeSize^2
public class PatchMerger: Module {
    let spatialMergeSize: Int
    let inputDim: Int
    let outputDim: Int

    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear

    public init(inputDim: Int, outputDim: Int, spatialMergeSize: Int) {
        self.spatialMergeSize = spatialMergeSize
        self.inputDim = inputDim
        self.outputDim = outputDim

        // After 2x2 merge: inputDim * 4 -> outputDim
        let mergedInputDim = inputDim * spatialMergeSize * spatialMergeSize
        self._mergingLayer.wrappedValue = Linear(mergedInputDim, outputDim, bias: false)
    }

    /// Merge patches spatially - matches Python's unfold operation ordering
    /// - Parameter x: Input [batch, num_patches, hidden_size]
    /// - Parameter patchesH: Number of patches in height dimension
    /// - Parameter patchesW: Number of patches in width dimension
    /// - Returns: Merged patches [batch, merged_patches, output_dim]
    public func callAsFunction(_ x: MLXArray, patchesH: Int, patchesW: Int) -> MLXArray {
        let batchSize = x.shape[0]
        let hiddenSize = x.shape[2]

        // Calculate output dimensions after merging (using floor division like Python)
        let mergedH = patchesH / spatialMergeSize
        let mergedW = patchesW / spatialMergeSize

        // Calculate effective input dimensions (truncate to match mergedH * spatialMergeSize)
        // This handles odd dimensions like Python's unfold does (drops the last row/column)
        let effectiveH = mergedH * spatialMergeSize
        let effectiveW = mergedW * spatialMergeSize

        // Reshape to [batch, patchesH, patchesW, hidden_size]
        var reshaped = x.reshaped([batchSize, patchesH, patchesW, hiddenSize])

        // Truncate to effective dimensions if needed (handles odd dimensions)
        if effectiveH < patchesH || effectiveW < patchesW {
            reshaped = reshaped[0..., 0..<effectiveH, 0..<effectiveW, 0...]
        }

        // Reshape to group patches for merging
        // [batch, mergedH, spatialMergeSize, mergedW, spatialMergeSize, hidden_size]
        reshaped = reshaped.reshaped([
            batchSize, mergedH, spatialMergeSize, mergedW, spatialMergeSize, hiddenSize
        ])

        // Transpose to [batch, mergedH, mergedW, spatialMergeSize, spatialMergeSize, hidden_size]
        reshaped = reshaped.transposed(axes: [0, 1, 3, 2, 4, 5])

        // CRITICAL: To match Python's unfold ordering, we need to put hidden_size before spatial dims
        // Python unfold produces: [c0_p00, c0_p01, c0_p10, c0_p11, c1_p00, ...] (channels interleaved with positions)
        // Our old order was: [p00_c0, p00_c1, ..., p01_c0, ...] (positions first, then channels)
        // Transpose to [batch, mergedH, mergedW, hidden_size, spatialMergeSize, spatialMergeSize]
        reshaped = reshaped.transposed(axes: [0, 1, 2, 5, 3, 4])

        // Flatten the merged dimensions: hidden_size * spatialMergeSize * spatialMergeSize
        let mergedDim = hiddenSize * spatialMergeSize * spatialMergeSize
        reshaped = reshaped.reshaped([batchSize, mergedH, mergedW, mergedDim])

        // Flatten spatial dimensions
        // [batch, mergedH * mergedW, mergedDim]
        let mergedPatches = mergedH * mergedW
        reshaped = reshaped.reshaped([batchSize, mergedPatches, mergedDim])

        // Debug: print stats before and after merging layer
        if projDebug {
            print("[PatchMerger] Before merging_layer: shape=\(reshaped.shape), std=\(MLX.std(reshaped).item(Float.self))")
        }
        let output = mergingLayer(reshaped)
        if projDebug {
            print("[PatchMerger] After merging_layer: shape=\(output.shape), std=\(MLX.std(output).item(Float.self))")
            fflush(stdout)
        }

        // Project through merging layer
        return output
    }
}

// MARK: - MultiModal Projector

/// Projects vision encoder output to language model embedding space
public class MultiModalProjector: Module {
    let config: MultiModalProjectorConfig

    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(config: MultiModalProjectorConfig = .mistralSmall) {
        self.config = config

        // Pre-projection normalization
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.visionHiddenSize,
            eps: config.normEps
        )

        // Patch merger: reduces token count by spatialMergeSize^2
        self._patchMerger.wrappedValue = PatchMerger(
            inputDim: config.visionHiddenSize,
            outputDim: config.visionHiddenSize,  // Output same as input before linear
            spatialMergeSize: config.spatialMergeSize
        )

        // After merge: inputDim * spatialMergeSize^2 -> textHiddenSize
        // But patchMerger already handles this, so linear1 takes visionHiddenSize
        // Actually looking at the weights more carefully:
        // linear_1: (visionHiddenSize * spatialMergeSize^2) -> textHiddenSize
        // But patchMerger.merging_layer already does that
        // So linear_1 takes the output of patch_merger (which is visionHiddenSize) -> textHiddenSize

        // Let me reconsider based on weight shapes:
        // patch_merger.merging_layer: 4096 -> 1024 (merge 4 patches of 1024 into 1024)
        // linear_1: 1024 -> 5120
        // linear_2: 5120 -> 5120

        self._linear1.wrappedValue = Linear(
            config.visionHiddenSize,  // 1024 (output of patch merger)
            config.textHiddenSize,    // 5120
            bias: false
        )

        self._linear2.wrappedValue = Linear(
            config.textHiddenSize,    // 5120
            config.textHiddenSize,    // 5120
            bias: false
        )
    }

    /// Project vision features to language model space
    /// - Parameter visionFeatures: Output from vision encoder [batch, num_patches, vision_hidden_size]
    /// - Parameter patchesH: Number of patches in height dimension
    /// - Parameter patchesW: Number of patches in width dimension
    /// - Returns: Projected features [batch, merged_patches, text_hidden_size]
    public func callAsFunction(
        _ visionFeatures: MLXArray,
        patchesH: Int,
        patchesW: Int
    ) -> MLXArray {
        if projDebug {
            print("[Proj] Input: shape=\(visionFeatures.shape), std=\(MLX.std(visionFeatures).item(Float.self))")
        }

        // 1. Normalize vision features
        var x = norm(visionFeatures)
        if projDebug { print("[Proj] After RMSNorm: std=\(MLX.std(x).item(Float.self))") }

        // 2. Merge patches spatially (2x2 -> 1)
        x = patchMerger(x, patchesH: patchesH, patchesW: patchesW)
        if projDebug { print("[Proj] After PatchMerger: shape=\(x.shape), std=\(MLX.std(x).item(Float.self))") }

        // 3. Project to text hidden size with GELU activation
        x = linear1(x)
        if projDebug { print("[Proj] After linear1: std=\(MLX.std(x).item(Float.self))") }
        x = gelu(x)
        if projDebug { print("[Proj] After GELU: std=\(MLX.std(x).item(Float.self))") }

        // 4. Second linear projection
        x = linear2(x)
        if projDebug {
            print("[Proj] After linear2 (output): std=\(MLX.std(x).item(Float.self))")
            fflush(stdout)
        }

        return x
    }

    /// Get output info for given patch dimensions
    public func getOutputInfo(patchesH: Int, patchesW: Int) -> (mergedPatches: Int, hiddenSize: Int) {
        let mergedH = patchesH / config.spatialMergeSize
        let mergedW = patchesW / config.spatialMergeSize
        return (mergedH * mergedW, config.textHiddenSize)
    }
}
