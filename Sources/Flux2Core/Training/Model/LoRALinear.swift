// LoRAInjectedLinear.swift - Drop-in replacement for Linear with LoRA adaptation
// Copyright 2025 Vincent Gourbin
//
// Based on mlx-examples LoRA implementation:
// https://github.com/ml-explore/mlx-examples/tree/main/lora

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A Linear layer with Low-Rank Adaptation (LoRA)
///
/// This is a drop-in replacement for Linear that adds trainable LoRA weights.
/// During forward pass: output = Linear(x) + scale * (x @ loraA @ loraB)
///
/// The base Linear weights are frozen, only loraA and loraB are trained.
public class LoRAInjectedLinear: Linear {

    /// LoRA A matrix (down projection): [input_dim, rank] - mlx-examples convention
    /// Note: Named loraA (not lora_a) because MLXNN filters out underscore-prefixed params
    public var loraA: MLXArray

    /// LoRA B matrix (up projection): [rank, output_dim] - mlx-examples convention
    public var loraB: MLXArray

    /// LoRA rank
    public let rank: Int

    /// LoRA scale factor (mutable for DOP - Differential Output Preservation)
    public var loraScale: Float

    /// Original scale factor for restoration after DOP pass
    public let originalLoraScale: Float

    /// Create a LoRAInjectedLinear from an existing Linear layer
    /// - Parameters:
    ///   - linear: The base Linear layer to wrap
    ///   - rank: LoRA rank (default 8)
    ///   - alpha: LoRA alpha for scaling (default equals rank)
    public static func fromLinear(_ linear: Linear, rank: Int = 8, alpha: Float? = nil) -> LoRAInjectedLinear {
        let outputDim = linear.weight.shape[0]
        let inputDim = linear.weight.shape[1]
        let scale = (alpha ?? Float(rank)) / Float(rank)

        return LoRAInjectedLinear(
            weight: linear.weight,
            bias: linear.bias,
            inputDim: inputDim,
            outputDim: outputDim,
            rank: rank,
            scale: scale
        )
    }

    /// Initialize LoRAInjectedLinear with existing weights
    private init(
        weight: MLXArray,
        bias: MLXArray?,
        inputDim: Int,
        outputDim: Int,
        rank: Int,
        scale: Float
    ) {
        self.rank = rank
        self.loraScale = scale
        self.originalLoraScale = scale

        // Initialize LoRA matrices using mlx-examples convention:
        // A: [input_dim, rank] - small random values
        // B: [rank, output_dim] - zeros
        // This way forward is: x @ loraA @ loraB (no transposes needed)
        let loraAInit = MLXRandom.uniform(
            low: -1.0 / sqrt(Float(inputDim)),
            high: 1.0 / sqrt(Float(inputDim)),
            [inputDim, rank]
        ).asType(weight.dtype)

        let loraBInit = MLXArray.zeros([rank, outputDim]).asType(weight.dtype)

        self.loraA = loraAInit
        self.loraB = loraBInit

        // Initialize base Linear with existing weights
        super.init(weight: weight, bias: bias)
    }

    /// Forward pass: base + LoRA
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear output
        let baseOutput = super.callAsFunction(x)

        // LoRA output: x @ loraA @ loraB (mlx-examples convention, no transposes)
        // loraA: [input_dim, rank], loraB: [rank, output_dim]
        let xFloat = x.asType(loraA.dtype)
        let loraOutput = matmul(matmul(xFloat, loraA), loraB)

        // Combine: base + scale * lora
        return baseOutput + loraScale * loraOutput.asType(baseOutput.dtype)
    }

    /// Get only the LoRA parameters for saving
    public var loraParameters: [String: MLXArray] {
        ["lora_a": loraA, "lora_b": loraB]
    }
}

// MARK: - Transformer LoRA Extension

extension Flux2Transformer2DModel {

    /// Convert Linear layers to LoRAInjectedLinear for training
    /// - Parameters:
    ///   - rank: LoRA rank
    ///   - alpha: LoRA alpha (scaling factor)
    ///   - targetBlocks: Which blocks to apply LoRA to (nil = all)
    public func applyLoRA(rank: Int = 8, alpha: Float? = nil, targetBlocks: LoRATargetBlocks = .all) {
        // NOTE: We do NOT use freeze() because it prevents trainableParameters() from seeing
        // the LoRA parameters even after unfreezing them. Instead, we'll filter gradients
        // in the training loop to only apply to LoRA parameters.
        //
        // IMPORTANT: We use update(modules:) instead of direct property assignment because
        // MLXNN caches module items. Direct assignment doesn't update the cache, so
        // trainableParameters() wouldn't find the new LoRA parameters.

        // Double-stream blocks
        for (idx, block) in transformerBlocks.enumerated() {
            guard targetBlocks.includesDoubleBlock(idx) else { continue }

            // Attention projections - use update(modules:) to properly update cache
            let attn = block.attn
            var modulesDict: [String: NestedItem<String, Module>] = [
                "toQ": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toQ, rank: rank, alpha: alpha)),
                "toK": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toK, rank: rank, alpha: alpha)),
                "toV": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toV, rank: rank, alpha: alpha)),
                "addQProj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.addQProj, rank: rank, alpha: alpha)),
                "addKProj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.addKProj, rank: rank, alpha: alpha)),
                "addVProj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.addVProj, rank: rank, alpha: alpha))
            ]

            if targetBlocks.includesOutputProjections {
                modulesDict["toOut"] = NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toOut, rank: rank, alpha: alpha))
                modulesDict["toAddOut"] = NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toAddOut, rank: rank, alpha: alpha))
            }

            let modules = ModuleChildren(values: modulesDict)
            attn.update(modules: modules)
            
            // FFN layers (image stream)
            if targetBlocks.includesFFN {
                // Inject into ff.activation.proj (SwiGLU input projection)
                let ffActivationModules = ModuleChildren(values: [
                    "proj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(block.ff.activation.proj, rank: rank, alpha: alpha))
                ])
                block.ff.activation.update(modules: ffActivationModules)
                
                // Inject into ff.linearOut (output projection)
                let ffModules = ModuleChildren(values: [
                    "linearOut": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(block.ff.linearOut, rank: rank, alpha: alpha))
                ])
                block.ff.update(modules: ffModules)
                
                // FFN layers (text stream)
                let ffContextActivationModules = ModuleChildren(values: [
                    "proj": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(block.ffContext.activation.proj, rank: rank, alpha: alpha))
                ])
                block.ffContext.activation.update(modules: ffContextActivationModules)
                
                let ffContextModules = ModuleChildren(values: [
                    "linearOut": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(block.ffContext.linearOut, rank: rank, alpha: alpha))
                ])
                block.ffContext.update(modules: ffContextModules)
            }
        }

        // Single-stream blocks
        for (idx, block) in singleTransformerBlocks.enumerated() {
            guard targetBlocks.includesSingleBlock(idx) else { continue }

            let attn = block.attn
            var modulesDict: [String: NestedItem<String, Module>] = [
                "toQkvMlp": NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toQkvMlp, rank: rank, alpha: alpha))
            ]

            if targetBlocks.includesOutputProjections {
                modulesDict["toOut"] = NestedItem<String, Module>.value(LoRAInjectedLinear.fromLinear(attn.toOut, rank: rank, alpha: alpha))
            }

            let modules = ModuleChildren(values: modulesDict)
            attn.update(modules: modules)
        }

        // NOTE: We intentionally do NOT train embedding layers (xEmbedder, contextEmbedder),
        // modulation layers, time embeddings, or final projection.
        // Analysis of Ostris/AI-Toolkit shows these layers are NOT included in their LoRA training.
        // Training these layers caused training collapse in our tests.
        // Ostris trains: attention QKV, attention output, and MLP layers only.

        // Log trainable parameter count
        let trainableParams = trainableParameters()
        let loraPathCount = trainableParams.flattened().filter { $0.0.hasSuffix("loraA") || $0.0.hasSuffix("loraB") }.count
        let trainableCount = trainableParams.flattenedValues().reduce(0) { $0 + $1.size }
        Flux2Debug.log("[Transformer] Applied LoRA with rank \(rank): \(loraPathCount) LoRA layers, \(trainableCount) trainable parameters")
    }

    /// Get all LoRA parameters for saving
    /// Note: Transposes weights to match inference format:
    /// - Training uses: loraA [input_dim, rank], loraB [rank, output_dim]
    /// - Inference expects: loraA [rank, input_dim], loraB [output_dim, rank]
    public func getLoRAParameters() -> [String: MLXArray] {
        var params: [String: MLXArray] = [:]

        // Double-stream blocks
        for (idx, block) in transformerBlocks.enumerated() {
            let prefix = "transformer_blocks.\(idx).attn"

            if let lora = block.attn.toQ as? LoRAInjectedLinear {
                params["\(prefix).to_q.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_q.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toK as? LoRAInjectedLinear {
                params["\(prefix).to_k.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_k.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toV as? LoRAInjectedLinear {
                params["\(prefix).to_v.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_v.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear {
                params["\(prefix).add_q_proj.lora_A.weight"] = lora.loraA.T
                params["\(prefix).add_q_proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear {
                params["\(prefix).add_k_proj.lora_A.weight"] = lora.loraA.T
                params["\(prefix).add_k_proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear {
                params["\(prefix).add_v_proj.lora_A.weight"] = lora.loraA.T
                params["\(prefix).add_v_proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                params["\(prefix).to_out.0.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_out.0.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear {
                params["\(prefix).to_add_out.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_add_out.lora_B.weight"] = lora.loraB.T
            }
            
            // FFN layers (image stream)
            let ffPrefix = "transformer_blocks.\(idx)"
            if let lora = block.ff.activation.proj as? LoRAInjectedLinear {
                params["\(ffPrefix).ff.activation.proj.lora_A.weight"] = lora.loraA.T
                params["\(ffPrefix).ff.activation.proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.ff.linearOut as? LoRAInjectedLinear {
                params["\(ffPrefix).ff.linear_out.lora_A.weight"] = lora.loraA.T
                params["\(ffPrefix).ff.linear_out.lora_B.weight"] = lora.loraB.T
            }
            
            // FFN layers (text stream)
            if let lora = block.ffContext.activation.proj as? LoRAInjectedLinear {
                params["\(ffPrefix).ff_context.activation.proj.lora_A.weight"] = lora.loraA.T
                params["\(ffPrefix).ff_context.activation.proj.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.ffContext.linearOut as? LoRAInjectedLinear {
                params["\(ffPrefix).ff_context.linear_out.lora_A.weight"] = lora.loraA.T
                params["\(ffPrefix).ff_context.linear_out.lora_B.weight"] = lora.loraB.T
            }
        }

        // Single-stream blocks
        for (idx, block) in singleTransformerBlocks.enumerated() {
            let prefix = "single_transformer_blocks.\(idx).attn"

            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear {
                params["\(prefix).to_qkv_mlp.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_qkv_mlp.lora_B.weight"] = lora.loraB.T
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                params["\(prefix).to_out.lora_A.weight"] = lora.loraA.T
                params["\(prefix).to_out.lora_B.weight"] = lora.loraB.T
            }
        }

        // NOTE: Embedding layers (xEmbedder, contextEmbedder), modulation layers,
        // time embeddings, and final projection are NOT included in LoRA training.
        // This matches Ostris/AI-Toolkit behavior.

        return params
    }

    /// Count trainable LoRA parameters
    public var loraParameterCount: Int {
        var count = 0
        for (_, array) in getLoRAParameters() {
            count += array.size
        }
        return count
    }

    /// Unfreeze only LoRA parameters (loraA and loraB)
    /// Call this after freeze(recursive: true) to make only LoRA params trainable
    public func unfreezeLoRAParameters() {
        // Unfreeze LoRA in double-stream blocks
        for block in transformerBlocks {
            if let lora = block.attn.toQ as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toK as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toV as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }

            // FFN layers (image stream)
            if let lora = block.ff.activation.proj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.ff.linearOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }

            // FFN layers (text stream)
            if let lora = block.ffContext.activation.proj as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.ffContext.linearOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
        }

        // Unfreeze LoRA in single-stream blocks
        for block in singleTransformerBlocks {
            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
            if let lora = block.attn.toOut as? LoRAInjectedLinear {
                lora.unfreeze(recursive: false, keys: ["loraA", "loraB"])
            }
        }

        // NOTE: Embedding layers (xEmbedder, contextEmbedder), modulation layers,
        // time embeddings, and final projection are NOT included in LoRA training.
        // This matches Ostris/AI-Toolkit behavior.

        Flux2Debug.log("[Transformer] Unfroze LoRA parameters (loraA, loraB)")
    }

    /// Set all LoRA scales to a specific value (used for DOP - Differential Output Preservation)
    /// - Parameter scale: The scale to set (0.0 to disable LoRA, or restore to original)
    public func setAllLoRAScales(_ scale: Float) {
        // Double-stream blocks
        for block in transformerBlocks {
            if let lora = block.attn.toQ as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.toK as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.toV as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear { lora.loraScale = scale }
        }
        // Single-stream blocks
        for block in singleTransformerBlocks {
            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear { lora.loraScale = scale }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { lora.loraScale = scale }
        }
    }

    /// Disable LoRA for DOP preservation pass (sets all scales to 0)
    public func disableLoRA() {
        setAllLoRAScales(0.0)
    }

    /// Restore LoRA to original scales after DOP preservation pass
    public func restoreLoRAScales() {
        // Double-stream blocks
        for block in transformerBlocks {
            if let lora = block.attn.toQ as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.toK as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.toV as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
        }
        // Single-stream blocks
        for block in singleTransformerBlocks {
            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { lora.loraScale = lora.originalLoraScale }
        }
    }
}

// MARK: - Target Blocks Configuration

/// Configuration for which blocks to apply LoRA to
public struct LoRATargetBlocks: Sendable {
    public let doubleBlockIndices: [Int]?  // nil = all
    public let singleBlockIndices: [Int]?  // nil = all
    public let includesOutputProjections: Bool
    public let includesFFN: Bool

    public init(
        doubleBlockIndices: [Int]? = nil,
        singleBlockIndices: [Int]? = nil,
        includesOutputProjections: Bool = false,
        includesFFN: Bool = false
    ) {
        self.doubleBlockIndices = doubleBlockIndices
        self.singleBlockIndices = singleBlockIndices
        self.includesOutputProjections = includesOutputProjections
        self.includesFFN = includesFFN
    }

    public static let all = LoRATargetBlocks(
        doubleBlockIndices: nil,
        singleBlockIndices: nil,
        includesOutputProjections: true,
        includesFFN: true
    )

    // NOTE: Ostris/AI-Toolkit trains QKV + output projections by default.
    // Including output projections is essential for proper training.
    public static let attentionOnly = LoRATargetBlocks(
        doubleBlockIndices: nil,
        singleBlockIndices: nil,
        includesOutputProjections: true,  // Ostris includes this
        includesFFN: false
    )

    public static let attentionWithOutput = LoRATargetBlocks(
        doubleBlockIndices: nil,
        singleBlockIndices: nil,
        includesOutputProjections: true,
        includesFFN: false
    )

    public func includesDoubleBlock(_ index: Int) -> Bool {
        doubleBlockIndices == nil || doubleBlockIndices!.contains(index)
    }

    public func includesSingleBlock(_ index: Int) -> Bool {
        singleBlockIndices == nil || singleBlockIndices!.contains(index)
    }
}

// MARK: - LoRATargetLayers to LoRATargetBlocks Conversion

extension LoRATargetLayers {
    /// Convert to LoRATargetBlocks for use in applyLoRA
    public func toTargetBlocks() -> LoRATargetBlocks {
        switch self {
        case .attention:
            return .attentionOnly
        case .attentionOutput:
            return .attentionWithOutput
        case .attentionFFN:
            return LoRATargetBlocks(
                includesOutputProjections: true,
                includesFFN: true
            )
        case .all:
            return .all
        }
    }
}
