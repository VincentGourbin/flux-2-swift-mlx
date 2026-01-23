// LoRALoader.swift - Load LoRA weights from safetensors
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Errors that can occur during LoRA loading
public enum LoRALoaderError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)
    case incompatibleModel(String)
    case missingWeights(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "LoRA file not found: \(path)"
        case .invalidFormat(let message):
            return "Invalid LoRA format: \(message)"
        case .incompatibleModel(let message):
            return "Incompatible model: \(message)"
        case .missingWeights(let layer):
            return "Missing LoRA weights for layer: \(layer)"
        }
    }
}

/// Loads LoRA weights from safetensors files
public class LoRALoader {

    /// Loaded LoRA weights, keyed by target layer path
    private(set) var weights: [String: LoRAWeightPair] = [:]

    /// LoRA information
    private(set) var info: LoRAInfo?

    /// The LoRA configuration
    public let config: LoRAConfig

    /// Initialize LoRA loader with configuration
    public init(config: LoRAConfig) {
        self.config = config
    }

    /// Load LoRA weights from file
    public func load() throws {
        let path = config.filePath

        guard FileManager.default.fileExists(atPath: path) else {
            throw LoRALoaderError.fileNotFound(path)
        }

        Flux2Debug.log("[LoRA] Loading from: \(path)")

        // Load safetensors
        let rawWeights = try loadArrays(url: URL(fileURLWithPath: path))

        Flux2Debug.log("[LoRA] Loaded \(rawWeights.count) tensors")

        // Parse LoRA weights and map to layer paths
        try parseAndMapWeights(rawWeights)

        Flux2Debug.log("[LoRA] Mapped \(weights.count) layer pairs")
    }

    /// Parse raw weights and map to our layer naming scheme
    private func parseAndMapWeights(_ rawWeights: [String: MLXArray]) throws {
        // Group weights by layer (strip lora_A/lora_B suffix)
        var layerGroups: [String: (loraA: MLXArray?, loraB: MLXArray?)] = [:]

        for (key, value) in rawWeights {
            // Skip metadata keys
            if key.hasPrefix("__") { continue }

            // Extract base layer path
            // Format: base_model.model.{layer_path}.lora_A.weight
            //      or base_model.model.{layer_path}.lora_B.weight
            let basePath: String
            let isLoraA: Bool

            if key.hasSuffix(".lora_A.weight") {
                basePath = String(key.dropLast(".lora_A.weight".count))
                isLoraA = true
            } else if key.hasSuffix(".lora_B.weight") {
                basePath = String(key.dropLast(".lora_B.weight".count))
                isLoraA = false
            } else {
                Flux2Debug.verbose("[LoRA] Skipping non-LoRA key: \(key)")
                continue
            }

            // Strip "base_model.model." prefix if present
            let layerPath = basePath.hasPrefix("base_model.model.")
                ? String(basePath.dropFirst("base_model.model.".count))
                : basePath

            if layerGroups[layerPath] == nil {
                layerGroups[layerPath] = (nil, nil)
            }

            if isLoraA {
                layerGroups[layerPath]?.loraA = value
            } else {
                layerGroups[layerPath]?.loraB = value
            }
        }

        // Convert to weight pairs and map to our naming scheme
        var rank: Int = 0
        var totalParams = 0

        for (bflPath, pair) in layerGroups {
            guard let loraA = pair.loraA, let loraB = pair.loraB else {
                Flux2Debug.log("[LoRA] Warning: Missing pair for \(bflPath)")
                continue
            }

            // Get rank from first pair
            if rank == 0 {
                rank = loraA.shape[0]
                Flux2Debug.log("[LoRA] Detected rank: \(rank)")
            }

            // Check if this is a combined QKV layer that needs splitting
            if isCombinedQKVLayer(bflPath) {
                // Split the combined QKV LoRA into separate Q, K, V parts
                let splitPairs = splitQKVLoRA(bflPath: bflPath, loraA: loraA, loraB: loraB)
                for (swiftPath, splitLoraA, splitLoraB) in splitPairs {
                    weights[swiftPath] = LoRAWeightPair(
                        loraA: MLXArrayWrapper(splitLoraA),
                        loraB: MLXArrayWrapper(splitLoraB)
                    )
                    totalParams += splitLoraA.size + splitLoraB.size
                }
            } else {
                // Direct 1:1 mapping
                let swiftPath = mapBFLPathToSwiftPath(bflPath)
                weights[swiftPath] = LoRAWeightPair(
                    loraA: MLXArrayWrapper(loraA),
                    loraB: MLXArrayWrapper(loraB)
                )
                totalParams += loraA.size + loraB.size
            }
        }

        // Detect target model based on layer structure
        let targetModel = detectTargetModel(layerGroups.keys.map { $0 })

        info = LoRAInfo(
            numLayers: weights.count,
            rank: rank,
            numParameters: totalParams,
            targetModel: targetModel
        )

        Flux2Debug.log("[LoRA] Info: \(weights.count) layers, rank=\(rank), params=\(totalParams), target=\(targetModel.rawValue)")
    }

    /// Check if a layer path refers to a combined QKV projection
    private func isCombinedQKVLayer(_ bflPath: String) -> Bool {
        return bflPath.contains(".img_attn.qkv") || bflPath.contains(".txt_attn.qkv")
    }

    /// Split combined QKV LoRA weights into separate Q, K, V parts
    /// The loraB output dimension is 3x the individual projection dimension
    private func splitQKVLoRA(bflPath: String, loraA: MLXArray, loraB: MLXArray) -> [(String, MLXArray, MLXArray)] {
        let blockIdx = extractBlockIndex(from: bflPath, prefix: "double_blocks.")

        // loraB shape: [3*innerDim, rank] - split along first axis
        let totalOutputDim = loraB.shape[0]
        let singleDim = totalOutputDim / 3

        // Split loraB into Q, K, V parts
        let loraBQ = loraB[0..<singleDim, 0...]
        let loraBK = loraB[singleDim..<(singleDim * 2), 0...]
        let loraBV = loraB[(singleDim * 2)..., 0...]

        // loraA is shared for Q, K, V
        if bflPath.contains(".img_attn.qkv") {
            return [
                ("transformerBlocks.\(blockIdx).attn.toQ", loraA, loraBQ),
                ("transformerBlocks.\(blockIdx).attn.toK", loraA, loraBK),
                ("transformerBlocks.\(blockIdx).attn.toV", loraA, loraBV)
            ]
        } else {
            // txt_attn.qkv
            return [
                ("transformerBlocks.\(blockIdx).attn.addQProj", loraA, loraBQ),
                ("transformerBlocks.\(blockIdx).attn.addKProj", loraA, loraBK),
                ("transformerBlocks.\(blockIdx).attn.addVProj", loraA, loraBV)
            ]
        }
    }

    /// Map BFL layer path to Swift module path
    /// Note: Combined QKV layers (.img_attn.qkv, .txt_attn.qkv) are handled
    /// separately by splitQKVLoRA() and should not reach this function.
    private func mapBFLPathToSwiftPath(_ bflPath: String) -> String {
        // Double block attention output projections
        if bflPath.contains("double_blocks.") {
            let blockIdx = extractBlockIndex(from: bflPath, prefix: "double_blocks.")

            if bflPath.contains(".img_attn.proj") {
                return "transformerBlocks.\(blockIdx).attn.toOut"
            } else if bflPath.contains(".txt_attn.proj") {
                return "transformerBlocks.\(blockIdx).attn.toAddOut"
            }
            // Note: .img_attn.qkv and .txt_attn.qkv are split into Q/K/V in splitQKVLoRA()
        }

        // Single block linear layers
        if bflPath.contains("single_blocks.") {
            let blockIdx = extractBlockIndex(from: bflPath, prefix: "single_blocks.")

            if bflPath.contains(".linear1") {
                return "singleTransformerBlocks.\(blockIdx).attn.toQkvMlp"
            } else if bflPath.contains(".linear2") {
                return "singleTransformerBlocks.\(blockIdx).attn.toOut"
            }
        }

        // Modulation layers
        if bflPath.contains("double_stream_modulation_img") {
            return "doubleStreamModulationImg.linear"
        } else if bflPath.contains("double_stream_modulation_txt") {
            return "doubleStreamModulationTxt.linear"
        } else if bflPath.contains("single_stream_modulation") {
            return "singleStreamModulation.linear"
        }

        // Input/output layers
        if bflPath == "img_in" {
            return "xEmbedder"
        } else if bflPath == "txt_in" {
            return "contextEmbedder"
        } else if bflPath == "final_layer.linear" {
            return "projOut"
        }

        // Time embeddings
        if bflPath.contains("time_in.in_layer") {
            return "timeGuidanceEmbed.timestepEmbedder.inLayer"
        } else if bflPath.contains("time_in.out_layer") {
            return "timeGuidanceEmbed.timestepEmbedder.outLayer"
        }

        // Return original path if no mapping found
        Flux2Debug.verbose("[LoRA] No mapping for: \(bflPath)")
        return bflPath
    }

    /// Extract block index from layer path
    private func extractBlockIndex(from path: String, prefix: String) -> Int {
        guard let startRange = path.range(of: prefix) else { return 0 }
        let afterPrefix = path[startRange.upperBound...]
        let indexStr = afterPrefix.prefix(while: { $0.isNumber })
        return Int(indexStr) ?? 0
    }

    /// Detect target model based on layer structure
    private func detectTargetModel(_ layers: [String]) -> LoRAInfo.TargetModel {
        // Count double and single blocks
        let doubleBlocks = Set(layers.compactMap { path -> Int? in
            guard path.hasPrefix("double_blocks.") else { return nil }
            return extractBlockIndex(from: path, prefix: "double_blocks.")
        })

        let singleBlocks = Set(layers.compactMap { path -> Int? in
            guard path.hasPrefix("single_blocks.") else { return nil }
            return extractBlockIndex(from: path, prefix: "single_blocks.")
        })

        let maxDouble = doubleBlocks.max() ?? 0
        let maxSingle = singleBlocks.max() ?? 0

        Flux2Debug.log("[LoRA] Detected structure: \(maxDouble + 1) double blocks, \(maxSingle + 1) single blocks")

        // Klein 4B: 5 double, 20 single
        // Klein 9B: 8 double, 24 single
        // Dev: 8 double, 48 single

        if maxDouble == 4 && maxSingle == 19 {
            return .klein4B
        } else if maxDouble == 7 && maxSingle == 23 {
            return .klein9B
        } else if maxDouble == 7 && maxSingle == 47 {
            return .dev
        }

        return .unknown
    }

    /// Get LoRA weight pair for a specific layer
    public func getWeights(for layerPath: String) -> LoRAWeightPair? {
        return weights[layerPath]
    }

    /// Get all layer paths that have LoRA weights
    public var layerPaths: [String] {
        Array(weights.keys).sorted()
    }
}
