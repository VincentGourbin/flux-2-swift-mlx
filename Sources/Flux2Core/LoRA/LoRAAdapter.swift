// LoRAAdapter.swift - Apply LoRA weights to transformer layers
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Applies LoRA weights to a base linear layer
///
/// LoRA (Low-Rank Adaptation) adds low-rank updates to model weights:
/// `output = base_linear(x) + scale * lora_B(lora_A(x))`
///
/// This allows efficient fine-tuning without modifying the base model weights.
public class LoRALinear: Module, @unchecked Sendable {
    /// The base linear layer (frozen)
    let base: Linear

    /// LoRA A matrix (down projection): [rank, input_dim]
    let loraA: MLXArray

    /// LoRA B matrix (up projection): [output_dim, rank]
    let loraB: MLXArray

    /// Scale factor for LoRA output
    var scale: Float

    /// Initialize LoRA-enhanced linear layer
    /// - Parameters:
    ///   - base: The original linear layer
    ///   - loraA: LoRA A matrix
    ///   - loraB: LoRA B matrix
    ///   - scale: Scale factor for LoRA output
    public init(base: Linear, loraA: MLXArray, loraB: MLXArray, scale: Float = 1.0) {
        self.base = base
        self.loraA = loraA
        self.loraB = loraB
        self.scale = scale
    }

    /// Forward pass with LoRA adaptation
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear output
        let baseOutput = base(x)

        // LoRA adaptation: scale * (x @ loraA.T @ loraB.T)
        // loraA: [rank, in_features] -> transpose to [in_features, rank]
        // loraB: [out_features, rank] -> transpose to [rank, out_features]
        let loraOutput = matmul(matmul(x, loraA.T), loraB.T)

        return baseOutput + scale * loraOutput
    }
}

/// Manages multiple LoRA adapters and their application to a transformer
public class LoRAManager: @unchecked Sendable {
    /// Loaded LoRA loaders
    private var loaders: [LoRALoader] = []

    /// Layer path to LoRA linear mapping (for lookup during forward)
    private var layerMappings: [String: [(loader: LoRALoader, pair: LoRAWeightPair)]] = [:]

    /// Initialize empty LoRA manager
    public init() {}

    /// Load and register a LoRA
    /// - Parameter config: LoRA configuration
    /// - Returns: LoRA info after loading
    @discardableResult
    public func loadLoRA(_ config: LoRAConfig) throws -> LoRAInfo {
        let loader = LoRALoader(config: config)
        try loader.load()

        guard let info = loader.info else {
            throw LoRALoaderError.invalidFormat("Failed to get LoRA info")
        }

        loaders.append(loader)

        // Register layer mappings
        for path in loader.layerPaths {
            if layerMappings[path] == nil {
                layerMappings[path] = []
            }
            if let pair = loader.getWeights(for: path) {
                layerMappings[path]?.append((loader: loader, pair: pair))
            }
        }

        Flux2Debug.log("[LoRA] Registered \(config.name) with \(info.numLayers) layers")

        return info
    }

    /// Unload a LoRA by name
    public func unloadLoRA(name: String) {
        loaders.removeAll { $0.config.name == name }
        rebuildMappings()
    }

    /// Unload all LoRAs
    public func unloadAll() {
        loaders.removeAll()
        layerMappings.removeAll()
    }

    /// Rebuild layer mappings after unload
    private func rebuildMappings() {
        layerMappings.removeAll()
        for loader in loaders {
            for path in loader.layerPaths {
                if layerMappings[path] == nil {
                    layerMappings[path] = []
                }
                if let pair = loader.getWeights(for: path) {
                    layerMappings[path]?.append((loader: loader, pair: pair))
                }
            }
        }
    }

    /// Update scale for a specific LoRA
    public func setScale(name: String, scale: Float) {
        for loader in loaders {
            if loader.config.name == name {
                // LoRAConfig is a struct, need to work around mutability
                // For now, we'll track scales separately
                // TODO: Make scale mutable in LoRAConfig
            }
        }
    }

    /// Check if a layer has LoRA adapters
    public func hasLoRA(for layerPath: String) -> Bool {
        return layerMappings[layerPath] != nil && !layerMappings[layerPath]!.isEmpty
    }

    /// Get all LoRA pairs for a layer path
    public func getLoRAPairs(for layerPath: String) -> [(scale: Float, loraA: MLXArray, loraB: MLXArray)] {
        guard let mappings = layerMappings[layerPath] else { return [] }

        return mappings.map { mapping in
            (
                scale: mapping.loader.config.effectiveScale,
                loraA: mapping.pair.loraA.array,
                loraB: mapping.pair.loraB.array
            )
        }
    }

    /// Apply LoRA to a linear layer output
    /// - Parameters:
    ///   - baseOutput: Output from the base linear layer
    ///   - input: Original input to the linear layer
    ///   - layerPath: Path identifying the layer
    /// - Returns: Output with LoRA adaptation applied
    public func applyLoRA(baseOutput: MLXArray, input: MLXArray, layerPath: String) -> MLXArray {
        let pairs = getLoRAPairs(for: layerPath)
        guard !pairs.isEmpty else { return baseOutput }

        var output = baseOutput

        for (scale, loraA, loraB) in pairs {
            // LoRA: output += scale * (input @ loraA.T @ loraB.T)
            let loraOutput = matmul(matmul(input, loraA.T), loraB.T)
            output = output + scale * loraOutput
        }

        return output
    }

    /// Number of loaded LoRAs
    public var count: Int {
        loaders.count
    }

    /// Names of loaded LoRAs
    public var loadedNames: [String] {
        loaders.map { $0.config.name }
    }

    /// All layer paths that have LoRA weights
    public var loadedLayerPaths: [String] {
        Array(layerMappings.keys).sorted()
    }

    /// Get combined activation keywords from all loaded LoRAs
    public var activationKeywords: [String] {
        loaders.compactMap { $0.config.activationKeyword }
    }

    /// Prepend activation keywords to a prompt
    public func enhancePrompt(_ prompt: String) -> String {
        let keywords = activationKeywords
        if keywords.isEmpty { return prompt }
        return keywords.joined(separator: ", ") + ", " + prompt
    }

    /// Clear LoRA weights from memory after fusion into base model
    ///
    /// After LoRA weights have been merged into the transformer via `mergeLoRAWeights`,
    /// the original LoRA matrices are no longer needed. This method frees that memory.
    ///
    /// - Note: After calling this, LoRAs cannot be "unfused" without reloading the base model.
    public func clearWeightsAfterFusion() {
        var totalFreed: Float = 0
        for loader in loaders {
            if loader.hasWeightsInMemory {
                totalFreed += loader.info?.memorySizeMB ?? 0
                loader.clearWeightsAfterFusion()
            }
        }
        layerMappings.removeAll()
        Flux2Debug.log("[LoRA] Total memory freed after fusion: ~\(String(format: "%.1f", totalFreed)) MB")
    }

    /// Whether any loaders still have weights in memory
    public var hasWeightsInMemory: Bool {
        loaders.contains { $0.hasWeightsInMemory }
    }
}
