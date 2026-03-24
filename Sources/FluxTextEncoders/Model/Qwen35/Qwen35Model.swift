/**
 * Qwen35Model.swift
 * Main Qwen3.5 language model with hybrid attention (Gated DeltaNet + GQA)
 *
 * 32 layers: 24 linear attention + 8 full attention
 * Manages two cache types: KVCache for full attn, DeltaNetCache for linear attn
 */

import Foundation
import MLX
import MLXNN

// MARK: - Hybrid Cache

/// Cache container for one layer: either KV cache (full attn) or DeltaNet cache (linear attn)
public class Qwen35LayerCache {
    public let isLinear: Bool

    // Full attention cache
    public var kvCache: KVCache?

    // DeltaNet cache: [conv_state, recurrent_state]
    public var deltaCache: Qwen35GatedDeltaNet.DeltaNetCache

    /// Offset for position tracking (full attention layers)
    public var offset: Int = 0

    public init(isLinear: Bool) {
        self.isLinear = isLinear
        if isLinear {
            self.kvCache = nil
            self.deltaCache = [nil, nil]
        } else {
            self.kvCache = KVCache()
            self.deltaCache = [nil, nil]
        }
    }

    public var length: Int {
        if isLinear {
            return offset
        } else {
            return kvCache?.length ?? 0
        }
    }
}

// MARK: - Model

public class Qwen35LanguageModel: Module {
    public let config: Qwen35TextConfig

    @ModuleInfo public var embed_tokens: Embedding
    public var layers: [Qwen35DecoderLayer]
    public var norm: RMSNorm

    public var memoryConfig: TextEncoderMemoryConfig = .disabled

    public init(config: Qwen35TextConfig) {
        self.config = config

        self._embed_tokens = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        ))

        self.layers = (0..<config.numHiddenLayers).map { i in
            Qwen35DecoderLayer(config: config, layerIndex: i)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        cache: [Qwen35LayerCache]? = nil,
        mask: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var hiddenStates = inputEmbeddings ?? embed_tokens(inputIds)

        // Position IDs for full attention layers
        let cacheOffset = cache?.first(where: { !$0.isLinear })?.length ?? 0
        let seqLen = hiddenStates.dim(1)
        let posIds = positionIds ?? Qwen35MRoPE.textOnlyPositionIds(seqLen: seqLen, offset: cacheOffset)

        // Causal mask for full attention layers
        let causalMask = createCausalMask(seqLen: seqLen, offset: cacheOffset)

        // Mask for linear attention (padding mask)
        // For linear attn, mask indicates which positions are valid (true = valid)
        // We use nil for now (no padding in generation)

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            var deltaCache = layerCache?.deltaCache ?? [nil, nil]

            hiddenStates = layer(
                hiddenStates,
                mask: layer.isLinear ? nil : causalMask,
                positionIds: posIds,
                kvCache: layerCache?.kvCache,
                deltaCache: &deltaCache
            )

            // Write back delta cache
            if let layerCache = layerCache {
                layerCache.deltaCache = deltaCache
                layerCache.offset += seqLen
            }

            // Memory optimization
            if memoryConfig.evalFrequency > 0 && (i + 1) % memoryConfig.evalFrequency == 0 {
                eval(hiddenStates)
                if memoryConfig.clearCacheOnEval {
                    MLX.Memory.clearCache()
                }
            }
        }

        return norm(hiddenStates)
    }

    private func createCausalMask(seqLen: Int, offset: Int) -> MLXArray? {
        if seqLen == 1 { return nil }

        let totalLen = seqLen + offset
        let rowIndices = MLXArray.arange(seqLen, dtype: .float32).expandedDimensions(axis: 1)
        let colIndices = MLXArray.arange(totalLen, dtype: .float32).expandedDimensions(axis: 0)

        let mask = MLX.where(
            colIndices .<= (rowIndices + Float(offset)),
            MLXArray(Float(0.0)),
            MLXArray(-Float.infinity)
        ).reshaped([1, 1, seqLen, totalLen])

        return mask
    }
}

// MARK: - Conditional Generation (VLM wrapper)

public class Qwen35ForConditionalGeneration: Module {
    public let config: Qwen35Config
    public var model: Qwen35LanguageModel
    @ModuleInfo public var lm_head: Linear

    public var useTiedWeights: Bool = false

    public init(config: Qwen35Config) {
        self.config = config
        self.model = Qwen35LanguageModel(config: config.textConfig)
        self._lm_head = ModuleInfo(wrappedValue: Linear(
            config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false))
        super.init()
    }

    private func computeLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if useTiedWeights {
            if let quantizedEmbed = model.embed_tokens as? QuantizedEmbedding {
                return MLX.quantizedMM(
                    hiddenStates,
                    quantizedEmbed.weight,
                    scales: quantizedEmbed.scales,
                    biases: quantizedEmbed.biases,
                    transpose: true,
                    groupSize: quantizedEmbed.groupSize,
                    bits: quantizedEmbed.bits
                )
            } else {
                return MLX.matmul(hiddenStates, model.embed_tokens.weight.T)
            }
        }
        return lm_head(hiddenStates)
    }

    /// Forward pass for generation
    public func forward(
        _ inputIds: MLXArray,
        cache: [Qwen35LayerCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let hiddenStates = model(inputIds, cache: cache, inputEmbeddings: inputEmbeddings)
        return computeLogits(hiddenStates)
    }

    /// Create cache for all layers
    public func createCache() -> [Qwen35LayerCache] {
        return (0..<config.textConfig.numHiddenLayers).map { i in
            Qwen35LayerCache(isLinear: config.textConfig.isLinearLayer(i))
        }
    }
}

// MARK: - Quantization Config

private struct Qwen35QuantizationConfig: Decodable {
    let groupSize: Int
    let bits: Int
    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private struct Qwen35ConfigWithQuantization: Decodable {
    let quantization: Qwen35QuantizationConfig?
}

// MARK: - Model Loading

extension Qwen35ForConditionalGeneration {
    /// Load Qwen3.5 model from path (language component)
    /// Skips vision_tower.* weights — vision loaded separately
    public static func load(from modelPath: String) throws -> Qwen35ForConditionalGeneration {
        let config = try Qwen35Config.load(from: "\(modelPath)/config.json")
        let model = Qwen35ForConditionalGeneration(config: config)

        // Check quantization
        let configData = try Data(contentsOf: URL(fileURLWithPath: "\(modelPath)/config.json"))
        if let quantConfig = try? JSONDecoder().decode(Qwen35ConfigWithQuantization.self, from: configData),
           let quant = quantConfig.quantization {
            FluxDebug.log("Qwen3.5 model is quantized: groupSize=\(quant.groupSize), bits=\(quant.bits)")
            quantize(model: model, groupSize: quant.groupSize, bits: quant.bits)
        }

        // Find safetensors
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        guard !safetensorFiles.isEmpty else {
            throw Qwen35ModelError.noWeightsFound
        }

        FluxDebug.log("Loading Qwen3.5 weights from \(safetensorFiles.count) safetensor files...")

        // Load weights, remapping VL key prefix
        var allWeights: [String: MLXArray] = [:]
        var skippedKeys = 0

        for filename in safetensorFiles {
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))
            for (key, value) in weights {
                // Skip vision tower weights
                if key.hasPrefix("vision_tower.") {
                    skippedKeys += 1
                    continue
                }

                // Remap: "language_model.model.X" → "model.X"
                var mappedKey = key
                if key.hasPrefix("language_model.") {
                    mappedKey = String(key.dropFirst("language_model.".count))
                }

                allWeights[mappedKey] = value
            }
        }

        FluxDebug.log("Qwen3.5: loaded \(allWeights.count) language tensors, skipped \(skippedKeys) vision tensors")

        // Apply weights
        let parameters = ModuleParameters.unflattened(allWeights)
        do {
            try model.update(parameters: parameters, verify: .noUnusedKeys)
        } catch {
            FluxDebug.log("Qwen3.5 weight loading warning: \(error)")
            try model.update(parameters: parameters, verify: .none)
        }
        eval(model)

        // Weight tying
        let hasLmHead = allWeights.keys.contains { $0.contains("lm_head") }
        if !hasLmHead && config.textConfig.tieWordEmbeddings {
            FluxDebug.log("Qwen3.5: Enabling weight tying")
            model.useTiedWeights = true
        }

        FluxDebug.log("Qwen3.5 model loaded successfully")
        return model
    }
}

// MARK: - Errors

public enum Qwen35ModelError: LocalizedError {
    case noWeightsFound
    case invalidConfig
    case loadError(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound: return "No safetensors files found in Qwen3.5 model directory"
        case .invalidConfig: return "Invalid Qwen3.5 model configuration"
        case .loadError(let msg): return "Failed to load Qwen3.5 model: \(msg)"
        }
    }
}
