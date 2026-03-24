/**
 * Qwen3VLModel.swift
 * Main Qwen3-VL model architecture (language component only)
 *
 * Phase 1: Text-only evaluation — no vision encoder, no DeepStack.
 * Loads only the language model weights from Qwen3-VL safetensors,
 * ignoring visual.* and cross_attn.* keys.
 *
 * The key difference from Qwen3Model is MRoPE-based attention
 * with position_ids propagated through the forward pass.
 */

import Foundation
import MLX
import MLXNN

// MARK: - Model Output

/// Output structure for Qwen3-VL model
public struct Qwen3VLModelOutput {
    public let logits: MLXArray
    public let hiddenStates: [MLXArray]?
    public let lastHiddenState: MLXArray

    public init(logits: MLXArray, hiddenStates: [MLXArray]? = nil, lastHiddenState: MLXArray) {
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.lastHiddenState = lastHiddenState
    }
}

// MARK: - Main Model

/// Qwen3-VL transformer model (language component)
public class Qwen3VLModel: Module {
    public let config: Qwen3VLTextConfig

    /// Memory optimization configuration
    public var memoryConfig: TextEncoderMemoryConfig = .disabled

    @ModuleInfo public var embed_tokens: Embedding
    public var layers: [Qwen3VLDecoderLayer]
    public var norm: RMSNorm

    public init(config: Qwen3VLTextConfig) {
        self.config = config

        self._embed_tokens = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        ))

        self.layers = (0..<config.numHiddenLayers).map { _ in
            Qwen3VLDecoderLayer(config: config)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        cache: [KVCache]? = nil,
        outputHiddenStates: Bool = false,
        attentionMask: MLXArray? = nil
    ) -> (hiddenStates: MLXArray, allHiddenStates: [MLXArray]?) {
        var hiddenStates = embed_tokens(inputIds)

        // Generate text-only position IDs if not provided
        // CRITICAL: Use cache offset so position IDs are correct during KV-cached generation
        let cacheOffset = cache?.first?.length ?? 0
        let posIds = positionIds ?? Qwen3VLMRoPE.textOnlyPositionIds(seqLen: inputIds.shape[1], offset: cacheOffset)

        // Create causal mask with optional padding mask
        let mask = createCausalMask(
            seqLen: inputIds.shape[1],
            offset: cacheOffset,
            attentionMask: attentionMask
        )

        var allHiddenStates: [MLXArray]? = outputHiddenStates ? [] : nil

        for (i, layer) in layers.enumerated() {
            if outputHiddenStates {
                eval(hiddenStates)
                allHiddenStates?.append(hiddenStates)
            }

            let layerCache = cache?[i]
            hiddenStates = layer(hiddenStates, mask: mask, positionIds: posIds, cache: layerCache)

            // Memory optimization: periodic evaluation
            if memoryConfig.evalFrequency > 0 && (i + 1) % memoryConfig.evalFrequency == 0 {
                eval(hiddenStates)
                if memoryConfig.clearCacheOnEval {
                    MLX.Memory.clearCache()
                }
            }
        }

        hiddenStates = norm(hiddenStates)

        if outputHiddenStates {
            eval(hiddenStates)
            allHiddenStates?.append(hiddenStates)
        }

        return (hiddenStates, allHiddenStates)
    }

    /// Forward pass with hidden states extraction at specific layers
    /// Returns a dictionary mapping layer indices to hidden states
    public func forwardWithHiddenStates(
        _ inputIds: MLXArray,
        layerIndices: [Int],
        positionIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> [Int: MLXArray] {
        var hiddenStates = embed_tokens(inputIds)
        eval(hiddenStates)

        // Generate text-only position IDs if not provided
        let posIds = positionIds ?? Qwen3VLMRoPE.textOnlyPositionIds(seqLen: inputIds.shape[1])

        let mask = createCausalMask(
            seqLen: inputIds.shape[1],
            offset: 0,
            attentionMask: attentionMask
        )
        if let m = mask {
            eval(m)
        }

        let layerSet = Set(layerIndices)
        var extractedStates: [Int: MLXArray] = [:]

        if layerSet.contains(0) {
            extractedStates[0] = hiddenStates
        }

        for (i, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, positionIds: posIds, cache: nil)

            let layerIdx = i + 1

            let shouldEval = layerSet.contains(layerIdx) ||
                            (memoryConfig.evalFrequency > 0 && (i + 1) % memoryConfig.evalFrequency == 0)

            if shouldEval {
                eval(hiddenStates)
                if memoryConfig.clearCacheOnEval {
                    MLX.Memory.clearCache()
                }
            }

            if layerSet.contains(layerIdx) {
                extractedStates[layerIdx] = hiddenStates
            }
        }

        let numLayers = layers.count
        if layerSet.contains(numLayers) {
            let normalizedStates = norm(hiddenStates)
            eval(normalizedStates)
            extractedStates[numLayers] = normalizedStates
        }

        MLX.Memory.clearCache()

        return extractedStates
    }

    private func createCausalMask(seqLen: Int, offset: Int, attentionMask: MLXArray? = nil) -> MLXArray? {
        if seqLen == 1 && attentionMask == nil {
            return nil
        }

        let totalLen = seqLen + offset
        let rowIndices = MLXArray.arange(seqLen, dtype: .float32).expandedDimensions(axis: 1)
        let colIndices = MLXArray.arange(totalLen, dtype: .float32).expandedDimensions(axis: 0)

        var mask = MLX.where(
            colIndices .<= (rowIndices + Float(offset)),
            MLXArray(Float(0.0)),
            MLXArray(-Float.infinity)
        )

        if let attnMask = attentionMask {
            let maskValue: Float = -1e9
            let paddingMask = MLX.where(
                attnMask .== Int32(1),
                MLXArray(Float(0.0)),
                MLXArray(maskValue)
            ).reshaped([attnMask.shape[0], 1, 1, attnMask.shape[1]])

            mask = mask.reshaped([1, 1, seqLen, totalLen]) + paddingMask
        } else {
            mask = mask.reshaped([1, 1, seqLen, totalLen])
        }

        return mask
    }
}

// MARK: - Language Model Head

/// Full Qwen3-VL model with language model head
public class Qwen3VLForCausalLM: Module {
    public let config: Qwen3VLTextConfig
    public var model: Qwen3VLModel
    @ModuleInfo public var lm_head: Linear

    public var useTiedWeights: Bool = false

    public init(config: Qwen3VLTextConfig) {
        self.config = config
        self.model = Qwen3VLModel(config: config)
        self._lm_head = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false))
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
        } else {
            return lm_head(hiddenStates)
        }
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        positionIds: MLXArray? = nil,
        cache: [KVCache]? = nil,
        outputHiddenStates: Bool = false,
        attentionMask: MLXArray? = nil
    ) -> Qwen3VLModelOutput {
        let (hiddenStates, allHiddenStates) = model(
            inputIds,
            positionIds: positionIds,
            cache: cache,
            outputHiddenStates: outputHiddenStates,
            attentionMask: attentionMask
        )
        let logits = computeLogits(hiddenStates)

        return Qwen3VLModelOutput(
            logits: logits,
            hiddenStates: allHiddenStates,
            lastHiddenState: hiddenStates
        )
    }

    public func forward(_ inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let (hiddenStates, _) = model(inputIds, cache: cache, outputHiddenStates: false)
        return computeLogits(hiddenStates)
    }

    public func createCache() -> [KVCache] {
        return (0..<config.numHiddenLayers).map { _ in KVCache() }
    }
}

// MARK: - Quantization Config (reuse same format)

private struct Qwen3VLQuantizationConfig: Codable {
    let groupSize: Int
    let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private struct Qwen3VLModelConfigWithQuantization: Codable {
    let quantization: Qwen3VLQuantizationConfig?
}

// MARK: - Model Loading

extension Qwen3VLForCausalLM {
    /// Load Qwen3-VL model from path (language component only)
    /// Ignores visual.* and cross_attn.* weight keys
    public static func load(from modelPath: String) throws -> Qwen3VLForCausalLM {
        let config = try Qwen3VLTextConfig.load(from: "\(modelPath)/config.json")
        let model = Qwen3VLForCausalLM(config: config)

        // Check for quantization
        let configPath = "\(modelPath)/config.json"
        let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
        if let quantConfig = try? JSONDecoder().decode(Qwen3VLModelConfigWithQuantization.self, from: configData),
           let quant = quantConfig.quantization {
            FluxDebug.log("Qwen3-VL model is quantized: groupSize=\(quant.groupSize), bits=\(quant.bits)")
            quantize(model: model, groupSize: quant.groupSize, bits: quant.bits)
        }

        // Find safetensors files
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        if safetensorFiles.isEmpty {
            throw Qwen3VLModelError.noWeightsFound
        }

        FluxDebug.log("Loading Qwen3-VL weights from \(safetensorFiles.count) safetensor files...")

        // Load weights, filtering out vision encoder and remapping language model keys
        // VL safetensors use "language_model.model.layers.X..." instead of "model.layers.X..."
        // and "vision_tower.*" for vision encoder (skip these)
        var allWeights: [String: MLXArray] = [:]
        var skippedKeys = 0

        for filename in safetensorFiles {
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))
            for (key, value) in weights {
                // Skip vision encoder weights
                if key.hasPrefix("visual.") || key.hasPrefix("vision_tower.") {
                    skippedKeys += 1
                    continue
                }
                // Skip DeepStack cross-attention weights
                if key.contains("cross_attn") {
                    skippedKeys += 1
                    continue
                }

                // Remap VL key prefix: "language_model.model.X" → "model.X"
                var mappedKey = key
                if key.hasPrefix("language_model.") {
                    mappedKey = String(key.dropFirst("language_model.".count))
                }

                allWeights[mappedKey] = value
            }
        }

        FluxDebug.log("Qwen3-VL: loaded \(allWeights.count) language model tensors, skipped \(skippedKeys) vision tensors")

        // Apply weights
        try model.loadWeights(allWeights)

        // Enable weight tying if applicable
        let hasLmHead = allWeights.keys.contains { $0.contains("lm_head") }
        if !hasLmHead && config.tieWordEmbeddings {
            FluxDebug.log("Qwen3-VL: Enabling weight tying (lm_head uses embed_tokens weights)")
            model.useTiedWeights = true
        }

        FluxDebug.log("Qwen3-VL model loaded successfully")

        return model
    }

    private func loadWeights(_ weights: [String: MLXArray]) throws {
        var convertedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Qwen3-VL language model keys match MLX Swift format directly
            convertedWeights[key] = value
        }

        let parameters = ModuleParameters.unflattened(convertedWeights)

        do {
            try update(parameters: parameters, verify: .noUnusedKeys)
        } catch {
            FluxDebug.log("Qwen3-VL weight loading warning: \(error)")
            try update(parameters: parameters, verify: .none)
        }

        eval(self)

        FluxDebug.log("Qwen3-VL weights applied successfully")
    }
}

// MARK: - Errors

public enum Qwen3VLModelError: LocalizedError {
    case noWeightsFound
    case invalidConfig
    case loadError(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound:
            return "No safetensors files found in Qwen3-VL model directory"
        case .invalidConfig:
            return "Invalid Qwen3-VL model configuration"
        case .loadError(let message):
            return "Failed to load Qwen3-VL model: \(message)"
        }
    }
}
