/**
 * MistralModel.swift
 * Main Mistral model architecture with hidden states extraction support
 */

import Foundation
import MLX
import MLXNN

// MARK: - Decoder Layer

/// Single Mistral decoder layer
public class MistralDecoderLayer: Module {
    let config: MistralTextConfig

    public var self_attn: MistralAttention
    public var mlp: MistralMLP
    public var input_layernorm: RMSNorm
    public var post_attention_layernorm: RMSNorm

    public init(config: MistralTextConfig) {
        self.config = config

        self.self_attn = MistralAttention(config: config)
        self.mlp = MistralMLP(config: config)
        self.input_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.post_attention_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Self-attention with residual
        let residual = hiddenStates
        let normalizedHidden = input_layernorm(hiddenStates)
        let attnOutput = self_attn(normalizedHidden, mask: mask, cache: cache)
        var hidden = residual + attnOutput

        // MLP with residual
        let residual2 = hidden
        let normalizedHidden2 = post_attention_layernorm(hidden)
        let mlpOutput = mlp(normalizedHidden2)
        hidden = residual2 + mlpOutput

        return hidden
    }
}

// MARK: - Model Output

/// Output structure containing logits and optional hidden states
public struct MistralModelOutput {
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

/// Mistral transformer model
public class MistralModel: Module {
    public let config: MistralTextConfig

    @ModuleInfo public var embed_tokens: Embedding
    public var layers: [MistralDecoderLayer]
    public var norm: RMSNorm

    public init(config: MistralTextConfig) {
        self.config = config

        self._embed_tokens = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        ))

        self.layers = (0..<config.numHiddenLayers).map { _ in
            MistralDecoderLayer(config: config)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]? = nil,
        outputHiddenStates: Bool = false,
        attentionMask: MLXArray? = nil
    ) -> (hiddenStates: MLXArray, allHiddenStates: [MLXArray]?) {
        var hiddenStates = embed_tokens(inputIds)

        // Create causal mask with optional padding mask
        let mask = createCausalMask(
            seqLen: inputIds.shape[1],
            offset: cache?.first?.length ?? 0,
            attentionMask: attentionMask
        )

        // Collect hidden states if requested
        var allHiddenStates: [MLXArray]? = outputHiddenStates ? [] : nil

        // Pass through layers
        for (i, layer) in layers.enumerated() {
            if outputHiddenStates {
                allHiddenStates?.append(hiddenStates)
            }

            let layerCache = cache?[i]
            hiddenStates = layer(hiddenStates, mask: mask, cache: layerCache)
        }

        // Final normalization
        hiddenStates = norm(hiddenStates)

        if outputHiddenStates {
            allHiddenStates?.append(hiddenStates)
        }

        return (hiddenStates, allHiddenStates)
    }

    private func createCausalMask(seqLen: Int, offset: Int, attentionMask: MLXArray? = nil) -> MLXArray? {
        if seqLen == 1 && attentionMask == nil {
            return nil
        }

        let totalLen = seqLen + offset

        // GPU-based causal mask creation (replaces O(seqLen * totalLen) CPU loop)
        // Row indices: [seqLen, 1] - each row i
        let rowIndices = MLXArray(Array(0..<seqLen).map { Float($0) }).expandedDimensions(axis: 1)
        // Column indices: [1, totalLen] - each column j
        let colIndices = MLXArray(Array(0..<totalLen).map { Float($0) }).expandedDimensions(axis: 0)

        // Causal mask: allow position j if j <= i + offset
        // mask[i,j] = 0 if j <= i + offset, else -inf
        var mask = MLX.where(
            colIndices .<= (rowIndices + Float(offset)),
            MLXArray(Float(0.0)),
            MLXArray(-Float.infinity)
        )

        // Combine with attention mask (for padding) if provided
        // attentionMask shape: [batch, seqLen] with 1 for real tokens, 0 for padding
        if let attnMask = attentionMask {
            // Convert 0/1 mask to 0/large-negative mask
            // Use -1e9 instead of -inf to avoid NaN in softmax for padding positions
            // HuggingFace uses torch.finfo(dtype).min which is similar
            // paddingMask shape: [batch, 1, 1, seqLen]
            let maskValue: Float = -1e9
            let paddingMask = MLX.where(
                attnMask .== Int32(1),
                MLXArray(Float(0.0)),
                MLXArray(maskValue)
            ).reshaped([attnMask.shape[0], 1, 1, attnMask.shape[1]])

            // Combine: add padding mask to causal mask
            // mask shape: [1, 1, seqLen, totalLen]
            mask = mask.reshaped([1, 1, seqLen, totalLen]) + paddingMask
        } else {
            mask = mask.reshaped([1, 1, seqLen, totalLen])
        }

        return mask
    }
}

// MARK: - Language Model Head

/// Full Mistral model with language model head
public class MistralForCausalLM: Module {
    public let config: MistralTextConfig
    public var model: MistralModel
    @ModuleInfo public var lm_head: Linear

    public init(config: MistralTextConfig) {
        self.config = config
        self.model = MistralModel(config: config)

        // LM head - ties weights with embeddings if configured
        if config.tieWordEmbeddings {
            // Weight tying will be done after loading
            self._lm_head = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false))
        } else {
            self._lm_head = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false))
        }

        super.init()
    }

    /// Forward pass returning full output
    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]? = nil,
        outputHiddenStates: Bool = false,
        attentionMask: MLXArray? = nil
    ) -> MistralModelOutput {
        let (hiddenStates, allHiddenStates) = model(
            inputIds,
            cache: cache,
            outputHiddenStates: outputHiddenStates,
            attentionMask: attentionMask
        )
        let logits = lm_head(hiddenStates)

        return MistralModelOutput(
            logits: logits,
            hiddenStates: allHiddenStates,
            lastHiddenState: hiddenStates
        )
    }

    /// Simple forward for generation (logits only)
    public func forward(_ inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let (hiddenStates, _) = model(inputIds, cache: cache, outputHiddenStates: false)
        return lm_head(hiddenStates)
    }

    /// Create new KV cache for generation
    public func createCache() -> [KVCache] {
        return (0..<config.numHiddenLayers).map { _ in KVCache() }
    }
}

// MARK: - Quantization Config

private struct QuantizationConfig: Codable {
    let groupSize: Int
    let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private struct ModelConfigWithQuantization: Codable {
    let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case quantization
    }
}

// MARK: - Model Loading

extension MistralForCausalLM {
    /// Load model from path
    public static func load(from modelPath: String) throws -> MistralForCausalLM {
        // Load config
        let config = try MistralTextConfig.load(from: "\(modelPath)/config.json")
        let model = MistralForCausalLM(config: config)

        // Check for quantization config
        let configPath = "\(modelPath)/config.json"
        let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
        if let quantConfig = try? JSONDecoder().decode(ModelConfigWithQuantization.self, from: configData),
           let quant = quantConfig.quantization {
            FluxDebug.log("Model is quantized: groupSize=\(quant.groupSize), bits=\(quant.bits)")
            // Replace Linear/Embedding layers with quantized versions
            quantize(model: model, groupSize: quant.groupSize, bits: quant.bits)
        }

        // Find safetensors files
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        if safetensorFiles.isEmpty {
            throw MistralModelError.noWeightsFound
        }

        FluxDebug.log("Loading weights from \(safetensorFiles.count) safetensor files...")

        // Load weights
        var allWeights: [String: MLXArray] = [:]

        for filename in safetensorFiles {
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))
            for (key, value) in weights {
                allWeights[key] = value
            }
        }

        // Apply weights to model
        try model.loadWeights(allWeights)

        FluxDebug.log("Model loaded successfully with \(allWeights.count) tensors")

        return model
    }

    private func loadWeights(_ weights: [String: MLXArray]) throws {
        // Convert HuggingFace weight keys to MLX Swift format
        var convertedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            let swiftKey = convertKeyName(key)
            convertedWeights[swiftKey] = value
        }

        FluxDebug.log("Converting \(convertedWeights.count) weight tensors...")

        // Unflatten the weights to create nested ModuleParameters structure
        let parameters = ModuleParameters.unflattened(convertedWeights)

        // Apply weights to model (using .none for verification to be permissive)
        // Some models may have extra keys we don't use
        try update(parameters: parameters, verify: .none)

        // Evaluate to ensure weights are loaded
        eval(self)

        FluxDebug.log("Weights applied successfully")
    }

    private func convertKeyName(_ key: String) -> String {
        // Convert from VLM HuggingFace format:
        //   language_model.model.layers.0.self_attn.q_proj.weight
        //   language_model.lm_head.weight
        // To MLX Swift format:
        //   model.layers.0.self_attn.q_proj.weight
        //   lm_head.weight
        //
        // For VLM models (Mistral3ForConditionalGeneration), all text model weights
        // have a "language_model." prefix that we need to strip.

        var result = key

        // Strip language_model. prefix for VLM models
        if result.hasPrefix("language_model.") {
            result = String(result.dropFirst("language_model.".count))
        }

        return result
    }
}

// MARK: - Errors

public enum MistralModelError: LocalizedError {
    case noWeightsFound
    case invalidConfig
    case loadError(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound:
            return "No safetensors files found in model directory"
        case .invalidConfig:
            return "Invalid model configuration"
        case .loadError(let message):
            return "Failed to load model: \(message)"
        }
    }
}
