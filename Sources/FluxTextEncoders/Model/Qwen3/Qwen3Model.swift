/**
 * Qwen3Model.swift
 * Main Qwen3 model architecture with hidden states extraction support
 *
 * Used as text encoder for FLUX.2 Klein (4B and 9B variants)
 */

import Foundation
import MLX
import MLXNN

// MARK: - Model Output

/// Output structure containing logits and optional hidden states
public struct Qwen3ModelOutput {
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

/// Qwen3 transformer model
public class Qwen3Model: Module {
    public let config: Qwen3TextConfig

    @ModuleInfo public var embed_tokens: Embedding
    public var layers: [Qwen3DecoderLayer]
    public var norm: RMSNorm

    public init(config: Qwen3TextConfig) {
        self.config = config

        self._embed_tokens = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        ))

        self.layers = (0..<config.numHiddenLayers).map { _ in
            Qwen3DecoderLayer(config: config)
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

    /// Forward pass with hidden states extraction at specific layers
    /// Returns a dictionary mapping layer indices to hidden states
    public func forwardWithHiddenStates(
        _ inputIds: MLXArray,
        layerIndices: [Int],
        attentionMask: MLXArray? = nil
    ) -> [Int: MLXArray] {
        var hiddenStates = embed_tokens(inputIds)

        // Create causal mask with optional padding mask
        let mask = createCausalMask(
            seqLen: inputIds.shape[1],
            offset: 0,
            attentionMask: attentionMask
        )

        // Set of layers to extract
        let layerSet = Set(layerIndices)

        // Collect hidden states at specified layers
        var extractedStates: [Int: MLXArray] = [:]

        // Check if we need layer 0 (before any transformer layers)
        if layerSet.contains(0) {
            extractedStates[0] = hiddenStates
        }

        // Pass through layers
        for (i, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: nil)

            // Layer indices are 1-based (layer 1 = after first layer)
            let layerIdx = i + 1
            if layerSet.contains(layerIdx) {
                extractedStates[layerIdx] = hiddenStates
            }
        }

        // Final normalization (only if needed for the last layer)
        let numLayers = layers.count
        if layerSet.contains(numLayers) {
            extractedStates[numLayers] = norm(hiddenStates)
        }

        return extractedStates
    }

    private func createCausalMask(seqLen: Int, offset: Int, attentionMask: MLXArray? = nil) -> MLXArray? {
        if seqLen == 1 && attentionMask == nil {
            return nil
        }

        let totalLen = seqLen + offset

        // GPU-based causal mask creation
        let rowIndices = MLXArray(Array(0..<seqLen).map { Float($0) }).expandedDimensions(axis: 1)
        let colIndices = MLXArray(Array(0..<totalLen).map { Float($0) }).expandedDimensions(axis: 0)

        // Causal mask: allow position j if j <= i + offset
        var mask = MLX.where(
            colIndices .<= (rowIndices + Float(offset)),
            MLXArray(Float(0.0)),
            MLXArray(-Float.infinity)
        )

        // Combine with attention mask (for padding) if provided
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

/// Full Qwen3 model with language model head
public class Qwen3ForCausalLM: Module {
    public let config: Qwen3TextConfig
    public var model: Qwen3Model
    @ModuleInfo public var lm_head: Linear

    /// When true, use embed_tokens.weight for lm_head (weight tying)
    public var useTiedWeights: Bool = false

    public init(config: Qwen3TextConfig) {
        self.config = config
        self.model = Qwen3Model(config: config)

        // LM head - for Qwen3, tie_word_embeddings is typically true
        // Weight tying will be handled via useTiedWeights flag
        self._lm_head = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false))

        super.init()
    }

    /// Compute logits from hidden states, using tied weights if enabled
    private func computeLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if useTiedWeights {
            // Use embed_tokens.weight for projection (weight tying)
            // For quantized embeddings, use quantizedMatmul with the quantized weights
            // For regular embeddings, use standard matmul
            if let quantizedEmbed = model.embed_tokens as? QuantizedEmbedding {
                // QuantizedEmbedding stores weight as quantized [vocab_size, hidden_size/pack_ratio]
                // Use MLX.quantizedMM for proper computation
                // hiddenStates: [batch, seq, hidden_size]
                // weight: [vocab_size, hidden_size] (quantized)
                // scales: [vocab_size, num_groups]
                // biases: [vocab_size, num_groups]
                // transpose = true means we compute hiddenStates @ weight.T
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
                // Regular embedding - use standard matmul
                // embed_tokens.weight shape: [vocab_size, hidden_size]
                // hiddenStates shape: [batch, seq, hidden_size]
                // output shape: [batch, seq, vocab_size]
                return MLX.matmul(hiddenStates, model.embed_tokens.weight.T)
            }
        } else {
            return lm_head(hiddenStates)
        }
    }

    /// Forward pass returning full output
    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]? = nil,
        outputHiddenStates: Bool = false,
        attentionMask: MLXArray? = nil
    ) -> Qwen3ModelOutput {
        let (hiddenStates, allHiddenStates) = model(
            inputIds,
            cache: cache,
            outputHiddenStates: outputHiddenStates,
            attentionMask: attentionMask
        )
        let logits = computeLogits(hiddenStates)

        return Qwen3ModelOutput(
            logits: logits,
            hiddenStates: allHiddenStates,
            lastHiddenState: hiddenStates
        )
    }

    /// Simple forward for generation (logits only)
    public func forward(_ inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let (hiddenStates, _) = model(inputIds, cache: cache, outputHiddenStates: false)
        return computeLogits(hiddenStates)
    }

    /// Create new KV cache for generation
    public func createCache() -> [KVCache] {
        return (0..<config.numHiddenLayers).map { _ in KVCache() }
    }
}

// MARK: - Quantization Config

private struct Qwen3QuantizationConfig: Codable {
    let groupSize: Int
    let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private struct Qwen3ModelConfigWithQuantization: Codable {
    let quantization: Qwen3QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case quantization
    }
}

// MARK: - Model Loading

extension Qwen3ForCausalLM {
    /// Load model from path
    public static func load(from modelPath: String) throws -> Qwen3ForCausalLM {
        // Load config
        let config = try Qwen3TextConfig.load(from: "\(modelPath)/config.json")

        let model = Qwen3ForCausalLM(config: config)

        // Check for quantization config
        let configPath = "\(modelPath)/config.json"
        let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
        if let quantConfig = try? JSONDecoder().decode(Qwen3ModelConfigWithQuantization.self, from: configData),
           let quant = quantConfig.quantization {
            FluxDebug.log("Qwen3 model is quantized: groupSize=\(quant.groupSize), bits=\(quant.bits)")
            // Replace Linear/Embedding layers with quantized versions
            quantize(model: model, groupSize: quant.groupSize, bits: quant.bits)
        }

        // Find safetensors files
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        if safetensorFiles.isEmpty {
            throw Qwen3ModelError.noWeightsFound
        }

        FluxDebug.log("Loading Qwen3 weights from \(safetensorFiles.count) safetensor files...")

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

        // Enable weight tying if tie_word_embeddings is true and no lm_head weights exist
        let hasLmHead = allWeights.keys.contains { $0.contains("lm_head") }
        if !hasLmHead && config.tieWordEmbeddings {
            FluxDebug.log("Qwen3: Enabling weight tying (lm_head uses embed_tokens weights)")
            model.useTiedWeights = true
        }

        FluxDebug.log("Qwen3 model loaded successfully with \(allWeights.count) tensors")

        return model
    }

    private func loadWeights(_ weights: [String: MLXArray]) throws {
        // Convert HuggingFace weight keys to MLX Swift format
        var convertedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            let swiftKey = convertKeyName(key)
            convertedWeights[swiftKey] = value
        }

        FluxDebug.log("Converting \(convertedWeights.count) Qwen3 weight tensors...")

        // Unflatten the weights to create nested ModuleParameters structure
        let parameters = ModuleParameters.unflattened(convertedWeights)

        // Apply weights to model - use .noUnusedKeys to detect any missing model weights
        // that aren't in the safetensors file
        do {
            try update(parameters: parameters, verify: .noUnusedKeys)
        } catch {
            FluxDebug.log("Qwen3 weight loading warning: \(error)")
            // Try again with .none to allow loading to proceed
            try update(parameters: parameters, verify: .none)
        }

        // Evaluate to ensure weights are loaded
        eval(self)

        FluxDebug.log("Qwen3 weights applied successfully")
    }

    private func convertKeyName(_ key: String) -> String {
        // Qwen3 weight key format:
        //   model.embed_tokens.weight
        //   model.layers.0.self_attn.q_proj.weight
        //   model.layers.0.mlp.gate_proj.weight
        //   model.norm.weight
        //   lm_head.weight
        // 
        // These match MLX Swift format directly, so no conversion needed
        return key
    }
}

// MARK: - Errors

public enum Qwen3ModelError: LocalizedError {
    case noWeightsFound
    case invalidConfig
    case loadError(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound:
            return "No safetensors files found in Qwen3 model directory"
        case .invalidConfig:
            return "Invalid Qwen3 model configuration"
        case .loadError(let message):
            return "Failed to load Qwen3 model: \(message)"
        }
    }
}
