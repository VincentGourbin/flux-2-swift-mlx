/**
 * MistralConfiguration.swift
 * Configuration for Mistral Small 3.2 model
 *
 * Based on: https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
 */

import Foundation

// MARK: - Text Model Configuration

/**
 * Configuration for Mistral Small 3.2 text decoder
 * Matches the HuggingFace config.json structure
 */
public struct MistralTextConfig: Codable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool
    public let hiddenAct: String
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let mlpBias: Bool
    public let headDim: Int
    public let slidingWindow: Int?

    // Llama-4 attention scaling parameters for long sequences
    public let llama4ScalingBeta: Float
    public let originalMaxPositionEmbeddings: Int

    // CodingKeys for JSON mapping
    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenAct = "hidden_act"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case mlpBias = "mlp_bias"
        case headDim = "head_dim"
        case slidingWindow = "sliding_window"
        case llama4ScalingBeta = "llama_4_scaling_beta"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }

    // Custom decoder to handle missing optional fields
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)

        // Llama-4 attention scaling - defaults match transformers ministral3 config
        llama4ScalingBeta = try container.decodeIfPresent(Float.self, forKey: .llama4ScalingBeta) ?? 0.1
        originalMaxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings) ?? 16384
    }

    /// Default configuration for Mistral Small 3.2 (24B)
    public static let mistralSmall32: MistralTextConfig = MistralTextConfig(
        vocabSize: 131_072,
        hiddenSize: 5120,
        intermediateSize: 14336,
        numHiddenLayers: 40,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        maxPositionEmbeddings: 131_072,
        rmsNormEps: 1e-5,
        ropeTheta: 1_000_000.0,
        tieWordEmbeddings: false,
        hiddenAct: "silu",
        attentionBias: false,
        attentionDropout: 0.0,
        mlpBias: false,
        headDim: 128,
        slidingWindow: nil,
        llama4ScalingBeta: 0.1,
        originalMaxPositionEmbeddings: 16384
    )

    public init(
        vocabSize: Int = 131_072,
        hiddenSize: Int = 5120,
        intermediateSize: Int = 14336,
        numHiddenLayers: Int = 40,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        maxPositionEmbeddings: Int = 131_072,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 1_000_000.0,
        tieWordEmbeddings: Bool = false,
        hiddenAct: String = "silu",
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        mlpBias: Bool = false,
        headDim: Int = 128,
        slidingWindow: Int? = nil,
        llama4ScalingBeta: Float = 0.1,
        originalMaxPositionEmbeddings: Int = 16384
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
        self.hiddenAct = hiddenAct
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.mlpBias = mlpBias
        self.headDim = headDim
        self.slidingWindow = slidingWindow
        self.llama4ScalingBeta = llama4ScalingBeta
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
    }

    /// Load configuration from JSON file
    /// Handles both flat config and nested config (with text_config key for VLM models)
    public static func load(from path: String) throws -> MistralTextConfig {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()

        // First try to decode as flat MistralTextConfig
        if let config = try? decoder.decode(MistralTextConfig.self, from: data) {
            return config
        }

        // Try nested structure (VLM models like Mistral3ForConditionalGeneration)
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let textConfigDict = json["text_config"] as? [String: Any] {
            let textConfigData = try JSONSerialization.data(withJSONObject: textConfigDict)
            return try decoder.decode(MistralTextConfig.self, from: textConfigData)
        }

        // Fallback to default config
        FluxDebug.log("Warning: Could not parse config.json, using default configuration")
        return .mistralSmall32
    }
}

// MARK: - Vision Model Configuration

/**
 * Configuration for Mistral Vision encoder (for multimodal support)
 */
public struct MistralVisionConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
    }

    /// Default vision configuration for Mistral Small 3.2
    public static let defaultVision: MistralVisionConfig = MistralVisionConfig(
        hiddenSize: 1024,
        imageSize: 384,
        patchSize: 14,
        numChannels: 3,
        numHiddenLayers: 24,
        numAttentionHeads: 16,
        intermediateSize: 4096
    )

    public init(
        hiddenSize: Int = 1024,
        imageSize: Int = 384,
        patchSize: Int = 14,
        numChannels: Int = 3,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        intermediateSize: Int = 4096
    ) {
        self.hiddenSize = hiddenSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
    }
}

// MARK: - Combined Configuration

/**
 * Full Mistral model configuration combining text and optional vision
 */
public struct MistralConfig: Codable, Sendable {
    public let textConfig: MistralTextConfig
    public let visionConfig: MistralVisionConfig?
    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
    }

    public init(
        textConfig: MistralTextConfig = .mistralSmall32,
        visionConfig: MistralVisionConfig? = nil,
        modelType: String = "mistral"
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.modelType = modelType
    }

    /// Load full configuration from model directory
    public static func load(from modelPath: String) throws -> MistralConfig {
        let configPath = "\(modelPath)/config.json"
        let url = URL(fileURLWithPath: configPath)
        let data = try Data(contentsOf: url)

        // Try to decode as full config first
        let decoder = JSONDecoder()

        // If it's a simple text config, wrap it
        if let textConfig = try? decoder.decode(MistralTextConfig.self, from: data) {
            return MistralConfig(textConfig: textConfig)
        }

        // Otherwise try full config
        return try decoder.decode(MistralConfig.self, from: data)
    }
}

// MARK: - Generation Configuration

/**
 * Configuration for text generation
 */
public struct GenerationConfig: Codable, Sendable {
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public static let mistralDefault = GenerationConfig(
        bosTokenId: 1,
        eosTokenId: 2,
        padTokenId: nil
    )

    public init(bosTokenId: Int = 1, eosTokenId: Int = 2, padTokenId: Int? = nil) {
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
    }

    public static func load(from modelPath: String) throws -> GenerationConfig {
        let configPath = "\(modelPath)/generation_config.json"
        let url = URL(fileURLWithPath: configPath)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GenerationConfig.self, from: data)
    }
}
