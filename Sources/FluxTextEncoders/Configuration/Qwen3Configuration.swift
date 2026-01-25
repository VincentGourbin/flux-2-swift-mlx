/**
 * Qwen3Configuration.swift
 * Configuration for Qwen3 models (used as text encoder for FLUX.2 Klein)
 *
 * Qwen3-4B and Qwen3-8B are used by FLUX.2 Klein 4B and 9B respectively
 */

import Foundation

// MARK: - Qwen3 Text Model Configuration

/**
 * Configuration for Qwen3 text decoder
 * Based on Qwen3-4B and Qwen3-8B architecture
 */
public struct Qwen3TextConfig: Codable, Sendable {
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
    public let headDim: Int

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
        case headDim = "head_dim"
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
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        
        // Qwen3 head_dim is computed from hidden_size / num_attention_heads
        let computedHeadDim = hiddenSize / numAttentionHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? computedHeadDim
    }

    /// Default configuration for Qwen3-4B (used by Klein 4B)
    /// hidden_size: 2560, num_layers: 36, head_dim: 80
    public static let qwen3_4B = Qwen3TextConfig(
        vocabSize: 151_936,
        hiddenSize: 2560,
        intermediateSize: 9216,
        numHiddenLayers: 36,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        maxPositionEmbeddings: 40960,
        rmsNormEps: 1e-6,
        ropeTheta: 1_000_000.0,
        tieWordEmbeddings: true,
        hiddenAct: "silu",
        attentionBias: false,
        attentionDropout: 0.0,
        headDim: 80  // 2560 / 32 = 80
    )

    /// Default configuration for Qwen3-8B (used by Klein 9B)
    /// hidden_size: 4096, num_layers: 36, head_dim: 128
    public static let qwen3_8B = Qwen3TextConfig(
        vocabSize: 151_936,
        hiddenSize: 4096,
        intermediateSize: 12288,
        numHiddenLayers: 36,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        maxPositionEmbeddings: 131_072,
        rmsNormEps: 1e-6,
        ropeTheta: 1_000_000.0,
        tieWordEmbeddings: true,
        hiddenAct: "silu",
        attentionBias: false,
        attentionDropout: 0.0,
        headDim: 128  // 4096 / 32 = 128
    )

    public init(
        vocabSize: Int = 151_936,
        hiddenSize: Int = 2560,
        intermediateSize: Int = 9216,
        numHiddenLayers: Int = 36,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        maxPositionEmbeddings: Int = 40960,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        tieWordEmbeddings: Bool = true,
        hiddenAct: String = "silu",
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        headDim: Int? = nil
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
        self.headDim = headDim ?? (hiddenSize / numAttentionHeads)
    }

    /// Load configuration from JSON file
    public static func load(from path: String) throws -> Qwen3TextConfig {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(Qwen3TextConfig.self, from: data)
    }
}

// MARK: - Generation Configuration for Qwen3

/**
 * Configuration for Qwen3 text generation
 */
public struct Qwen3GenerationConfig: Codable, Sendable {
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public static let qwen3Default = Qwen3GenerationConfig(
        bosTokenId: 151643,  // <|im_start|>
        eosTokenId: 151645,  // <|im_end|>
        padTokenId: 151643   // Same as BOS
    )

    public init(bosTokenId: Int = 151643, eosTokenId: Int = 151645, padTokenId: Int? = 151643) {
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
    }

    public static func load(from modelPath: String) throws -> Qwen3GenerationConfig {
        let configPath = "\(modelPath)/generation_config.json"
        let url = URL(fileURLWithPath: configPath)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Qwen3GenerationConfig.self, from: data)
    }
}
