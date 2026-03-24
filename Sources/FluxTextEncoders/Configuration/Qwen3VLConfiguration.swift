/**
 * Qwen3VLConfiguration.swift
 * Configuration for Qwen3-VL models (vision-language variant)
 *
 * Qwen3-VL-4B has the same hidden_size (2560) as Qwen3-4B but differs in:
 * - head_dim: 128 vs 80 (explicit, not derived from hidden_size/num_heads)
 * - intermediate_size: 9728 vs 9216
 * - RoPE: MRoPE (multi-dimensional) vs standard 1D
 */

import Foundation

// MARK: - Qwen3-VL Text Model Configuration

/// Configuration for Qwen3-VL language model component
/// The vision encoder and DeepStack fusion are NOT included (Phase 1: text-only)
public struct Qwen3VLTextConfig: Decodable, Sendable {
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

    /// MRoPE section sizes: [temporal, height, width]
    /// head_dim is split into these 3 groups for multi-dimensional position encoding
    /// Total = sum of sections * 2 = head_dim (e.g. [24,20,20] * 2 = 128)
    public let mropeSectionSizes: [Int]

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
        case ropeScaling = "rope_scaling"
    }

    /// Intermediate struct to decode rope_scaling.mrope_section from config.json
    private struct RopeScaling: Codable {
        let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
        }
    }

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
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128

        // Parse MRoPE sections from rope_scaling
        if let ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling),
           let sections = ropeScaling.mropeSection {
            mropeSectionSizes = sections
        } else {
            mropeSectionSizes = [24, 20, 20]  // Default for Qwen3-VL-4B
        }
    }

    /// Default configuration for Qwen3-VL-4B (language component only)
    /// hidden_size: 2560, num_layers: 36, head_dim: 128 (NOT 80 like Qwen3-4B)
    public static let qwen3VL_4B = Qwen3VLTextConfig(
        vocabSize: 151_936,
        hiddenSize: 2560,
        intermediateSize: 9728,
        numHiddenLayers: 36,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        maxPositionEmbeddings: 262_144,
        rmsNormEps: 1e-6,
        ropeTheta: 5_000_000.0,
        tieWordEmbeddings: true,
        hiddenAct: "silu",
        attentionBias: false,
        attentionDropout: 0.0,
        headDim: 128,
        mropeSectionSizes: [24, 20, 20]
    )

    /// Default configuration for Qwen3-VL-8B (language component only)
    /// hidden_size: 4096, num_layers: 36, head_dim: 128, matches Klein 9B
    public static let qwen3VL_8B = Qwen3VLTextConfig(
        vocabSize: 151_936,
        hiddenSize: 4096,
        intermediateSize: 12288,
        numHiddenLayers: 36,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        maxPositionEmbeddings: 262_144,
        rmsNormEps: 1e-6,
        ropeTheta: 5_000_000.0,
        tieWordEmbeddings: true,
        hiddenAct: "silu",
        attentionBias: false,
        attentionDropout: 0.0,
        headDim: 128,
        mropeSectionSizes: [24, 20, 20]
    )

    public init(
        vocabSize: Int = 151_936,
        hiddenSize: Int = 2560,
        intermediateSize: Int = 9728,
        numHiddenLayers: Int = 36,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        maxPositionEmbeddings: Int = 262_144,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        tieWordEmbeddings: Bool = true,
        hiddenAct: String = "silu",
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        headDim: Int = 128,
        mropeSectionSizes: [Int] = [24, 20, 20]
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
        self.headDim = headDim
        self.mropeSectionSizes = mropeSectionSizes
    }

    /// Load configuration from JSON file
    /// Handles the nested VL config structure where text params are under "text_config"
    public static func load(from path: String) throws -> Qwen3VLTextConfig {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)

        // Try nested VL format first: { "text_config": { ... } }
        struct VLConfigWrapper: Decodable {
            let textConfig: Qwen3VLTextConfig
            enum CodingKeys: String, CodingKey {
                case textConfig = "text_config"
            }
        }

        if let wrapper = try? JSONDecoder().decode(VLConfigWrapper.self, from: data) {
            return wrapper.textConfig
        }

        // Fallback: direct decode (flat format)
        return try JSONDecoder().decode(Qwen3VLTextConfig.self, from: data)
    }
}
