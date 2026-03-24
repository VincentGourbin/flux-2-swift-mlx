/**
 * Qwen35Configuration.swift
 * Configuration for Qwen3.5 VLM models (hybrid SSM/Transformer with vision)
 *
 * Qwen3.5-4B: 32 layers (24 linear_attn + 8 full_attn), hidden_size=2560,
 * head_dim=256, vocab=248320, integrated 24-layer ViT vision encoder.
 */

import Foundation

// MARK: - Text Config

public struct Qwen35TextConfig: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool

    // Hybrid attention
    public let layerTypes: [String]
    public let fullAttentionInterval: Int

    // Linear attention (Gated DeltaNet)
    public let linearConvKernelDim: Int
    public let linearKeyHeadDim: Int
    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearValueHeadDim: Int

    // MRoPE
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    public let mropeSectionSizes: [Int]

    // Special tokens
    public let eosTokenId: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case eosTokenId = "eos_token_id"
        case ropeParameters = "rope_parameters"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        let partialRotaryFactor: Float?
        let mropeSection: [Int]?
        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case mropeSection = "mrope_section"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        fullAttentionInterval = try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        linearKeyHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        linearNumKeyHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        linearNumValueHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 32
        linearValueHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 248044

        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta ?? 10_000_000.0
        partialRotaryFactor = rope?.partialRotaryFactor ?? 0.25
        mropeSectionSizes = rope?.mropeSection ?? [11, 11, 10]
    }

    /// Whether a given layer index uses linear attention (Gated DeltaNet)
    public func isLinearLayer(_ index: Int) -> Bool {
        if !layerTypes.isEmpty && index < layerTypes.count {
            return layerTypes[index] == "linear_attention"
        }
        return (index + 1) % fullAttentionInterval != 0
    }

    /// Rotary embedding dimension (partial)
    public var rotaryDim: Int {
        Int(Float(headDim) * partialRotaryFactor)
    }
}

// MARK: - Vision Config

public struct Qwen35VisionConfig: Decodable, Sendable {
    public let depth: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let inChannels: Int
    public let outHiddenSize: Int
    public let numPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case inChannels = "in_channels"
        case outHiddenSize = "out_hidden_size"
        case numPositionEmbeddings = "num_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depth = try c.decode(Int.self, forKey: .depth)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 2560
        numPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .numPositionEmbeddings) ?? 2304
    }
}

// MARK: - Top-level Config

public struct Qwen35Config: Decodable, Sendable {
    public let textConfig: Qwen35TextConfig
    public let visionConfig: Qwen35VisionConfig
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(Qwen35TextConfig.self, forKey: .textConfig)
        visionConfig = try c.decode(Qwen35VisionConfig.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248056
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248057
        visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 248053
        visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 248054
    }

    public static func load(from path: String) throws -> Qwen35Config {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(Qwen35Config.self, from: data)
    }
}
