// Flux2Config.swift - Flux.2 Transformer Configuration
// Copyright 2025 Vincent Gourbin

import Foundation

/// Configuration for the Flux.2 diffusion transformer
public struct Flux2TransformerConfig: Codable, Sendable {
    /// Patch size for input embedding (1 for Flux.2)
    public var patchSize: Int

    /// Number of input channels (128 for Flux.2 latents)
    public var inChannels: Int

    /// Number of output channels (same as input)
    public var outChannels: Int

    /// Number of double-stream transformer blocks
    public var numLayers: Int

    /// Number of single-stream transformer blocks
    public var numSingleLayers: Int

    /// Dimension of each attention head
    public var attentionHeadDim: Int

    /// Number of attention heads
    public var numAttentionHeads: Int

    /// Inner dimension for transformer (numAttentionHeads * attentionHeadDim)
    public var innerDim: Int {
        numAttentionHeads * attentionHeadDim
    }

    /// Dimension of joint attention (from Mistral embeddings: 15360)
    public var jointAttentionDim: Int

    /// Dimension of pooled projection (time + guidance embeddings)
    public var pooledProjectionDim: Int

    /// Whether to use guidance embedding
    public var guidanceEmbeds: Bool

    /// Axes dimensions for RoPE [T, H, W, L]
    public var axesDimsRope: [Int]

    /// Base theta for RoPE
    public var ropeTheta: Float

    /// MLP expansion ratio (3.0 for Flux.2, determines FFN hidden dimension)
    public var mlpRatio: Float

    /// Activation function for feedforward
    public var activationFunction: String

    public init(
        patchSize: Int = 1,
        inChannels: Int = 128,
        outChannels: Int = 128,
        numLayers: Int = 8,
        numSingleLayers: Int = 48,
        attentionHeadDim: Int = 128,
        numAttentionHeads: Int = 48,
        jointAttentionDim: Int = 15360,
        pooledProjectionDim: Int = 768,
        guidanceEmbeds: Bool = true,
        axesDimsRope: [Int] = [32, 32, 32, 32],
        ropeTheta: Float = 2000.0,
        mlpRatio: Float = 3.0,
        activationFunction: String = "silu"
    ) {
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.pooledProjectionDim = pooledProjectionDim
        self.guidanceEmbeds = guidanceEmbeds
        self.axesDimsRope = axesDimsRope
        self.ropeTheta = ropeTheta
        self.mlpRatio = mlpRatio
        self.activationFunction = activationFunction
    }

    /// Default Flux.2 configuration
    public static let flux2Dev = Flux2TransformerConfig()

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case numLayers = "num_layers"
        case numSingleLayers = "num_single_layers"
        case attentionHeadDim = "attention_head_dim"
        case numAttentionHeads = "num_attention_heads"
        case jointAttentionDim = "joint_attention_dim"
        case pooledProjectionDim = "pooled_projection_dim"
        case guidanceEmbeds = "guidance_embeds"
        case axesDimsRope = "axes_dims_rope"
        case ropeTheta = "rope_theta"
        case mlpRatio = "mlp_ratio"
        case activationFunction = "activation_function"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 1
        inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 128
        outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 128
        numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
        numSingleLayers = try container.decodeIfPresent(Int.self, forKey: .numSingleLayers) ?? 48
        attentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .attentionHeadDim) ?? 128
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 48
        jointAttentionDim = try container.decodeIfPresent(Int.self, forKey: .jointAttentionDim) ?? 15360
        pooledProjectionDim = try container.decodeIfPresent(Int.self, forKey: .pooledProjectionDim) ?? 768
        guidanceEmbeds = try container.decodeIfPresent(Bool.self, forKey: .guidanceEmbeds) ?? true
        axesDimsRope = try container.decodeIfPresent([Int].self, forKey: .axesDimsRope) ?? [32, 32, 32, 32]
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 2000.0
        mlpRatio = try container.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 3.0
        activationFunction = try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "silu"
    }

    /// Load configuration from a JSON file
    public static func load(from url: URL) throws -> Flux2TransformerConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Flux2TransformerConfig.self, from: data)
    }
}

extension Flux2TransformerConfig: CustomStringConvertible {
    public var description: String {
        """
        Flux2TransformerConfig(
            layers: \(numLayers) double + \(numSingleLayers) single,
            heads: \(numAttentionHeads) × \(attentionHeadDim) = \(innerDim),
            jointDim: \(jointAttentionDim),
            rope: \(axesDimsRope) θ=\(ropeTheta)
        )
        """
    }
}
