// VAEConfig.swift - Flux.2 VAE Configuration
// Copyright 2025 Vincent Gourbin

import Foundation

/// Configuration for the Flux.2 VAE (AutoencoderKL)
public struct VAEConfig: Codable, Sendable {
    /// Number of input/output channels (3 for RGB)
    public var inChannels: Int

    /// Number of output channels (same as input)
    public var outChannels: Int

    /// Number of latent channels (32 for Flux.2, vs 4 for Flux.1)
    public var latentChannels: Int

    /// Channel multipliers for each block
    public var blockOutChannels: [Int]

    /// Number of ResNet blocks per level
    public var layersPerBlock: Int

    /// Activation function
    public var activationFunction: String

    /// Normalization layer epsilon
    public var normEps: Float

    /// Number of groups for group normalization
    public var normNumGroups: Int

    /// Scaling factor for latent space
    public var scalingFactor: Float

    /// Whether to use BatchNorm for latent normalization
    public var useBatchNorm: Bool

    /// Patch size for latent packing
    public var patchSize: (Int, Int)

    public init(
        inChannels: Int = 3,
        outChannels: Int = 3,
        latentChannels: Int = 32,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: Int = 2,
        activationFunction: String = "silu",
        normEps: Float = 1e-6,
        normNumGroups: Int = 32,
        scalingFactor: Float = 0.18215,
        useBatchNorm: Bool = true,
        patchSize: (Int, Int) = (2, 2)
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.latentChannels = latentChannels
        self.blockOutChannels = blockOutChannels
        self.layersPerBlock = layersPerBlock
        self.activationFunction = activationFunction
        self.normEps = normEps
        self.normNumGroups = normNumGroups
        self.scalingFactor = scalingFactor
        self.useBatchNorm = useBatchNorm
        self.patchSize = patchSize
    }

    /// Default Flux.2 VAE configuration
    public static let flux2Dev = VAEConfig()

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case latentChannels = "latent_channels"
        case blockOutChannels = "block_out_channels"
        case layersPerBlock = "layers_per_block"
        case activationFunction = "act_fn"
        case normEps = "norm_eps"
        case normNumGroups = "norm_num_groups"
        case scalingFactor = "scaling_factor"
        case useBatchNorm = "use_batch_norm"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 3
        latentChannels = try container.decodeIfPresent(Int.self, forKey: .latentChannels) ?? 32
        blockOutChannels = try container.decodeIfPresent([Int].self, forKey: .blockOutChannels) ?? [128, 256, 512, 512]
        layersPerBlock = try container.decodeIfPresent(Int.self, forKey: .layersPerBlock) ?? 2
        activationFunction = try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "silu"
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-6
        normNumGroups = try container.decodeIfPresent(Int.self, forKey: .normNumGroups) ?? 32
        scalingFactor = try container.decodeIfPresent(Float.self, forKey: .scalingFactor) ?? 0.18215
        useBatchNorm = try container.decodeIfPresent(Bool.self, forKey: .useBatchNorm) ?? true
        patchSize = (2, 2) // Not in JSON, hardcoded for Flux.2
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(inChannels, forKey: .inChannels)
        try container.encode(outChannels, forKey: .outChannels)
        try container.encode(latentChannels, forKey: .latentChannels)
        try container.encode(blockOutChannels, forKey: .blockOutChannels)
        try container.encode(layersPerBlock, forKey: .layersPerBlock)
        try container.encode(activationFunction, forKey: .activationFunction)
        try container.encode(normEps, forKey: .normEps)
        try container.encode(normNumGroups, forKey: .normNumGroups)
        try container.encode(scalingFactor, forKey: .scalingFactor)
        try container.encode(useBatchNorm, forKey: .useBatchNorm)
    }

    /// Load configuration from a JSON file
    public static func load(from url: URL) throws -> VAEConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VAEConfig.self, from: data)
    }
}

extension VAEConfig: CustomStringConvertible {
    public var description: String {
        """
        VAEConfig(
            channels: \(inChannels) -> latent \(latentChannels) -> \(outChannels),
            blocks: \(blockOutChannels),
            batchNorm: \(useBatchNorm),
            patch: \(patchSize)
        )
        """
    }
}
