/**
 * HiddenStatesConfig.swift
 * Configuration for hidden states extraction
 */

import Foundation

/// Configuration for extracting hidden states from specific layers
public struct HiddenStatesConfig: Sendable {
    /// Layer indices to extract hidden states from
    /// Use negative indices for layers from the end (e.g., -1 for last layer)
    public var layerIndices: [Int]

    /// Whether to concatenate hidden states from multiple layers
    public var concatenate: Bool

    /// Whether to normalize the extracted embeddings
    public var normalize: Bool

    /// Pooling strategy for sequence dimension
    public var pooling: PoolingStrategy

    public init(
        layerIndices: [Int],
        concatenate: Bool = true,
        normalize: Bool = false,
        pooling: PoolingStrategy = .lastToken
    ) {
        self.layerIndices = layerIndices
        self.concatenate = concatenate
        self.normalize = normalize
        self.pooling = pooling
    }

    // MARK: - Presets

    /// mflux-gradio compatible: layers 10, 20, 30 concatenated -> 15360 dims
    /// For Mistral Small 3.2 (40 layers, 5120 hidden size): 3 * 5120 = 15360
    public static let mfluxDefault = HiddenStatesConfig(
        layerIndices: [10, 20, 30],
        concatenate: true,
        normalize: false,
        pooling: .none
    )

    /// Last layer only
    public static let lastLayerOnly = HiddenStatesConfig(
        layerIndices: [-1],
        concatenate: false,
        normalize: false,
        pooling: .lastToken
    )

    /// Middle layer (layer 20 for 40-layer model)
    public static let middleLayer = HiddenStatesConfig(
        layerIndices: [20],
        concatenate: false,
        normalize: false,
        pooling: .lastToken
    )

    /// All layers - useful for analysis
    public static let allLayers = HiddenStatesConfig(
        layerIndices: Array(0..<40),
        concatenate: false,  // Would be too large if concatenated
        normalize: false,
        pooling: .none
    )

    /// Custom configuration builder
    public static func custom(
        layers: [Int],
        concatenate: Bool = true,
        normalize: Bool = false,
        pooling: PoolingStrategy = .none
    ) -> HiddenStatesConfig {
        return HiddenStatesConfig(
            layerIndices: layers,
            concatenate: concatenate,
            normalize: normalize,
            pooling: pooling
        )
    }
}

/// Pooling strategy for reducing sequence dimension
public enum PoolingStrategy: String, Codable, Sendable {
    /// No pooling - keep full sequence
    case none

    /// Take the last token's hidden state
    case lastToken

    /// Average all token hidden states
    case mean

    /// Max pooling across sequence
    case max

    /// CLS token (first token) - for models with CLS
    case cls
}
