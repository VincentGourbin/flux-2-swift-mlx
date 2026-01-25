/**
 * HiddenStatesConfigTests.swift
 * Unit tests for HiddenStatesConfig
 */

import XCTest
@testable import FluxTextEncoders

final class HiddenStatesConfigTests: XCTestCase {

    // MARK: - Preset Tests

    func testMfluxDefaultPreset() {
        let config = HiddenStatesConfig.mfluxDefault

        XCTAssertEqual(config.layerIndices, [10, 20, 30],
                      "mflux default should extract from layers 10, 20, 30")
        XCTAssertTrue(config.concatenate,
                     "mflux default should concatenate layers")
        XCTAssertFalse(config.normalize,
                      "mflux default should not normalize")
        XCTAssertEqual(config.pooling, .none,
                      "mflux default should have no pooling")
    }

    func testLastLayerOnlyPreset() {
        let config = HiddenStatesConfig.lastLayerOnly

        XCTAssertEqual(config.layerIndices, [-1],
                      "lastLayerOnly should use index -1")
        XCTAssertFalse(config.concatenate,
                      "lastLayerOnly should not concatenate (only one layer)")
        XCTAssertFalse(config.normalize,
                      "lastLayerOnly should not normalize by default")
        XCTAssertEqual(config.pooling, .lastToken,
                      "lastLayerOnly should use lastToken pooling")
    }

    func testMiddleLayerPreset() {
        let config = HiddenStatesConfig.middleLayer

        XCTAssertEqual(config.layerIndices, [20],
                      "middleLayer should extract from layer 20")
        XCTAssertFalse(config.concatenate)
        XCTAssertEqual(config.pooling, .lastToken)
    }

    func testAllLayersPreset() {
        let config = HiddenStatesConfig.allLayers

        XCTAssertEqual(config.layerIndices.count, 40,
                      "allLayers should have 40 layer indices")
        XCTAssertEqual(config.layerIndices.first, 0,
                      "allLayers should start at 0")
        XCTAssertEqual(config.layerIndices.last, 39,
                      "allLayers should end at 39")
        XCTAssertFalse(config.concatenate,
                      "allLayers should not concatenate (too large)")
        XCTAssertEqual(config.pooling, .none)
    }

    // MARK: - Custom Config Tests

    func testCustomInit() {
        let config = HiddenStatesConfig(
            layerIndices: [5, 15, 25],
            concatenate: false,
            normalize: true,
            pooling: .mean
        )

        XCTAssertEqual(config.layerIndices, [5, 15, 25])
        XCTAssertFalse(config.concatenate)
        XCTAssertTrue(config.normalize)
        XCTAssertEqual(config.pooling, .mean)
    }

    func testCustomBuilder() {
        let config = HiddenStatesConfig.custom(
            layers: [0, 10, 20, 30, 39],
            concatenate: true,
            normalize: true,
            pooling: .lastToken
        )

        XCTAssertEqual(config.layerIndices.count, 5)
        XCTAssertTrue(config.concatenate)
        XCTAssertTrue(config.normalize)
        XCTAssertEqual(config.pooling, .lastToken)
    }

    func testCustomBuilderDefaults() {
        let config = HiddenStatesConfig.custom(layers: [10])

        // Check defaults
        XCTAssertTrue(config.concatenate, "Default concatenate should be true")
        XCTAssertFalse(config.normalize, "Default normalize should be false")
        XCTAssertEqual(config.pooling, .none, "Default pooling should be none")
    }

    // MARK: - Pooling Strategy Tests

    func testPoolingStrategyNone() {
        XCTAssertEqual(PoolingStrategy.none.rawValue, "none")
    }

    func testPoolingStrategyLastToken() {
        XCTAssertEqual(PoolingStrategy.lastToken.rawValue, "lastToken")
    }

    func testPoolingStrategyMean() {
        XCTAssertEqual(PoolingStrategy.mean.rawValue, "mean")
    }

    func testPoolingStrategyMax() {
        XCTAssertEqual(PoolingStrategy.max.rawValue, "max")
    }

    func testPoolingStrategyCLS() {
        XCTAssertEqual(PoolingStrategy.cls.rawValue, "cls")
    }

    // MARK: - Edge Cases

    func testEmptyLayerIndices() {
        let config = HiddenStatesConfig(
            layerIndices: [],
            concatenate: true,
            normalize: false,
            pooling: .none
        )

        XCTAssertTrue(config.layerIndices.isEmpty)
    }

    func testNegativeLayerIndices() {
        let config = HiddenStatesConfig(
            layerIndices: [-1, -2, -3],
            concatenate: true,
            normalize: false,
            pooling: .none
        )

        // Negative indices should be preserved (resolved at extraction time)
        XCTAssertEqual(config.layerIndices, [-1, -2, -3])
    }

    func testSingleLayerWithConcatenate() {
        let config = HiddenStatesConfig(
            layerIndices: [10],
            concatenate: true,  // Should work even with single layer
            normalize: false,
            pooling: .none
        )

        XCTAssertEqual(config.layerIndices.count, 1)
        XCTAssertTrue(config.concatenate)
    }

    // MARK: - Sendable Conformance

    func testHiddenStatesConfigIsSendable() {
        let config: Sendable = HiddenStatesConfig.mfluxDefault
        XCTAssertNotNil(config)
    }

    func testPoolingStrategyIsSendable() {
        let strategy: Sendable = PoolingStrategy.mean
        XCTAssertNotNil(strategy)
    }

    // MARK: - FLUX.2 Compatibility Tests

    func testFluxCompatibleDimensions() {
        let config = HiddenStatesConfig.mfluxDefault
        let hiddenSize = 5120  // Mistral Small 3.2 hidden size
        let numLayers = config.layerIndices.count

        // FLUX.2 expects 15360 dimensions (3 layers * 5120)
        let expectedDimension = numLayers * hiddenSize
        XCTAssertEqual(expectedDimension, 15360,
                      "FLUX.2 config should produce 15360 dimensions")
    }
}
