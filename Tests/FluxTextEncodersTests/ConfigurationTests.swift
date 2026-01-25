/**
 * ConfigurationTests.swift
 * Unit tests for configuration structures
 */

import XCTest
@testable import FluxTextEncoders

final class ConfigurationTests: XCTestCase {

    // MARK: - MistralTextConfig Tests

    func testMistralTextConfigDefaults() {
        let config = MistralTextConfig.mistralSmall32

        // Verify Mistral Small 3.2 defaults
        XCTAssertEqual(config.vocabSize, 131_072, "Vocab size should be 131K")
        XCTAssertEqual(config.hiddenSize, 5120, "Hidden size should be 5120")
        XCTAssertEqual(config.intermediateSize, 14336, "Intermediate size should be 14336")
        XCTAssertEqual(config.numHiddenLayers, 40, "Should have 40 layers")
        XCTAssertEqual(config.numAttentionHeads, 32, "Should have 32 attention heads")
        XCTAssertEqual(config.numKeyValueHeads, 8, "Should have 8 KV heads (GQA)")
        XCTAssertEqual(config.maxPositionEmbeddings, 131_072, "Max position should be 131K")
        XCTAssertEqual(config.headDim, 128, "Head dimension should be 128")
    }

    func testMistralTextConfigCustomInit() {
        let config = MistralTextConfig(
            vocabSize: 50000,
            hiddenSize: 1024,
            intermediateSize: 4096,
            numHiddenLayers: 12,
            numAttentionHeads: 16,
            numKeyValueHeads: 4,
            maxPositionEmbeddings: 8192
        )

        XCTAssertEqual(config.vocabSize, 50000)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 4)
    }

    func testMistralTextConfigRopeSettings() {
        let config = MistralTextConfig.mistralSmall32

        XCTAssertEqual(config.ropeTheta, 1_000_000.0, "RoPE theta should be 1M")
        XCTAssertEqual(config.rmsNormEps, 1e-5, "RMS norm eps should be 1e-5")
    }

    func testMistralTextConfigActivation() {
        let config = MistralTextConfig.mistralSmall32

        XCTAssertEqual(config.hiddenAct, "silu", "Activation should be silu")
        XCTAssertFalse(config.attentionBias, "Attention bias should be false")
        XCTAssertFalse(config.mlpBias, "MLP bias should be false")
    }

    // MARK: - MistralVisionConfig Tests

    func testMistralVisionConfigDefaults() {
        let config = MistralVisionConfig.defaultVision

        XCTAssertEqual(config.hiddenSize, 1024, "Vision hidden size should be 1024")
        XCTAssertEqual(config.imageSize, 384, "Image size should be 384")
        XCTAssertEqual(config.patchSize, 14, "Patch size should be 14")
        XCTAssertEqual(config.numChannels, 3, "Should have 3 color channels")
        XCTAssertEqual(config.numHiddenLayers, 24, "Vision should have 24 layers")
        XCTAssertEqual(config.numAttentionHeads, 16, "Vision should have 16 attention heads")
    }

    func testMistralVisionConfigCustomInit() {
        let config = MistralVisionConfig(
            hiddenSize: 768,
            imageSize: 224,
            patchSize: 16,
            numChannels: 3,
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            intermediateSize: 3072
        )

        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.imageSize, 224)
        XCTAssertEqual(config.patchSize, 16)
    }

    // MARK: - MistralConfig Tests

    func testMistralConfigInit() {
        let textConfig = MistralTextConfig.mistralSmall32
        let config = MistralConfig(textConfig: textConfig, modelType: "mistral")

        XCTAssertEqual(config.textConfig.vocabSize, 131_072)
        XCTAssertEqual(config.modelType, "mistral")
        XCTAssertNil(config.visionConfig, "Vision config should be nil for text-only")
    }

    func testMistralConfigWithVision() {
        let textConfig = MistralTextConfig.mistralSmall32
        let visionConfig = MistralVisionConfig.defaultVision
        let config = MistralConfig(
            textConfig: textConfig,
            visionConfig: visionConfig,
            modelType: "mistral_vlm"
        )

        XCTAssertNotNil(config.visionConfig, "Vision config should not be nil")
        XCTAssertEqual(config.visionConfig?.imageSize, 384)
    }

    // MARK: - GenerationConfig Tests

    func testGenerationConfigDefaults() {
        let config = GenerationConfig.mistralDefault

        XCTAssertEqual(config.bosTokenId, 1, "BOS token should be 1")
        XCTAssertEqual(config.eosTokenId, 2, "EOS token should be 2")
        XCTAssertNil(config.padTokenId, "Default pad token should be nil")
    }

    func testGenerationConfigCustomInit() {
        let config = GenerationConfig(bosTokenId: 0, eosTokenId: 1, padTokenId: 2)

        XCTAssertEqual(config.bosTokenId, 0)
        XCTAssertEqual(config.eosTokenId, 1)
        XCTAssertEqual(config.padTokenId, 2)
    }

    // MARK: - GenerateParameters Tests

    func testGenerateParametersDefaults() {
        let params = GenerateParameters()

        XCTAssertEqual(params.maxTokens, 2048, "Default max tokens should be 2048")
        XCTAssertEqual(params.temperature, 0.7, "Default temperature should be 0.7")
        XCTAssertEqual(params.topP, 0.95, "Default topP should be 0.95")
        XCTAssertEqual(params.repetitionPenalty, 1.1, "Default repetition penalty should be 1.1")
        XCTAssertNil(params.seed, "Default seed should be nil")
    }

    func testGenerateParametersGreedyPreset() {
        let params = GenerateParameters.greedy

        XCTAssertEqual(params.temperature, 0.0, "Greedy should have temperature 0")
        XCTAssertEqual(params.topP, 1.0, "Greedy should have topP 1.0")
        XCTAssertEqual(params.repetitionPenalty, 1.0, "Greedy should have no repetition penalty")
    }

    func testGenerateParametersCreativePreset() {
        let params = GenerateParameters.creative

        XCTAssertEqual(params.temperature, 0.9, "Creative should have high temperature")
        XCTAssertEqual(params.maxTokens, 4096, "Creative should have more tokens")
        XCTAssertEqual(params.repetitionPenalty, 1.2, "Creative should have higher repetition penalty")
    }

    func testGenerateParametersBalancedPreset() {
        let params = GenerateParameters.balanced

        XCTAssertEqual(params.temperature, 0.7, "Balanced temperature should be 0.7")
        XCTAssertEqual(params.topP, 0.9, "Balanced topP should be 0.9")
        XCTAssertEqual(params.maxTokens, 2048, "Balanced max tokens should be 2048")
        XCTAssertEqual(params.repetitionPenalty, 1.1, "Balanced repetition penalty should be 1.1")
    }

    func testGenerateParametersCustomInit() {
        let params = GenerateParameters(
            maxTokens: 100,
            temperature: 0.5,
            topP: 0.8,
            repetitionPenalty: 1.5,
            repetitionContextSize: 50,
            seed: 42
        )

        XCTAssertEqual(params.maxTokens, 100)
        XCTAssertEqual(params.temperature, 0.5)
        XCTAssertEqual(params.topP, 0.8)
        XCTAssertEqual(params.repetitionPenalty, 1.5)
        XCTAssertEqual(params.repetitionContextSize, 50)
        XCTAssertEqual(params.seed, 42)
    }

    func testGenerateParametersMaxContextLength() {
        XCTAssertEqual(GenerateParameters.maxContextLength, 131_072,
                      "Max context should be 131K for Mistral Small 3.2")
    }

    // MARK: - Sendable Conformance

    func testConfigurationsAreSendable() {
        // These should compile without issues if Sendable is properly implemented
        let textConfig: Sendable = MistralTextConfig.mistralSmall32
        let visionConfig: Sendable = MistralVisionConfig.defaultVision
        let genParams: Sendable = GenerateParameters.balanced

        XCTAssertNotNil(textConfig)
        XCTAssertNotNil(visionConfig)
        XCTAssertNotNil(genParams)
    }
}
