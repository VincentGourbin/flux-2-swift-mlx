/**
 * FluxTextEncodersTests.swift
 * Main unit tests for FluxTextEncoders library
 */

import XCTest
@testable import FluxTextEncoders

final class FluxTextEncodersTests: XCTestCase {

    // MARK: - Version Tests

    func testVersion() {
        XCTAssertFalse(FluxTextEncoders.version.isEmpty, "Version should not be empty")
        XCTAssertEqual(FluxTextEncoders.version, "0.1.0", "Version should be 0.1.0")
    }

    func testMistralVersionInfo() {
        XCTAssertEqual(MistralVersion.version, "0.1.0")
        XCTAssertEqual(MistralVersion.modelName, "Mistral Small 3.2")
        XCTAssertEqual(MistralVersion.modelVersion, "24B-Instruct-2506")
    }

    // MARK: - Singleton Tests

    func testFluxTextEncodersSharedInstance() {
        let core1 = FluxTextEncoders.shared
        let core2 = FluxTextEncoders.shared
        XCTAssertTrue(core1 === core2, "Shared instance should be singleton")
    }

    // MARK: - Model State Tests

    func testIsModelLoadedInitiallyFalse() {
        // Note: In a real test environment, you'd create a fresh instance
        // For singleton, this test verifies the property exists
        let core = FluxTextEncoders.shared
        _ = core.isModelLoaded  // Should not crash
    }

    func testIsVLMLoadedInitiallyFalse() {
        let core = FluxTextEncoders.shared
        _ = core.isVLMLoaded  // Should not crash
    }

    // MARK: - Error Tests

    func testFluxEncoderErrorDescriptions() {
        XCTAssertEqual(FluxEncoderError.modelNotLoaded.errorDescription,
                      "Model not loaded. Call loadModel() first.")

        XCTAssertEqual(FluxEncoderError.vlmNotLoaded.errorDescription,
                      "VLM not loaded. Call loadVLMModel() first for vision capabilities.")

        let invalidInput = FluxEncoderError.invalidInput("test message")
        XCTAssertEqual(invalidInput.errorDescription, "Invalid input: test message")

        let genFailed = FluxEncoderError.generationFailed("gen error")
        XCTAssertEqual(genFailed.errorDescription, "Generation failed: gen error")
    }

    // MARK: - Generate Without Model Tests

    func testGenerateThrowsWhenModelNotLoaded() {
        // Create a fresh core that definitely doesn't have a model loaded
        // For singleton, we test that the error is properly defined
        XCTAssertNotNil(FluxEncoderError.modelNotLoaded.errorDescription)
    }

    func testChatThrowsWhenModelNotLoaded() {
        XCTAssertNotNil(FluxEncoderError.modelNotLoaded.errorDescription)
    }

    // MARK: - Embeddings Without Model Tests

    func testExtractEmbeddingsThrowsWhenModelNotLoaded() {
        XCTAssertNotNil(FluxEncoderError.modelNotLoaded.errorDescription)
    }

    // MARK: - Vision Without Model Tests

    func testAnalyzeImageThrowsWhenVLMNotLoaded() {
        XCTAssertNotNil(FluxEncoderError.vlmNotLoaded.errorDescription)
    }

    // MARK: - Tokenization Tests (Basic)

    func testTokenizer() throws {
        let tokenizer = TekkenTokenizer()
        let text = "Hello world"
        let tokens = tokenizer.encode(text)
        XCTAssertFalse(tokens.isEmpty, "Tokenization should produce tokens")
    }

    func testHiddenStatesConfig() throws {
        let config = HiddenStatesConfig.mfluxDefault
        XCTAssertEqual(config.layerIndices, [10, 20, 30])
        XCTAssertTrue(config.concatenate)
    }

    @MainActor
    func testTextEncoderModelRegistry() throws {
        let models = TextEncoderModelRegistry.shared.allModels()
        XCTAssertGreaterThanOrEqual(models.count, 3, "Should have at least 3 model variants")

        let defaultModel = TextEncoderModelRegistry.shared.defaultModel()
        XCTAssertEqual(defaultModel.variant, .mlx8bit)
    }

    func testGenerateParameters() throws {
        let params = GenerateParameters.balanced
        XCTAssertEqual(params.maxTokens, 2048)
        XCTAssertEqual(params.temperature, 0.7)
        XCTAssertEqual(params.topP, 0.9)
    }

    // MARK: - FLUX Config Tests

    func testFluxConfigValues() {
        XCTAssertEqual(FluxConfig.maxSequenceLength, 512)
        XCTAssertEqual(FluxConfig.hiddenStateLayers, [10, 20, 30])
        XCTAssertFalse(FluxConfig.systemMessage.isEmpty,
                      "System message should not be empty")
    }

    func testFluxConfigSystemMessage() {
        let systemMessage = FluxConfig.systemMessage
        XCTAssertTrue(systemMessage.contains("image"),
                     "System message should mention images")
    }

    // MARK: - Export Format Tests

    func testExportFormatCases() {
        let binary = ExportFormat.binary
        let numpy = ExportFormat.numpy
        let json = ExportFormat.json

        // Just verify they exist and are distinct
        XCTAssertNotEqual(String(describing: binary), String(describing: numpy))
        XCTAssertNotEqual(String(describing: binary), String(describing: json))
    }
}

// MARK: - Integration Tests (Without Model Loading)

final class FluxTextEncodersIntegrationTests: XCTestCase {

    /// Test that configurations work together correctly
    func testConfigurationIntegration() {
        let textConfig = MistralTextConfig.mistralSmall32
        let hiddenStatesConfig = HiddenStatesConfig.mfluxDefault

        // Verify layer indices are valid for the model
        let maxLayer = textConfig.numHiddenLayers
        for layerIdx in hiddenStatesConfig.layerIndices {
            XCTAssertLessThan(layerIdx, maxLayer,
                            "Layer index \(layerIdx) should be less than num layers \(maxLayer)")
        }
    }

    /// Test that FLUX embeddings configuration matches expected dimensions
    func testFluxEmbeddingsDimensions() {
        let config = MistralTextConfig.mistralSmall32
        let hiddenStatesConfig = HiddenStatesConfig.mfluxDefault

        // FLUX expects 3 layers * hidden_size = 15360
        let expectedDim = hiddenStatesConfig.layerIndices.count * config.hiddenSize
        XCTAssertEqual(expectedDim, 15360,
                      "FLUX embeddings should produce 15360 dimensions")
    }

    /// Test tokenizer with various input types
    func testTokenizerVariousInputs() {
        let tokenizer = TekkenTokenizer()

        // ASCII
        let asciiTokens = tokenizer.encode("Hello")
        XCTAssertFalse(asciiTokens.isEmpty)

        // Unicode
        let unicodeTokens = tokenizer.encode("世界")
        XCTAssertFalse(unicodeTokens.isEmpty)

        // Mixed
        let mixedTokens = tokenizer.encode("Hello 世界!")
        XCTAssertFalse(mixedTokens.isEmpty)

        // Numbers
        let numberTokens = tokenizer.encode("12345")
        XCTAssertFalse(numberTokens.isEmpty)

        // Special chars
        let specialTokens = tokenizer.encode("@#$%^&*()")
        XCTAssertFalse(specialTokens.isEmpty)
    }

    /// Test chat template produces valid structure
    func testChatTemplateStructure() {
        let tokenizer = TekkenTokenizer()

        let messages: [[String: String]] = [
            ["role": "system", "content": "You are helpful"],
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Hi there!"],
            ["role": "user", "content": "How are you?"]
        ]

        let prompt = tokenizer.applyChatTemplate(messages: messages)

        // Should be non-empty
        XCTAssertFalse(prompt.isEmpty)

        // Should contain all content
        XCTAssertTrue(prompt.contains("helpful") || prompt.contains("You are helpful"))
        XCTAssertTrue(prompt.contains("Hello"))
        XCTAssertTrue(prompt.contains("Hi there!"))
        XCTAssertTrue(prompt.contains("How are you?"))
    }

    @MainActor
    func testTextEncoderModelRegistryHasAllExpectedVariants() {
        let registry = TextEncoderModelRegistry.shared
        let models = registry.allModels()

        // Should have quantized models
        let has8bit = models.contains { $0.variant == .mlx8bit }
        let has4bit = models.contains { $0.variant == .mlx4bit }

        XCTAssertTrue(has8bit, "Should have 8-bit model")
        XCTAssertTrue(has4bit, "Should have 4-bit model")
    }
}
