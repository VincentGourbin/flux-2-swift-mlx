/**
 * TextEncoderModelRegistryTests.swift
 * Unit tests for TextEncoderModelRegistry and ModelVariant
 */

import XCTest
@testable import FluxTextEncoders

@MainActor
final class TextEncoderModelRegistryTests: XCTestCase {

    // MARK: - ModelVariant Tests

    func testModelVariantRawValues() {
        XCTAssertEqual(ModelVariant.bf16.rawValue, "bf16")
        XCTAssertEqual(ModelVariant.mlx8bit.rawValue, "8bit")
        XCTAssertEqual(ModelVariant.mlx6bit.rawValue, "6bit")
        XCTAssertEqual(ModelVariant.mlx4bit.rawValue, "4bit")
    }

    func testModelVariantDisplayNames() {
        XCTAssertEqual(ModelVariant.bf16.displayName, "Full Precision (BF16)")
        XCTAssertEqual(ModelVariant.mlx8bit.displayName, "8-bit Quantized")
        XCTAssertEqual(ModelVariant.mlx6bit.displayName, "6-bit Quantized")
        XCTAssertEqual(ModelVariant.mlx4bit.displayName, "4-bit Quantized")
    }

    func testModelVariantShortNames() {
        XCTAssertEqual(ModelVariant.bf16.shortName, "BF16")
        XCTAssertEqual(ModelVariant.mlx8bit.shortName, "8-bit")
        XCTAssertEqual(ModelVariant.mlx6bit.shortName, "6-bit")
        XCTAssertEqual(ModelVariant.mlx4bit.shortName, "4-bit")
    }

    func testModelVariantEstimatedSizes() {
        XCTAssertEqual(ModelVariant.bf16.estimatedSize, "~48GB")
        XCTAssertEqual(ModelVariant.mlx8bit.estimatedSize, "~25GB")
        XCTAssertEqual(ModelVariant.mlx6bit.estimatedSize, "~19GB")
        XCTAssertEqual(ModelVariant.mlx4bit.estimatedSize, "~14GB")
    }

    func testModelVariantCaseIterable() {
        let allCases = ModelVariant.allCases
        XCTAssertEqual(allCases.count, 4, "Should have 4 model variants")
        XCTAssertTrue(allCases.contains(.bf16))
        XCTAssertTrue(allCases.contains(.mlx8bit))
        XCTAssertTrue(allCases.contains(.mlx6bit))
        XCTAssertTrue(allCases.contains(.mlx4bit))
    }

    // MARK: - ModelInfo Tests

    func testModelInfoInit() {
        let model = ModelInfo(
            id: "test-model",
            repoId: "org/test-model",
            name: "Test Model",
            description: "A test model",
            variant: .mlx8bit,
            parameters: "24B"
        )

        XCTAssertEqual(model.id, "test-model")
        XCTAssertEqual(model.repoId, "org/test-model")
        XCTAssertEqual(model.name, "Test Model")
        XCTAssertEqual(model.description, "A test model")
        XCTAssertEqual(model.variant, .mlx8bit)
        XCTAssertEqual(model.parameters, "24B")
    }

    // MARK: - TextEncoderModelRegistry Tests

    func testTextEncoderModelRegistrySharedInstance() {
        let registry1 = TextEncoderModelRegistry.shared
        let registry2 = TextEncoderModelRegistry.shared
        XCTAssertTrue(registry1 === registry2, "Shared instance should be singleton")
    }

    func testTextEncoderModelRegistryAllModels() {
        let models = TextEncoderModelRegistry.shared.allModels()

        XCTAssertGreaterThanOrEqual(models.count, 3,
                                    "Should have at least 3 models (8bit, 6bit, 4bit)")
    }

    func testTextEncoderModelRegistryContainsExpectedVariants() {
        let models = TextEncoderModelRegistry.shared.allModels()
        let variants = models.map { $0.variant }

        XCTAssertTrue(variants.contains(.mlx8bit), "Should have 8-bit model")
        XCTAssertTrue(variants.contains(.mlx4bit), "Should have 4-bit model")
    }

    func testTextEncoderModelRegistryDefaultModel() {
        let defaultModel = TextEncoderModelRegistry.shared.defaultModel()

        XCTAssertEqual(defaultModel.variant, .mlx8bit,
                      "Default model should be 8-bit")
        XCTAssertFalse(defaultModel.id.isEmpty, "Default model should have ID")
        XCTAssertFalse(defaultModel.repoId.isEmpty, "Default model should have repo ID")
    }

    func testTextEncoderModelRegistryFindByVariant() {
        let model8bit = TextEncoderModelRegistry.shared.model(withVariant: .mlx8bit)
        let model4bit = TextEncoderModelRegistry.shared.model(withVariant: .mlx4bit)

        XCTAssertNotNil(model8bit, "Should find 8-bit model")
        XCTAssertNotNil(model4bit, "Should find 4-bit model")

        XCTAssertEqual(model8bit?.variant, .mlx8bit)
        XCTAssertEqual(model4bit?.variant, .mlx4bit)
    }

    func testTextEncoderModelRegistryFindById() {
        let models = TextEncoderModelRegistry.shared.allModels()
        guard let firstModel = models.first else {
            XCTFail("Should have at least one model")
            return
        }

        let foundModel = TextEncoderModelRegistry.shared.model(withId: firstModel.id)

        XCTAssertNotNil(foundModel, "Should find model by ID")
        XCTAssertEqual(foundModel?.id, firstModel.id)
    }

    func testTextEncoderModelRegistryFindByIdNotFound() {
        let model = TextEncoderModelRegistry.shared.model(withId: "non-existent-model-id")
        XCTAssertNil(model, "Should return nil for non-existent ID")
    }

    func testTextEncoderModelRegistryFindByVariantNotFound() {
        // All variants should exist, but test the lookup mechanism
        let model = TextEncoderModelRegistry.shared.model(withVariant: .bf16)
        // bf16 may or may not be registered depending on implementation
        // Just verify the lookup doesn't crash
        _ = model
    }

    // MARK: - Model Metadata Tests

    func testRegisteredModelsHaveValidMetadata() {
        let models = TextEncoderModelRegistry.shared.allModels()

        for model in models {
            XCTAssertFalse(model.id.isEmpty, "Model ID should not be empty")
            XCTAssertFalse(model.repoId.isEmpty, "Repo ID should not be empty")
            XCTAssertFalse(model.name.isEmpty, "Name should not be empty")
            XCTAssertFalse(model.description.isEmpty, "Description should not be empty")
            XCTAssertEqual(model.parameters, "24B", "Parameters should be 24B")
        }
    }

    func testRegisteredModelsHaveHuggingFaceRepos() {
        let models = TextEncoderModelRegistry.shared.allModels()

        for model in models {
            // Repo IDs should follow HuggingFace format: org/repo
            XCTAssertTrue(model.repoId.contains("/"),
                         "Repo ID '\(model.repoId)' should be in HuggingFace format")
        }
    }

    // MARK: - Tekken JSON URL Test

    func testTekkenJsonURLIsValid() {
        let url = TextEncoderModelRegistry.tekkenJsonURL

        XCTAssertTrue(url.hasPrefix("https://"),
                     "Tekken JSON URL should use HTTPS")
        XCTAssertTrue(url.contains("huggingface.co"),
                     "Tekken JSON URL should be on HuggingFace")
        XCTAssertTrue(url.contains("tekken.json"),
                     "URL should point to tekken.json")
    }

    // MARK: - Sendable Conformance

    func testModelVariantIsSendable() {
        let variant: Sendable = ModelVariant.mlx8bit
        XCTAssertNotNil(variant)
    }

    func testModelInfoIsSendable() {
        let model: Sendable = ModelInfo(
            id: "test",
            repoId: "test/test",
            name: "Test",
            description: "Test",
            variant: .mlx8bit,
            parameters: "24B"
        )
        XCTAssertNotNil(model)
    }

    // MARK: - Gated Status Tests

    func testModelVariantIsGated() {
        // bf16 from mistralai is gated
        XCTAssertTrue(ModelVariant.bf16.isGated)

        // Quantized versions from lmstudio-community are NOT gated
        XCTAssertFalse(ModelVariant.mlx8bit.isGated)
        XCTAssertFalse(ModelVariant.mlx6bit.isGated)
        XCTAssertFalse(ModelVariant.mlx4bit.isGated)
    }

    func testModelInfoIsGated() {
        let models = TextEncoderModelRegistry.shared.allModels()

        // Check bf16 model is gated
        if let bf16Model = models.first(where: { $0.variant == .bf16 }) {
            XCTAssertTrue(bf16Model.isGated)
        }

        // Check quantized models are NOT gated
        if let mlx8bit = models.first(where: { $0.variant == .mlx8bit }) {
            XCTAssertFalse(mlx8bit.isGated)
        }
    }

    func testQwen3VariantIsGated() {
        // All Qwen3 models from lmstudio-community are NOT gated
        XCTAssertFalse(Qwen3Variant.qwen3_4B_8bit.isGated)
        XCTAssertFalse(Qwen3Variant.qwen3_4B_4bit.isGated)
        XCTAssertFalse(Qwen3Variant.qwen3_8B_8bit.isGated)
        XCTAssertFalse(Qwen3Variant.qwen3_8B_4bit.isGated)
    }

    // MARK: - HuggingFace URL Tests

    func testModelVariantHuggingFaceURL() {
        for variant in ModelVariant.allCases {
            XCTAssertTrue(variant.huggingFaceURL.starts(with: "https://huggingface.co/"))
            XCTAssertTrue(variant.huggingFaceURL.contains(variant.repoId))
        }
    }

    func testModelInfoHuggingFaceURL() {
        let models = TextEncoderModelRegistry.shared.allModels()

        for model in models {
            XCTAssertTrue(model.huggingFaceURL.starts(with: "https://huggingface.co/"))
            XCTAssertTrue(model.huggingFaceURL.contains(model.repoId))
        }
    }

    func testQwen3VariantHuggingFaceURL() {
        for variant in Qwen3Variant.allCases {
            XCTAssertTrue(variant.huggingFaceURL.starts(with: "https://huggingface.co/"))
            XCTAssertTrue(variant.huggingFaceURL.contains(variant.repoId))
        }
    }

    func testModelVariantRepoIdValues() {
        // bf16 should be from mistralai
        XCTAssertTrue(ModelVariant.bf16.repoId.contains("mistralai"))

        // Quantized should be from lmstudio-community
        XCTAssertTrue(ModelVariant.mlx8bit.repoId.contains("lmstudio-community"))
        XCTAssertTrue(ModelVariant.mlx6bit.repoId.contains("lmstudio-community"))
        XCTAssertTrue(ModelVariant.mlx4bit.repoId.contains("lmstudio-community"))
    }

    func testQwen3VariantRepoIdValues() {
        // All Qwen3 should be from lmstudio-community
        for variant in Qwen3Variant.allCases {
            XCTAssertTrue(variant.repoId.contains("lmstudio-community"))
        }
    }

    func testModelVariantEstimatedSizeGB() {
        XCTAssertEqual(ModelVariant.bf16.estimatedSizeGB, 48)
        XCTAssertEqual(ModelVariant.mlx8bit.estimatedSizeGB, 25)
        XCTAssertEqual(ModelVariant.mlx6bit.estimatedSizeGB, 19)
        XCTAssertEqual(ModelVariant.mlx4bit.estimatedSizeGB, 14)
    }

    func testQwen3VariantEstimatedSizeGB() {
        XCTAssertEqual(Qwen3Variant.qwen3_4B_8bit.estimatedSizeGB, 4)
        XCTAssertEqual(Qwen3Variant.qwen3_4B_4bit.estimatedSizeGB, 2)
        XCTAssertEqual(Qwen3Variant.qwen3_8B_8bit.estimatedSizeGB, 8)
        XCTAssertEqual(Qwen3Variant.qwen3_8B_4bit.estimatedSizeGB, 4)
    }
}
