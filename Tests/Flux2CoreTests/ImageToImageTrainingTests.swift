// ImageToImageTrainingTests.swift - Unit tests for Image-to-Image LoRA training
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
import MLX

// MARK: - LoRATrainingConfig I2I Tests

final class ImageToImageConfigTests: XCTestCase {

    let tmpDataset = URL(fileURLWithPath: "/tmp/test-dataset")
    let tmpOutput = URL(fileURLWithPath: "/tmp/test-output")

    // MARK: - controlPath / controlDropout / isImageToImage

    func testConfigDefaultsToTextToImage() {
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )

        XCTAssertNil(config.controlPath)
        XCTAssertEqual(config.controlDropout, 0.0)
        XCTAssertFalse(config.isImageToImage)
    }

    func testConfigWithControlPathIsI2I() {
        let controlURL = URL(fileURLWithPath: "/tmp/controls")
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            controlPath: controlURL,
            controlDropout: 0.3,
            outputPath: tmpOutput
        )

        XCTAssertEqual(config.controlPath, controlURL)
        XCTAssertEqual(config.controlDropout, 0.3)
        XCTAssertTrue(config.isImageToImage)
    }

    func testConfigNilControlPathIsT2I() {
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            controlPath: nil,
            controlDropout: 0.5,
            outputPath: tmpOutput
        )

        XCTAssertFalse(config.isImageToImage)
    }

    func testConfigControlDropoutDefaultsToZero() {
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            controlPath: URL(fileURLWithPath: "/tmp/controls"),
            outputPath: tmpOutput
        )

        XCTAssertEqual(config.controlDropout, 0.0)
    }

    // MARK: - ValidationPromptConfig.referenceImage

    func testValidationPromptDefaultNoReferenceImage() {
        let prompt = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "a cat sitting",
            is512: true
        )

        XCTAssertNil(prompt.referenceImage)
    }

    func testValidationPromptWithReferenceImage() {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let prompt = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "remove the hat",
            is512: true,
            referenceImage: refURL
        )

        XCTAssertEqual(prompt.referenceImage, refURL)
    }

    // MARK: - Codable round-trip (referenceImage)

    func testValidationPromptCodableRoundTrip() throws {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let original = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "edit instruction",
            is512: true,
            is1024: false,
            applyTrigger: true,
            seed: 42,
            referenceImage: refURL
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(LoRATrainingConfig.ValidationPromptConfig.self, from: data)

        XCTAssertEqual(decoded.prompt, "edit instruction")
        XCTAssertEqual(decoded.is512, true)
        XCTAssertEqual(decoded.is1024, false)
        XCTAssertEqual(decoded.applyTrigger, true)
        XCTAssertEqual(decoded.seed, 42)
        XCTAssertEqual(decoded.referenceImage, refURL)
    }

    func testValidationPromptCodableNilReferenceImage() throws {
        let original = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "a landscape",
            is512: true
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(
            LoRATrainingConfig.ValidationPromptConfig.self, from: data
        )

        XCTAssertNil(decoded.referenceImage)
    }

    // MARK: - Presets remain T2I

    func testPresetsHaveNoControlPath() {
        let minimal = LoRATrainingConfig.minimal8GB(
            datasetPath: tmpDataset, outputPath: tmpOutput
        )
        XCTAssertNil(minimal.controlPath)
        XCTAssertFalse(minimal.isImageToImage)

        let balanced = LoRATrainingConfig.balanced16GB(
            datasetPath: tmpDataset, outputPath: tmpOutput
        )
        XCTAssertNil(balanced.controlPath)
        XCTAssertFalse(balanced.isImageToImage)

        let quality = LoRATrainingConfig.quality32GB(
            datasetPath: tmpDataset, outputPath: tmpOutput
        )
        XCTAssertNil(quality.controlPath)
        XCTAssertFalse(quality.isImageToImage)
    }
}

// MARK: - CachedLatentEntry I2I Tests

final class CachedLatentEntryI2ITests: XCTestCase {

    func testCachedLatentEntryDefaultNoControl() {
        let entry = CachedLatentEntry(
            filename: "test.png",
            latent: MLXArray.zeros([1, 32, 64, 64]),
            width: 512,
            height: 512
        )

        XCTAssertNil(entry.controlLatent)
        XCTAssertEqual(entry.filename, "test.png")
        XCTAssertEqual(entry.width, 512)
        XCTAssertEqual(entry.height, 512)
    }

    func testCachedLatentEntryWithControlLatent() {
        let targetLatent = MLXArray.zeros([1, 32, 64, 64])
        let controlLatent = MLXArray.ones([1, 32, 64, 64])

        let entry = CachedLatentEntry(
            filename: "edit_001.png",
            latent: targetLatent,
            width: 512,
            height: 512,
            controlLatent: controlLatent
        )

        XCTAssertNotNil(entry.controlLatent)
        XCTAssertEqual(entry.controlLatent!.shape, [1, 32, 64, 64])
    }

    func testCachedLatentEntryControlLatentMatchesTargetShape() {
        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let entry = CachedLatentEntry(
            filename: "pair.png",
            latent: target,
            width: w * 8,
            height: h * 8,
            controlLatent: control
        )

        // Control and target should have same spatial dimensions
        XCTAssertEqual(entry.latent.shape, entry.controlLatent!.shape)
    }
}

// MARK: - SimpleLoRAConfig I2I Tests

final class SimpleLoRAConfigI2ITests: XCTestCase {

    func testSimpleLoRAConfigDefaultControlDropout() {
        let config = SimpleLoRAConfig(outputDir: URL(fileURLWithPath: "/tmp"))

        XCTAssertEqual(config.controlDropout, 0.0)
    }

    func testSimpleLoRAConfigControlDropoutSetting() {
        var config = SimpleLoRAConfig(outputDir: URL(fileURLWithPath: "/tmp"))
        config.controlDropout = 0.3

        XCTAssertEqual(config.controlDropout, 0.3)
    }

    func testSimpleLoRAConfigValidationPromptReferenceImage() {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let prompt = SimpleLoRAConfig.ValidationPromptConfig(
            prompt: "remove background",
            is512: true,
            applyTrigger: false,
            referenceImage: refURL
        )

        XCTAssertEqual(prompt.referenceImage, refURL)
    }

    func testSimpleLoRAConfigValidationPromptNoReferenceImage() {
        let prompt = SimpleLoRAConfig.ValidationPromptConfig(
            prompt: "a cat",
            is512: true,
            applyTrigger: true,
            seed: 42
        )

        XCTAssertNil(prompt.referenceImage)
    }
}

// MARK: - Position ID Tests for I2I

final class PositionIDI2ITests: XCTestCase {

    func testReferenceImagePositionIDsGeneration() {
        let height = 512
        let width = 512
        let latentH = height / 8  // 64
        let latentW = width / 8   // 64

        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH,
            latentWidth: latentW,
            imageIndex: 0
        )

        // Position count: latentH * latentW = 4096 (patchification is separate)
        let expectedPatches = latentH * latentW
        XCTAssertEqual(refImgIds.shape[0], expectedPatches)
        XCTAssertEqual(refImgIds.shape[1], 4)  // [T, H, W, L]
    }

    func testReferenceImagePositionIDsTCoordinate() {
        let latentH = 64
        let latentW = 64

        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH,
            latentWidth: latentW,
            imageIndex: 0
        )

        eval(refImgIds)

        // T coordinate for first reference image should be 10 (T=10 + imageIndex*10)
        let firstT = refImgIds[0, 0].item(Int32.self)
        XCTAssertEqual(firstT, 10, "First reference image should have T=10")
    }

    func testReferenceImagePositionIDsSecondImage() {
        let latentH = 64
        let latentW = 64

        let refIds0 = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH, latentWidth: latentW, imageIndex: 0
        )
        let refIds1 = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH, latentWidth: latentW, imageIndex: 1
        )

        eval(refIds0, refIds1)

        let t0 = refIds0[0, 0].item(Int32.self)
        let t1 = refIds1[0, 0].item(Int32.self)

        // Different images should have different T coordinates
        XCTAssertNotEqual(t0, t1)
    }

    func testConcatenatedPositionIDs() {
        let height = 512
        let width = 512

        let imgIds = LatentUtils.generateImagePositionIDs(height: height, width: width)
        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: height / 8,
            latentWidth: width / 8,
            imageIndex: 0
        )

        // Concatenate along sequence dimension
        let combined = concatenated([imgIds, refImgIds], axis: 0)

        let expectedTotal = imgIds.shape[0] + refImgIds.shape[0]
        XCTAssertEqual(combined.shape[0], expectedTotal)
        XCTAssertEqual(combined.shape[1], 4)
    }
}

// MARK: - Latent Packing I2I Tests

final class LatentPackingI2ITests: XCTestCase {

    func testPackedControlLatentSameShapeAsTarget() {
        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let packedTarget = LatentUtils.packLatents(target, patchSize: 2)
        let packedControl = LatentUtils.packLatents(control, patchSize: 2)

        XCTAssertEqual(packedTarget.shape, packedControl.shape)
    }

    func testConcatenatedLatentSequence() {
        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let packedTarget = LatentUtils.packLatents(target, patchSize: 2)
        let packedControl = LatentUtils.packLatents(control, patchSize: 2)

        // Concatenate along sequence dimension
        let combined = concatenated([packedTarget, packedControl], axis: 1)

        let outputSeqLen = packedTarget.shape[1]
        let totalSeqLen = combined.shape[1]

        XCTAssertEqual(totalSeqLen, 2 * outputSeqLen)
        XCTAssertEqual(combined.shape[0], 1)  // batch
        XCTAssertEqual(combined.shape[2], 128)  // channels
    }

    func testOutputSlicingFromCombinedLatent() {
        let seqLen = 1024
        let channels = 128

        // Simulate model output with combined sequence
        let modelOutput = MLXRandom.normal([1, 2 * seqLen, channels])

        // Slice output portion only
        let outputPortion = modelOutput[0..., 0..<seqLen, 0...]

        eval(outputPortion)
        XCTAssertEqual(outputPortion.shape, [1, seqLen, channels])
    }
}
