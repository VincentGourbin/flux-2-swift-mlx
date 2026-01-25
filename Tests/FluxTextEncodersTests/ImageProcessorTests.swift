/**
 * ImageProcessorTests.swift
 * Unit tests for ImageProcessor and ImageProcessorConfig
 */

import XCTest
@testable import FluxTextEncoders

final class ImageProcessorTests: XCTestCase {

    // MARK: - ImageProcessorConfig Tests

    func testPixtralConfigDefaults() {
        let config = ImageProcessorConfig.pixtral

        XCTAssertEqual(config.imageSize, 1540, "Pixtral image size should be 1540")
        XCTAssertEqual(config.patchSize, 14, "Pixtral patch size should be 14")
        XCTAssertEqual(config.rescaleFactor, 1.0 / 255.0, accuracy: 0.0001,
                      "Rescale factor should be 1/255")
    }

    func testPixtralConfigImageMean() {
        let config = ImageProcessorConfig.pixtral

        XCTAssertEqual(config.imageMean.count, 3, "Should have 3 mean values (RGB)")
        XCTAssertEqual(config.imageMean[0], 0.48145466, accuracy: 0.0001, "R mean")
        XCTAssertEqual(config.imageMean[1], 0.4578275, accuracy: 0.0001, "G mean")
        XCTAssertEqual(config.imageMean[2], 0.40821073, accuracy: 0.0001, "B mean")
    }

    func testPixtralConfigImageStd() {
        let config = ImageProcessorConfig.pixtral

        XCTAssertEqual(config.imageStd.count, 3, "Should have 3 std values (RGB)")
        XCTAssertEqual(config.imageStd[0], 0.26862954, accuracy: 0.0001, "R std")
        XCTAssertEqual(config.imageStd[1], 0.26130258, accuracy: 0.0001, "G std")
        XCTAssertEqual(config.imageStd[2], 0.27577711, accuracy: 0.0001, "B std")
    }

    func testCustomConfigInit() {
        let config = ImageProcessorConfig(
            imageSize: 224,
            patchSize: 16,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5],
            rescaleFactor: 1.0 / 255.0
        )

        XCTAssertEqual(config.imageSize, 224)
        XCTAssertEqual(config.patchSize, 16)
        XCTAssertEqual(config.imageMean, [0.5, 0.5, 0.5])
        XCTAssertEqual(config.imageStd, [0.5, 0.5, 0.5])
    }

    // MARK: - ImageProcessor Tests

    func testImageProcessorInitWithDefaultConfig() {
        let processor = ImageProcessor()

        XCTAssertEqual(processor.config.imageSize, 1540,
                      "Default config should be Pixtral")
    }

    func testImageProcessorInitWithCustomConfig() {
        let customConfig = ImageProcessorConfig(
            imageSize: 384,
            patchSize: 14,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5],
            rescaleFactor: 1.0 / 255.0
        )
        let processor = ImageProcessor(config: customConfig)

        XCTAssertEqual(processor.config.imageSize, 384)
    }

    func testGetNumPatches() {
        let processor = ImageProcessor()

        // Test with dimensions divisible by patch size
        let (patchesX, patchesY, total) = processor.getNumPatches(width: 224, height: 224)

        XCTAssertEqual(patchesX, 224 / 14, "Width patches")
        XCTAssertEqual(patchesY, 224 / 14, "Height patches")
        XCTAssertEqual(total, patchesX * patchesY, "Total patches")
    }

    func testGetNumPatchesVariousSizes() {
        let processor = ImageProcessor()

        // 336x336
        let (px1, py1, t1) = processor.getNumPatches(width: 336, height: 336)
        XCTAssertEqual(px1, 24)
        XCTAssertEqual(py1, 24)
        XCTAssertEqual(t1, 576)

        // 448x336 (rectangular)
        let (px2, py2, t2) = processor.getNumPatches(width: 448, height: 336)
        XCTAssertEqual(px2, 32)
        XCTAssertEqual(py2, 24)
        XCTAssertEqual(t2, 768)
    }

    // MARK: - Error Tests

    func testImageProcessorErrorDescriptions() {
        XCTAssertEqual(ImageProcessorError.invalidImage.errorDescription,
                      "Invalid image format")
        XCTAssertEqual(ImageProcessorError.contextCreationFailed.errorDescription,
                      "Failed to create graphics context")
        XCTAssertEqual(ImageProcessorError.unsupportedFormat.errorDescription,
                      "Unsupported image format")

        let fileNotFound = ImageProcessorError.fileNotFound("/path/to/image.jpg")
        XCTAssertEqual(fileNotFound.errorDescription,
                      "Image file not found: /path/to/image.jpg")
    }

    func testLoadImageFromNonExistentPath() {
        let processor = ImageProcessor()

        XCTAssertThrowsError(try processor.loadImage(from: "/nonexistent/path/image.jpg")) { error in
            if case ImageProcessorError.fileNotFound(let path) = error {
                XCTAssertEqual(path, "/nonexistent/path/image.jpg")
            } else {
                XCTFail("Expected fileNotFound error")
            }
        }
    }

    func testPreprocessFromFileNonExistent() {
        let processor = ImageProcessor()

        XCTAssertThrowsError(try processor.preprocessFromFile("/nonexistent.jpg")) { error in
            XCTAssertTrue(error is ImageProcessorError)
        }
    }

    // MARK: - Patch Calculation Tests

    func testPatchCalculationWithPixtralConfig() {
        let processor = ImageProcessor(config: .pixtral)

        // Maximum size (1540x1540)
        let (maxPX, maxPY, maxTotal) = processor.getNumPatches(width: 1540, height: 1540)
        XCTAssertEqual(maxPX, 110)  // 1540 / 14
        XCTAssertEqual(maxPY, 110)
        XCTAssertEqual(maxTotal, 12100)
    }

    func testPatchCalculationMinimum() {
        let processor = ImageProcessor()

        // Single patch (14x14)
        let (px, py, total) = processor.getNumPatches(width: 14, height: 14)
        XCTAssertEqual(px, 1)
        XCTAssertEqual(py, 1)
        XCTAssertEqual(total, 1)
    }

    // MARK: - Config Codable Tests

    func testImageProcessorConfigCodable() throws {
        let config = ImageProcessorConfig.pixtral

        let encoder = JSONEncoder()
        let data = try encoder.encode(config)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ImageProcessorConfig.self, from: data)

        XCTAssertEqual(decoded.imageSize, config.imageSize)
        XCTAssertEqual(decoded.patchSize, config.patchSize)
        XCTAssertEqual(decoded.imageMean, config.imageMean)
        XCTAssertEqual(decoded.imageStd, config.imageStd)
        XCTAssertEqual(decoded.rescaleFactor, config.rescaleFactor)
    }

    // MARK: - Sendable Conformance

    func testImageProcessorConfigIsSendable() {
        let config: Sendable = ImageProcessorConfig.pixtral
        XCTAssertNotNil(config)
    }
}
