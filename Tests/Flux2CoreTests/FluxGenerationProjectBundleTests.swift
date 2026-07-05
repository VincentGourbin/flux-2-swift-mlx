import CoreGraphics
import Flux2Core
import XCTest

final class FluxGenerationProjectBundleTests: XCTestCase {
    func testBundleRoundTrip() throws {
        try XCTSkipUnless(ProjectBundleImageWriter.isSupported(), "JPEG XL encoding is not available")

        let slotID = UUID()
        let slotImage = try makeTestImage(width: 32, height: 24)
        let previewImage = try makeTestImage(width: 16, height: 16)

        let project = FluxGenerationProject(
            version: FluxGenerationProject.bundleVersion,
            selectedModel: "klein-4b",
            textQuantization: "8bit",
            transformerQuantization: "8bit",
            prompt: "bundle test",
            upsamplePrompt: false,
            width: 32,
            height: 24,
            steps: 4,
            guidance: 1,
            seed: "",
            contextArea: .init(CGRect(x: 0, y: 0, width: 1, height: 1)),
            images: [
                GenerationImageRecord(
                    id: slotID,
                    role: .reference,
                    isPrimary: true,
                    bundlePath: FluxGenerationProjectBundle.slotRelativePath(for: slotID)
                ),
            ],
            selectedImageSlotID: slotID
        )

        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-bundle-test-\(UUID().uuidString).\(FluxGenerationProjectBundle.packageExtension)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        try FluxGenerationProjectBundle.save(
            project: project,
            slotImages: [FluxGenerationProjectBundle.SlotImage(id: slotID, image: slotImage)],
            previewImage: previewImage,
            to: tempRoot
        )

        XCTAssertTrue(FluxGenerationProjectBundle.isBundleURL(tempRoot))
        XCTAssertTrue(FileManager.default.fileExists(atPath: FluxGenerationProjectBundle.manifestURL(in: tempRoot).path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: tempRoot.appendingPathComponent(FluxGenerationProjectBundle.previewRelativePath).path
            )
        )

        let loaded = try FluxGenerationProjectBundle.load(from: tempRoot)
        XCTAssertEqual(loaded.project.prompt, "bundle test")
        XCTAssertEqual(loaded.project.version, FluxGenerationProject.bundleVersion)
        XCTAssertNotNil(loaded.previewImage)

        let references = try loaded.project.loadReferenceCGImages(bundleRoot: loaded.bundleRoot)
        XCTAssertEqual(references.count, 1)
        XCTAssertEqual(references[0].width, 32)
        XCTAssertEqual(references[0].height, 24)
    }

    func testFlatJSONProjectStillLoads() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/VMSmoke/project.json")
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("VMSmoke fixture missing")
        }

        let loaded = try FluxGenerationProject.load(at: fixture)
        XCTAssertEqual(loaded.project.version, 2)
        XCTAssertNil(loaded.bundleRoot)
        XCTAssertFalse(loaded.project.referenceImageRecords().isEmpty)
    }

    private func makeTestImage(width: Int, height: Int) throws -> CGImage {
        var bytes = [UInt8](repeating: 200, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let image = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to allocate test image")
        }
        return image
    }
}
