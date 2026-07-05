import CoreGraphics
import Flux2Core
@testable import Flux2App
import XCTest

@MainActor
final class EditHistoryStoreTests: XCTestCase {
    func testHistoryAssetsFromMemoryWithoutBundleRoot() throws {
        try XCTSkipUnless(ProjectBundleImageWriter.isSupported(), "JPEG XL encoding is not available")

        let store = EditHistoryStore()
        let master = try makeTestImage(width: 16, height: 16)
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1))
        )
        let settings = EditHistorySettings(selectedModel: "klein-4b", steps: 4, guidance: 1)
        _ = try store.append(
            master: master,
            label: "Import",
            kind: .import,
            prompt: "test",
            settings: settings,
            spatial: spatial
        )

        let assets = try store.historyAssets(bundleRoot: nil)
        XCTAssertEqual(assets.count, 1)
        XCTAssertEqual(assets[0].master.width, 16)
    }

    func testHistoryAssetsLoadsMastersFromBundleAfterLoad() throws {
        try XCTSkipUnless(ProjectBundleImageWriter.isSupported(), "JPEG XL encoding is not available")

        let store = EditHistoryStore()
        let master = try makeTestImage(width: 32, height: 24)
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1))
        )
        let settings = EditHistorySettings(
            selectedModel: "klein-4b",
            steps: 4,
            guidance: 1,
            generateRoute: I2IGenerateRoute.fullImage.rawValue
        )
        _ = try store.append(
            master: master,
            label: "Import",
            kind: .import,
            prompt: "test",
            settings: settings,
            spatial: spatial
        )

        let bundleRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-history-test-\(UUID().uuidString).\(FluxGenerationProjectBundle.packageExtension)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: bundleRoot) }

        let assets = try store.historyAssets(bundleRoot: nil)
        let project = FluxGenerationProject(
            version: FluxGenerationProject.bundleVersion,
            selectedModel: "klein-4b",
            textQuantization: "8bit",
            transformerQuantization: "8bit",
            prompt: "test",
            upsamplePrompt: false,
            width: 32,
            height: 24,
            steps: 4,
            guidance: 1,
            seed: "",
            contextArea: spatial.contextArea,
            images: [GenerationImageRecord()],
            currentHistoryIndex: store.manifestFields().currentIndex,
            history: store.manifestFields().entries
        )
        try FluxGenerationProjectBundle.save(
            project: project,
            slotImages: [],
            previewImage: master,
            historyAssets: assets,
            to: bundleRoot
        )

        let reloadedStore = EditHistoryStore()
        let loaded = try FluxGenerationProject.load(at: bundleRoot)
        reloadedStore.load(from: loaded.project, bundleRoot: loaded.bundleRoot)

        let roundTripAssets = try reloadedStore.historyAssets(bundleRoot: loaded.bundleRoot)
        XCTAssertEqual(roundTripAssets.count, 1)
        XCTAssertEqual(roundTripAssets[0].master.width, 32)
        XCTAssertEqual(roundTripAssets[0].master.height, 24)
    }

    func testLoadRejectsOutOfRangeCurrentHistoryIndex() {
        let store = EditHistoryStore()
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1))
        )
        let settings = EditHistorySettings(selectedModel: "klein-4b", steps: 4, guidance: 1)
        let entry = EditHistoryEntry(
            label: "Import",
            master: EditHistoryPaths.masterPath(sequence: 1),
            thumb: EditHistoryPaths.thumbPath(sequence: 1),
            kind: .import,
            prompt: "test",
            settings: settings,
            spatial: spatial
        )
        let project = FluxGenerationProject(
            version: FluxGenerationProject.bundleVersion,
            selectedModel: "klein-4b",
            textQuantization: "8bit",
            transformerQuantization: "8bit",
            prompt: "test",
            upsamplePrompt: false,
            width: 32,
            height: 24,
            steps: 4,
            guidance: 1,
            seed: "",
            contextArea: spatial.contextArea,
            images: [GenerationImageRecord()],
            currentHistoryIndex: 99,
            history: [entry]
        )

        store.load(from: project, bundleRoot: nil)
        XCTAssertEqual(store.entries.count, 1)
        XCTAssertNil(store.currentIndex)
    }

    func testAppendPruneReportsDroppedStepCount() throws {
        try XCTSkipUnless(ProjectBundleImageWriter.isSupported(), "JPEG XL encoding is not available")

        let store = EditHistoryStore()
        let master = try makeTestImage(width: 8, height: 8)
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1))
        )
        let settings = EditHistorySettings(selectedModel: "klein-4b", steps: 4, guidance: 1)

        for step in 1...EditHistoryStore.maxEntryCount + 2 {
            _ = try store.append(
                master: master,
                label: "Step \(step)",
                kind: .generate,
                prompt: "test",
                settings: settings,
                spatial: spatial
            )
        }

        XCTAssertEqual(store.entries.count, EditHistoryStore.maxEntryCount)
        XCTAssertGreaterThan(store.lastPrunedStepCount, 0)
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
