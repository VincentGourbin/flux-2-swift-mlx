import CoreGraphics
import Flux2Core
import ImageIO
import XCTest

/// Builds a v3 bundle with one Import history step for VM smoke (`bin/vm-smoke-history.sh`).
/// Invoke: `EXPORT_VMSMOKE_HISTORY_BUNDLE=/path/to/Name.flux2project swift test --filter testExportVMSmokeHistoryBundleWhenRequested`
final class VMSmokeHistoryBundleTests: XCTestCase {
    func testExportVMSmokeHistoryBundleWhenRequested() throws {
        guard let exportPath = ProcessInfo.processInfo.environment["EXPORT_VMSMOKE_HISTORY_BUNDLE"],
              !exportPath.isEmpty else {
            throw XCTSkip("Set EXPORT_VMSMOKE_HISTORY_BUNDLE to export the VM history smoke bundle")
        }
        try XCTSkipUnless(ProjectBundleImageWriter.isSupported(), "JPEG XL encoding is not available")

        let referenceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/VMSmoke/reference.png")
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("VMSmoke reference.png missing")
        }

        let reference = try loadPNG(from: referenceURL)
        let thumb = try ProjectBundleImageWriter.makeThumbnail(from: reference)
        let slotID = UUID()
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8))
        )
        let settings = EditHistorySettings(
            selectedModel: "klein-4b",
            steps: 4,
            guidance: 4,
            megapixelBudget: 1.0,
            generateRoute: I2IGenerateRoute.fullImage.rawValue
        )
        let historyEntry = EditHistoryEntry(
            label: "Import",
            master: EditHistoryPaths.masterPath(sequence: 1),
            thumb: EditHistoryPaths.thumbPath(sequence: 1),
            kind: .import,
            prompt: "VM smoke history — import milestone",
            settings: settings,
            spatial: spatial
        )

        let project = FluxGenerationProject(
            version: FluxGenerationProject.bundleVersion,
            selectedModel: "klein-4b",
            textQuantization: "8bit",
            transformerQuantization: "8bit",
            prompt: "VM smoke history — import milestone",
            upsamplePrompt: false,
            width: reference.width,
            height: reference.height,
            steps: 4,
            guidance: 4,
            seed: "",
            contextArea: spatial.contextArea,
            images: [
                GenerationImageRecord(
                    id: slotID,
                    role: .reference,
                    isPrimary: true,
                    bundlePath: FluxGenerationProjectBundle.slotRelativePath(for: slotID)
                ),
            ],
            selectedImageSlotID: slotID,
            currentHistoryIndex: 0,
            history: [historyEntry]
        )

        let bundleURL = URL(fileURLWithPath: exportPath, isDirectory: true)
        if FileManager.default.fileExists(atPath: bundleURL.path) {
            try FileManager.default.removeItem(at: bundleURL)
        }

        try FluxGenerationProjectBundle.save(
            project: project,
            slotImages: [FluxGenerationProjectBundle.SlotImage(id: slotID, image: reference)],
            previewImage: reference,
            historyAssets: [
                FluxGenerationProjectBundle.HistoryAsset(entry: historyEntry, master: reference, thumb: thumb),
            ],
            to: bundleURL
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: bundleURL.appendingPathComponent("history/0001.jxl").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: bundleURL.appendingPathComponent("thumbs/0001.jxl").path))
    }

    private func loadPNG(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Flux2Error.imageProcessingFailed("Could not load \(url.path)")
        }
        return image
    }
}
