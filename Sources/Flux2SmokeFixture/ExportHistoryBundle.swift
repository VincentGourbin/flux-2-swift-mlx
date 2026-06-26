import CoreGraphics
import Flux2Core
import Foundation
import ImageIO

@main
enum Flux2SmokeFixture {
    /// Stable IDs so the committed VM smoke bundle does not churn in git.
    private static let smokeSlotID = UUID(uuidString: "a1111111-1111-4111-8111-111111111101")!
    private static let smokeHistoryID = UUID(uuidString: "a1111111-1111-4111-8111-111111111102")!

    static func main() throws {
        guard CommandLine.arguments.count >= 2 else {
            fputs("usage: Flux2SmokeFixture export-history-bundle <output.flux2project>\n", stderr)
            exit(2)
        }
        guard CommandLine.arguments[1] == "export-history-bundle" else {
            fputs("Unknown command: \(CommandLine.arguments[1])\n", stderr)
            exit(2)
        }

        let bundlePath = CommandLine.arguments[2]
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let referenceURL = root
            .appendingPathComponent("Tests/Fixtures/VMSmoke/reference.png")

        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            fputs("Missing fixture: \(referenceURL.path)\n", stderr)
            exit(1)
        }

        let reference = try loadPNG(from: referenceURL)
        let thumb = try ProjectBundleImageWriter.makeThumbnail(from: reference)
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
            id: smokeHistoryID,
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
                    id: smokeSlotID,
                    role: .reference,
                    isPrimary: true,
                    bundlePath: FluxGenerationProjectBundle.slotRelativePath(for: smokeSlotID)
                ),
            ],
            selectedImageSlotID: smokeSlotID,
            currentHistoryIndex: 0,
            history: [historyEntry]
        )

        let bundleURL = URL(fileURLWithPath: bundlePath, isDirectory: true)
        try FluxGenerationProjectBundle.save(
            project: project,
            slotImages: [FluxGenerationProjectBundle.SlotImage(id: smokeSlotID, image: reference)],
            previewImage: reference,
            historyAssets: [
                FluxGenerationProjectBundle.HistoryAsset(entry: historyEntry, master: reference, thumb: thumb),
            ],
            to: bundleURL
        )

        print(bundleURL.path)
    }

    private static func loadPNG(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Flux2Error.imageProcessingFailed("Could not load \(url.path)")
        }
        return image
    }
}
