import CoreGraphics
import Foundation

/// v3 `.flux2project` package layout (manifest + JXL assets).
public enum FluxGenerationProjectBundle {
    public static let packageExtension = "flux2project"
    public static let manifestName = "project.json"
    public static let previewRelativePath = "preview.jxl"
    public static let slotsDirectoryName = "slots"

    public struct LoadedPackage: Sendable {
        public var project: FluxGenerationProject
        public var bundleRoot: URL?
        public var previewImage: CGImage?

        public init(project: FluxGenerationProject, bundleRoot: URL?, previewImage: CGImage?) {
            self.project = project
            self.bundleRoot = bundleRoot
            self.previewImage = previewImage
        }
    }

    public struct SlotImage: Sendable {
        public var id: UUID
        public var image: CGImage

        public init(id: UUID, image: CGImage) {
            self.id = id
            self.image = image
        }
    }

    public static func isBundleURL(_ url: URL) -> Bool {
        if url.pathExtension == packageExtension {
            return true
        }
        if url.lastPathComponent == manifestName {
            return url.deletingLastPathComponent().pathExtension == packageExtension
        }
        return false
    }

    public static func manifestURL(in bundleRoot: URL) -> URL {
        bundleRoot.appendingPathComponent(manifestName, isDirectory: false)
    }

    public static func slotRelativePath(for slotID: UUID) -> String {
        "\(slotsDirectoryName)/\(slotID.uuidString).\(ProjectBundleImageWriter.fileExtension)"
    }

    public static func save(
        project: FluxGenerationProject,
        slotImages: [SlotImage],
        previewImage: CGImage?,
        to bundleRoot: URL
    ) throws {
        guard ProjectBundleImageWriter.isSupported() else {
            throw Flux2Error.imageProcessingFailed("JPEG XL encoding is not available on this Mac.")
        }
        guard project.version >= FluxGenerationProject.bundleVersion else {
            throw Flux2Error.invalidConfiguration("Project manifest must be version \(FluxGenerationProject.bundleVersion) for bundle save.")
        }

        let fileManager = FileManager.default
        let tempRoot = fileManager.temporaryDirectory
            .appendingPathComponent("flux2project-\(UUID().uuidString)", isDirectory: true)

        do {
            try fileManager.createDirectory(at: tempRoot, withIntermediateDirectories: true)
            let slotsRoot = tempRoot.appendingPathComponent(slotsDirectoryName, isDirectory: true)
            try fileManager.createDirectory(at: slotsRoot, withIntermediateDirectories: true)

            for slot in slotImages {
                let url = tempRoot.appendingPathComponent(Self.slotRelativePath(for: slot.id), isDirectory: false)
                try ProjectBundleImageWriter.write(slot.image, to: url, mode: .lossless)
            }

            if let previewImage {
                try ProjectBundleImageWriter.write(
                    previewImage,
                    to: tempRoot.appendingPathComponent(previewRelativePath, isDirectory: false),
                    mode: .lossless
                )
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let manifestData = try encoder.encode(project)
            try manifestData.write(to: manifestURL(in: tempRoot), options: .atomic)

            if fileManager.fileExists(atPath: bundleRoot.path) {
                try fileManager.removeItem(at: bundleRoot)
            }
            try fileManager.createDirectory(
                at: bundleRoot.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try fileManager.moveItem(at: tempRoot, to: bundleRoot)
        } catch {
            try? fileManager.removeItem(at: tempRoot)
            throw error
        }
    }

    public static func load(from url: URL) throws -> LoadedPackage {
        let bundleRoot = resolveBundleRoot(url)
        guard isBundleURL(bundleRoot) else {
            throw Flux2Error.invalidConfiguration("Not a FLUX.2 project bundle: \(url.path)")
        }

        let manifestData = try Data(contentsOf: manifestURL(in: bundleRoot))
        let project = try FluxGenerationProject.loadManifest(from: manifestData)
        guard project.version >= FluxGenerationProject.bundleVersion else {
            throw FluxGenerationProjectError.unsupportedBundleVersion(project.version)
        }

        let previewURL = bundleRoot.appendingPathComponent(previewRelativePath, isDirectory: false)
        let previewImage: CGImage?
        if FileManager.default.fileExists(atPath: previewURL.path) {
            previewImage = try ProjectBundleImageWriter.loadCGImage(from: previewURL)
        } else {
            previewImage = nil
        }

        return LoadedPackage(project: project, bundleRoot: bundleRoot, previewImage: previewImage)
    }

    private static func resolveBundleRoot(_ url: URL) -> URL {
        if url.lastPathComponent == manifestName {
            return url.deletingLastPathComponent()
        }
        return url
    }
}
