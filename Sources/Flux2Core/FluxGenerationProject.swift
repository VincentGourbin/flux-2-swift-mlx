import CoreGraphics
import Foundation
import ImageIO

/// Flux2App generation project JSON (flat v2 or bundle v3 manifest).
public struct FluxGenerationProject: Codable, Sendable {
    /// Version written by **Save** today (v3 bundle manifest).
    public static let currentVersion = 3
    /// Oldest flat JSON manifest still opened directly.
    public static let minimumLoadableVersion = 2
    /// Bundle packages require this manifest version.
    public static let bundleVersion = 3
    public static let maxImageSlots = 16

    public var version: Int = currentVersion
    public var selectedModel: String
    public var textQuantization: String
    public var transformerQuantization: String
    public var prompt: String
    public var upsamplePrompt: Bool
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: String
    public var preparationOverlayOpacity: Double?
    public var megapixelBudget: Double?
    public var clearPromptAfterGeneration: Bool?
    public var selectedFamily: String?
    public var processArea: NormalizedRect?
    public var contextArea: NormalizedRect
    public var editMode: String?
    public var inpaintMaskTool: String?
    public var outpaintPadding: OutpaintPadding?
    public var inpaintIntent: String?
    public var enrichInpaintPromptWithVLM: Bool?
    public var vlmContextManual: Bool?
    /// Generative-fill Qwen context slider: -1 = minimum, 0 = auto, 1 = full image.
    public var fillContextMaskScale: Double?
    public var inpaintMaskLayers: [InpaintMaskLayer]?
    public var images: [GenerationImageRecord]
    public var selectedImageSlotID: UUID?

    public struct NormalizedRect: Codable, Sendable, Equatable {
        public var x: Double
        public var y: Double
        public var width: Double
        public var height: Double

        public var cgRect: CGRect {
            CGRect(x: x, y: y, width: width, height: height)
        }

        public init(_ rect: CGRect) {
            x = rect.minX
            y = rect.minY
            width = rect.width
            height = rect.height
        }
    }

    public init(
        version: Int = currentVersion,
        selectedModel: String,
        textQuantization: String,
        transformerQuantization: String,
        prompt: String,
        upsamplePrompt: Bool,
        width: Int,
        height: Int,
        steps: Int,
        guidance: Float,
        seed: String,
        preparationOverlayOpacity: Double? = nil,
        megapixelBudget: Double? = nil,
        clearPromptAfterGeneration: Bool? = nil,
        selectedFamily: String? = nil,
        processArea: NormalizedRect? = nil,
        contextArea: NormalizedRect,
        editMode: String? = nil,
        inpaintMaskTool: String? = nil,
        outpaintPadding: OutpaintPadding? = nil,
        inpaintIntent: String? = nil,
        enrichInpaintPromptWithVLM: Bool? = nil,
        vlmContextManual: Bool? = nil,
        fillContextMaskScale: Double? = nil,
        inpaintMaskLayers: [InpaintMaskLayer]? = nil,
        images: [GenerationImageRecord],
        selectedImageSlotID: UUID? = nil
    ) {
        self.version = version
        self.selectedModel = selectedModel
        self.textQuantization = textQuantization
        self.transformerQuantization = transformerQuantization
        self.prompt = prompt
        self.upsamplePrompt = upsamplePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.preparationOverlayOpacity = preparationOverlayOpacity
        self.megapixelBudget = megapixelBudget
        self.clearPromptAfterGeneration = clearPromptAfterGeneration
        self.selectedFamily = selectedFamily
        self.processArea = processArea
        self.contextArea = contextArea
        self.editMode = editMode
        self.inpaintMaskTool = inpaintMaskTool
        self.outpaintPadding = outpaintPadding
        self.inpaintIntent = inpaintIntent
        self.enrichInpaintPromptWithVLM = enrichInpaintPromptWithVLM
        self.vlmContextManual = vlmContextManual
        self.fillContextMaskScale = fillContextMaskScale
        self.inpaintMaskLayers = inpaintMaskLayers
        self.images = images
        self.selectedImageSlotID = selectedImageSlotID
    }

    public static func load(from path: String) throws -> FluxGenerationProject {
        let url = URL(fileURLWithPath: path)
        return try load(at: url).project
    }

    /// Load a flat `project.json` file or a `.flux2project` package.
    public static func load(at url: URL) throws -> FluxGenerationProjectBundle.LoadedPackage {
        if FluxGenerationProjectBundle.isBundleURL(url) {
            return try FluxGenerationProjectBundle.load(from: url)
        }
        let data = try Data(contentsOf: url)
        let project = try loadManifest(from: data)
        return FluxGenerationProjectBundle.LoadedPackage(project: project, bundleRoot: nil, previewImage: nil)
    }

    public static func loadManifest(from data: Data) throws -> FluxGenerationProject {
        if let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let version = object["version"] as? Int,
           version < minimumLoadableVersion {
            throw FluxGenerationProjectError.unsupportedVersion(version)
        }
        let project = try JSONDecoder().decode(FluxGenerationProject.self, from: data)
        guard project.version >= minimumLoadableVersion else {
            throw FluxGenerationProjectError.unsupportedVersion(project.version)
        }
        return project
    }

    public static func load(from data: Data) throws -> FluxGenerationProject {
        try loadManifest(from: data)
    }

    /// Reference-role images for generation: primary first, then tab order.
    public func referenceImageRecords(bundleRoot: URL? = nil) -> [GenerationImageRecord] {
        let references = images.filter { $0.role == .reference && $0.hasStoredImage(bundleRoot: bundleRoot) }
        guard let primary = references.first(where: \.isPrimary) else {
            return references
        }
        return [primary] + references.filter { $0.id != primary.id }
    }

    public func interpretImageRecords(bundleRoot: URL? = nil) -> [GenerationImageRecord] {
        images.filter { $0.role == .interpretive && $0.hasStoredImage(bundleRoot: bundleRoot) }
    }

    public func primaryReferenceRecord(bundleRoot: URL? = nil) -> GenerationImageRecord? {
        referenceImageRecords(bundleRoot: bundleRoot).first
    }

    public func preparationSettings(
        compositeBack: Bool = true,
        pixelAlignment: Int = 32
    ) -> ImagePreparationSettings {
        let budget = megapixelBudget ?? 1.0
        var settings = (primaryReferenceRecord(bundleRoot: nil)?.formatting ?? ImageSlotFormatting())
            .preparationSettings(megapixelBudget: budget, pixelAlignment: pixelAlignment, compositeBack: compositeBack)
        settings.contextArea = ImagePreparation.clampUnitRect(contextArea.cgRect)
        settings.processArea = processArea?.cgRect
        settings.clampValues()
        return settings
    }

    public func loadReferenceCGImages(bundleRoot: URL? = nil) throws -> [CGImage] {
        try referenceImageRecords(bundleRoot: bundleRoot).map { try loadCGImage(from: $0, bundleRoot: bundleRoot) }
    }

    public func loadInterpretPaths(bundleRoot: URL? = nil) throws -> [String] {
        try interpretImageRecords(bundleRoot: bundleRoot).compactMap { record in
            if let path = record.sourcePath, FileManager.default.fileExists(atPath: path) {
                return path
            }
            if let bundleRoot,
               let bundlePath = record.bundlePath {
                let url = bundleRoot.appendingPathComponent(bundlePath, isDirectory: false)
                if FileManager.default.fileExists(atPath: url.path) {
                    return url.path
                }
            }
            guard let pngBase64 = record.pngBase64,
                  let data = Data(base64Encoded: pngBase64) else {
                return nil
            }
            let temp = FileManager.default.temporaryDirectory
                .appendingPathComponent("flux2-interpret-\(record.id.uuidString).png")
            try data.write(to: temp, options: .atomic)
            return temp.path
        }
    }

    private func loadCGImage(from record: GenerationImageRecord, bundleRoot: URL?) throws -> CGImage {
        if let bundleRoot,
           let bundlePath = record.bundlePath {
            let url = bundleRoot.appendingPathComponent(bundlePath, isDirectory: false)
            if FileManager.default.fileExists(atPath: url.path) {
                return try ProjectBundleImageWriter.loadCGImage(from: url)
            }
        }
        if let path = record.sourcePath,
           FileManager.default.fileExists(atPath: path),
           let image = Self.loadCGImage(from: path) {
            return image
        }
        guard let pngBase64 = record.pngBase64,
              let data = Data(base64Encoded: pngBase64),
              let image = Self.cgImage(from: data) else {
            throw Flux2Error.invalidConfiguration("Project contains an invalid image record")
        }
        return image
    }

    private static func loadCGImage(from path: String) -> CGImage? {
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private static func cgImage(from data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
