import CoreGraphics
import Foundation
import ImageIO

/// Flux2App generation project JSON (version 1).
public struct FluxGenerationProject: Codable, Sendable {
    public var version: Int = 1
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
    public var sizingFavor: String
    public var sizingMethod: String
    public var preparationScale: Double?
    public var preparationOverlayOpacity: Double?
    public var megapixelBudget: Double?
    public var clearPromptAfterGeneration: Bool?
    public var selectedFamily: String?
    public var processArea: NormalizedRect?
    public var contextArea: NormalizedRect
    public var editMode: String?
    public var inpaintIntent: String?
    public var enrichInpaintPromptWithVLM: Bool?
    public var interpretImagePaths: [String]
    public var referenceImages: [ReferenceImage]

    public struct NormalizedRect: Codable, Sendable {
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

    public struct ReferenceImage: Codable, Sendable {
        public var sourcePath: String?
        public var pngBase64: String

        public init(sourcePath: String?, pngBase64: String) {
            self.sourcePath = sourcePath
            self.pngBase64 = pngBase64
        }
    }

    public init(
        version: Int = 1,
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
        sizingFavor: String,
        sizingMethod: String,
        preparationScale: Double? = nil,
        preparationOverlayOpacity: Double? = nil,
        megapixelBudget: Double? = nil,
        clearPromptAfterGeneration: Bool? = nil,
        selectedFamily: String? = nil,
        processArea: NormalizedRect? = nil,
        contextArea: NormalizedRect,
        editMode: String? = nil,
        inpaintIntent: String? = nil,
        enrichInpaintPromptWithVLM: Bool? = nil,
        interpretImagePaths: [String],
        referenceImages: [ReferenceImage]
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
        self.sizingFavor = sizingFavor
        self.sizingMethod = sizingMethod
        self.preparationScale = preparationScale
        self.preparationOverlayOpacity = preparationOverlayOpacity
        self.megapixelBudget = megapixelBudget
        self.clearPromptAfterGeneration = clearPromptAfterGeneration
        self.selectedFamily = selectedFamily
        self.processArea = processArea
        self.contextArea = contextArea
        self.editMode = editMode
        self.inpaintIntent = inpaintIntent
        self.enrichInpaintPromptWithVLM = enrichInpaintPromptWithVLM
        self.interpretImagePaths = interpretImagePaths
        self.referenceImages = referenceImages
    }

    public static func load(from path: String) throws -> FluxGenerationProject {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(FluxGenerationProject.self, from: data)
    }

    public func preparationSettings(compositeBack: Bool = true) -> ImagePreparationSettings {
        var settings = ImagePreparationSettings()
        settings.sizingFavor = ImageSizingFavor(rawValue: sizingFavor) ?? .original
        settings.sizingMethod = ImageSizingMethod(rawValue: sizingMethod) ?? .crop
        settings.preparationScale = max(0.1, min(1.0, preparationScale ?? 1.0))
        settings.megapixelBudget = megapixelBudget ?? 1.0
        settings.contextArea = ImagePreparation.clampUnitRect(contextArea.cgRect)
        settings.processArea = processArea?.cgRect
        settings.compositeBack = compositeBack
        settings.clampValues()
        return settings
    }

    public func loadReferenceCGImages() throws -> [CGImage] {
        try referenceImages.map { stored in
            if let path = stored.sourcePath,
               FileManager.default.fileExists(atPath: path),
               let image = Self.loadCGImage(from: path) {
                return image
            }
            guard let data = Data(base64Encoded: stored.pngBase64),
                  let image = Self.cgImage(from: data) else {
                throw Flux2Error.invalidConfiguration("Project contains an invalid reference image")
            }
            return image
        }
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
