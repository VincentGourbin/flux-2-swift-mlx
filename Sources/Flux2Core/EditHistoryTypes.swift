import Foundation

public enum EditHistoryKind: String, Codable, Sendable, CaseIterable {
    case generate
    case adopt

    public var displayName: String {
        switch self {
        case .generate: "Generate"
        case .adopt: "Adopt"
        }
    }
}

public struct EditHistorySpatial: Codable, Sendable, Equatable {
    public var contextArea: FluxGenerationProject.NormalizedRect
    public var processArea: FluxGenerationProject.NormalizedRect?
    public var inpaintMaskLayers: [InpaintMaskLayer]?
    public var fillContextMaskScale: Double?
    public var outpaintPadding: OutpaintPadding?
    public var inpaintIntent: String?
    public var enrichInpaintPromptWithVLM: Bool?

    public init(
        contextArea: FluxGenerationProject.NormalizedRect,
        processArea: FluxGenerationProject.NormalizedRect? = nil,
        inpaintMaskLayers: [InpaintMaskLayer]? = nil,
        fillContextMaskScale: Double? = nil,
        outpaintPadding: OutpaintPadding? = nil,
        inpaintIntent: String? = nil,
        enrichInpaintPromptWithVLM: Bool? = nil
    ) {
        self.contextArea = contextArea
        self.processArea = processArea
        self.inpaintMaskLayers = inpaintMaskLayers
        self.fillContextMaskScale = fillContextMaskScale
        self.outpaintPadding = outpaintPadding
        self.inpaintIntent = inpaintIntent
        self.enrichInpaintPromptWithVLM = enrichInpaintPromptWithVLM
    }
}

public struct EditHistorySettings: Codable, Sendable, Equatable {
    public var selectedModel: String
    public var steps: Int
    public var guidance: Float
    public var megapixelBudget: Double?
    public var upsamplePrompt: Bool
    public var generateRoute: String?

    public init(
        selectedModel: String,
        steps: Int,
        guidance: Float,
        megapixelBudget: Double? = nil,
        upsamplePrompt: Bool = false,
        generateRoute: String? = nil
    ) {
        self.selectedModel = selectedModel
        self.steps = steps
        self.guidance = guidance
        self.megapixelBudget = megapixelBudget
        self.upsamplePrompt = upsamplePrompt
        self.generateRoute = generateRoute
    }
}

public struct EditHistoryEntry: Codable, Sendable, Identifiable, Equatable {
    public var id: UUID
    public var label: String
    public var master: String
    public var thumb: String
    public var kind: EditHistoryKind
    public var prompt: String
    public var settings: EditHistorySettings
    public var spatial: EditHistorySpatial

    public init(
        id: UUID = UUID(),
        label: String,
        master: String,
        thumb: String,
        kind: EditHistoryKind,
        prompt: String,
        settings: EditHistorySettings,
        spatial: EditHistorySpatial
    ) {
        self.id = id
        self.label = label
        self.master = master
        self.thumb = thumb
        self.kind = kind
        self.prompt = prompt
        self.settings = settings
        self.spatial = spatial
    }
}

public enum EditHistoryPaths {
    public static let historyDirectory = "history"
    public static let thumbsDirectory = "thumbs"

    public static func masterPath(sequence: Int) -> String {
        "\(historyDirectory)/\(fileStem(sequence: sequence)).\(ProjectBundleImageWriter.fileExtension)"
    }

    public static func thumbPath(sequence: Int) -> String {
        "\(thumbsDirectory)/\(fileStem(sequence: sequence)).\(ProjectBundleImageWriter.fileExtension)"
    }

    private static func fileStem(sequence: Int) -> String {
        String(format: "%04d", sequence)
    }
}
