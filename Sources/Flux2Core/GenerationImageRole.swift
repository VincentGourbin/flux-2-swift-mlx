import Foundation

/// Role assigned to an image slot in the Images palette (project v2).
public enum GenerationImageRole: String, Codable, Sendable, CaseIterable, Identifiable {
    case unassigned
    case reference
    case interpretive

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .unassigned: return "Unassigned"
        case .reference: return "Reference"
        case .interpretive: return "Interpret (VLM)"
        }
    }

    /// Short label shown on image tabs.
    public var tabBadge: String? {
        switch self {
        case .unassigned: return nil
        case .reference: return "Ref"
        case .interpretive: return "VLM"
        }
    }
}

/// Per-slot Image Formatting settings (project v2).
public struct ImageSlotFormatting: Codable, Sendable, Equatable {
    public var sizingFavor: String
    public var sizingMethod: String
    public var preparationScale: Double?

    public init(
        sizingFavor: String = ImageSizingFavor.original.rawValue,
        sizingMethod: String = ImageSizingMethod.crop.rawValue,
        preparationScale: Double? = 1.0
    ) {
        self.sizingFavor = sizingFavor
        self.sizingMethod = sizingMethod
        self.preparationScale = preparationScale
    }

    public func preparationSettings(
        megapixelBudget: Double,
        pixelAlignment: Int = ImagePreparation.generationSizeMultiple,
        compositeBack: Bool = true
    ) -> ImagePreparationSettings {
        var settings = ImagePreparationSettings()
        settings.sizingFavor = ImageSizingFavor(rawValue: sizingFavor) ?? .original
        settings.sizingMethod = ImageSizingMethod(rawValue: sizingMethod) ?? .crop
        settings.preparationScale = max(0.1, min(1.0, preparationScale ?? 1.0))
        settings.megapixelBudget = megapixelBudget
        settings.pixelAlignment = pixelAlignment
        settings.compositeBack = compositeBack
        settings.clampValues()
        return settings
    }
}

/// One image tab persisted in a generation project (v2+ flat JSON, v3 bundle paths).
public struct GenerationImageRecord: Codable, Sendable, Identifiable, Equatable {
    public var id: UUID
    public var role: GenerationImageRole
    public var isPrimary: Bool
    public var sourcePath: String?
    public var pngBase64: String?
    /// Relative path inside a `.flux2project` bundle (v3+), e.g. `slots/<uuid>.jxl`.
    public var bundlePath: String?
    public var formatting: ImageSlotFormatting
    /// Optional user-defined tab label (project v2).
    public var tabLabel: String?

    public init(
        id: UUID = UUID(),
        role: GenerationImageRole = .unassigned,
        isPrimary: Bool = false,
        sourcePath: String? = nil,
        pngBase64: String? = nil,
        bundlePath: String? = nil,
        formatting: ImageSlotFormatting = ImageSlotFormatting(),
        tabLabel: String? = nil
    ) {
        self.id = id
        self.role = role
        self.isPrimary = isPrimary
        self.sourcePath = sourcePath
        self.pngBase64 = pngBase64
        self.bundlePath = bundlePath
        self.formatting = formatting
        self.tabLabel = tabLabel
    }

    public func hasStoredImage(bundleRoot: URL? = nil) -> Bool {
        if let bundlePath,
           let bundleRoot,
           FileManager.default.fileExists(atPath: bundleRoot.appendingPathComponent(bundlePath).path) {
            return true
        }
        if let path = sourcePath, FileManager.default.fileExists(atPath: path) {
            return true
        }
        guard let pngBase64, !pngBase64.isEmpty else { return false }
        return Data(base64Encoded: pngBase64) != nil
    }
}

public enum FluxGenerationProjectError: Error, LocalizedError, Sendable {
    case unsupportedVersion(Int)
    case unsupportedBundleVersion(Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let version):
            return "Generation project version \(version) is no longer supported (requires version \(FluxGenerationProject.minimumLoadableVersion)+)"
        case .unsupportedBundleVersion(let version):
            return "FLUX.2 project bundle version \(version) is not supported (requires version \(FluxGenerationProject.bundleVersion)+)"
        }
    }
}
