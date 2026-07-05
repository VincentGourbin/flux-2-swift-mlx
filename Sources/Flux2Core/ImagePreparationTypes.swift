import CoreGraphics
import Foundation

public enum ImageSizingFavor: String, CaseIterable, Codable, Sendable, Identifiable {
    case original = "Original"
    case horizontal = "Horizontal"
    case vertical = "Vertical"

    public var id: String { rawValue }
}

public enum ImageSizingMethod: String, CaseIterable, Codable, Sendable, Identifiable {
    case crop = "Crop"
    case pad = "Pad"

    public var id: String { rawValue }
}

/// Normalized (0…1) barn-door / live region and formatting controls for I2I.
public struct ImagePreparationSettings: Sendable {
    public var sizingFavor: ImageSizingFavor = .original
    public var sizingMethod: ImageSizingMethod = .crop
    public var preparationScale: Double = 1.0
    public var megapixelBudget: Double = 1.0
    public var contextArea: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    public var processArea: CGRect?
    public var pixelAlignment: Int = ImagePreparation.generationSizeMultiple
    /// When true, paste the generated patch back into the full-resolution original.
    public var compositeBack: Bool = true

    public init() {}

    public static let minMegapixelBudget = 0.25
    public static let maxMegapixelBudget = 4.0

    public mutating func clampValues() {
        preparationScale = min(max(preparationScale, 0.1), 1.0)
        megapixelBudget = min(max(megapixelBudget, Self.minMegapixelBudget), Self.maxMegapixelBudget)
        contextArea = ImagePreparation.clampUnitRect(contextArea)
        if let processArea {
            self.processArea = ImagePreparation.clampUnitRect(processArea)
        }
    }
}

public struct ImagePreparationTransform: Sendable {
    public let targetWidth: Int
    public let targetHeight: Int
    public let scale: CGFloat
    public let offsetX: CGFloat
    public let offsetY: CGFloat

    public func canvasRect(forSourceRect sourceRect: CGRect) -> CGRect {
        CGRect(
            x: sourceRect.minX * scale + offsetX,
            y: sourceRect.minY * scale + offsetY,
            width: sourceRect.width * scale,
            height: sourceRect.height * scale
        )
    }

    public func sourceRect(forCanvasRect canvasRect: CGRect) -> CGRect {
        CGRect(
            x: (canvasRect.minX - offsetX) / scale,
            y: (canvasRect.minY - offsetY) / scale,
            width: canvasRect.width / scale,
            height: canvasRect.height / scale
        )
    }
}

public struct ImageCompositionPlan: Sendable {
    public let originalImage: CGImage
    public let contextRect: CGRect
    public let processRect: CGRect
    public let transform: ImagePreparationTransform
}

public struct PreparedImageToImageInput: Sendable {
    public let images: [CGImage]
    public let width: Int
    public let height: Int
    public let compositionPlan: ImageCompositionPlan?

    public init(images: [CGImage], width: Int, height: Int, compositionPlan: ImageCompositionPlan?) {
        self.images = images
        self.width = width
        self.height = height
        self.compositionPlan = compositionPlan
    }
}
