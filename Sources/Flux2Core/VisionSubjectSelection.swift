import CoreGraphics
import Foundation

/// User-drawn region that disambiguates which Vision foreground instance to keep.
public enum VisionSubjectSelection: Codable, Sendable, Equatable {
    case rectangle(FluxGenerationProject.NormalizedRect)
    case polygon([FluxGenerationProject.NormalizedPoint])

    public var boundingRect: CGRect {
        switch self {
        case .rectangle(let rect):
            return rect.cgRect
        case .polygon(let points):
            guard let first = points.first else {
                return .zero
            }
            var minX = first.x
            var maxX = first.x
            var minY = first.y
            var maxY = first.y
            for point in points {
                minX = min(minX, point.x)
                maxX = max(maxX, point.x)
                minY = min(minY, point.y)
                maxY = max(maxY, point.y)
            }
            return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        }
    }

    public var visionRegion: VisionSubjectRegion {
        switch self {
        case .rectangle(let rect):
            return .rectangle(rect.cgRect)
        case .polygon(let points):
            return .polygon(points.map(\.cgPoint))
        }
    }
}

/// Normalized selection geometry passed to Vision instance picking.
public enum VisionSubjectRegion: Sendable, Equatable {
    case rectangle(CGRect)
    case polygon([CGPoint])
}

public struct VisionSubjectPickOptions: Sendable, Equatable {
    public var expansionStepFraction: Double
    public var maxExpansionSteps: Int
    /// Minimum fraction of selection pixels that must overlap a subject instance.
    public var minimumOverlapFraction: Double

    public init(
        expansionStepFraction: Double = 0.05,
        maxExpansionSteps: Int = 8,
        minimumOverlapFraction: Double = 0.01
    ) {
        self.expansionStepFraction = expansionStepFraction
        self.maxExpansionSteps = maxExpansionSteps
        self.minimumOverlapFraction = minimumOverlapFraction
    }

    public static let `default` = VisionSubjectPickOptions()
}
