import CoreGraphics
import Foundation

public enum InpaintMaskTool: String, CaseIterable, Codable, Sendable, Identifiable {
    case rectangle
    case polygon
    case visionSubject

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .rectangle: "Rectangle"
        case .polygon: "Polygon"
        case .visionSubject: "Subject"
        }
    }
}

/// How a new primitive combines with the mask built so far.
public enum InpaintMaskCombineMode: String, CaseIterable, Codable, Sendable, Identifiable {
    /// Union — extend the inpaint region.
    case add
    /// Intersection — clip the inpaint region.
    case clip

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .add: "Add"
        case .clip: "Clip"
        }
    }
}

public enum InpaintMaskPrimitive: Codable, Sendable, Equatable {
    case rectangle(FluxGenerationProject.NormalizedRect)
    case polygon([FluxGenerationProject.NormalizedPoint])
    case visionSubject
}

public struct InpaintMaskLayer: Codable, Sendable, Equatable, Identifiable {
    public var id: UUID
    public var combineMode: InpaintMaskCombineMode
    public var primitive: InpaintMaskPrimitive

    public init(
        id: UUID = UUID(),
        combineMode: InpaintMaskCombineMode,
        primitive: InpaintMaskPrimitive
    ) {
        self.id = id
        self.combineMode = combineMode
        self.primitive = primitive
    }
}

public struct InpaintMaskDefinition: Codable, Sendable, Equatable {
    public var layers: [InpaintMaskLayer]

    public init(layers: [InpaintMaskLayer] = []) {
        self.layers = layers
    }

    public var isEmpty: Bool { layers.isEmpty }
}

extension FluxGenerationProject {
    public struct NormalizedPoint: Codable, Sendable, Equatable {
        public var x: Double
        public var y: Double

        public var cgPoint: CGPoint {
            CGPoint(x: x, y: y)
        }

        public init(_ point: CGPoint) {
            x = point.x
            y = point.y
        }

        public init(x: Double, y: Double) {
            self.x = x
            self.y = y
        }
    }
}
