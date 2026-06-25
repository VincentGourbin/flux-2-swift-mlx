import CoreGraphics
import Foundation

public enum InpaintMaskTool: String, CaseIterable, Codable, Sendable, Identifiable {
    case pointer
    case liveArea
    case rectangle
    case polygon
    case visionSubject
    case cropCanvas

    public var id: String { rawValue }

    /// Tools shown in the toolbar (excludes the implicit pointer / neutral state).
    public static let toolbarCases: [InpaintMaskTool] = [
        .liveArea, .rectangle, .polygon, .visionSubject, .cropCanvas
    ]

    public var displayName: String {
        switch self {
        case .pointer: "Pointer"
        case .liveArea: "Live Area"
        case .rectangle: "Rectangle"
        case .polygon: "Polygon"
        case .visionSubject: "Subject"
        case .cropCanvas: "Crop"
        }
    }

    public var systemImage: String {
        switch self {
        case .pointer: "arrow.up.left"
        case .liveArea: "viewfinder"
        case .rectangle: "rectangle.dashed"
        case .polygon: "pentagon"
        case .visionSubject: "lasso"
        case .cropCanvas: "crop"
        }
    }

    public var isSelectionTool: Bool {
        switch self {
        case .rectangle, .polygon, .visionSubject: true
        case .pointer, .liveArea, .cropCanvas: false
        }
    }

    public var isBarnDoorTool: Bool {
        self == .liveArea
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
        case .clip: "Subtract"
        }
    }
}

/// Photoshop-style selection interaction mode (toolbar).
public enum SelectionMode: String, CaseIterable, Codable, Sendable, Identifiable {
    /// Replace the current selection with the next shape.
    case replace
    /// Union the next shape into the selection.
    case add
    /// Subtract the next shape from the selection.
    case subtract

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .replace: "New"
        case .add: "Add"
        case .subtract: "Subtract"
        }
    }

    public var systemImage: String {
        switch self {
        case .replace: "selection.pin.in.out"
        case .add: "plus.square.dashed"
        case .subtract: "minus.square.dashed"
        }
    }

    public var combineMode: InpaintMaskCombineMode {
        switch self {
        case .replace, .add: .add
        case .subtract: .clip
        }
    }
}

public enum InpaintMaskPrimitive: Codable, Sendable, Equatable {
    case rectangle(FluxGenerationProject.NormalizedRect)
    case polygon([FluxGenerationProject.NormalizedPoint])
    case visionSubject(VisionSubjectSelection)

    private enum CodingKeys: String, CodingKey {
        case rectangle
        case polygon
        case visionSubject
        case selection
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let rect = try container.decodeIfPresent(
            FluxGenerationProject.NormalizedRect.self,
            forKey: .rectangle
        ) {
            self = .rectangle(rect)
            return
        }
        if let points = try container.decodeIfPresent(
            [FluxGenerationProject.NormalizedPoint].self,
            forKey: .polygon
        ) {
            self = .polygon(points)
            return
        }
        if container.contains(.visionSubject) {
            if let selection = try container.decodeIfPresent(
                VisionSubjectSelection.self,
                forKey: .selection
            ) {
                self = .visionSubject(selection)
            } else {
                // Legacy projects stored a subject layer without geometry.
                self = .visionSubject(.rectangle(.init(CGRect(x: 0, y: 0, width: 1, height: 1))))
            }
            return
        }
        throw DecodingError.dataCorrupted(
            .init(codingPath: decoder.codingPath, debugDescription: "Unknown inpaint mask primitive")
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .rectangle(let rect):
            try container.encode(rect, forKey: .rectangle)
        case .polygon(let points):
            try container.encode(points, forKey: .polygon)
        case .visionSubject(let selection):
            try container.encode(true, forKey: .visionSubject)
            try container.encode(selection, forKey: .selection)
        }
    }
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
