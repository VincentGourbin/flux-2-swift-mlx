import Foundation

/// Image-to-image edit workflow in Flux2App.
public enum ImageEditMode: String, CaseIterable, Codable, Sendable, Identifiable {
    /// Barn-door live area + Image Preparation conditioning and optional composite-back.
    case promptEdit = "promptEdit"
    /// RePaint-style local repair via ``Flux2MaskedInpaintingChain`` (Flux2Chains):
    /// draw a rectangle over a blemish or bad patch; optional Qwen3.5 VLM rewrites the prompt.
    case generativeFill = "generativeFill"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .promptEdit: "Prompt edit"
        case .generativeFill: "Generative fill"
        }
    }

    /// Decode persisted project values, including the experimental `maskedInpaint` raw value.
    public static func fromProjectValue(_ raw: String?) -> ImageEditMode {
        guard let raw else { return .promptEdit }
        if raw == "maskedInpaint" { return .generativeFill }
        return ImageEditMode(rawValue: raw) ?? .promptEdit
    }
}
