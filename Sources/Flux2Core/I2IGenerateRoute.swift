import Foundation

/// Which generate path Image to Image takes. Inferred from the active tool
/// and whether a selection exists — not chosen by the user.
public enum I2IGenerateRoute: String, Codable, Sendable {
    /// Barn-door live area + full-frame I2I (no local selection).
    case fullImage
    /// RePaint-style edit inside a selection mask.
    case localFill
    /// Canvas expansion via ``Flux2OutpaintingChain``.
    case outpaint

    /// Decode legacy `editMode` project / session values.
    public static func fromProjectValue(_ raw: String?) -> I2IGenerateRoute {
        fromLegacyProjectValue(raw) ?? .fullImage
    }

    /// Decode legacy `editMode` project values.
    public static func fromLegacyProjectValue(_ raw: String?) -> I2IGenerateRoute? {
        guard let raw else { return nil }
        switch raw {
        case "promptEdit", "fullImage":
            return .fullImage
        case "generativeFill", "maskedInpaint", "localFill":
            return .localFill
        case "outpaint":
            return .outpaint
        default:
            return I2IGenerateRoute(rawValue: raw)
        }
    }
}

@available(*, deprecated, renamed: "I2IGenerateRoute")
public typealias ImageEditMode = I2IGenerateRoute
