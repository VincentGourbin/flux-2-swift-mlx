import ArgumentParser
import CoreGraphics
import Flux2Core
import Foundation

enum ImageToImagePreparationSupport {
    static func parseNormalizedRect(_ value: String, label: String) throws -> CGRect {
        let parts = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 4,
              let x = Double(parts[0]),
              let y = Double(parts[1]),
              let w = Double(parts[2]),
              let h = Double(parts[3]) else {
            throw ValidationError("\(label) must be x,y,width,height (e.g. 0.1,0.1,0.8,0.8)")
        }
        return ImagePreparation.clampUnitRect(CGRect(x: x, y: y, width: w, height: h))
    }

    static func parseFavor(_ value: String) throws -> ImageSizingFavor {
        switch value.lowercased() {
        case "original": return .original
        case "horizontal": return .horizontal
        case "vertical": return .vertical
        default:
            throw ValidationError("Invalid favour: \(value). Use original, horizontal, or vertical")
        }
    }

    static func parseMethod(_ value: String) throws -> ImageSizingMethod {
        switch value.lowercased() {
        case "crop": return .crop
        case "pad": return .pad
        default:
            throw ValidationError("Invalid method: \(value). Use crop or pad")
        }
    }

    static func usesPreparation(
        prepared: Bool,
        project: String?,
        favour: String?,
        method: String?,
        scale: Double?,
        megapixels: Double?,
        liveArea: String?,
        processArea: String?,
        noComposite: Bool
    ) -> Bool {
        prepared
            || project != nil
            || favour != nil
            || method != nil
            || scale != nil
            || megapixels != nil
            || liveArea != nil
            || processArea != nil
            || noComposite
    }

    static func buildSettings(
        project: FluxGenerationProject?,
        favour: String?,
        method: String?,
        scale: Double?,
        megapixels: Double?,
        liveArea: String?,
        processArea: String?,
        noComposite: Bool
    ) throws -> ImagePreparationSettings {
        var settings = project?.preparationSettings(compositeBack: !noComposite) ?? ImagePreparationSettings()
        if let favour {
            settings.sizingFavor = try parseFavor(favour)
        }
        if let method {
            settings.sizingMethod = try parseMethod(method)
        }
        if let scale {
            settings.preparationScale = scale
        }
        if let megapixels {
            settings.megapixelBudget = megapixels
        }
        if let liveArea {
            settings.contextArea = try parseNormalizedRect(liveArea, label: "--live-area")
        }
        if let processArea {
            settings.processArea = try parseNormalizedRect(processArea, label: "--process-area")
        }
        if noComposite {
            settings.compositeBack = false
        }
        settings.clampValues()
        return settings
    }
}
