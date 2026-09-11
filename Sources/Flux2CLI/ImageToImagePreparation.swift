// ImageToImagePreparation.swift - CLI flag parsing for `flux2 i2i --prepared`
// Copyright 2025 Vincent Gourbin
//
// Originally contributed by Bill Evans (@realnotsteve) in PR #99.

import ArgumentParser
import CoreGraphics
import Flux2Core
import Foundation

enum ImageToImagePreparationSupport {
    static func parseNormalizedRect(_ value: String, label: String) throws -> CGRect {
        let parts = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        // `Double("nan")`/`Double("inf")` parse successfully in Swift, so the
        // count/Double(...) guard alone lets a typo like `--live-area
        // nan,0.1,0.8,0.8` through silently — clampUnitRect degrades it to
        // the full frame with no error. Reject explicitly instead.
        guard parts.count == 4,
              let x = Double(parts[0]), x.isFinite,
              let y = Double(parts[1]), y.isFinite,
              let w = Double(parts[2]), w.isFinite,
              let h = Double(parts[3]), h.isFinite else {
            throw ValidationError("\(label) must be four finite numbers x,y,width,height (e.g. 0.1,0.1,0.8,0.8)")
        }
        return ImagePreparation.clampUnitRect(CGRect(x: x, y: y, width: w, height: h))
    }

    private static func parseFiniteFraction(_ value: Double, flag: String) throws -> Double {
        guard value.isFinite else {
            throw ValidationError("\(flag) must be a finite number")
        }
        return value
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
        favour: String?,
        method: String?,
        scale: Double?,
        megapixels: Double?,
        liveArea: String?,
        processArea: String?,
        noComposite: Bool
    ) -> Bool {
        prepared
            || favour != nil
            || method != nil
            || scale != nil
            || megapixels != nil
            || liveArea != nil
            || processArea != nil
            || noComposite
    }

    static func buildSettings(
        favour: String?,
        method: String?,
        scale: Double?,
        megapixels: Double?,
        liveArea: String?,
        processArea: String?,
        noComposite: Bool
    ) throws -> ImagePreparationSettings {
        var settings = ImagePreparationSettings()
        if let favour {
            settings.sizingFavor = try parseFavor(favour)
        }
        if let method {
            settings.sizingMethod = try parseMethod(method)
        }
        if let scale {
            settings.preparationScale = try parseFiniteFraction(scale, flag: "--prep-scale")
        }
        if let megapixels {
            settings.megapixelBudget = try parseFiniteFraction(megapixels, flag: "--megapixels")
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
