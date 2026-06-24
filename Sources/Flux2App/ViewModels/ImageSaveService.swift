/**
 * ImageSaveService.swift
 * BE-style generated image saving for Flux2App
 */

import AppKit
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum ImageSaveOutputMode: String, CaseIterable, Identifiable {
    case `default` = "Default"
    case preset = "Preset"

    var id: String { rawValue }
}

enum ImageSavePreset: String, CaseIterable, Identifiable {
    case peeps
    case comparisons
    case tests

    var id: String { rawValue }

    var relativeDirectory: String {
        switch self {
        case .peeps: "images/peeps"
        case .comparisons: "images/comparisons"
        case .tests: "images/tests"
        }
    }
}

enum ImageSaveFormat: String, CaseIterable, Identifiable {
    case png24 = "PNG 24 (No Alpha)"
    case jpeg = "JPEG (Max Quality Lossy)"
    case jpegXL = "JPEG XL (Max Quality Lossy)"
    case heic = "HEIC (Max Quality Lossy)"
    case webP = "WebP (Max Quality Lossy)"

    var id: String { rawValue }

    var fileExtension: String {
        switch self {
        case .png24: "png"
        case .jpeg: "jpg"
        case .jpegXL: "jxl"
        case .heic: "heic"
        case .webP: "webp"
        }
    }

    var typeIdentifier: String {
        switch self {
        case .png24: "public.png"
        case .jpeg: "public.jpeg"
        case .jpegXL: "public.jpeg-xl"
        case .heic: "public.heic"
        case .webP: "org.webmproject.webp"
        }
    }

    var contentType: UTType {
        UTType(typeIdentifier) ?? UTType(filenameExtension: fileExtension) ?? .data
    }

    var isSupported: Bool {
        Self.supportedTypeIdentifiers.contains(typeIdentifier)
    }

    static var supportedCases: [ImageSaveFormat] {
        allCases.filter(\.isSupported)
    }

    private static var supportedTypeIdentifiers: Set<String> {
        let identifiers = CGImageDestinationCopyTypeIdentifiers() as? [String] ?? []
        return Set(identifiers)
    }
}

enum ImageSaveTimestampFormat: String, CaseIterable, Identifiable {
    case compactDate = "YYYYMMDD"
    case dashedDate = "YYYY-MM-DD"
    case compactDateTime = "YYYYMMDD-HHMMSS"
    case compactTime = "HHMMSS"
    case dashedTime = "HH-MM-SS"

    var id: String { rawValue }

    var dateFormat: String {
        switch self {
        case .compactDate: "yyyyMMdd"
        case .dashedDate: "yyyy-MM-dd"
        case .compactDateTime: "yyyyMMdd-HHmmss"
        case .compactTime: "HHmmss"
        case .dashedTime: "HH-mm-ss"
        }
    }
}

enum ImageSaveInputBase: String, CaseIterable, Identifiable {
    case staticPrefix = "Static"
    case prompt = "Prompt"

    var id: String { rawValue }
}

struct ImageSaveMetadata {
    let prompt: String
}

enum ImageSaveService {
    private static let generalOutputDirectory = "images/general"
    private static let illegalSegmentCharacters = CharacterSet(charactersIn: "\\/:*?\"<>|")
        .union(.controlCharacters)

    static var defaultOutputRoot: String {
        (FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Pictures"))
            .appendingPathComponent("Flux2Swift")
            .path
    }

    /// Configured save folder from Settings (output root + mode/preset subpath).
    @MainActor
    static func outputDirectory() throws -> URL {
        let defaults = UserDefaults.standard
        let rootPath = nonEmpty(defaults.string(forKey: "imageSaveOutputRoot"), fallback: defaultOutputRoot)
        let outputMode = ImageSaveOutputMode(rawValue: nonEmpty(defaults.string(forKey: "imageSaveOutputMode"), fallback: ImageSaveOutputMode.default.rawValue)) ?? .default
        let preset = ImageSavePreset(rawValue: nonEmpty(defaults.string(forKey: "imageSavePreset"), fallback: ImageSavePreset.peeps.rawValue)) ?? .peeps

        let directory = URL(fileURLWithPath: rootPath, isDirectory: true)
            .appendingPathComponent(relativeDirectory(outputMode: outputMode, preset: preset), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    @MainActor
    static func save(_ image: CGImage, metadata: ImageSaveMetadata, stemSuffix: String? = nil) throws -> URL {
        let defaults = UserDefaults.standard
        let format = savedFormat(from: defaults)
        let directory = try outputDirectory()

        // Lanczos upscale is driven by the factor on the Generated Image row:
        // 1 (or unset) means no upscale, > 1 scales up on save.
        let imageToSave: CGImage
        let upscaleBy = defaults.double(forKey: "imageSaveUpscaleBy")
        if upscaleBy > 1.0 {
            imageToSave = try upscale(image, scale: upscaleBy)
        } else {
            imageToSave = image
        }

        let filename = try resolveUniqueFilename(
            in: directory,
            format: format,
            metadata: metadata,
            stemSuffix: stemSuffix
        )
        let url = directory.appendingPathComponent(filename)
        try write(imageToSave, to: url, format: format)
        return url
    }

    /// Write `image` next to `outputURL`, sharing its folder, base name, and
    /// extension, with `suffix` appended to the stem (e.g. "name-input.png").
    @MainActor
    static func saveCompanion(_ image: CGImage, alongside outputURL: URL, suffix: String) throws -> URL {
        let directory = outputURL.deletingLastPathComponent()
        let stem = outputURL.deletingPathExtension().lastPathComponent
        let ext = outputURL.pathExtension
        let format = format(forExtension: ext)
        let resolvedExtension = ext.isEmpty ? format.fileExtension : ext

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(stem)\(suffix).\(resolvedExtension)")
        try write(image, to: url, format: format)
        return url
    }

    private static func format(forExtension ext: String) -> ImageSaveFormat {
        switch ext.lowercased() {
        case "jpg", "jpeg": return .jpeg
        case "jxl": return .jpegXL
        case "heic": return .heic
        case "webp": return .webP
        default: return .png24
        }
    }

    static func previewFilename(metadata: ImageSaveMetadata) -> String {
        let defaults = UserDefaults.standard
        let format = savedFormat(from: defaults)
        let stem = composeStem(metadata: metadata, incrementIndex: 0)
        return "\(stem).\(format.fileExtension)"
    }

    private static func savedFormat(from defaults: UserDefaults) -> ImageSaveFormat {
        let format = ImageSaveFormat(rawValue: nonEmpty(defaults.string(forKey: "imageSaveFormat"), fallback: ImageSaveFormat.png24.rawValue)) ?? .png24
        return format.isSupported ? format : .png24
    }

    private static func relativeDirectory(outputMode: ImageSaveOutputMode, preset: ImageSavePreset) -> String {
        switch outputMode {
        case .default: generalOutputDirectory
        case .preset: preset.relativeDirectory
        }
    }

    private static func resolveUniqueFilename(
        in directory: URL,
        format: ImageSaveFormat,
        metadata: ImageSaveMetadata,
        stemSuffix: String? = nil
    ) throws -> String {
        let defaults = UserDefaults.standard
        let usesIncrement = defaults.bool(forKey: "imageSaveUseAutoIncrement")
        let fileExtension = format.fileExtension
        let normalizedSuffix = stemSuffix.flatMap { suffix in
            let trimmed = suffix.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }

        if usesIncrement {
            var index = 0
            while true {
                let stem = composeStem(metadata: metadata, incrementIndex: index, stemSuffix: normalizedSuffix)
                let filename = "\(stem).\(fileExtension)"
                if !FileManager.default.fileExists(atPath: directory.appendingPathComponent(filename).path) {
                    return filename
                }
                index += 1
            }
        }

        let stem = composeStem(metadata: metadata, incrementIndex: 0, stemSuffix: normalizedSuffix)
        var filename = "\(stem).\(fileExtension)"
        var suffix = 2
        while FileManager.default.fileExists(atPath: directory.appendingPathComponent(filename).path) {
            filename = "\(stem)--\(suffix).\(fileExtension)"
            suffix += 1
        }
        return filename
    }

    private static func composeStem(metadata: ImageSaveMetadata, incrementIndex: Int, stemSuffix: String? = nil) -> String {
        let defaults = UserDefaults.standard
        let baseMode = ImageSaveInputBase(rawValue: nonEmpty(defaults.string(forKey: "imageSaveInputBase"), fallback: ImageSaveInputBase.staticPrefix.rawValue)) ?? .staticPrefix
        let baseValue: String
        switch baseMode {
        case .staticPrefix:
            baseValue = nonEmpty(defaults.string(forKey: "imageSaveFilenamePrefix"), fallback: "image")
        case .prompt:
            baseValue = metadata.prompt
        }

        var segments = [sanitizeSegment(baseValue)]

        let freeText = sanitizeSegment(defaults.string(forKey: "imageSaveFreeText") ?? "")
        if !freeText.isEmpty {
            segments.append(freeText)
        }

        if defaults.bool(forKey: "imageSaveUseTimestamp") {
            let timestampFormat = ImageSaveTimestampFormat(rawValue: nonEmpty(defaults.string(forKey: "imageSaveTimestampFormat"), fallback: ImageSaveTimestampFormat.compactDate.rawValue)) ?? .compactDate
            let formatter = DateFormatter()
            formatter.dateFormat = timestampFormat.dateFormat
            segments.append(sanitizeSegment(formatter.string(from: Date())))
        }

        if defaults.bool(forKey: "imageSaveUseAutoIncrement") {
            let digits = max(1, defaults.integer(forKey: "imageSaveAutoIncrementDigits"))
            let start = defaults.object(forKey: "imageSaveAutoIncrementStart") == nil ? 1 : defaults.integer(forKey: "imageSaveAutoIncrementStart")
            let step = max(1, defaults.integer(forKey: "imageSaveAutoIncrementStep"))
            let value = start + max(0, incrementIndex) * step
            segments.append(String(format: "%0\(digits)d", value))
        }

        let stem = segments.filter { !$0.isEmpty }.joined(separator: "-")
        let base = stem.isEmpty ? "image" : stem
        guard let stemSuffix, !stemSuffix.isEmpty else { return base }
        return "\(base)\(stemSuffix)"
    }

    private static func write(_ image: CGImage, to url: URL, format: ImageSaveFormat) throws {
        let imageToWrite = try flattenAlphaToRGB(image)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, format.typeIdentifier as CFString, 1, nil) else {
            throw saveError("Could not create \(format.rawValue) destination. This macOS install may not support that format.")
        }

        let properties: CFDictionary
        switch format {
        case .png24:
            properties = [kCGImagePropertyPNGDictionary: [kCGImagePropertyPNGCompressionFilter: 4]] as CFDictionary
        case .jpeg:
            properties = [kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary
        case .jpegXL, .heic, .webP:
            properties = [kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary
        }

        CGImageDestinationAddImage(destination, imageToWrite, properties)
        if !CGImageDestinationFinalize(destination) {
            throw saveError("Could not save as \(format.rawValue).")
        }
    }

    private static func upscale(_ image: CGImage, scale: Double) throws -> CGImage {
        guard scale > 1 else { return image }
        let input = CIImage(cgImage: image)
        guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
            throw saveError("Lanczos upscale is not available.")
        }
        filter.setValue(input, forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(1.0, forKey: kCIInputAspectRatioKey)

        guard let output = filter.outputImage,
              let scaled = CIContext().createCGImage(output, from: output.extent.integral) else {
            throw saveError("Lanczos upscale failed.")
        }
        return scaled
    }

    private static func flattenAlphaToRGB(_ image: CGImage) throws -> CGImage {
        guard let context = CGContext(
            data: nil,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: image.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw saveError("Could not create RGB image context.")
        }

        context.setFillColor(NSColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))

        guard let flattened = context.makeImage() else {
            throw saveError("Could not flatten image alpha.")
        }
        return flattened
    }

    private static func sanitizeSegment(_ value: String) -> String {
        let scalars = value
            .replacingOccurrences(of: " ", with: "_")
            .unicodeScalars
            .filter { !illegalSegmentCharacters.contains($0) }
        return String(String.UnicodeScalarView(scalars)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func nonEmpty(_ value: String?, fallback: String) -> String {
        let trimmed = (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? fallback : trimmed
    }

    private static func saveError(_ message: String) -> NSError {
        NSError(domain: "ImageSave", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
