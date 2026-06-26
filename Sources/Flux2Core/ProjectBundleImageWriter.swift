import CoreGraphics
import Foundation
import ImageIO

/// JPEG XL encoder for `.flux2project` bundle assets (separate from export preferences).
///
/// Decode uses ImageIO. Encode prefers ImageIO when available; otherwise falls back to
/// Homebrew `cjxl` (`brew install jpeg-xl`).
public enum ProjectBundleImageWriter {
    public static let typeIdentifier = "public.jpeg-xl"
    public static let fileExtension = "jxl"

    public enum EncodeMode: Sendable {
        case lossless
        case lossyHighQuality
    }

    public static func isSupported() -> Bool {
        imageIODestinationSupportsJXL || CJXLImageEncoder.isAvailable()
    }

    public static func encode(_ image: CGImage, mode: EncodeMode = .lossless) throws -> Data {
        if imageIODestinationSupportsJXL {
            return try encodeWithImageIO(image, mode: mode)
        }
        if CJXLImageEncoder.isAvailable() {
            return try CJXLImageEncoder.encode(image, mode: mode)
        }
        throw unavailableError()
    }

    public static func write(_ image: CGImage, to url: URL, mode: EncodeMode = .lossless) throws {
        if imageIODestinationSupportsJXL {
            let data = try encodeWithImageIO(image, mode: mode)
            try data.write(to: url, options: .atomic)
            return
        }
        if CJXLImageEncoder.isAvailable() {
            try CJXLImageEncoder.write(image, to: url, mode: mode)
            return
        }
        throw unavailableError()
    }

    public static func loadCGImage(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw Flux2Error.imageProcessingFailed("Could not open bundle image at \(url.path)")
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Flux2Error.imageProcessingFailed("Could not decode bundle image at \(url.path)")
        }
        return image
    }

    public static func makeThumbnail(from image: CGImage, maxDimension: Int = 128) throws -> CGImage {
        let longer = max(image.width, image.height)
        guard longer > maxDimension else { return image }
        let scale = CGFloat(maxDimension) / CGFloat(longer)
        let width = max(1, Int((CGFloat(image.width) * scale).rounded()))
        let height = max(1, Int((CGFloat(image.height) * scale).rounded()))

        let colorSpace = image.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw Flux2Error.imageProcessingFailed("Could not allocate thumbnail context.")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let thumbnail = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Could not create thumbnail image.")
        }
        return thumbnail
    }

    private static var imageIODestinationSupportsJXL: Bool {
        let identifiers = CGImageDestinationCopyTypeIdentifiers() as? [String] ?? []
        return identifiers.contains(typeIdentifier)
    }

    private static func encodeWithImageIO(_ image: CGImage, mode: EncodeMode) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, typeIdentifier as CFString, 1, nil) else {
            throw Flux2Error.imageProcessingFailed("Could not create JPEG XL destination.")
        }

        CGImageDestinationAddImage(destination, image, destinationProperties(mode: mode))
        guard CGImageDestinationFinalize(destination) else {
            throw Flux2Error.imageProcessingFailed("Could not finalize JPEG XL image.")
        }
        return data as Data
    }

    private static func unavailableError() -> Flux2Error {
        .imageProcessingFailed(
            "JPEG XL encoding is not available. Install with: brew install jpeg-xl"
        )
    }

    private static func destinationProperties(mode: EncodeMode) -> CFDictionary {
        switch mode {
        case .lossless:
            return [
                "kCGImagePropertyJPEGXL_Lossless" as CFString: true,
                kCGImageDestinationLossyCompressionQuality: 1.0,
            ] as CFDictionary
        case .lossyHighQuality:
            return [kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary
        }
    }
}
