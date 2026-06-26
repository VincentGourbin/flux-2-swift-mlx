import CoreGraphics
import Foundation
import ImageIO

/// JPEG XL encode via Homebrew `cjxl` when ImageIO has no JXL destination (typical on macOS).
enum CJXLImageEncoder {
    static func isAvailable() -> Bool {
        locateExecutable() != nil
    }

    static func write(
        _ image: CGImage,
        to url: URL,
        mode: ProjectBundleImageWriter.EncodeMode
    ) throws {
        guard let cjxlPath = locateExecutable() else {
            throw Flux2Error.imageProcessingFailed(
                "cjxl was not found. Install JPEG XL with: brew install jpeg-xl"
            )
        }

        let pngURL = url.deletingPathExtension().appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: pngURL) }

        try writePNG(image, to: pngURL)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: cjxlPath)
        process.arguments = [pngURL.path, url.path] + distanceArguments(for: mode)

        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let detail = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw Flux2Error.imageProcessingFailed("cjxl failed (\(process.terminationStatus)): \(detail)")
        }
    }

    static func encode(
        _ image: CGImage,
        mode: ProjectBundleImageWriter.EncodeMode
    ) throws -> Data {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-jxl-\(UUID().uuidString).\(ProjectBundleImageWriter.fileExtension)")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try write(image, to: tempURL, mode: mode)
        return try Data(contentsOf: tempURL)
    }

    static func locateExecutable() -> String? {
        let candidates = [
            "/opt/homebrew/bin/cjxl",
            "/usr/local/bin/cjxl",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return nil
    }

    private static func distanceArguments(for mode: ProjectBundleImageWriter.EncodeMode) -> [String] {
        switch mode {
        case .lossless:
            return ["-d", "0"]
        case .lossyHighQuality:
            return ["-d", "1"]
        }
    }

    private static func writePNG(_ image: CGImage, to url: URL) throws {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else {
            throw Flux2Error.imageProcessingFailed("Could not create PNG destination.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw Flux2Error.imageProcessingFailed("Could not finalize PNG image.")
        }
        try (data as Data).write(to: url, options: .atomic)
    }
}
