// CleanImageCommand.swift — run SCUNet image cleanup on a file (visual check / utility).

import Foundation
import ArgumentParser
import Flux2Core
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct CleanImage: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "clean-image",
        abstract: "Clean an image with SCUNet (denoise / JPEG-artifact removal, 1:1 resolution)"
    )

    @Option(name: .shortAndLong, help: "Input image path")
    var input: String

    @Option(name: .shortAndLong, help: "Output image path (PNG)")
    var output: String = "cleaned.png"

    func run() throws {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: input) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ValidationError("Could not read image: \(input)")
        }
        print("Input \(image.width)x\(image.height) — running SCUNet…")

        let start = Date()
        let cleaned = try ImageCleanup.scunet(image)
        let elapsed = Date().timeIntervalSince(start)
        let mp = Double(image.width * image.height) / 1_000_000
        print(String(format: "Cleaned in %.2fs  (%.2f MP, %.2f MP/s)", elapsed, mp, mp / max(elapsed, 1e-6)))

        guard let dest = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: output) as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw ValidationError("Could not create output destination: \(output)")
        }
        CGImageDestinationAddImage(dest, cleaned, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ValidationError("Failed to write output: \(output)")
        }
        print("Wrote \(output)")
    }
}
