// InpaintCommand.swift — `flux2 inpaint` (RePaint-style masked inpainting)
// Copyright 2025 Vincent Gourbin
//
// Drives `Flux2MaskedInpaintingChain` from the framework. Moved here from
// the sibling `sharp-cli` (where it originally landed for historical
// reasons) so that all pure-Flux2 chain commands live next to the chains
// themselves.

import Foundation
import ArgumentParser
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Flux2Core
import Flux2Chains

struct Inpaint: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inpaint",
        abstract: "RePaint-style masked inpainting: regenerate the white-masked region of an image with a text prompt"
    )

    @Option(name: .long, help: "Root of the FluxForge Studio Models directory.")
    var modelsDir: String = "/Users/vincent/Pictures/FluxforgeStudio/Models"

    @Option(name: .long, help: "FLUX.2 model: klein-9b (distilled, default) | klein-9b-base | klein-9b-kv | dev | klein-4b | klein-4b-base.")
    var fluxModel: String = "klein-9b"

    @Option(name: [.short, .long], help: "Input image to inpaint.")
    var image: String

    @Option(name: [.short, .long], help: "Grayscale mask, same dimensions as image. WHITE = inpaint (will be replaced), BLACK = keep (preserved by RePaint blend).")
    var mask: String

    @Option(name: [.short, .long], help: "Text prompt describing the desired full image (the model regenerates everything, the mask just forces the original back outside the painted region).")
    var prompt: String

    @Option(name: [.short, .long], help: "Output PNG path.")
    var output: String

    @Option(name: .long, help: "Optional path(s) to reference image(s) — when set, the chain runs in I2I mode so the transformer attends to these while still respecting the RePaint mask. Repeat for multiple references. Useful for outpainting where you want the new strips to continue the existing scene.")
    var reference: [String] = []

    @Option(name: .long, help: "Denoising steps for FLUX.2 (4 for distilled klein, 25-28 for base/dev).")
    var steps: Int = 4
    @Option(name: .long, help: "Guidance scale (1.0 for distilled, 3.5 for base, 4.0 for dev).")
    var guidance: Float = 1.0
    @Option(name: .long, help: "Random seed (omit for non-deterministic).")
    var seed: UInt64?
    @Option(name: .long, help: "Maximum total pixel count for the working resolution. Defaults to 1024² (≈ 1 048 576).")
    var maxPixels: Int = 1024 * 1024

    func run() async throws {
        @Sendable func logErr(_ msg: String) {
            FileHandle.standardError.write(Data((msg + "\n").utf8))
        }

        ModelRegistry.customModelsDirectory = URL(fileURLWithPath: modelsDir)

        guard let imageCG = Self.loadCGImage(at: image) else {
            throw ValidationError("Could not decode image at \(image)")
        }
        guard let maskCG = Self.loadCGImage(at: mask) else {
            throw ValidationError("Could not decode mask at \(mask)")
        }
        logErr("Image: \(image) (\(imageCG.width)×\(imageCG.height))")
        logErr("Mask : \(mask) (\(maskCG.width)×\(maskCG.height))")
        logErr("Prompt: \(prompt)")

        let modelChoice: Flux2Model
        switch fluxModel.lowercased() {
        case "klein-9b", "klein9b":               modelChoice = .klein9B
        case "klein-9b-base", "klein9b-base":     modelChoice = .klein9BBase
        case "klein-9b-kv", "klein9b-kv":         modelChoice = .klein9BKV
        case "klein-4b", "klein4b":               modelChoice = .klein4B
        case "klein-4b-base", "klein4b-base":     modelChoice = .klein4BBase
        case "dev":                               modelChoice = .dev
        default:
            throw ValidationError("Unsupported --flux-model '\(fluxModel)'.")
        }

        let pipeline = Flux2Pipeline(
            model: modelChoice,
            quantization: .memoryEfficient,
            vaeVariant: .smallDecoder
        )
        let loadStart = Date()
        try await pipeline.loadModels()
        logErr("✓ Flux2 pipeline ready in \(String(format: "%.1fs", Date().timeIntervalSince(loadStart)))")

        let referenceCGs: [CGImage]? = reference.isEmpty ? nil : try reference.map { p in
            guard let img = Self.loadCGImage(at: p) else {
                throw ValidationError("Could not decode reference image at \(p)")
            }
            logErr("Reference: \(p) (\(img.width)×\(img.height))")
            return img
        }
        if referenceCGs != nil { logErr("I2I+RePaint mode enabled") }

        let chain = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: prompt,
            image: imageCG,
            mask: maskCG,
            referenceImages: referenceCGs,
            steps: steps,
            guidance: guidance,
            seed: seed,
            maxPixels: maxPixels,
            onProgress: { step, total in
                logErr("  step \(step)/\(total)")
            }
        )

        let runStart = Date()
        let result = try await chain.run()
        logErr("✓ Inpainting done in \(String(format: "%.1fs", Date().timeIntervalSince(runStart)))")
        try Self.savePNG(result.image, to: output)
        logErr("✓ Inpainted image → \(output)")
    }

    private static func loadCGImage(at path: String) -> CGImage? {
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private static func savePNG(_ image: CGImage, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw ValidationError("Could not create PNG destination at \(path)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ValidationError("Failed to write PNG at \(path)")
        }
    }
}
