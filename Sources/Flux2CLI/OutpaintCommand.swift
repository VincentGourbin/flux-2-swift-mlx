// OutpaintCommand.swift — `flux2 outpaint` (Flux2OutpaintingChain wrapper)
// Copyright 2025 Vincent Gourbin
//
// One-liner outpainting:
//   flux2 outpaint -i in.jpg -o out.png --left 384 --right 384 \
//                  --prompt "..."
//
// Mirrors the Black Forest Labs `flux_outpainting` API surface: caller
// passes paddings in pixels per side + a prompt; the chain handles the
// canvas/mask/seed/inference internally.

import Foundation
import ArgumentParser
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Flux2Core
import Flux2Chains

struct Outpaint: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "outpaint",
        abstract: "Extend an image on any of the four sides via Flux2OutpaintingChain"
    )

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage). Defaults to the framework's standard models path.")
    var modelsDir: String?

    @Option(name: .long, help: "FLUX.2 model: klein-9b (distilled, default) | klein-9b-base | klein-9b-kv | dev | klein-4b | klein-4b-base.")
    var fluxModel: String = "klein-9b"

    @Option(name: [.short, .long], help: "Input image to extend.")
    var image: String

    @Option(name: [.short, .long], help: "Text prompt describing the full extended scene. Follow Flux 2 prompting guidelines (https://docs.bfl.ml/guides/prompting_guide_flux2): Subject + Action + Style + Context, 30–80 words, mention the content of the keep region too.")
    var prompt: String

    @Option(name: [.short, .long], help: "Output PNG path.")
    var output: String

    @Option(name: .long, help: "Pixels to add at the top. Rounded up to a multiple of 32.")
    var top: Int = 0
    @Option(name: .long, help: "Pixels to add at the bottom. Rounded up to a multiple of 32.")
    var bottom: Int = 0
    @Option(name: .long, help: "Pixels to add on the left. Rounded up to a multiple of 32.")
    var left: Int = 0
    @Option(name: .long, help: "Pixels to add on the right. Rounded up to a multiple of 32.")
    var right: Int = 0

    @Option(name: .long, help: "Denoising steps (4 for distilled klein, 25-28 for base/dev).")
    var steps: Int = 4
    @Option(name: .long, help: "Guidance scale (1.0 for distilled, 3.5 for base, 4.0 for dev).")
    var guidance: Float = 1.0
    @Option(name: .long, help: "Random seed.")
    var seed: UInt64?
    @Flag(name: .long, help: "Text-encoder-only prompt rewriting (Mistral/Klein-Qwen3). Does NOT look at the image. For image-aware rewriting that continues the source's lighting/materials into the new strips, use --enrich-prompt-with-vlm instead.")
    var upsamplePrompt: Bool = false

    @Flag(name: .long, help: "Image-aware prompt rewriting via the bundled Qwen3.5 VLM. The VLM looks at --image and the requested extension sides, then assembles a 30-80 word BFL-style Flux 2 prompt that continues the kept region's materials, perspective, lighting and palette into the new strips. Strictly optional: if the VLM is not loaded, the chain falls back to --prompt verbatim with a warning. Load the VLM ahead of time via FluxEncodersCLI or the test-qwen35 command. When both --upsample-prompt and --enrich-prompt-with-vlm are set, the VLM wins.")
    var enrichPromptWithVLM: Bool = false
    @Option(name: .long, help: "Width of the soft transition band, in pixels of the keep region. Default 32.")
    var transitionPixels: Int = 32
    @Option(name: .long, help: "Cap on the total working pixel count. Defaults to 4 M; raise if you want larger canvases.")
    var maxPixels: Int = 4 * 1024 * 1024

    func run() async throws {
        @Sendable func logErr(_ msg: String) {
            FileHandle.standardError.write(Data((msg + "\n").utf8))
        }

        configureModelsDirectory(modelsDir)

        guard let imageCG = Self.loadCGImage(at: image) else {
            throw ValidationError("Could not decode image at \(image)")
        }
        logErr("Image: \(image) (\(imageCG.width)×\(imageCG.height))")
        logErr("Padding: top=\(top) bottom=\(bottom) left=\(left) right=\(right)")
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

        let chain = Flux2OutpaintingChain(
            pipeline: pipeline,
            image: imageCG,
            top: top,
            bottom: bottom,
            left: left,
            right: right,
            prompt: prompt,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            enrichPromptWithVLM: enrichPromptWithVLM,
            transitionPixels: transitionPixels,
            maxPixels: maxPixels,
            onProgress: { step, total in
                logErr("  step \(step)/\(total)")
            }
        )

        let runStart = Date()
        let result = try await chain.run()
        logErr("✓ Outpainting done in \(String(format: "%.1fs", Date().timeIntervalSince(runStart)))")
        try Self.savePNG(result.image, to: output)
        logErr("✓ Outpainted image → \(output)")
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
