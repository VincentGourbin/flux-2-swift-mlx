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
import FluxTextEncoders
import MLX

/// User-facing string mapping for ``Flux2MaskConvention``.
///
/// Kept local to the CLI so the framework enum stays free of
/// ArgumentParser dependencies; the `--mask-convention` flag accepts the
/// short identifiers `grayscale` / `alpha`.
enum MaskConventionArg: String, ExpressibleByArgument, CaseIterable {
    case grayscale
    case alpha

    var convention: Flux2MaskConvention {
        switch self {
        case .grayscale: return .grayscaleWhiteInpaint
        case .alpha:     return .alphaTransparentInpaint
        }
    }
}

/// User-facing string mapping for ``Flux2InpaintIntent``.
///
/// Kept local to the CLI so the framework enum stays free of
/// ArgumentParser dependencies. Same identifiers as the enum cases.
enum InpaintIntentArg: String, ExpressibleByArgument, CaseIterable {
    case replace
    case remove
    case modify
    case changeScene = "change-scene"

    var intent: Flux2InpaintIntent {
        switch self {
        case .replace:     return .replace
        case .remove:      return .remove
        case .modify:      return .modify
        case .changeScene: return .changeScene
        }
    }
}

struct Inpaint: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inpaint",
        abstract: "RePaint-style masked inpainting: regenerate the masked region of an image with a text prompt"
    )

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage). Defaults to the framework's standard models path.")
    var modelsDir: String?

    @Option(name: .long, help: "FLUX.2 model: klein-9b (distilled, default) | klein-9b-base | klein-9b-kv | dev | klein-4b | klein-4b-base.")
    var fluxModel: String = "klein-9b"

    @Option(name: [.short, .long], help: "Input image to inpaint.")
    var image: String

    @Option(name: [.short, .long], help: "Mask image, same dimensions as the input. Interpretation depends on --mask-convention: 'grayscale' (default) reads luminance, WHITE = inpaint, BLACK = keep; 'alpha' reads the alpha channel, TRANSPARENT = inpaint, OPAQUE = keep. Soft values honoured in both modes.")
    var mask: String

    @Option(name: .long, help: "How --mask is read: 'grayscale' (default — white = inpaint) or 'alpha' (transparent = inpaint, RGB ignored). Use 'alpha' when you erased the region to redo in a photo editor and saved a PNG with alpha.")
    var maskConvention: MaskConventionArg = .grayscale

    @Option(name: [.short, .long], help: "Text prompt describing the desired full image. Follow Flux 2 prompting guidelines (https://docs.bfl.ml/guides/prompting_guide_flux2): Subject + Action + Style + Context, 30–80 words, describe the WHOLE scene including the kept surroundings (the model has no other lighting / perspective cues for the inpainted region).")
    var prompt: String

    @Option(name: [.short, .long], help: "Output PNG path.")
    var output: String

    @Option(name: .long, help: "Optional path(s) to reference image(s) — when set, the chain runs in I2I mode so the transformer attends to these while still respecting the RePaint mask. Repeat for multiple references. Useful for scene-extension / repair where the new content must continue the kept region.")
    var reference: [String] = []

    @Flag(name: .long, help: "Auto-feed the source image as I2I conditioning when --reference is not set. OFF by default — for 'replace object X with object Y' inpainting, the source still contains X under the mask and bleeds X into Y via attention. Enable for repair / scene-extension where the masked region is empty.")
    var useImageAsReference: Bool = false

    @Flag(name: .long, help: "Text-encoder-only prompt rewriting (Mistral/Klein-Qwen3). Does NOT look at the image. For image-aware rewriting that inherits the source's lighting/camera/materials, use --enrich-prompt-with-vlm instead.")
    var upsamplePrompt: Bool = false

    @Flag(name: .long, help: "Image-aware prompt rewriting via the bundled Qwen3.5 VLM. The VLM looks at --image and rewrites --prompt into a 30-80 word BFL-style Flux 2 prompt that inherits the source's photographic identity (camera angle, lighting direction, materials, palette, depth of field). Strictly optional: if the VLM is not loaded, the chain falls back to --prompt verbatim with a warning. Note: this CLI does NOT auto-load the VLM; load it ahead of time via FluxEncodersCLI or the test-qwen35 command. When both --upsample-prompt and --enrich-prompt-with-vlm are set, the VLM wins.")
    var enrichPromptWithVLM: Bool = false

    @Option(name: .long, help: "Drives --enrich-prompt-with-vlm (ignored otherwise). 'replace' = swap object X for Y (default). 'remove' = clear object X, surface continues. 'modify' = keep object X but change colour/outfit/expression. 'change-scene' = keep subject exactly as-is, change the scene around it (use this when the mask preserves the subject — e.g. 'put the cat at the pool').")
    var intent: InpaintIntentArg = .replace

    @Option(name: .long, help: "Qwen3.5 VLM variant to load in-process when --enrich-prompt-with-vlm is set: '8bit' (default, 5 GB, recommended) or '4bit' (3 GB, faster but lower quality). Auto-downloads if missing. Omit to skip loading — the chain will then fall back to --prompt verbatim and emit a warning.")
    var qwen35Variant: String?

    @Option(name: .long, help: "Override the local path to Qwen3.5 VLM weights (alternative to --qwen35-variant for sandboxed apps).")
    var qwen35Path: String?

    @Option(name: .long, help: "Denoising steps for FLUX.2 (4 for distilled klein, 25-28 for base/dev).")
    var steps: Int = 4
    @Option(name: .long, help: "Guidance scale (1.0 for distilled, 3.5 for base, 4.0 for dev).")
    var guidance: Float = 1.0
    @Option(name: .long, help: "Random seed (omit for non-deterministic).")
    var seed: UInt64?
    @Option(name: .long, help: "Maximum total pixel count for the working resolution. Defaults to 1024² (≈ 1 048 576).")
    var maxPixels: Int = 1024 * 1024
    @Option(name: .long, help: "Denoising strength in (0, 1]. 1.0 (default) = masked region starts from pure noise (replace/remove). < 1.0 = start from the noised original and skip early timesteps — preserves layout/pose/palette, use ~0.5-0.75 for 'modify' edits. With 4-step distilled models only ≤ 0.75 actually skips a step.")
    var strength: Float = 1.0
    @Option(name: .long, help: "Crop-and-stitch padding in pixels (like diffusers padding_mask_crop). When set, inpaint only a crop around the mask (full token budget on the edit) and paste the result back onto the untouched original — output keeps the original resolution. Typical: 32-64. Recommended when the mask is small relative to the photo.")
    var maskCropPadding: Int?

    @Option(name: .long, help: "Text encoder quantization: bf16, 8bit, 6bit, 4bit")
    var textQuant: String = "4bit"

    @Option(name: .long, help: "Transformer quantization: \(TransformerQuantization.cliValueList)")
    var transformerQuant: String = "qint8"

    @Flag(name: .long, help: "Show detailed per-phase performance profiling (model loads, text encoding, VAE encodes, per-step timings).")
    var profile: Bool = false

    @Option(name: .long, help: "Run the chain N times in the same process (same pipeline instance) to separate cold-start (first run: kernel compilation, cache warm-up) from steady-state. Default 1.")
    var repeatCount: Int = 1

    @Flag(name: .long, help: "Clear the MLX GPU buffer cache between --repeat-count runs (diagnostic for run-to-run slowdown).")
    var cleanupBetweenRuns: Bool = false

    @Flag(name: .long, help: "Compile the denoising transformer forward with MLX.compile (experimental, benchmarking only). Measured neutral on klein-9b bf16 — the elementwise hot spots are already hand-fused; steps are GEMM-bound. Forces memoryOptimization to .disabled (higher peak memory) and pays a one-time trace on the first step. Output is numerically identical.")
    var compileStep: Bool = false
    @Flag(name: .long, help: "Composite the generated canvas back onto the original in pixel space using the soft mask (kept pixels stay bit-identical, no VAE roundtrip). Implied by --mask-crop-padding.")
    var compositeOnOriginal: Bool = false

    @Flag(name: .long, help: "Keep the text encoder loaded between generations instead of reloading it each time (memory-first default). Saves ~1s warm / several seconds cold per generation at the cost of encoder + transformer resident simultaneously (Klein-9B: ≈ +5 GB). Enable on machines with RAM headroom.")
    var keepTextEncoder: Bool = false

    func run() async throws {
        @Sendable func logErr(_ msg: String) {
            FileHandle.standardError.write(Data((msg + "\n").utf8))
        }

        configureModelsDirectory(modelsDir)

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

        // Optional Qwen3.5 VLM load — only relevant when the user opts
        // into --enrich-prompt-with-vlm. The chain itself doesn't
        // auto-load; we do it here at the CLI level so a single
        // invocation is enough to benchmark the VLM-enriched path.
        if enrichPromptWithVLM {
            // Surface the VLM-built prompt via FluxDebug.info so the
            // user can audit what FLUX.2 actually receives.
            FluxDebug.isEnabled = true
            if qwen35Variant == nil, qwen35Path == nil {
                logErr("WARNING: --enrich-prompt-with-vlm is set but neither --qwen35-variant nor --qwen35-path was provided — the chain will fall back to --prompt verbatim.")
            }
        }
        if let qwen35Path {
            logErr("Loading Qwen3.5 VLM from \(qwen35Path) ...")
            try await FluxTextEncoders.shared.loadQwen35VLM(from: qwen35Path)
            logErr("✓ Qwen3.5 VLM loaded")
        } else if let variantStr = qwen35Variant {
            let selectedVariant: Qwen35Variant
            switch variantStr.lowercased() {
            case "4bit": selectedVariant = .qwen35_4B_4bit
            case "8bit": selectedVariant = .qwen35_4B_8bit
            default: throw ValidationError("Unsupported --qwen35-variant '\(variantStr)' (use '8bit' or '4bit')")
            }
            logErr("Downloading/loading Qwen3.5 VLM (\(selectedVariant.displayName)) ...")
            let downloader = TextEncoderModelDownloader()
            let path = try await downloader.downloadQwen35(variant: selectedVariant) { progress, message in
                logErr("  [\(Int(progress * 100))%] \(message)")
            }
            try await FluxTextEncoders.shared.loadQwen35VLM(from: path.path)
            logErr("✓ Qwen3.5 VLM loaded")
        }

        guard let textQuantization = MistralQuantization(rawValue: textQuant) else {
            throw ValidationError("Invalid text quantization: \(textQuant). Use bf16, 8bit, 6bit, or 4bit")
        }
        let quantConfig = Flux2QuantizationConfig(
            textEncoder: textQuantization,
            transformer: try TransformerQuantization.parseCLI(transformerQuant)
        )

        if profile {
            Flux2Profiler.shared.enable()
        }

        let pipeline = Flux2Pipeline(
            model: modelChoice,
            quantization: quantConfig,
            vaeVariant: .smallDecoder
        )
        pipeline.compileDenoisingStep = compileStep
        pipeline.keepTextEncoderLoaded = keepTextEncoder
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
            maskConvention: maskConvention.convention,
            referenceImages: referenceCGs,
            useImageAsReference: useImageAsReference,
            steps: steps,
            guidance: guidance,
            seed: seed,
            strength: strength,
            maskCropPadding: maskCropPadding,
            compositeOnOriginal: compositeOnOriginal,
            upsamplePrompt: upsamplePrompt,
            enrichPromptWithVLM: enrichPromptWithVLM,
            intent: intent.intent,
            maxPixels: maxPixels,
            onProgress: { step, total in
                logErr("  step \(step)/\(total)")
            }
        )

        guard repeatCount >= 1 else {
            throw ValidationError("--repeat-count must be ≥ 1")
        }
        @Sendable func logMLXMemory(_ label: String) {
            let active = MLX.Memory.activeMemory / 1_048_576
            let peak = MLX.Memory.peakMemory / 1_048_576
            let cache = MLX.Memory.cacheMemory / 1_048_576
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            let rss = kr == KERN_SUCCESS ? "\(info.resident_size / 1_048_576) MB" : "n/a"
            logErr("[mem \(label)] MLX active=\(active) MB peak=\(peak) MB cache=\(cache) MB | RSS=\(rss)")
        }

        var wallTimes: [TimeInterval] = []
        for runIndex in 1...repeatCount {
            if profile { Flux2Profiler.shared.reset() }
            if cleanupBetweenRuns && runIndex > 1 {
                MLX.Memory.clearCache()
                logErr("[mem] cleared MLX cache between runs")
            }
            logMLXMemory("before run \(runIndex)")
            let runStart = Date()
            let result = try await chain.run()
            let wall = Date().timeIntervalSince(runStart)
            wallTimes.append(wall)
            logMLXMemory("after run \(runIndex)")
            logErr("✓ Inpainting run \(runIndex)/\(repeatCount) done in \(String(format: "%.1fs", wall))")
            if profile {
                print(Flux2Profiler.shared.generateReport())
            }
            if runIndex == repeatCount {
                try Self.savePNG(result.image, to: output)
                logErr("✓ Inpainted image → \(output)")
            }
        }
        if repeatCount > 1 {
            let summary = wallTimes.enumerated()
                .map { "run \($0.offset + 1): \(String(format: "%.1fs", $0.element))" }
                .joined(separator: "  |  ")
            logErr("Σ wall times — \(summary)")
        }
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
