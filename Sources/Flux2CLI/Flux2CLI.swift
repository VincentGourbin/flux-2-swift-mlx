// Flux2CLI.swift - Command Line Interface for Flux.2
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import ImageIO
import UniformTypeIdentifiers

@main
struct Flux2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flux2",
        abstract: "Flux.2 image generation on Mac with MLX",
        version: Flux2Core.version,
        subcommands: [
            TextToImage.self,
            ImageToImage.self,
            Download.self,
            Info.self,
            VLMTest.self,
        ],
        defaultSubcommand: TextToImage.self
    )
}

// MARK: - Text-to-Image Command

struct TextToImage: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "t2i",
        abstract: "Generate image from text prompt"
    )

    @Argument(help: "Text prompt for image generation")
    var prompt: String

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String = "output.png"

    @Option(name: .shortAndLong, help: "Image width")
    var width: Int = 1024

    @Option(name: .shortAndLong, help: "Image height")
    var height: Int = 1024

    @Option(name: .shortAndLong, help: "Number of inference steps (default: 50 for Dev, 4 for Klein)")
    var steps: Int?

    @Option(name: .shortAndLong, help: "Guidance scale (default: 4.0 for Dev, 1.0 for Klein)")
    var guidance: Float?

    @Option(name: .long, help: "Random seed")
    var seed: UInt64?

    @Option(name: .long, help: "Model variant: dev (32B), klein-4b (4B, Apache 2.0), klein-9b (9B)")
    var model: String = "dev"

    @Option(name: .long, help: "Text encoder quantization: bf16, 8bit, 6bit, 4bit")
    var textQuant: String = "8bit"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, qint4")
    var transformerQuant: String = "qint8"

    @Flag(name: .long, help: "Enable debug logs (verbose output)")
    var debug: Bool = false

    @Flag(name: .long, help: "Enable performance profiling")
    var profile: Bool = false

    @Flag(name: .long, help: "Enhance prompt with more visual details before encoding")
    var upsamplePrompt: Bool = false

    @Option(name: .long, help: "Image to analyze with VLM and inject description into prompt (semantic interpretation)")
    var interpret: [String] = []

    @Option(name: .long, help: "Save intermediate images at each N steps (e.g., 5 saves every 5 steps)")
    var checkpoint: Int?

    func run() async throws {
        // Configure debug logging
        if debug {
            Flux2Debug.enableDebugMode()
        } else {
            Flux2Debug.setNormalMode()
        }

        // Configure profiling
        if profile {
            Flux2Profiler.shared.enable()
        }
        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, or klein-9b")
        }

        // Apply model-specific defaults for Klein (distilled for 4 steps, guidance 1.0)
        let actualSteps: Int
        let actualGuidance: Float
        switch modelVariant {
        case .dev:
            actualSteps = steps ?? 50
            actualGuidance = guidance ?? 4.0
        case .klein4B, .klein9B:
            actualSteps = steps ?? 4
            actualGuidance = guidance ?? 1.0
        }

        // Parse quantization settings
        guard let textQuantization = MistralQuantization(rawValue: textQuant) else {
            throw ValidationError("Invalid text quantization: \(textQuant). Use bf16, 8bit, 6bit, or 4bit")
        }

        guard let transformerQuantization = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use bf16, qint8, or qint4")
        }

        let quantConfig = Flux2QuantizationConfig(
            textEncoder: textQuantization,
            transformer: transformerQuantization
        )

        if debug {
            print("Configuration:")
            print("  Model: \(modelVariant.displayName)")
            print("  Text encoder: \(textQuantization.displayName)")
            print("  Transformer: \(transformerQuantization.displayName)")
            print("  Estimated memory: ~\(quantConfig.estimatedTotalMemoryGB)GB")
            print()
        }

        // Validate interpret image paths exist
        var interpretImagePaths: [String] = []
        for path in interpret {
            guard FileManager.default.fileExists(atPath: path) else {
                throw ValidationError("Interpret image not found: \(path)")
            }
            interpretImagePaths.append(path)
        }

        print("Generating image...")
        print("  Prompt: \"\(prompt)\"")
        if !interpretImagePaths.isEmpty {
            print("  Interpret images: \(interpretImagePaths.count) (VLM will analyze and enrich prompt)")
            for path in interpretImagePaths {
                print("    - \(path)")
            }
        }
        if upsamplePrompt {
            print("  Prompt upsampling: enabled (will enhance prompt with visual details)")
        }
        print("  Size: \(width)x\(height)")
        print("  Steps: \(actualSteps)")
        print("  Guidance: \(actualGuidance)")
        if let seed = seed {
            print("  Seed: \(seed)")
        }
        if let checkpointInterval = checkpoint {
            print("  Checkpoints: every \(checkpointInterval) step(s)")
        }
        print()

        // Create pipeline
        let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig)

        // Check for missing models
        if !pipeline.hasRequiredModels {
            let missing = pipeline.missingModels
            print("Missing models:")
            for m in missing {
                print("  - \(m.displayName)")
            }
            print()
            print("Please download the required models first.")
            throw ExitCode.failure
        }

        // Generate
        let startTime = Date()

        // Prepare checkpoint directory if needed
        let checkpointDir: String?
        if let _ = checkpoint {
            // Create checkpoint directory based on output path
            let outputURL = URL(fileURLWithPath: output)
            let baseName = outputURL.deletingPathExtension().lastPathComponent
            let parentDir = outputURL.deletingLastPathComponent().path
            checkpointDir = "\(parentDir)/\(baseName)_checkpoints"

            // Create directory
            try FileManager.default.createDirectory(
                atPath: checkpointDir!,
                withIntermediateDirectories: true
            )
            print("Checkpoints will be saved to: \(checkpointDir!)")
        } else {
            checkpointDir = nil
        }

        let image = try await pipeline.generateTextToImage(
            prompt: prompt,
            interpretImagePaths: interpretImagePaths.isEmpty ? nil : interpretImagePaths,
            height: height,
            width: width,
            steps: actualSteps,
            guidance: actualGuidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpoint
        ) { current, total in
            let progress = Float(current) / Float(total) * 100
            print("\rStep \(current)/\(total) [\(String(format: "%.0f", progress))%]", terminator: "")
            fflush(stdout)
        } onCheckpoint: { step, checkpointImage in
            if let dir = checkpointDir {
                let checkpointPath = "\(dir)/step_\(String(format: "%03d", step)).png"
                do {
                    try saveImage(checkpointImage, to: checkpointPath)
                    print("\n  Checkpoint saved: step_\(String(format: "%03d", step)).png")
                } catch {
                    print("\n  Failed to save checkpoint at step \(step): \(error.localizedDescription)")
                }
            }
        }

        print()

        let elapsed = Date().timeIntervalSince(startTime)
        print("Generation completed in \(String(format: "%.1f", elapsed))s")

        // Save image
        try saveImage(image, to: output)
        print("Image saved to \(output)")
    }
}

// MARK: - Image-to-Image Command

struct ImageToImage: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "i2i",
        abstract: "Transform an image using a text prompt (image-to-image)"
    )

    @Argument(help: "Text prompt describing the desired output")
    var prompt: String

    @Option(name: .shortAndLong, help: "Reference image for visual conditioning")
    var images: [String]

    @Option(name: .long, help: "Image to analyze with VLM and inject description into prompt (not used as visual reference)")
    var interpret: [String] = []

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String = "output.png"

    @Option(name: .shortAndLong, help: "Number of effective denoising steps (default: 28 for Dev, 4 for Klein)")
    var steps: Int?

    @Flag(name: .long, help: "Interpret --steps as total steps before strength reduction (legacy behavior)")
    var totalSteps: Bool = false

    @Option(name: .shortAndLong, help: "Guidance scale (default: 4.0 for Dev, 1.0 for Klein)")
    var guidance: Float?

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Option(name: .shortAndLong, help: "Output image width (default: from first reference image)")
    var width: Int?

    @Option(name: .shortAndLong, help: "Output image height (default: from first reference image)")
    var height: Int?

    @Option(name: .long, help: "Denoising strength (0.0-1.0). Lower = preserve more of original image")
    var strength: Float = 0.8

    @Flag(name: .long, help: "Enhance prompt with visual details using Mistral before encoding")
    var upsamplePrompt: Bool = false

    @Option(name: .long, help: "Save checkpoint image every N steps")
    var checkpoint: Int?

    @Flag(name: .long, help: "Show detailed performance profiling")
    var profile: Bool = false

    @Option(name: .long, help: "Model variant: dev (32B), klein-4b (4B, Apache 2.0), klein-9b (9B)")
    var model: String = "dev"

    @Option(name: .long, help: "Text encoder quantization: bf16, 8bit, 6bit, 4bit")
    var textQuant: String = "8bit"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, qint4")
    var transformerQuant: String = "qint8"

    func run() async throws {
        let startTime = Date()

        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, or klein-9b")
        }

        // Apply model-specific defaults for Klein (distilled for 4 steps, guidance 1.0)
        let actualSteps: Int
        let actualGuidance: Float
        switch modelVariant {
        case .dev:
            actualSteps = steps ?? 28
            actualGuidance = guidance ?? 4.0
        case .klein4B, .klein9B:
            actualSteps = steps ?? 4
            actualGuidance = guidance ?? 1.0
        }

        // Validate image count (1-3 reference images)
        guard !images.isEmpty && images.count <= 3 else {
            if images.isEmpty {
                throw ValidationError("Please provide 1-3 reference images with --images")
            } else {
                throw ValidationError("Maximum 3 reference images allowed")
            }
        }

        // Validate strength
        guard strength > 0.0 && strength <= 1.0 else {
            throw ValidationError("Strength must be between 0.0 and 1.0")
        }

        // Configure profiling
        if profile {
            Flux2Profiler.shared.enable()
        }

        // Load reference images (visual conditioning)
        var refImages: [CGImage] = []
        for path in images {
            guard let image = loadImage(from: path) else {
                throw ValidationError("Failed to load image: \(path)")
            }
            refImages.append(image)
            print("Loaded reference image: \(path) (\(image.width)x\(image.height))")
        }

        // Validate interpret image paths exist (VLM will load them directly)
        var interpretImagePaths: [String] = []
        for path in interpret {
            guard FileManager.default.fileExists(atPath: path) else {
                throw ValidationError("Interpret image not found: \(path)")
            }
            interpretImagePaths.append(path)
            print("Will interpret image: \(path) [VLM analysis]")
        }

        print("Mode: Image-to-Image (Flux.2 conditioning)")

        // Show output dimensions
        let outputWidth = width ?? refImages[0].width
        let outputHeight = height ?? refImages[0].height
        print("Output size: \(outputWidth)x\(outputHeight)")

        // Note: Flux.2 I2I uses conditioning mode, not SD-style noise injection
        // The reference image provides visual context to the transformer
        // Strength parameter is accepted for API compatibility but doesn't skip timesteps
        print("Steps: \(actualSteps)")
        print("Guidance: \(actualGuidance)")

        if upsamplePrompt {
            print("Prompt upsampling: enabled")
        }

        // Parse quantization
        guard let textQuantization = MistralQuantization(rawValue: textQuant) else {
            throw ValidationError("Invalid text quantization: \(textQuant). Use: bf16, 8bit, 6bit, 4bit")
        }

        guard let transformerQuantization = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use: bf16, qint8, qint4")
        }

        let quantConfig = Flux2QuantizationConfig(
            textEncoder: textQuantization,
            transformer: transformerQuantization
        )

        // Create pipeline
        let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig)

        // Setup checkpoint directory if needed
        let checkpointDir: String?
        if let interval = checkpoint, interval > 0 {
            let baseName = (output as NSString).deletingPathExtension
            checkpointDir = "\(baseName)_checkpoints"
            try FileManager.default.createDirectory(
                atPath: checkpointDir!,
                withIntermediateDirectories: true
            )
            print("Checkpoints will be saved to: \(checkpointDir!)")
        } else {
            checkpointDir = nil
        }

        print("Generating image...")

        let image = try await pipeline.generateImageToImage(
            prompt: prompt,
            images: refImages,
            interpretImagePaths: interpretImagePaths.isEmpty ? nil : interpretImagePaths,
            height: height,
            width: width,
            steps: actualSteps,
            guidance: actualGuidance,
            seed: seed,
            strength: strength,
            upsamplePrompt: upsamplePrompt,
            checkpointInterval: checkpoint,
            onProgress: { current, total in
                let progress = Float(current) / Float(total) * 100
                print("\rStep \(current)/\(total) [\(String(format: "%.0f", progress))%]", terminator: "")
                fflush(stdout)
            },
            onCheckpoint: { step, checkpointImage in
                if let dir = checkpointDir {
                    let checkpointPath = "\(dir)/step_\(String(format: "%03d", step)).png"
                    do {
                        try saveImage(checkpointImage, to: checkpointPath)
                        print("\n  Checkpoint saved: step_\(String(format: "%03d", step)).png")
                    } catch {
                        print("\n  Failed to save checkpoint at step \(step): \(error.localizedDescription)")
                    }
                }
            }
        )

        print()

        let elapsed = Date().timeIntervalSince(startTime)
        print("Generation completed in \(String(format: "%.1f", elapsed))s")

        // Save image
        try saveImage(image, to: output)
        print("Image saved to \(output)")

        // Show profiler report if requested
        if profile {
            print()
            print(Flux2Profiler.shared.generateReport())
        }
    }
}

// MARK: - Download Command

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download required models from HuggingFace"
    )

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Option(name: .long, help: "Model to download: dev, klein-4b, klein-9b")
    var model: String = "dev"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8/8bit, qint4")
    var transformerQuant: String = "qint8"

    @Flag(name: .long, help: "Download all model variants")
    var all: Bool = false

    @Flag(name: .long, help: "Only download VAE")
    var vaeOnly: Bool = false

    func run() async throws {
        // Get token from environment if not provided
        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        if token == nil {
            print("⚠️  No HuggingFace token provided.")
            print("   For gated models like Flux.2, you may need to:")
            print("   1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-dev")
            print("   2. Set HF_TOKEN environment variable or use --hf-token")
            print()
        }

        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use dev, klein-4b, or klein-9b")
        }

        let downloader = Flux2ModelDownloader(hfToken: token)

        if vaeOnly {
            print("Downloading VAE...")
            let component = ModelRegistry.ModelComponent.vae(.standard)
            try await downloadComponent(downloader, component)
            return
        }

        if all {
            print("Downloading all model variants for \(modelVariant.displayName)...")
            let variants = ModelRegistry.TransformerVariant.allCases.filter { $0.modelType == modelVariant }
            for variant in variants {
                let component = ModelRegistry.ModelComponent.transformer(variant)
                try await downloadComponent(downloader, component)
            }
        } else {
            // Parse quantization and get the right variant for this model type
            let quant: TransformerQuantization
            switch transformerQuant {
            case "bf16": quant = .bf16
            case "qint8", "8bit": quant = .qint8
            case "qint4", "4bit": quant = .qint4
            default:
                throw ValidationError("Invalid transformer quantization: \(transformerQuant). Use bf16, qint8/8bit, or qint4")
            }

            let variant = ModelRegistry.TransformerVariant.variant(for: modelVariant, quantization: quant)
            print("Downloading \(modelVariant.displayName) Transformer (\(variant.rawValue))...")
            let component = ModelRegistry.ModelComponent.transformer(variant)
            try await downloadComponent(downloader, component)
        }

        // Always download VAE (shared between all models)
        print("Downloading VAE...")
        let vaeComponent = ModelRegistry.ModelComponent.vae(.standard)
        try await downloadComponent(downloader, vaeComponent)

        print()
        print("✅ Download complete!")
        print("   Models stored in: \(ModelRegistry.modelsDirectory.path)")

        // Text encoder info
        switch modelVariant {
        case .dev:
            print("   Note: Text encoder (Mistral) will be auto-downloaded on first run")
        case .klein4B, .klein9B:
            print("   Note: Text encoder (Qwen3) will be auto-downloaded on first run")
        }
    }

    private func downloadComponent(_ downloader: Flux2ModelDownloader, _ component: ModelRegistry.ModelComponent) async throws {
        do {
            _ = try await downloader.download(component) { progress, message in
                let percent = Int(progress * 100)
                print("\r  [\(percent)%] \(message)", terminator: "")
                fflush(stdout)
            }
            print()
        } catch {
            print()
            print("❌ Failed to download \(component.displayName): \(error.localizedDescription)")
            throw error
        }
    }
}

// MARK: - Info Command

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show system and model information"
    )

    func run() throws {
        print("Flux.2 Swift MLX Framework")
        print("Version: \(Flux2Core.version)")
        print()

        print("System Information:")
        print("  RAM: \(ModelRegistry.systemRAMGB)GB")
        print("  Recommended config: \(ModelRegistry.defaultConfig.description)")
        print()

        print("Available Quantization Presets:")
        print("  High Quality (~90GB): bf16 text + bf16 transformer")
        print("  Balanced (~60GB): 8bit text + qint8 transformer")
        print("  Memory Efficient (~50GB): 4bit text + qint8 transformer")
        print("  Minimal (~35GB): 4bit text + qint4 transformer")
        print()

        print("Model Status:")
        for variant in ModelRegistry.TransformerVariant.allCases {
            let component = ModelRegistry.ModelComponent.transformer(variant)
            let status = ModelRegistry.isDownloaded(component) ? "✓" : "✗"
            print("  [\(status)] \(component.displayName)")
        }

        for variant in ModelRegistry.TextEncoderVariant.allCases {
            let component = ModelRegistry.ModelComponent.textEncoder(variant)
            let status = ModelRegistry.isDownloaded(component) ? "✓" : "✗"
            print("  [\(status)] \(component.displayName)")
        }

        let vaeComponent = ModelRegistry.ModelComponent.vae(.standard)
        let vaeStatus = ModelRegistry.isDownloaded(vaeComponent) ? "✓" : "✗"
        print("  [\(vaeStatus)] \(vaeComponent.displayName)")
    }
}

// MARK: - Helper Functions

func loadImage(from path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }
    return image
}

func saveImage(_ image: CGImage, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    let utType: CFString = path.hasSuffix(".png") ? UTType.png.identifier as CFString : UTType.jpeg.identifier as CFString
    let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        utType,
        1,
        nil
    )

    guard let dest = destination else {
        throw Flux2Error.imageProcessingFailed("Failed to create image destination")
    }

    CGImageDestinationAddImage(dest, image, nil)

    guard CGImageDestinationFinalize(dest) else {
        throw Flux2Error.imageProcessingFailed("Failed to write image")
    }
}
