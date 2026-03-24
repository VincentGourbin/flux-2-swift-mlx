// TestQwen35.swift - Test Qwen3.5 VLM (image analysis + text generation)
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import MLX
import ImageIO

struct TestQwen35: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test-qwen35",
        abstract: "Test Qwen3.5 VLM: analyze images or generate text"
    )

    @Argument(help: "Text prompt")
    var prompt: String

    @Option(name: .shortAndLong, help: "Image path to analyze (omit for text-only)")
    var image: String?

    @Option(name: .long, help: "Second image path (for comparison mode)")
    var image2: String?

    @Flag(name: .long, help: "Compare two images using FLUX.2 criteria (requires --image and --image2)")
    var compare: Bool = false

    @Option(name: .long, help: "Local path to Qwen3.5 model (if not set, auto-downloads)")
    var modelPath: String?

    @Option(name: .long, help: "Model variant: 8bit (default), 4bit")
    var variant: String = "8bit"

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 512

    @Option(name: .long, help: "Temperature (0 = greedy)")
    var temperature: Float = 0.7

    @Option(name: .long, help: "System prompt (use 'flux' for FLUX.2 image description mode)")
    var systemPrompt: String?

    @Flag(name: .long, help: "Use FLUX.2-optimized image description system prompt")
    var fluxDescribe: Bool = false

    func run() async throws {
        let startTime = Date()

        print("=== Qwen3.5 VLM Test ===")
        print("Prompt: \"\(prompt)\"")
        if let imgPath = image {
            print("Image: \(imgPath)")
        } else {
            print("Mode: text-only")
        }
        print()

        // Load model
        print("Loading Qwen3.5 VLM...")
        if let path = modelPath {
            try await FluxTextEncoders.shared.loadQwen35VLM(from: path)
        } else {
            // Auto-download
            let selectedVariant: Qwen35Variant = variant == "4bit" ? .qwen35_4B_4bit : .qwen35_4B_8bit
            print("Variant: \(selectedVariant.displayName)")
            let downloader = TextEncoderModelDownloader()
            let path = try await downloader.downloadQwen35(variant: selectedVariant) { progress, message in
                print("\r  [\(Int(progress * 100))%] \(message)", terminator: "")
                fflush(stdout)
            }
            print()
            try await FluxTextEncoders.shared.loadQwen35VLM(from: path.path)
        }
        print("Model loaded.\n")

        // Load image if provided
        var cgImage: CGImage? = nil
        if let imgPath = image {
            guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: imgPath) as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw ValidationError("Failed to load image: \(imgPath)")
            }
            cgImage = img
            print("Image loaded: \(img.width)x\(img.height)")
        }

        // === Comparison mode ===
        if compare {
            guard let img1Path = image, let img2Path = image2 else {
                throw ValidationError("Comparison mode requires both --image and --image2")
            }
            guard cgImage != nil else {
                throw ValidationError("Failed to load first image")
            }
            guard let source2 = CGImageSourceCreateWithURL(URL(fileURLWithPath: img2Path) as CFURL, nil),
                  let img2 = CGImageSourceCreateImageAtIndex(source2, 0, nil) else {
                throw ValidationError("Failed to load second image: \(img2Path)")
            }
            print("Image 2 loaded: \(img2.width)x\(img2.height)")
            print("\n--- FLUX.2 Image Comparison ---")

            let comparison = try FluxTextEncoders.shared.compareImagesForFlux(
                reference: cgImage!,
                generated: img2
            ) { token in
                print(token, terminator: "")
                fflush(stdout)
                return true
            }

            print("\n\n--- Scores ---")
            print("Scene: \(comparison.sceneScore)/10 — \(comparison.sceneReason)")
            print("Style: \(comparison.styleScore)/10 — \(comparison.styleReason)")

            let totalElapsed = Date().timeIntervalSince(startTime)
            print("\nTotal time: \(String(format: "%.1f", totalElapsed))s")

            await MainActor.run { FluxTextEncoders.shared.unloadQwen35VLM() }
            return
        }

        // Resolve system prompt
        let effectiveSystemPrompt: String?
        if fluxDescribe {
            effectiveSystemPrompt = FluxTextEncoders.fluxImageDescriptionSystemPrompt
            print("Mode: FLUX.2 image description")
        } else {
            effectiveSystemPrompt = systemPrompt
        }

        // Generate
        print("--- Generation ---")
        let result: GenerationResult
        if let img = cgImage {
            result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
                image: img,
                prompt: prompt,
                systemPrompt: effectiveSystemPrompt,
                maxTokens: maxTokens,
                temperature: temperature
            ) { token in
                print(token, terminator: "")
                fflush(stdout)
                return true
            }
        } else {
            result = try FluxTextEncoders.shared.generateWithQwen35(
                prompt: prompt,
                systemPrompt: effectiveSystemPrompt,
                maxTokens: maxTokens,
                temperature: temperature
            ) { token in
                print(token, terminator: "")
                fflush(stdout)
                return true
            }
        }
        print("\n")

        // Stats
        print("--- Stats ---")
        print("Prompt tokens: \(result.promptTokens)")
        print("Generated tokens: \(result.generatedTokens)")
        print("Time: \(String(format: "%.1f", result.totalTime))s")
        print("Speed: \(String(format: "%.1f", result.tokensPerSecond)) tok/s")

        let totalElapsed = Date().timeIntervalSince(startTime)
        print("Total time: \(String(format: "%.1f", totalElapsed))s")

        // Cleanup
        await MainActor.run { FluxTextEncoders.shared.unloadQwen35VLM() }
    }
}
