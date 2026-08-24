// TestGemma4.swift — `flux2 test-gemma4` (Gemma 4 E2B as the VLM provider)
// Copyright 2025 Vincent Gourbin
//
// The `test-qwen35` counterpart for the opt-in Gemma 4 provider. Same three
// modes (text-only, single-image analysis, two-image FLUX.2 comparison) so the
// two providers can be compared on the exact same prompts — which is the whole
// point of routing them through `FluxVLMProvider`.

import Foundation
import ArgumentParser
import CoreGraphics
import ImageIO
import FluxTextEncoders
import FluxGemma4VLM

struct TestGemma4: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test-gemma4",
        abstract: "Test Gemma 4 E2B as the VLM provider: analyze images, compare two images, or generate text"
    )

    @Argument(help: "Text prompt")
    var prompt: String

    @Option(name: .shortAndLong, help: "Image path to analyze (omit for text-only)")
    var image: String?

    @Option(name: .long, help: "Second image path (for --compare)")
    var image2: String?

    @Flag(name: .long, help: "Compare two images with the framework's FLUX.2 scene/style rubric (requires --image and --image2)")
    var compare: Bool = false

    @Option(name: .long, help: "Variant: 6bit (default), 4bit, 8bit, bf16")
    var variant: String = "6bit"

    @Option(name: .long, help: "Local path to Gemma 4 weights (directory with config.json + safetensors + tokenizer.json). Takes precedence over --variant.")
    var modelPath: String?

    @Option(name: .long, help: "Custom Gemma weights cache directory (layout: {org}/{model})")
    var modelsDir: String?

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 512

    @Flag(name: .long, help: "Ask for a reasoning pass before the answer (slower, needs a larger --max-tokens)")
    var think: Bool = false

    @Option(name: .long, help: "Temperature (0 = greedy, the framework default)")
    var temperature: Float = 0

    @Option(name: .long, help: "System prompt")
    var systemPrompt: String?

    @Flag(name: .long, help: "Use the framework's FLUX.2 image-description system prompt (what describeImageForFlux sends)")
    var fluxDescribe: Bool = false

    func run() async throws {
        let startTime = Date()

        guard let selectedVariant = Gemma4VLMProvider.Variant.fromCLIName(variant) else {
            throw ValidationError("Unsupported --variant '\(variant)' (use '4bit', '6bit', '8bit' or 'bf16')")
        }

        if let modelsDir {
            FluxGemma4VLM.setModelsDirectory(URL(fileURLWithPath: modelsDir))
        }

        print("=== Gemma 4 VLM Test ===")
        print("Prompt: \"\(prompt)\"")
        print("Mode: \(compare ? "comparison" : (image == nil ? "text-only" : "image analysis"))")
        print()

        print("Loading \(selectedVariant.displayName)...")
        let provider = try await FluxGemma4VLM.activate(
            variant: selectedVariant,
            modelPath: modelPath.map { URL(fileURLWithPath: $0) },
            progress: { print($0); fflush(stdout) }
        )
        print("Model loaded.\n")

        if compare {
            guard let img1 = image, let img2 = image2 else {
                throw ValidationError("--compare requires both --image and --image2")
            }
            let ref = try Self.loadImage(img1)
            let gen = try Self.loadImage(img2)
            print("--- FLUX.2 Image Comparison ---")
            let comparison = try await provider.compareImagesForFlux(reference: ref, generated: gen)
            print("Scene: \(comparison.sceneScore)/100 — \(comparison.sceneReason)")
            print("Style: \(comparison.styleScore)/100 — \(comparison.styleReason)")
            print("\nRaw: \(comparison.rawResponse)")
        } else {
            let effectiveSystemPrompt: String? = fluxDescribe
                ? FluxTextEncoders.fluxImageDescriptionSystemPrompt
                : systemPrompt
            let images: [CGImage] = try image.map { [try Self.loadImage($0)] } ?? []
            if let first = images.first {
                print("Image loaded: \(first.width)x\(first.height)")
            }
            print("--- Generation ---")
            let text = try await provider.generateText(
                images: images,
                prompt: prompt,
                systemPrompt: effectiveSystemPrompt,
                enableThinking: think,
                maxTokens: maxTokens,
                temperature: temperature
            )
            print(text)
        }

        print("\nTotal time: \(String(format: "%.1f", Date().timeIntervalSince(startTime)))s")
        await FluxGemma4VLM.deactivate()
    }

    private static func loadImage(_ path: String) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ValidationError("Failed to load image: \(path)")
        }
        return image
    }
}
