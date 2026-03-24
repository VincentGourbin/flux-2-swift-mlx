// TestVLGeneration.swift - Test Qwen3-VL text generation standalone
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import FluxTextEncoders
import MLX

struct TestVLGeneration: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test-vl",
        abstract: "Test Qwen3-VL text generation (standalone, no image generation)"
    )

    @Argument(help: "Text prompt to generate from")
    var prompt: String

    @Option(name: .long, help: "Qwen3-VL variant: vl-4b-8bit (default), vl-4b-4bit, vl-8b-8bit, vl-8b-4bit")
    var vlVariant: String = "vl-4b-8bit"

    @Option(name: .long, help: "Local path to Qwen3-VL model (if not set, auto-downloads)")
    var vlModelPath: String?

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 200

    @Option(name: .long, help: "Temperature (0 = greedy)")
    var temperature: Float = 0.7

    func run() async throws {
        // Parse VL variant
        guard let variant = Qwen3VLVariant(rawValue: "qwen3\(vlVariant)") else {
            throw ValidationError("Invalid VL variant: \(vlVariant). Use vl-4b-8bit, vl-4b-4bit, vl-8b-8bit, vl-8b-4bit")
        }

        let kleinVariant = variant.kleinVariant

        print("=== Qwen3-VL Text Generation Test ===")
        print("Variant: \(variant.displayName)")
        print("Prompt: \"\(prompt)\"")
        print("Max tokens: \(maxTokens)")
        print("Temperature: \(temperature)")
        print()

        // Load model
        print("Loading \(variant.displayName)...")
        if let path = vlModelPath {
            try await FluxTextEncoders.shared.loadKleinVLModel(variant: kleinVariant, from: path)
        } else {
            try await FluxTextEncoders.shared.loadKleinVLModel(
                variant: kleinVariant,
                qwen3VLVariant: variant
            ) { progress, message in
                print("\r  [\(Int(progress * 100))%] \(message)", terminator: "")
                fflush(stdout)
            }
            print()
        }
        print("Model loaded.\n")

        // Generate
        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.9
        )

        print("--- Generation ---")
        let result = try FluxTextEncoders.shared.generateQwen3VL(
            prompt: prompt,
            parameters: params
        ) { token in
            print(token, terminator: "")
            fflush(stdout)
            return true
        }
        print("\n")

        print("--- Stats ---")
        print("Prompt tokens: \(result.promptTokens)")
        print("Generated tokens: \(result.generatedTokens)")
        print("Time: \(String(format: "%.1f", result.totalTime))s")
        print("Speed: \(String(format: "%.1f", result.tokensPerSecond)) tok/s")

        // Cleanup
        await MainActor.run { FluxTextEncoders.shared.unloadKleinModel() }
    }
}
