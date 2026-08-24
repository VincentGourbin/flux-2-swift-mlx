// Qwen35VLMProvider.swift — The bundled Qwen3.5 VLM as a FluxVLMProvider
// Copyright 2025 Vincent Gourbin
//
// This is the default provider, and it changes nothing about how Qwen3.5 runs:
// it forwards to the same `FluxTextEncoders.shared` entry points the enrichment
// functions called directly before the abstraction existed (same weights, same
// download path, same `Qwen35VLM.generateMultiImage` forward). Its only job is
// to present them behind `FluxVLMProvider` so a second VLM can take the same
// seat.

import Foundation
import CoreGraphics
import MLX

/// The bundled Qwen3.5 4B VLM, exposed as a provider.
///
/// `ensureLoaded()` downloads the configured variant through
/// `TextEncoderModelDownloader` and loads it into `FluxTextEncoders.shared`,
/// which is exactly what the training/evaluation paths did inline.
public final class Qwen35VLMProvider: FluxVLMProvider {

    /// Process-wide default instance (4-bit, matching what training and the
    /// chains have always downloaded).
    public static let shared = Qwen35VLMProvider()

    /// Variant `ensureLoaded()` fetches. Register a custom instance to change
    /// it: `FluxVLM.register(Qwen35VLMProvider(variant: .qwen35_4B_8bit))`.
    public let variant: Qwen35Variant

    private let hfToken: String?

    public init(variant: Qwen35Variant = .qwen35_4B_4bit, hfToken: String? = nil) {
        self.variant = variant
        self.hfToken = hfToken
    }

    public var displayName: String { "Qwen3.5 4B (\(variant.shortName))" }

    public var isLoaded: Bool { FluxTextEncoders.shared.isQwen35VLMLoaded }

    public func ensureLoaded() async throws {
        guard !isLoaded else { return }
        let downloader = TextEncoderModelDownloader(hfToken: hfToken)
        let path = try await downloader.downloadQwen35(variant: variant)
        try await FluxTextEncoders.shared.loadQwen35VLM(from: path.path)
    }

    public func unload() async {
        FluxTextEncoders.shared.unloadQwen35VLM()
    }

    public func generateText(
        images: [CGImage],
        prompt: String,
        systemPrompt: String?,
        enableThinking: Bool,
        maxTokens: Int,
        temperature: Float
    ) async throws -> String {
        guard isLoaded else {
            throw FluxEncoderError.invalidInput("Qwen3.5 VLM not loaded")
        }
        // The forward is synchronous and takes seconds on M-series. Run it off
        // the cooperative pool so UI updates and concurrent chains aren't
        // starved while we wait (same reasoning as Flux2VLMPromptBuilder).
        return try await Task.detached(priority: .userInitiated) {
            try FluxTextEncoders.shared.generateWithQwen35(
                images: images,
                prompt: prompt,
                systemPrompt: systemPrompt,
                enableThinking: enableThinking,
                maxTokens: maxTokens,
                temperature: temperature
            ).text
        }.value
    }
}
