// FluxVLMProvider.swift — Provider abstraction for the VLM enrichment services
// Copyright 2025 Vincent Gourbin
//
// Why this exists:
// Every image-aware convenience in this framework — the BFL prompt rewriter of
// the inpainting/outpainting chains, `describeImageForFlux`, the 0-100
// scene/style scoring used by the LoRA evaluator and by VLM-guided checkpoint
// selection, the training captioner — used to call the bundled Qwen3.5 VLM
// directly. That hardcoding meant a second VLM could not serve those functions
// without duplicating each call site.
//
// `FluxVLMProvider` is the one primitive all of them actually need: N images
// (0, 1 or 2) + a user prompt + a system prompt -> text. The system prompts and
// the response parsing stay HERE, shared, so a provider only has to know how to
// run a forward. `FluxVLM.active` selects which one runs; it defaults to the
// bundled Qwen3.5, so nothing changes for callers that never register anything.
//
// Adding a provider therefore means implementing four members, not re-deriving
// the prompt engineering. See `FluxGemma4VLM` (Gemma 4 E2B) for the second one.

import Foundation
import CoreGraphics

/// A VLM able to serve the framework's enrichment functions (prompt rewriting,
/// FLUX.2 image description, scene/style scoring, training captions).
///
/// Concurrency: implementations are shared, long-lived objects reached from any
/// task, hence `Sendable`. `ensureLoaded()`/`unload()` must be serialised by the
/// caller (the framework always does load → use → unload in sequence), exactly
/// like `FluxTextEncoders` itself.
public protocol FluxVLMProvider: AnyObject, Sendable {

    /// Short identifier used in logs and CLI output (e.g. `"qwen3.5-4b-4bit"`).
    var displayName: String { get }

    /// Whether weights are currently resident.
    var isLoaded: Bool { get }

    /// Download (if needed) and load this provider's weights. A no-op when
    /// `isLoaded` is already true. Callers that manage weights themselves
    /// (sandboxed apps loading from an explicit path) can implement this as a
    /// no-op and load ahead of time.
    func ensureLoaded() async throws

    /// Free the weights. Must be safe to call when nothing is loaded.
    func unload() async

    /// The single generation primitive.
    ///
    /// - Parameters:
    ///   - images: 0 images = text-only, 1 = image analysis, 2+ = comparison.
    ///     Providers must present them to the model in the given order.
    ///   - prompt: The user turn.
    ///   - systemPrompt: Instructions turn; `nil` uses the provider's default.
    ///   - enableThinking: Ask for a reasoning pass before the answer. The
    ///     framework's enrichment paths all pass `false` — they parse the
    ///     output. Providers must strip any reasoning channel from the result
    ///     either way, so the returned string is always the answer alone.
    ///   - maxTokens: Generation budget.
    ///   - temperature: `0` = greedy, which every framework path uses.
    /// - Returns: The answer text, control tokens and reasoning channel removed.
    func generateText(
        images: [CGImage],
        prompt: String,
        systemPrompt: String?,
        enableThinking: Bool,
        maxTokens: Int,
        temperature: Float
    ) async throws -> String
}

// MARK: - Shared enrichment services

extension FluxVLMProvider {

    /// Describe an image so FLUX.2 can recreate it (scene + style), using the
    /// shared system prompt in
    /// ``FluxTextEncoders/fluxImageDescriptionSystemPrompt``.
    public func describeImageForFlux(
        image: CGImage,
        context: String? = nil,
        maxTokens: Int = 300
    ) async throws -> String {
        try await generateText(
            images: [image],
            prompt: context ?? "Describe this image.",
            systemPrompt: FluxTextEncoders.fluxImageDescriptionSystemPrompt,
            enableThinking: false,
            maxTokens: maxTokens,
            temperature: 0
        )
    }

    /// Score a generated image against a reference on scene and style (0-100).
    ///
    /// - Parameter systemPrompt: Defaults to the framework's comparison rubric;
    ///   callers with training context (LoRA name/goal) pass their own, which is
    ///   why this is a parameter rather than a constant.
    public func compareImagesForFlux(
        reference: CGImage,
        generated: CGImage,
        systemPrompt: String? = nil,
        prompt: String = "Compare these two images.",
        maxTokens: Int = 300
    ) async throws -> FluxTextEncoders.FluxImageComparison {
        let text = try await generateText(
            images: [reference, generated],
            prompt: prompt,
            systemPrompt: systemPrompt ?? FluxTextEncoders.fluxImageComparisonSystemPrompt,
            enableThinking: false,
            maxTokens: maxTokens,
            temperature: 0
        )
        return FluxTextEncoders.shared.parseComparisonForEvaluation(text)
    }

    /// Free-form single-image analysis (the prompt rewriters use this).
    public func analyzeImage(
        image: CGImage,
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 220
    ) async throws -> String {
        try await generateText(
            images: [image],
            prompt: prompt,
            systemPrompt: systemPrompt,
            enableThinking: false,
            maxTokens: maxTokens,
            temperature: 0
        )
    }

    /// Text-only generation (LoRA context analysis uses this).
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 256
    ) async throws -> String {
        try await generateText(
            images: [],
            prompt: prompt,
            systemPrompt: systemPrompt,
            enableThinking: false,
            maxTokens: maxTokens,
            temperature: 0
        )
    }
}

// MARK: - Registry

/// Selects which ``FluxVLMProvider`` the framework's enrichment functions run
/// on.
///
/// The default is the bundled Qwen3.5 (``Qwen35VLMProvider/shared``), so a
/// process that registers nothing behaves exactly as before this abstraction
/// existed. Register once at startup — the framework reads `active` on every
/// enrichment call and never writes it.
///
/// ```swift
/// import FluxGemma4VLM
/// try await FluxGemma4VLM.activate(variant: .e2b6bit)   // registers + loads
/// ```
public enum FluxVLM {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var registered: (any FluxVLMProvider)?

    /// The provider serving enrichment calls: whatever was registered, else the
    /// bundled Qwen3.5.
    public static var active: any FluxVLMProvider {
        lock.withLock { registered ?? Qwen35VLMProvider.shared }
    }

    /// Whether a provider other than the bundled Qwen3.5 is in charge.
    public static var hasCustomProvider: Bool {
        lock.withLock { registered != nil }
    }

    /// Install `provider` as the active one. Pass `nil` to fall back to the
    /// bundled Qwen3.5. Does not load or unload anything.
    public static func register(_ provider: (any FluxVLMProvider)?) {
        lock.withLock { registered = provider }
    }
}
