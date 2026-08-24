// FluxGemma4VLM.swift — Entry point for the opt-in Gemma 4 VLM provider
// Copyright 2025 Vincent Gourbin

import Foundation
import FluxTextEncoders
import Gemma4Swift

/// Opt-in namespace: activates Gemma 4 E2B as the VLM behind the framework's
/// enrichment functions (BFL prompt rewriting in the inpainting/outpainting
/// chains, `describeImageForFlux`, LoRA scene/style scoring, training captions).
///
/// Nothing here runs unless you call it. Until then — and after
/// ``deactivate()`` — ``FluxVLM/active`` is the bundled Qwen3.5.
public enum FluxGemma4VLM {

    /// Register Gemma 4 as the active provider and load its weights.
    ///
    /// - Parameters:
    ///   - variant: Quantisation of Gemma 4 E2B-it. Default `.e2b6bit` (~4.2 GB).
    ///   - modelPath: Load from this directory instead of downloading (what a
    ///     sandboxed app passes; the directory needs `config.json`,
    ///     `*.safetensors` and `tokenizer.json`).
    ///   - hfToken: HuggingFace token, for gated or private repos.
    ///   - progress: Load/download progress lines, delivered off the main actor.
    /// - Returns: The registered provider, in case you want to unload it later.
    @discardableResult
    public static func activate(
        variant: Gemma4VLMProvider.Variant = .e2b6bit,
        modelPath: URL? = nil,
        hfToken: String? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> Gemma4VLMProvider {
        let provider = Gemma4VLMProvider(
            variant: variant, modelPath: modelPath, hfToken: hfToken, progress: progress
        )
        try await provider.ensureLoaded()
        FluxVLM.register(provider)
        return provider
    }

    /// Register Gemma 4 as the active provider **without** loading it. The
    /// framework loads it on first use through `ensureLoaded()` — which is what
    /// the training paths (LoRA evaluation, VLM-guided checkpoint selection)
    /// expect, since they load and unload the VLM around each generation phase.
    @discardableResult
    public static func register(
        variant: Gemma4VLMProvider.Variant = .e2b6bit,
        modelPath: URL? = nil,
        hfToken: String? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) -> Gemma4VLMProvider {
        let provider = Gemma4VLMProvider(
            variant: variant, modelPath: modelPath, hfToken: hfToken, progress: progress
        )
        FluxVLM.register(provider)
        return provider
    }

    /// Unload Gemma (if it is the active provider) and hand the seat back to
    /// the bundled Qwen3.5.
    public static func deactivate() async {
        let active = FluxVLM.active
        FluxVLM.register(nil)
        if let gemma = active as? Gemma4VLMProvider {
            await gemma.unload()
        }
    }

    /// Point the Gemma weights cache somewhere else — e.g. the app's shared
    /// models directory. The layout inside is `{org}/{model}`, the same
    /// HuggingFace owner namespace the other model directories use, so
    /// `mlx-community/gemma-4-e2b-it-6bit` lands next to the FLUX checkpoints.
    ///
    /// Set this before the first `activate`/`ensureLoaded`.
    public static func setModelsDirectory(_ directory: URL?) {
        Gemma4ModelCache.customModelsDirectory = directory
    }

    /// Whether `variant`'s weights are already on disk (custom cache, or the
    /// default HuggingFace one).
    public static func isDownloaded(_ variant: Gemma4VLMProvider.Variant) -> Bool {
        Gemma4ModelCache.isDownloaded(variant.model)
    }
}
