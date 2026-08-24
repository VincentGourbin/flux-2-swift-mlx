// Gemma4VLMProvider.swift — Gemma 4 E2B as a FluxVLMProvider
// Copyright 2025 Vincent Gourbin
//
// Why this target exists separately:
// Gemma4Swift brings `mlx-swift-lm` (MLXLLM/MLXLMCommon) into the package
// graph. The base libraries — FluxTextEncoders, Flux2Core, Flux2Chains — must
// stay free of it, so this provider lives in its own opt-in target. Link
// `FluxGemma4VLM` and call `FluxGemma4VLM.activate(...)`; link nothing and the
// framework keeps running on the bundled Qwen3.5, byte for byte.
//
// What it does: implements the four `FluxVLMProvider` members on top of
// `Gemma4Pipeline` (the same stack ltx-video-swift-mlx uses for its LTX-2.5
// prompt enhancer). The system prompts and the response parsing are NOT here —
// they are shared in FluxTextEncoders, so Gemma answers the exact same
// questions Qwen3.5 does.

import CoreGraphics
import Foundation
import FluxTextEncoders
import Gemma4Swift
import MLX

/// Gemma 4 E2B (2.3B effective, text+vision) serving the framework's VLM
/// enrichment functions.
///
/// ```swift
/// // Auto-download into the Gemma cache and take over from Qwen3.5:
/// try await FluxGemma4VLM.activate(variant: .e2b6bit)
///
/// // Sandboxed app with its own weights directory:
/// try await FluxGemma4VLM.activate(modelPath: URL(fileURLWithPath: "…/gemma-4-e2b-it-6bit"))
/// ```
public final class Gemma4VLMProvider: FluxVLMProvider, @unchecked Sendable {

    // MARK: - Variants

    /// The E2B family only: it is the smallest Gemma 4 that carries a vision
    /// tower, which keeps it in the same memory class as the Qwen3.5 4B it
    /// replaces. Larger families (E4B, 31B, 26B-A4B) work through
    /// ``Gemma4VLMProvider/init(model:modelPath:hfToken:)`` if you have the RAM.
    public enum Variant: String, CaseIterable, Sendable, Codable {
        case e2b4bit
        case e2b6bit
        case e2b8bit
        case e2bBf16

        /// CLI/YAML spelling (`4bit`, `6bit`, `8bit`, `bf16`).
        public var cliName: String {
            switch self {
            case .e2b4bit: return "4bit"
            case .e2b6bit: return "6bit"
            case .e2b8bit: return "8bit"
            case .e2bBf16: return "bf16"
            }
        }

        public static func fromCLIName(_ name: String) -> Variant? {
            Variant.allCases.first { $0.cliName == name.lowercased() }
        }

        public var model: Gemma4Pipeline.Model {
            switch self {
            case .e2b4bit: return .e2b4bit
            case .e2b6bit: return .e2b6bit
            case .e2b8bit: return .e2b8bit
            case .e2bBf16: return .e2bBf16
            }
        }

        public var displayName: String { model.displayName }
        public var estimatedSizeGB: Float { model.estimatedSizeGB }
    }

    // MARK: - Configuration

    /// Model this provider loads when it has to fetch weights itself.
    public let model: Gemma4Pipeline.Model

    /// When set, `ensureLoaded()` loads from this directory instead of
    /// downloading — the path sandboxed apps need.
    public let modelPath: URL?

    private let hfToken: String?
    private let progress: (@Sendable (String) -> Void)?

    private let lock = NSLock()
    private var pipeline: Gemma4Pipeline?

    /// Default: Gemma 4 E2B-it 6-bit (~4.2 GB) — the closest match in size and
    /// latency to the 4-bit Qwen3.5 the enrichment paths have always used, with
    /// noticeably steadier instruction following than 4-bit on the
    /// "output ONLY the prompt" rules the BFL rewriters depend on.
    public init(
        variant: Variant = .e2b6bit,
        modelPath: URL? = nil,
        hfToken: String? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) {
        self.model = variant.model
        self.modelPath = modelPath
        self.hfToken = hfToken
        self.progress = progress
    }

    /// Escape hatch for a non-E2B family (E4B, 31B, 26B-A4B). Vision-capable
    /// families only — a text-only model cannot serve the image paths.
    public init(
        model: Gemma4Pipeline.Model,
        modelPath: URL? = nil,
        hfToken: String? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) {
        self.model = model
        self.modelPath = modelPath
        self.hfToken = hfToken
        self.progress = progress
    }

    // MARK: - FluxVLMProvider

    public var displayName: String { model.displayName }

    public var isLoaded: Bool {
        lock.withLock { pipeline != nil }
    }

    public func ensureLoaded() async throws {
        guard !isLoaded else { return }

        let created = await Gemma4Pipeline()
        if let modelPath {
            progress?("Loading \(model.displayName) from \(modelPath.path)…")
            // `resolvingSymlinksInPath` matters for HuggingFace snapshot
            // layouts, whose files are symlinks into `blobs/`.
            try await created.load(from: modelPath.resolvingSymlinksInPath(), multimodal: true)
        } else {
            progress?("Loading \(model.displayName) (downloading if needed, ~\(String(format: "%.1f", model.estimatedSizeGB)) GB)…")
            let sink = progress
            try await created.load(
                model,
                multimodal: true,
                downloadIfNeeded: true,
                hfToken: hfToken,
                progress: { p in
                    guard !p.currentFile.isEmpty else { return }
                    sink?("  \(p.currentFile)")
                }
            )
        }

        lock.withLock { pipeline = created }
        progress?("\(model.displayName) ready")
    }

    public func unload() async {
        let taken = lock.withLock { () -> Gemma4Pipeline? in
            let current = pipeline
            pipeline = nil
            return current
        }
        guard let taken else { return }
        await taken.unload()
        Memory.clearCache()
    }

    public func generateText(
        images: [CGImage],
        prompt: String,
        systemPrompt: String?,
        enableThinking: Bool,
        maxTokens: Int,
        temperature: Float
    ) async throws -> String {
        guard let current = lock.withLock({ pipeline }) else {
            throw Gemma4VLMProviderError.notLoaded
        }

        // Thinking is off unless asked: the enrichment paths parse the answer
        // (a one-line prompt, a JSON object), and a reasoning channel both
        // eats the token budget and has to be stripped back out.
        let templateVariables: [String: any Sendable]? =
            enableThinking ? ["enable_thinking": true] : nil

        let stream: AsyncThrowingStream<String, Error>
        if images.isEmpty {
            // A nil system prompt would make ChatSession fall back to its own
            // French default instructions; keep the framework's paths English.
            stream = try await current.chatStream(
                prompt: prompt,
                systemPrompt: systemPrompt ?? Self.defaultSystemPrompt,
                temperature: temperature,
                maxTokens: maxTokens,
                templateVariables: templateVariables
            )
        } else {
            nonisolated(unsafe) let pixels = try Self.batchedPixelValues(images)
            // `Gemma4Processor.multimodalChatIds` prepends exactly one
            // `<|image|>` marker to the user turn; images 2..N need their own,
            // and the pipeline refuses a marker/batch mismatch rather than
            // scattering random embeddings.
            let extraMarkers = String(
                repeating: Gemma4Processor.imageToken + "\n",
                count: images.count - 1
            )
            stream = try await current.chatStreamMultimodal(
                prompt: extraMarkers + prompt,
                pixelValues: pixels,
                systemPrompt: systemPrompt,
                temperature: temperature,
                maxTokens: maxTokens,
                templateVariables: templateVariables
            )
        }

        var text = ""
        for try await chunk in stream { text += chunk }
        return Self.cleanGeneratedText(text)
    }

    /// Neutral instructions for the text-only path when the caller gives none.
    public static let defaultSystemPrompt = "You are a helpful assistant."

    // MARK: - Image batching

    /// Pack N images into one `[N, 3, H, W]` batch.
    ///
    /// `Gemma4ImageProcessor` sizes each image to its own aspect-ratio-preserving
    /// box, so a comparison of a 1024×1024 reference against a 512×768 render
    /// yields two differently shaped tensors. They are zero-padded to the batch
    /// maximum, top-left aligned — the same thing `gemma4-cli describe` does for
    /// its multi-image mode.
    internal static func batchedPixelValues(_ images: [CGImage]) throws -> MLXArray {
        precondition(!images.isEmpty, "batchedPixelValues requires at least one image")
        let processed = try images.map { try Gemma4ImageProcessor.processImage($0) }
        if processed.count == 1 { return processed[0] }

        let maxH = processed.map { $0.dim(2) }.max()!
        let maxW = processed.map { $0.dim(3) }.max()!
        let padded = processed.map { pv -> MLXArray in
            let h = pv.dim(2), w = pv.dim(3)
            if h == maxH && w == maxW { return pv }
            let canvas = MLXArray.zeros([1, 3, maxH, maxW], dtype: pv.dtype)
            canvas[0..., 0..., 0 ..< h, 0 ..< w] = pv
            return canvas
        }
        return concatenated(padded, axis: 0)
    }

    // MARK: - Output cleanup

    /// Strip Gemma's reasoning channel and control tokens.
    ///
    /// With thinking on, the answer is preceded by
    /// `<|channel>thought … <channel|>`; the stream forwards it verbatim and the
    /// caller is expected to cut it. Keeping only what follows the LAST
    /// `<channel|>` is what the ltx enhancer does, and it also degrades
    /// correctly when thinking is off (no marker, nothing removed).
    internal static func cleanGeneratedText(_ raw: String) -> String {
        var text = raw
        if let close = text.range(of: "<channel|>", options: .backwards) {
            text = String(text[close.upperBound...])
        }
        for token in ["<|channel>thought", "<|think|>", "<end_of_turn>", "<start_of_turn>", "<eos>"] {
            text = text.replacingOccurrences(of: token, with: "")
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Errors

public enum Gemma4VLMProviderError: LocalizedError {
    case notLoaded

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "Gemma 4 VLM not loaded — call ensureLoaded() (or FluxGemma4VLM.activate) first."
        }
    }
}
