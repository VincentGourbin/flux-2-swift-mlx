// VLMProviderOptions.swift — Shared CLI surface for choosing + loading the VLM
// Copyright 2025 Vincent Gourbin
//
// The image-aware prompt rewriters (`inpaint --enrich-prompt-with-vlm`,
// `outpaint --enrich-prompt-with-vlm`) need a VLM resident in-process; the
// chains never auto-load one, on purpose (they must stay usable with no VLM at
// all, falling back to the caller's prompt). This option group is the one place
// that decides which provider runs and gets it loaded, so both commands offer
// the same flags and the same defaults.
//
// Backwards compatible: `--qwen35-variant` / `--qwen35-path` keep their exact
// previous meaning, and the default provider is still the bundled Qwen3.5.

import Foundation
import ArgumentParser
import FluxTextEncoders
import FluxGemma4VLM

/// Which VLM serves the framework's image-aware enrichment.
enum VLMProviderArg: String, ExpressibleByArgument, CaseIterable {
    /// Bundled Qwen3.5 4B (default — what every previous release used).
    case qwen35
    /// Gemma 4 E2B through the opt-in `FluxGemma4VLM` target.
    case gemma4
}

struct VLMProviderOptions: ParsableArguments {

    @Option(name: .long, help: "Which VLM serves --enrich-prompt-with-vlm: 'qwen35' (default, bundled Qwen3.5 4B) or 'gemma4' (Gemma 4 E2B-it). Both answer the same system prompts; Gemma is smaller and faster, Qwen3.5 is the historical default.")
    var vlmProvider: VLMProviderArg = .qwen35

    @Option(name: .long, help: "Qwen3.5 VLM variant to load in-process when --enrich-prompt-with-vlm is set: '8bit' (5 GB, recommended) or '4bit' (3 GB, faster but lower quality). Auto-downloads if missing. Omit (and omit --qwen35-path) to skip loading — the chain then falls back to --prompt verbatim with a warning.")
    var qwen35Variant: String?

    @Option(name: .long, help: "Override the local path to Qwen3.5 VLM weights (alternative to --qwen35-variant for sandboxed apps).")
    var qwen35Path: String?

    @Option(name: .long, help: "Gemma 4 E2B-it variant when --vlm-provider gemma4: '6bit' (default, ~4.2 GB), '4bit' (~3.6 GB), '8bit' (~5.2 GB) or 'bf16' (~10 GB). Auto-downloads if missing.")
    var gemma4Variant: String?

    @Option(name: .long, help: "Override the local path to Gemma 4 weights (directory with config.json + safetensors + tokenizer.json). Takes precedence over --gemma4-variant.")
    var gemma4Path: String?

    /// Parse `--qwen35-variant`. Returns `nil` for an unknown spelling.
    private static func qwenVariant(named name: String) -> Qwen35Variant? {
        switch name.lowercased() {
        case "4bit": return .qwen35_4B_4bit
        case "8bit": return .qwen35_4B_8bit
        default: return nil
        }
    }

    /// Provider for an explicitly requested Qwen variant. `nil` (an unknown
    /// spelling, which `loadIfRequested` rejects right after, or no variant at
    /// all) falls back to `Qwen35VLMProvider.shared` through the registry.
    private static func qwenProvider(forVariant name: String) -> Qwen35VLMProvider? {
        qwenVariant(named: name).map { Qwen35VLMProvider(variant: $0) }
    }

    /// Load the selected provider and register it as `FluxVLM.active`.
    ///
    /// - Returns: `true` when a VLM is resident afterwards, `false` when the
    ///   caller asked for enrichment but gave no weights (the chain will then
    ///   fall back to the verbatim prompt — the commands warn about it).
    @discardableResult
    func loadIfRequested(
        enrichmentRequested: Bool,
        logErr: @escaping @Sendable (String) -> Void
    ) async throws -> Bool {
        switch vlmProvider {
        case .qwen35:
            // Make the flag authoritative rather than relying on whatever the
            // process last registered: `--vlm-provider qwen35` must put the
            // bundled Qwen3.5 in the seat, with the variant the user asked for
            // (so a later `ensureLoaded()` fetches that one, not the default).
            FluxVLM.register(qwen35Variant.flatMap(Self.qwenProvider(forVariant:)))

            if let qwen35Path {
                logErr("Loading Qwen3.5 VLM from \(qwen35Path) ...")
                try await FluxTextEncoders.shared.loadQwen35VLM(from: qwen35Path)
                logErr("✓ Qwen3.5 VLM loaded")
                return true
            }
            guard let variantStr = qwen35Variant else {
                if enrichmentRequested {
                    logErr("WARNING: --enrich-prompt-with-vlm is set but neither --qwen35-variant nor --qwen35-path was provided — the chain will fall back to --prompt verbatim.")
                }
                return false
            }
            guard let selectedVariant = Self.qwenVariant(named: variantStr) else {
                throw ValidationError("Unsupported --qwen35-variant '\(variantStr)' (use '8bit' or '4bit')")
            }
            logErr("Downloading/loading Qwen3.5 VLM (\(selectedVariant.displayName)) ...")
            let downloader = TextEncoderModelDownloader()
            let path = try await downloader.downloadQwen35(variant: selectedVariant) { progress, message in
                logErr("  [\(Int(progress * 100))%] \(message)")
            }
            try await FluxTextEncoders.shared.loadQwen35VLM(from: path.path)
            logErr("✓ Qwen3.5 VLM loaded")
            return true

        case .gemma4:
            let variant: Gemma4VLMProvider.Variant
            if let gemma4Variant {
                guard let parsed = Gemma4VLMProvider.Variant.fromCLIName(gemma4Variant) else {
                    throw ValidationError("Unsupported --gemma4-variant '\(gemma4Variant)' (use '4bit', '6bit', '8bit' or 'bf16')")
                }
                variant = parsed
            } else {
                variant = .e2b6bit
            }
            let path = gemma4Path.map { URL(fileURLWithPath: $0) }
            logErr("Loading \(variant.displayName)\(path == nil ? "" : " from \(path!.path)") ...")
            try await FluxGemma4VLM.activate(
                variant: variant,
                modelPath: path,
                progress: { message in logErr("  \(message)") }
            )
            logErr("✓ \(FluxVLM.active.displayName) loaded and registered as the active VLM")
            return true
        }
    }
}
