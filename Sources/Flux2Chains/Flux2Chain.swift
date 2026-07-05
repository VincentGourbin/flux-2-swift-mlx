// Flux2Chain.swift — Chain protocol + shared types
// Copyright 2025 Vincent Gourbin
//
// A "chain" is a unit of work that wraps a `Flux2Pipeline` invocation with
// extra stages around it: preprocessing (e.g. external model output → image),
// per-step latent transforms (e.g. RePaint masked-inpainting blend), and/or
// postprocessing. Chains are how host apps compose FLUX.2 with adjacent
// models without forking the core pipeline.

import Foundation
import Flux2Core
import CoreGraphics

/// A composable, single-shot inference job that drives a `Flux2Pipeline`.
///
/// Implementations decide their own input contract — there is intentionally no
/// shared "input struct" because chains differ wildly in what they consume
/// (a prompt + mask, a reference image + camera deltas, …). What is shared is
/// the *output*: a `Flux2GenerationResult`, so chains compose with the rest of
/// the framework's reporting (`usedPrompt`, upsampling flag, …).
///
/// `run` is async and throwing so chains can perform IO (loading auxiliary
/// models, decoding images) without forcing callers onto a particular
/// concurrency model.
public protocol Flux2Chain: Sendable {
    /// Execute the chain end-to-end. Calls `loadModels` on the pipeline if
    /// necessary; loads LoRAs the chain declares; runs preprocessing,
    /// generation, and postprocessing in sequence.
    func run() async throws -> Flux2GenerationResult
}

/// Error cases shared by built-in chains. Concrete chains can extend this
/// or throw their own typed errors.
public enum Flux2ChainError: Error, CustomStringConvertible, LocalizedError, Sendable {
    case invalidInput(String)
    case missingArtifact(String)

    public var description: String {
        switch self {
        case .invalidInput(let msg): return "Invalid chain input: \(msg)"
        case .missingArtifact(let msg): return "Missing chain artifact: \(msg)"
        }
    }

    public var errorDescription: String? {
        description
    }
}
