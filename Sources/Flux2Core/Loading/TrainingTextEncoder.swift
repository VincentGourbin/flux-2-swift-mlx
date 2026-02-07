// TrainingTextEncoder.swift - Protocol for text encoders used in training
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Protocol for text encoders that can be used for LoRA training
///
/// Both KleinTextEncoder (for Klein 4B/9B) and DevTextEncoder (for Dev)
/// conform to this protocol.
public protocol TrainingTextEncoder: AnyObject, Sendable {
    /// Whether the model is loaded
    var isLoaded: Bool { get }

    /// Maximum sequence length for embeddings
    var maxSequenceLength: Int { get }

    /// Estimated memory usage in GB
    var estimatedMemoryGB: Int { get }

    /// Load the model (if not already loaded)
    @MainActor
    func load() async throws

    /// Encode a text prompt to embeddings for training
    /// - Parameter prompt: Text prompt to encode
    /// - Returns: Embeddings tensor with shape [1, maxSequenceLength, embedDim]
    func encodeForTraining(_ prompt: String) throws -> MLXArray

    /// Unload the model to free memory
    @MainActor
    func unload()
}

// MARK: - KleinTextEncoder Conformance

extension KleinTextEncoder: TrainingTextEncoder {
    /// Load the model (protocol conformance wrapper)
    @MainActor
    public func load() async throws {
        try await load(from: nil)
    }

    /// Encode for training (no upsampling)
    public func encodeForTraining(_ prompt: String) throws -> MLXArray {
        return try encode(prompt, upsample: false)
    }
}

// MARK: - DevTextEncoder Conformance

extension DevTextEncoder: TrainingTextEncoder {
    /// Load the model (protocol conformance wrapper)
    @MainActor
    public func load() async throws {
        try await load(from: nil)
    }

    /// Encode for training
    public func encodeForTraining(_ prompt: String) throws -> MLXArray {
        return try encode(prompt)
    }
}
