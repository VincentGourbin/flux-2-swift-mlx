/**
 * TextEncoderModelRegistry.swift
 * Registry of available Mistral Small 3.2 and Qwen3 models for text encoders
 */

import Foundation

// MARK: - Model Type

/// Type of model architecture
public enum ModelType: String, CaseIterable, Codable, Sendable {
    case mistral = "mistral"
    case qwen3 = "qwen3"
    
    public var displayName: String {
        switch self {
        case .mistral: return "Mistral"
        case .qwen3: return "Qwen3"
        }
    }
}

// MARK: - Model Variant

public enum ModelVariant: String, CaseIterable, Codable, Sendable {
    case bf16 = "bf16"
    case mlx8bit = "8bit"
    case mlx6bit = "6bit"
    case mlx4bit = "4bit"

    public var displayName: String {
        switch self {
        case .bf16: return "Full Precision (BF16)"
        case .mlx8bit: return "8-bit Quantized"
        case .mlx6bit: return "6-bit Quantized"
        case .mlx4bit: return "4-bit Quantized"
        }
    }

    public var estimatedSize: String {
        switch self {
        case .bf16: return "~48GB"
        case .mlx8bit: return "~25GB"
        case .mlx6bit: return "~19GB"
        case .mlx4bit: return "~14GB"
        }
    }

    public var shortName: String {
        switch self {
        case .bf16: return "BF16"
        case .mlx8bit: return "8-bit"
        case .mlx6bit: return "6-bit"
        case .mlx4bit: return "4-bit"
        }
    }
}

// MARK: - Qwen3 Variant

/// Quantization variants for Qwen3 models
public enum Qwen3Variant: String, CaseIterable, Codable, Sendable {
    case qwen3_4B_8bit = "qwen3-4b-8bit"
    case qwen3_4B_4bit = "qwen3-4b-4bit"
    case qwen3_8B_8bit = "qwen3-8b-8bit"
    case qwen3_8B_4bit = "qwen3-8b-4bit"
    
    public var displayName: String {
        switch self {
        case .qwen3_4B_8bit: return "Qwen3 4B (8-bit)"
        case .qwen3_4B_4bit: return "Qwen3 4B (4-bit)"
        case .qwen3_8B_8bit: return "Qwen3 8B (8-bit)"
        case .qwen3_8B_4bit: return "Qwen3 8B (4-bit)"
        }
    }
    
    public var estimatedSize: String {
        switch self {
        case .qwen3_4B_8bit: return "~4GB"
        case .qwen3_4B_4bit: return "~2GB"
        case .qwen3_8B_8bit: return "~8GB"
        case .qwen3_8B_4bit: return "~4GB"
        }
    }
    
    public var shortName: String {
        switch self {
        case .qwen3_4B_8bit: return "4B-8bit"
        case .qwen3_4B_4bit: return "4B-4bit"
        case .qwen3_8B_8bit: return "8B-8bit"
        case .qwen3_8B_4bit: return "8B-4bit"
        }
    }
    
    /// Klein variant this Qwen3 model supports
    public var kleinVariant: KleinVariant {
        switch self {
        case .qwen3_4B_8bit, .qwen3_4B_4bit: return .klein4B
        case .qwen3_8B_8bit, .qwen3_8B_4bit: return .klein9B
        }
    }

    /// HuggingFace repository ID
    public var repoId: String {
        switch self {
        case .qwen3_4B_8bit: return "lmstudio-community/Qwen3-4B-MLX-8bit"
        case .qwen3_4B_4bit: return "lmstudio-community/Qwen3-4B-MLX-4bit"
        case .qwen3_8B_8bit: return "lmstudio-community/Qwen3-8B-MLX-8bit"
        case .qwen3_8B_4bit: return "lmstudio-community/Qwen3-8B-MLX-4bit"
        }
    }
}

// MARK: - Model Info

public struct ModelInfo: Codable, Sendable {
    public let id: String
    public let repoId: String
    public let name: String
    public let description: String
    public let variant: ModelVariant
    public let parameters: String
    public let modelType: ModelType
    
    public init(
        id: String,
        repoId: String,
        name: String,
        description: String,
        variant: ModelVariant,
        parameters: String,
        modelType: ModelType = .mistral
    ) {
        self.id = id
        self.repoId = repoId
        self.name = name
        self.description = description
        self.variant = variant
        self.parameters = parameters
        self.modelType = modelType
    }
}

// MARK: - Qwen3 Model Info

public struct Qwen3ModelInfo: Codable, Sendable {
    public let id: String
    public let repoId: String
    public let name: String
    public let description: String
    public let variant: Qwen3Variant
    public let parameters: String

    /// Display name (alias for name)
    public var displayName: String { name }

    public init(
        id: String,
        repoId: String,
        name: String,
        description: String,
        variant: Qwen3Variant,
        parameters: String
    ) {
        self.id = id
        self.repoId = repoId
        self.name = name
        self.description = description
        self.variant = variant
        self.parameters = parameters
    }
}

// MARK: - Model Registry

@MainActor
public final class TextEncoderModelRegistry {
    public static let shared = TextEncoderModelRegistry()

    private var models: [ModelInfo] = []
    private var qwen3Models: [Qwen3ModelInfo] = []

    private init() {
        registerDefaultModels()
        registerQwen3Models()
    }

    private func registerDefaultModels() {
        // Mistral Small 3.2 models
        // Quantized versions from lmstudio-community include VLM (vision) layers
        models = [
            ModelInfo(
                id: "mistral-small-3.2-bf16",
                repoId: "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                name: "Mistral Small 3.2 (BF16)",
                description: "Original full precision model from Mistral AI - reference quality",
                variant: .bf16,
                parameters: "24B",
                modelType: .mistral
            ),
            ModelInfo(
                id: "mistral-small-3.2-8bit",
                repoId: "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit",
                name: "Mistral Small 3.2 (8-bit)",
                description: "8-bit quantized with VLM layers, good balance of quality and memory",
                variant: .mlx8bit,
                parameters: "24B",
                modelType: .mistral
            ),
            ModelInfo(
                id: "mistral-small-3.2-6bit",
                repoId: "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit",
                name: "Mistral Small 3.2 (6-bit)",
                description: "6-bit quantized with VLM layers, balanced compression",
                variant: .mlx6bit,
                parameters: "24B",
                modelType: .mistral
            ),
            ModelInfo(
                id: "mistral-small-3.2-4bit",
                repoId: "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
                name: "Mistral Small 3.2 (4-bit)",
                description: "4-bit quantized with VLM layers, memory efficient",
                variant: .mlx4bit,
                parameters: "24B",
                modelType: .mistral
            ),
        ]
    }
    
    private func registerQwen3Models() {
        // Qwen3 models for FLUX.2 Klein
        qwen3Models = [
            Qwen3ModelInfo(
                id: "qwen3-4b-8bit",
                repoId: "lmstudio-community/Qwen3-4B-MLX-8bit",
                name: "Qwen3 4B (8-bit)",
                description: "8-bit quantized Qwen3 4B for Klein 4B embeddings",
                variant: .qwen3_4B_8bit,
                parameters: "4B"
            ),
            Qwen3ModelInfo(
                id: "qwen3-4b-4bit",
                repoId: "lmstudio-community/Qwen3-4B-MLX-4bit",
                name: "Qwen3 4B (4-bit)",
                description: "4-bit quantized Qwen3 4B for Klein 4B embeddings (memory efficient)",
                variant: .qwen3_4B_4bit,
                parameters: "4B"
            ),
            Qwen3ModelInfo(
                id: "qwen3-8b-8bit",
                repoId: "lmstudio-community/Qwen3-8B-MLX-8bit",
                name: "Qwen3 8B (8-bit)",
                description: "8-bit quantized Qwen3 8B for Klein 9B embeddings",
                variant: .qwen3_8B_8bit,
                parameters: "8B"
            ),
            Qwen3ModelInfo(
                id: "qwen3-8b-4bit",
                repoId: "lmstudio-community/Qwen3-8B-MLX-4bit",
                name: "Qwen3 8B (4-bit)",
                description: "4-bit quantized Qwen3 8B for Klein 9B embeddings (memory efficient)",
                variant: .qwen3_8B_4bit,
                parameters: "8B"
            ),
        ]
    }

    /// URL for downloading tekken.json from the original Mistral repo
    public static let tekkenJsonURL = "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/resolve/main/tekken.json"

    // MARK: - Mistral Models
    
    public func allModels() -> [ModelInfo] {
        return models
    }

    public func model(withId id: String) -> ModelInfo? {
        return models.first { $0.id == id }
    }

    public func model(withVariant variant: ModelVariant) -> ModelInfo? {
        return models.first { $0.variant == variant }
    }

    public func defaultModel() -> ModelInfo {
        return model(withVariant: .mlx8bit) ?? models[0]
    }
    
    // MARK: - Qwen3 Models
    
    public func allQwen3Models() -> [Qwen3ModelInfo] {
        return qwen3Models
    }
    
    public func qwen3Model(withId id: String) -> Qwen3ModelInfo? {
        return qwen3Models.first { $0.id == id }
    }
    
    public func qwen3Model(withVariant variant: Qwen3Variant) -> Qwen3ModelInfo? {
        return qwen3Models.first { $0.variant == variant }
    }
    
    /// Get the recommended Qwen3 model for a Klein variant
    public func qwen3Model(forKlein variant: KleinVariant) -> Qwen3ModelInfo? {
        switch variant {
        case .klein4B: return qwen3Model(withVariant: .qwen3_4B_8bit)
        case .klein9B: return qwen3Model(withVariant: .qwen3_8B_8bit)
        }
    }
    
    public func defaultQwen3Model() -> Qwen3ModelInfo {
        return qwen3Model(withVariant: .qwen3_4B_8bit) ?? qwen3Models[0]
    }

    // MARK: - Print Methods
    
    public func printAvailableModels() {
        print("\nAvailable Mistral Small 3.2 Models:")
        print("=".padding(toLength: 70, withPad: "=", startingAt: 0))

        for model in models {
            let isDownloaded = TextEncoderModelDownloader.isModelDownloaded(model)
            let status = isDownloaded ? "Downloaded" : "Not downloaded"

            print("\n  \(model.name)")
            print("  ID: \(model.id)")
            print("  Repo: \(model.repoId)")
            print("  Size: \(model.variant.estimatedSize)")
            print("  Status: \(status)")

            if isDownloaded, let path = TextEncoderModelDownloader.findModelPath(for: model) {
                print("  Path: \(path.path)")
            }
        }

        print("\n" + "=".padding(toLength: 70, withPad: "=", startingAt: 0))
    }
    
    public func printAvailableQwen3Models() {
        print("\nAvailable Qwen3 Models (for FLUX.2 Klein):")
        print("=".padding(toLength: 70, withPad: "=", startingAt: 0))
        
        for model in qwen3Models {
            print("\n  \(model.name)")
            print("  ID: \(model.id)")
            print("  Repo: \(model.repoId)")
            print("  Size: \(model.variant.estimatedSize)")
            print("  For: \(model.variant.kleinVariant.displayName)")
        }
        
        print("\n" + "=".padding(toLength: 70, withPad: "=", startingAt: 0))
    }
}
