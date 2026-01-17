// ModelRegistry.swift - Model variants and download sources
// Copyright 2025 Vincent Gourbin

import Foundation

/// Registry of available Flux.2 model variants
public enum ModelRegistry {

    // MARK: - Model Components

    /// Available Flux.2 transformer variants
    public enum TransformerVariant: String, CaseIterable, Sendable {
        case bf16 = "bf16"
        case qint8 = "qint8"
        case qint4 = "qint4"

        public var huggingFaceRepo: String {
            switch self {
            case .bf16:
                return "black-forest-labs/FLUX.2-dev"
            case .qint8, .qint4:
                return "VincentGOURBIN/flux_qint_8bit"
            }
        }

        /// Subfolder within the HuggingFace repo
        public var huggingFaceSubfolder: String? {
            switch self {
            case .bf16:
                return "transformer"
            case .qint8:
                return "flux-2-dev/transformer/qint8"
            case .qint4:
                return "flux-2-dev/transformer/qint4"
            }
        }

        public var estimatedSizeGB: Int {
            switch self {
            case .bf16: return 64
            case .qint8: return 32
            case .qint4: return 16
            }
        }

        public var quantization: TransformerQuantization {
            switch self {
            case .bf16: return .bf16
            case .qint8: return .qint8
            case .qint4: return .qint4
            }
        }
    }

    /// Available Mistral text encoder variants (from mistral-small-3.2-swift-mlx)
    public enum TextEncoderVariant: String, CaseIterable, Sendable {
        case bf16 = "bf16"
        case mlx8bit = "8bit"
        case mlx6bit = "6bit"
        case mlx4bit = "4bit"

        public var estimatedSizeGB: Int {
            switch self {
            case .bf16: return 48
            case .mlx8bit: return 25
            case .mlx6bit: return 19
            case .mlx4bit: return 14
            }
        }

        public var quantization: MistralQuantization {
            switch self {
            case .bf16: return .bf16
            case .mlx8bit: return .mlx8bit
            case .mlx6bit: return .mlx6bit
            case .mlx4bit: return .mlx4bit
            }
        }
    }

    /// VAE variant (only one available)
    public enum VAEVariant: String, CaseIterable, Sendable {
        case standard = "standard"

        public var huggingFaceRepo: String {
            "black-forest-labs/FLUX.2-dev"
        }

        public var estimatedSizeGB: Int { 3 }
    }

    // MARK: - Model Component Identifier

    /// Identifies a specific model component for download/status tracking
    public enum ModelComponent: Hashable, Sendable {
        case transformer(TransformerVariant)
        case textEncoder(TextEncoderVariant)
        case vae(VAEVariant)

        public var displayName: String {
            switch self {
            case .transformer(let variant):
                return "Flux.2 Transformer (\(variant.rawValue))"
            case .textEncoder(let variant):
                return "Mistral Small 3.2 (\(variant.rawValue))"
            case .vae:
                return "Flux.2 VAE"
            }
        }

        public var estimatedSizeGB: Int {
            switch self {
            case .transformer(let variant): return variant.estimatedSizeGB
            case .textEncoder(let variant): return variant.estimatedSizeGB
            case .vae(let variant): return variant.estimatedSizeGB
            }
        }

        public var localDirectoryName: String {
            switch self {
            case .transformer(let variant):
                return "flux2-transformer-\(variant.rawValue)"
            case .textEncoder(let variant):
                return "mistral-small-3.2-\(variant.rawValue)"
            case .vae:
                return "flux2-vae"
            }
        }
    }

    // MARK: - Paths

    /// Base directory for model storage (same as MistralCore: ~/Library/Caches/models)
    public static var modelsDirectory: URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("models", isDirectory: true)
    }

    /// Get the local path for a model component
    public static func localPath(for component: ModelComponent) -> URL {
        switch component {
        case .transformer(let variant):
            // Store as: models/black-forest-labs/FLUX.2-dev-transformer-{variant}
            return modelsDirectory
                .appendingPathComponent("black-forest-labs")
                .appendingPathComponent("FLUX.2-dev-transformer-\(variant.rawValue)")
        case .textEncoder(let variant):
            // Mistral models are handled by MistralCore
            // But we can still point to where they would be
            return modelsDirectory
                .appendingPathComponent("lmstudio-community")
                .appendingPathComponent("Mistral-Small-3.2-24B-Instruct-2506-MLX-\(variant.rawValue)")
        case .vae:
            return modelsDirectory
                .appendingPathComponent("black-forest-labs")
                .appendingPathComponent("FLUX.2-dev-vae")
        }
    }

    /// Check if a model component is downloaded
    /// Note: This delegates to Flux2ModelDownloader which checks multiple cache locations
    public static func isDownloaded(_ component: ModelComponent) -> Bool {
        // First check our local path
        let path = localPath(for: component)
        if FileManager.default.fileExists(atPath: path.path) {
            return true
        }

        // Also check HuggingFace cache via Flux2ModelDownloader
        return Flux2ModelDownloader.isDownloaded(component)
    }

    // MARK: - Configuration Files

    /// Expected files for each component
    public static func expectedFiles(for component: ModelComponent) -> [String] {
        switch component {
        case .transformer:
            return ["config.json", "model.safetensors.index.json"]
        case .textEncoder:
            return ["config.json", "model.safetensors.index.json", "tokenizer.json"]
        case .vae:
            return ["config.json", "diffusion_pytorch_model.safetensors"]
        }
    }
}

// MARK: - Preset Configurations

extension ModelRegistry {

    /// Recommended configuration for given RAM amount
    public static func recommendedConfig(forRAMGB ram: Int) -> Flux2QuantizationConfig {
        switch ram {
        case 0..<48:
            return .minimal       // ~35GB
        case 48..<64:
            return .memoryEfficient  // ~50GB
        case 64..<96:
            return .balanced      // ~60GB
        default:
            return .highQuality   // ~90GB
        }
    }

    /// Get system RAM in GB
    public static var systemRAMGB: Int {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        return Int(physicalMemory / 1_073_741_824)  // Convert bytes to GB
    }

    /// Default configuration based on system RAM
    public static var defaultConfig: Flux2QuantizationConfig {
        recommendedConfig(forRAMGB: systemRAMGB)
    }
}
