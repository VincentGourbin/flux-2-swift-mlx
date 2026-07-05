// ModelRegistry.swift - Model variants and download sources
// Copyright 2025 Vincent Gourbin

import Foundation

/// Registry of available Flux.2 model variants
public enum ModelRegistry {

  // MARK: - Model Components

  /// Available Flux.2 transformer variants
  public enum TransformerVariant: String, CaseIterable, Sendable {
    // Flux.2 Dev variants (32B)
    case bf16 = "bf16"
    case qint8 = "qint8"

    // Flux.2 Klein 4B variants (distilled - for inference)
    case klein4B_bf16 = "klein4b-bf16"
    case klein4B_8bit = "klein4b-8bit"

    // Flux.2 Klein 4B int4 (community pre-quantized MLX 4-bit; verified genuine
    // MLX quantization — .scales/.biases per layer, quantization_level "4" —
    // Apache 2.0, transformer/ subfolder only. See EXECUTION_PLAN.md OQ-1.)
    case klein4B_4bit = "klein4b-4bit"

    // Flux.2 Klein 4B Base (non-distilled - for LoRA training only)
    case klein4B_base_bf16 = "klein4b-base-bf16"

    // Flux.2 Klein 9B variants
    // Note: Only bf16 available for Klein 9B transformer (no community qint8 yet)
    case klein9B_bf16 = "klein9b-bf16"

    // Flux.2 Klein 9B Base (non-distilled - for LoRA training only)
    case klein9B_base_bf16 = "klein9b-base-bf16"

    // Flux.2 Klein 9B KV (KV-cached variant for faster multi-reference I2I)
    case klein9B_kv_bf16 = "klein9b-kv-bf16"

    /// Acervo model ID; weights originally from the corresponding HuggingFace repo.
    public var repoId: String {
      switch self {
      case .bf16:
        return "black-forest-labs/FLUX.2-dev"
      case .qint8:
        return "VincentGOURBIN/flux_qint_8bit"
      case .klein4B_bf16:
        return "black-forest-labs/FLUX.2-klein-4B"
      case .klein4B_8bit:
        // Community 8-bit quantization (contains only transformer weights)
        return "aydin99/FLUX.2-klein-4B-int8"
      case .klein4B_4bit:
        // Community pre-quantized MLX 4-bit (OQ-1); transformer/ subfolder only.
        return "themindstudio/flux2-klein-4b-mlx-4bit"
      case .klein4B_base_bf16:
        // Base model (non-distilled) for LoRA training
        return "black-forest-labs/FLUX.2-klein-base-4B"
      case .klein9B_bf16:
        return "black-forest-labs/FLUX.2-klein-9B"
      case .klein9B_base_bf16:
        // Base model (non-distilled) for LoRA training
        return "black-forest-labs/FLUX.2-klein-base-9B"
      case .klein9B_kv_bf16:
        return "black-forest-labs/FLUX.2-klein-9b-kv"
      }
    }

    /// Subfolder within the Acervo model directory where this transformer's weights live.
    public var repoSubfolder: String? {
      switch self {
      case .bf16:
        return "transformer"
      case .qint8:
        return "flux-2-dev/transformer/qint8"
      case .klein4B_8bit:
        // Community 8-bit repo (aydin99/FLUX.2-klein-4B-int8) ships only the
        // transformer weights, at the repo root — no diffusers subfolder.
        return nil
      case .klein4B_4bit:
        // Community int4 repo (themindstudio/flux2-klein-4b-mlx-4bit) nests
        // the pre-quantized MLX weights under a `transformer/` subfolder
        // (2.18 GB), unlike the flat aydin99 int8 repo (OQ-1).
        return "transformer"
      case .klein4B_bf16, .klein4B_base_bf16, .klein9B_bf16, .klein9B_base_bf16,
        .klein9B_kv_bf16:
        // The black-forest-labs Klein repos ship the full diffusers layout, so
        // the transformer weights live under `transformer/` (alongside
        // `text_encoder/` and `vae/`), not at the repo root.
        return "transformer"
      }
    }

    public var estimatedSizeGB: Double {
      switch self {
      case .bf16: return 64
      case .qint8: return 32
      case .klein4B_bf16: return 8
      case .klein4B_8bit: return 4
      case .klein4B_4bit: return 2.18  // themindstudio transformer/ subfolder (OQ-1)
      case .klein4B_base_bf16: return 8  // Same size as distilled
      case .klein9B_bf16: return 18
      case .klein9B_base_bf16: return 18  // Same size as distilled
      case .klein9B_kv_bf16: return 18  // Same architecture as klein-9b
      }
    }

    /// Whether this model requires accepting a license on HuggingFace before downloading
    public var isGated: Bool {
      switch self {
      case .bf16:
        // Dev bf16 from black-forest-labs is gated
        return true
      case .qint8:
        // VincentGOURBIN repo is NOT gated
        return false
      case .klein4B_bf16:
        // Klein 4B from black-forest-labs is NOT gated
        return false
      case .klein4B_8bit:
        // Community 8-bit quantization is NOT gated
        return false
      case .klein4B_4bit:
        // Community int4 quantization (themindstudio) is NOT gated
        return false
      case .klein4B_base_bf16:
        // Klein 4B Base from black-forest-labs is NOT gated
        return false
      case .klein9B_bf16:
        // Klein 9B from black-forest-labs IS gated
        return true
      case .klein9B_base_bf16:
        // Klein 9B Base from black-forest-labs IS gated
        return true
      case .klein9B_kv_bf16:
        // Klein 9B KV from black-forest-labs IS gated
        return true
      }
    }

    /// License information (delegates to modelType)
    public var license: String {
      modelType.license
    }

    /// Whether this model can be used commercially (delegates to modelType)
    public var isCommercialUseAllowed: Bool {
      modelType.isCommercialUseAllowed
    }

    /// Recommended number of inference steps (delegates to modelType)
    public var defaultSteps: Int {
      modelType.defaultSteps
    }

    /// Recommended guidance scale (delegates to modelType)
    public var defaultGuidance: Float {
      modelType.defaultGuidance
    }

    /// Maximum number of reference images for I2I (delegates to modelType)
    public var maxReferenceImages: Int {
      modelType.maxReferenceImages
    }

    public var quantization: TransformerQuantization {
      switch self {
      case .bf16, .klein4B_bf16, .klein4B_base_bf16, .klein9B_bf16, .klein9B_base_bf16,
        .klein9B_kv_bf16:
        return .bf16
      case .qint8, .klein4B_8bit: return .qint8
      case .klein4B_4bit: return .int4
      }
    }

    /// Whether this variant ships **genuine MLX-native pre-quantized** weights —
    /// a packed uint32 `.weight` plus per-layer `.scales`/`.biases` — that load
    /// DIRECTLY into `QuantizedLinear` with no bf16/float16 intermediate and no
    /// on-the-fly `quantize()` pass (Sortie B2, R2.1/R2.2/R2.3).
    ///
    /// Only `klein4B_4bit` (themindstudio/flux2-klein-4b-mlx-4bit, OQ-1) qualifies.
    /// The qint8 community repos (`qint8`, `klein4B_8bit`) are quanto-format —
    /// they are dequantized to float16 and re-quantized on-the-fly, so they are
    /// NOT pre-quantized MLX and stay on the existing load path.
    public var isPreQuantizedMLX: Bool {
      switch self {
      case .klein4B_4bit:
        return true
      case .bf16, .qint8, .klein4B_bf16, .klein4B_8bit, .klein4B_base_bf16,
        .klein9B_bf16, .klein9B_base_bf16, .klein9B_kv_bf16:
        return false
      }
    }

    /// The Flux.2 model type this variant belongs to
    public var modelType: Flux2Model {
      switch self {
      case .bf16, .qint8:
        return .dev
      case .klein4B_bf16, .klein4B_8bit, .klein4B_4bit:
        return .klein4B
      case .klein4B_base_bf16:
        return .klein4BBase
      case .klein9B_bf16:
        return .klein9B
      case .klein9B_base_bf16:
        return .klein9BBase
      case .klein9B_kv_bf16:
        return .klein9BKV
      }
    }

    /// Whether this variant can be used for inference
    /// Distilled models and Dev are for inference
    public var isForInference: Bool {
      switch self {
      case .bf16, .qint8:  // Dev
        return true
      case .klein4B_bf16, .klein4B_8bit, .klein4B_4bit:  // Klein 4B distilled
        return true
      case .klein9B_bf16:  // Klein 9B distilled
        return true
      case .klein9B_kv_bf16:  // Klein 9B KV distilled
        return true
      case .klein4B_base_bf16, .klein9B_base_bf16:  // Base models
        return false
      }
    }

    /// Whether this variant can be used for LoRA training
    /// Base (non-distilled) models and Dev bf16 are for training
    public var isForTraining: Bool {
      switch self {
      case .bf16:  // Dev bf16 - can train
        return true
      case .qint8:  // Dev int8 - cannot train (quantized)
        return false
      case .klein4B_bf16, .klein4B_8bit, .klein4B_4bit:  // Distilled - cannot train
        return false
      case .klein9B_bf16:  // Distilled - cannot train
        return false
      case .klein9B_kv_bf16:  // KV variant - cannot train
        return false
      case .klein4B_base_bf16, .klein9B_base_bf16:  // Base models - for training
        return true
      }
    }

    /// Get the appropriate variant for a model type and quantization
    ///
    /// For quantization levels without a pre-quantized variant (e.g. Klein 9B qint8, Dev int4),
    /// this returns the bf16 variant. The pipeline will then quantize on-the-fly after loading.
    ///
    /// `(klein4B, .int4)` is the exception: it resolves to `klein4B_4bit`, a genuine
    /// MLX-native pre-quantized variant (`isPreQuantizedMLX == true`) that the pipeline
    /// loads DIRECTLY into `QuantizedLinear` — no bf16 intermediate, no on-the-fly
    /// `quantize()` (Sortie B2, R2.1/R2.2/R2.3).
    public static func variant(for model: Flux2Model, quantization: TransformerQuantization)
      -> TransformerVariant
    {
      switch (model, quantization) {
      case (.dev, .bf16): return .bf16
      case (.dev, .qint8): return .qint8
      case (.dev, .int4): return .bf16  // Load bf16, quantize on-the-fly
      case (.klein4B, .bf16): return .klein4B_bf16
      case (.klein4B, .qint8): return .klein4B_8bit
      case (.klein4B, .int4): return .klein4B_4bit  // Direct pre-quantized MLX 4-bit load (B2)
      // Base models only available in bf16
      case (.klein4BBase, _): return .klein4B_base_bf16
      case (.klein9BBase, _): return .klein9B_base_bf16
      // Klein 9B only has bf16 — quantize on-the-fly for qint8/int4
      case (.klein9B, _): return .klein9B_bf16
      // Klein 9B KV only has bf16 — quantize on-the-fly for qint8/int4
      case (.klein9BKV, _): return .klein9B_kv_bf16
      }
    }

    /// Get the appropriate BASE variant for LoRA training (NOT distilled)
    /// LoRA training MUST use base (non-distilled) models
    /// Returns nil if no base model is available for the given model type
    public static func trainingVariant(for model: Flux2Model) -> TransformerVariant? {
      switch model {
      case .klein4B, .klein4BBase:
        // Base model only available in bf16
        return .klein4B_base_bf16
      case .klein9B, .klein9BBase, .klein9BKV:
        // Base model (non-distilled) for LoRA training
        return .klein9B_base_bf16
      case .dev:
        // Dev model is already "base" (not distilled)
        return .bf16
      }
    }
  }

  /// Available Mistral text encoder variants (from mistral-small-3.2-swift-mlx)
  public enum TextEncoderVariant: String, CaseIterable, Sendable {
    case bf16 = "bf16"
    case mlx8bit = "8bit"
    case mlx6bit = "6bit"
    case mlx4bit = "4bit"

    /// Acervo model ID; weights originally from the corresponding HuggingFace repo.
    public var repoId: String {
      switch self {
      case .bf16:
        // Original from Mistral AI (gated)
        return "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
      case .mlx8bit:
        return "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit"
      case .mlx6bit:
        return "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit"
      case .mlx4bit:
        return "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit"
      }
    }

    public var estimatedSizeGB: Int {
      switch self {
      case .bf16: return 48
      case .mlx8bit: return 25
      case .mlx6bit: return 19
      case .mlx4bit: return 14
      }
    }

    /// Whether this model requires accepting a license on HuggingFace before downloading
    public var isGated: Bool {
      switch self {
      case .bf16:
        // Original Mistral AI model is gated
        return true
      case .mlx8bit, .mlx6bit, .mlx4bit:
        // lmstudio-community quantized versions are NOT gated
        return false
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

    /// License information for Mistral text encoder
    public var license: String {
      "Apache 2.0"
    }

    /// Whether this model can be used commercially
    public var isCommercialUseAllowed: Bool {
      true  // Mistral Small 3.2 is Apache 2.0
    }
  }

  /// VAE variant (only one available)
  public enum VAEVariant: String, CaseIterable, Sendable {
    case standard = "standard"

    /// Acervo model ID; weights originally from `black-forest-labs/FLUX.2-klein-4B` on HuggingFace.
    public var repoId: String {
      // VAE is downloaded from Klein 4B repo (NOT gated)
      // Same VAE weights as all Flux.2 models
      "black-forest-labs/FLUX.2-klein-4B"
    }

    /// Subfolder within the Acervo model directory where the VAE weights live.
    public var repoSubfolder: String {
      "vae"
    }

    public var estimatedSizeGB: Int { 3 }

    /// Whether this model requires accepting a license on HuggingFace before downloading
    public var isGated: Bool {
      // VAE is downloaded from Klein 4B repo which is NOT gated
      false
    }

    /// License information (inherits from FLUX.2 Dev)
    public var license: String {
      "FLUX.2 Non-Commercial"
    }

    /// Whether this model can be used commercially
    public var isCommercialUseAllowed: Bool {
      false  // VAE inherits FLUX.2 Dev non-commercial license
    }
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

    public var estimatedSizeGB: Double {
      switch self {
      case .transformer(let variant): return variant.estimatedSizeGB
      case .textEncoder(let variant): return Double(variant.estimatedSizeGB)
      case .vae(let variant): return Double(variant.estimatedSizeGB)
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

    /// CDN directory name for this component (used when `ModelRegistry.cdnBaseURL` is set).
    public var cdnDirectoryName: String {
      switch self {
      case .transformer(let variant):
        switch variant {
        case .klein4B_bf16: return "FLUX.2-klein-4B"
        case .klein9B_bf16: return "FLUX.2-klein-9B"
        case .bf16: return "FLUX.2-dev-bf16"
        case .qint8: return "FLUX.2-dev-qint8"
        case .klein4B_8bit: return "FLUX.2-klein-4B-int8"
        case .klein4B_4bit: return "FLUX.2-klein-4B-int4"
        case .klein4B_base_bf16: return "FLUX.2-klein-base-4B"
        case .klein9B_base_bf16: return "FLUX.2-klein-base-9B"
        case .klein9B_kv_bf16: return "FLUX.2-klein-9b-kv"
        }
      case .textEncoder(let variant):
        return "mistral-small-3.2-\(variant.rawValue)"
      case .vae:
        return "FLUX.2-vae"
      }
    }

    /// Acervo model ID for this component. Forwards to the variant's own
    /// `repoId`.
    public var repoId: String {
      switch self {
      case .transformer(let variant): return variant.repoId
      case .textEncoder(let variant): return variant.repoId
      case .vae(let variant): return variant.repoId
      }
    }

    /// Subfolder within the Acervo model directory where this component's
    /// weights live, when the upstream repo nests them. `nil` means the
    /// weights live at the model root.
    public var repoSubfolder: String? {
      switch self {
      case .transformer(let variant): return variant.repoSubfolder
      case .textEncoder: return nil
      case .vae(let variant): return variant.repoSubfolder
      }
    }
  }

  // MARK: - Paths

  /// CDN base URL for downloading models without HuggingFace authentication.
  ///
  /// When set, the downloader fetches a `manifest.json` from the CDN path
  /// instead of querying the HuggingFace API, and downloads files directly
  /// from the CDN. No authentication headers are sent.
  ///
  /// Expected CDN structure:
  /// ```
  /// <cdnBaseURL>/models/<component-dir>/manifest.json
  /// <cdnBaseURL>/models/<component-dir>/<file>
  /// ```
  ///
  /// Set this at app startup before any download calls.
  nonisolated(unsafe) public static var cdnBaseURL: URL?

  /// Base directory for model storage.
  /// Falls back to ~/Library/Caches/models; Sortie 19 replaces this with Acervo.modelDirectory(for:).
  public static var modelsDirectory: URL {
    let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return cacheDir.appendingPathComponent("models", isDirectory: true)
  }

  /// Get the local path for a model component
  public static func localPath(for component: ModelComponent) -> URL {
    switch component {
    case .transformer(let variant):
      let modelName: String
      switch variant {
      case .bf16, .qint8:
        modelName = "FLUX.2-dev-transformer-\(variant.rawValue)"
      case .klein4B_bf16, .klein4B_8bit, .klein4B_4bit:
        modelName = "FLUX.2-klein-4B-\(variant.rawValue)"
      case .klein4B_base_bf16:
        modelName = "FLUX.2-klein-base-4B-\(variant.rawValue)"
      case .klein9B_bf16:
        modelName = "FLUX.2-klein-9B-\(variant.rawValue)"
      case .klein9B_base_bf16:
        modelName = "FLUX.2-klein-base-9B-\(variant.rawValue)"
      case .klein9B_kv_bf16:
        modelName = "FLUX.2-klein-9b-kv-\(variant.rawValue)"
      }
      return
        modelsDirectory
        .appendingPathComponent("black-forest-labs")
        .appendingPathComponent(modelName)
    case .textEncoder(let variant):
      // Mistral models are handled by MistralCore
      // But we can still point to where they would be
      return
        modelsDirectory
        .appendingPathComponent("lmstudio-community")
        .appendingPathComponent("Mistral-Small-3.2-24B-Instruct-2506-MLX-\(variant.rawValue)")
    case .vae:
      return
        modelsDirectory
        .appendingPathComponent("black-forest-labs")
        .appendingPathComponent("FLUX.2-klein-4B-vae")
    }
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
    // iPad tier (≤16 GB unified memory): select the tier explicitly so
    // iPad-class devices don't inherit the shared `<32` Mac bucket.
    if MemoryConfig.tier(forRAMGB: ram) == .iPad {
      return .ultraMinimal  // ~30GB (4-bit transformer)
    }

    switch ram {
    case 0..<32:
      return .ultraMinimal  // ~30GB (4-bit transformer)
    case 32..<48:
      return .minimal  // ~35GB
    case 48..<64:
      return .memoryEfficient  // ~50GB
    case 64..<96:
      return .balanced  // ~60GB
    default:
      return .highQuality  // ~90GB
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
