/**
 * ModelManager.swift
 * Manages model loading, downloading, and state
 */

import SwiftUI
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Memory Stats

struct MemoryStats {
    let active: Int
    let cache: Int
    let peak: Int

    static var current: MemoryStats {
        MemoryStats(
            active: Memory.activeMemory,
            cache: Memory.cacheMemory,
            peak: Memory.peakMemory
        )
    }
}

// MARK: - Model Manager

@MainActor
class ModelManager: ObservableObject {
    // MARK: - Loading State
    @Published var isLoaded = false
    @Published var isVLMLoaded = false  // Always load as VLM now
    @Published var isLoading = false
    @Published var loadingMessage = ""
    @Published var selectedVariant: ModelVariant?
    @Published var errorMessage: String?
    @Published var currentLoadedModelId: String?

    // MARK: - Qwen3/Klein State
    @Published var isQwen3Loaded = false
    @Published var isQwen3Loading = false
    @Published var qwen3LoadingMessage = ""
    @Published var loadedQwen3Variant: Qwen3Variant?
    @Published var downloadedQwen3Models: Set<String> = []
    @Published var qwen3ModelSizes: [String: Int64] = [:]

    // MARK: - Diffusion Models State (Flux2Core)
    @Published var downloadedTransformers: Set<String> = []  // Stores TransformerVariant.rawValue
    @Published var transformerSizes: [String: Int64] = [:]
    @Published var isVAEDownloaded = false
    @Published var vaeSize: Int64 = 0

    // MARK: - Download State
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var downloadMessage = ""

    // MARK: - Model Lists
    @Published var downloadedModels: Set<String> = []
    @Published var modelSizes: [String: Int64] = [:]

    // MARK: - Memory
    @Published var memoryStats = MemoryStats.current

    private let core = FluxTextEncoders.shared

    /// Models cache directory
    static var modelsCacheDirectory: URL {
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            return cacheDir.appendingPathComponent("models")
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/models")
    }

    init() {
        isLoaded = core.isModelLoaded
        isVLMLoaded = core.isVLMLoaded
        isQwen3Loaded = core.isKleinLoaded
        loadedQwen3Variant = core.kleinVariant.flatMap { variant in
            switch variant {
            case .klein4B: return .qwen3_4B_8bit
            case .klein9B: return .qwen3_8B_8bit
            }
        }
        refreshDownloadedModels()
        refreshDownloadedQwen3Models()
        refreshDownloadedDiffusionModels()
        selectSmallestDownloadedModel()
    }

    /// Select the smallest downloaded model by default
    private func selectSmallestDownloadedModel() {
        // ModelVariant ordered from smallest to largest: .mlx4bit, .mlx6bit, .mlx8bit, .bf16
        let variantsSmallestFirst: [ModelVariant] = [.mlx4bit, .mlx6bit, .mlx8bit, .bf16]

        for variant in variantsSmallestFirst {
            if let model = TextEncoderModelRegistry.shared.model(withVariant: variant),
               downloadedModels.contains(model.id) {
                selectedVariant = variant
                return
            }
        }

        // No model downloaded, leave selectedVariant as nil
        selectedVariant = nil
    }

    // MARK: - Available Models

    var availableModels: [ModelInfo] {
        TextEncoderModelRegistry.shared.allModels()
    }
    
    var availableQwen3Models: [Qwen3ModelInfo] {
        TextEncoderModelRegistry.shared.allQwen3Models()
    }

    var isCurrentModelLoaded: Bool {
        guard let currentId = currentLoadedModelId,
              let variant = selectedVariant,
              let model = TextEncoderModelRegistry.shared.model(withVariant: variant) else {
            return false
        }
        return currentId == model.id
    }

    // MARK: - Refresh

    func refreshDownloadedModels() {
        var downloaded: Set<String> = []
        var sizes: [String: Int64] = [:]

        for model in availableModels {
            if let path = TextEncoderModelDownloader.findModelPath(for: model) {
                downloaded.insert(model.id)
                sizes[model.id] = calculateDirectorySize(at: path)
            }
        }

        downloadedModels = downloaded
        modelSizes = sizes
        memoryStats = MemoryStats.current
    }

    private func calculateDirectorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }

        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(fileSize)
            }
        }
        return totalSize
    }

    // MARK: - Refresh Qwen3 Models

    func refreshDownloadedQwen3Models() {
        var downloaded: Set<String> = []
        var sizes: [String: Int64] = [:]

        for model in availableQwen3Models {
            if let path = TextEncoderModelDownloader.findQwen3ModelPath(for: model.variant) {
                downloaded.insert(model.id)
                sizes[model.id] = calculateDirectorySize(at: path)
            }
        }

        downloadedQwen3Models = downloaded
        qwen3ModelSizes = sizes
    }

    // MARK: - Load Qwen3 Model

    func loadQwen3Model(_ modelId: String) async {
        guard let model = availableQwen3Models.first(where: { $0.id == modelId }) else {
            print("[Qwen3] Model not found: \(modelId)")
            return
        }
        guard !isQwen3Loading else {
            print("[Qwen3] Already loading")
            return
        }

        isQwen3Loading = true
        qwen3LoadingMessage = "Loading \(model.displayName)..."
        errorMessage = nil

        print("[Qwen3] Starting load for \(model.displayName) (variant: \(model.variant))")

        do {
            let kleinVariant = model.variant.kleinVariant
            let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? UserDefaults.standard.string(forKey: "hfToken")

            print("[Qwen3] Loading Klein variant: \(kleinVariant), Qwen3 variant: \(model.variant)")

            try await core.loadKleinModel(
                variant: kleinVariant,
                qwen3Variant: model.variant,  // Pass the specific Qwen3 variant!
                hfToken: hfToken
            ) { progress, message in
                Task { @MainActor in
                    self.qwen3LoadingMessage = "\(message) (\(Int(progress * 100))%)"
                }
            }

            isQwen3Loaded = true
            loadedQwen3Variant = model.variant
            qwen3LoadingMessage = ""
            refreshDownloadedQwen3Models()
            print("[Qwen3] Load complete!")

        } catch {
            print("[Qwen3] Load error: \(error)")
            errorMessage = error.localizedDescription
            qwen3LoadingMessage = ""
        }

        isQwen3Loading = false
    }

    // MARK: - Download Qwen3 Model

    func downloadQwen3Model(_ modelId: String) async {
        guard let model = availableQwen3Models.first(where: { $0.id == modelId }) else { return }
        guard !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0
        downloadMessage = "Starting download of \(model.displayName)..."

        do {
            let downloader = TextEncoderModelDownloader(
                hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                    ?? UserDefaults.standard.string(forKey: "hfToken")
            )

            _ = try await downloader.downloadQwen3(variant: model.variant) { progress, message in
                Task { @MainActor in
                    self.downloadProgress = progress
                    self.downloadMessage = message
                }
            }

            downloadedQwen3Models.insert(modelId)
            refreshDownloadedQwen3Models()
            downloadMessage = "Download complete!"

        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        isDownloading = false
    }

    // MARK: - Delete Qwen3 Model

    func deleteQwen3Model(_ modelId: String) async throws {
        guard let model = availableQwen3Models.first(where: { $0.id == modelId }) else { return }

        // Can't delete if currently loaded
        if isQwen3Loaded && loadedQwen3Variant == model.variant {
            throw ModelManagerError.cannotDeleteLoadedModel
        }

        guard let path = TextEncoderModelDownloader.findQwen3ModelPath(for: model.variant) else { return }

        try FileManager.default.removeItem(at: path)
        downloadedQwen3Models.remove(modelId)
        qwen3ModelSizes.removeValue(forKey: modelId)
        refreshDownloadedQwen3Models()
    }

    // MARK: - Unload Qwen3 Model

    func unloadQwen3Model() {
        core.unloadKleinModel()
        isQwen3Loaded = false
        loadedQwen3Variant = nil
        memoryStats = MemoryStats.current
    }

    // MARK: - Diffusion Models (Flux2Core)

    /// All available transformer variants
    var availableTransformerVariants: [ModelRegistry.TransformerVariant] {
        ModelRegistry.TransformerVariant.allCases
    }

    /// Refresh downloaded diffusion models status
    func refreshDownloadedDiffusionModels() {
        var downloaded: Set<String> = []
        var sizes: [String: Int64] = [:]

        for variant in ModelRegistry.TransformerVariant.allCases {
            let component = ModelRegistry.ModelComponent.transformer(variant)
            if Flux2ModelDownloader.isDownloaded(component),
               let path = Flux2ModelDownloader.findModelPath(for: component) {
                downloaded.insert(variant.rawValue)
                sizes[variant.rawValue] = calculateDirectorySize(at: path)
            }
        }

        downloadedTransformers = downloaded
        transformerSizes = sizes

        // Check VAE
        let vaeComponent = ModelRegistry.ModelComponent.vae(.standard)
        isVAEDownloaded = Flux2ModelDownloader.isDownloaded(vaeComponent)
        if isVAEDownloaded, let path = Flux2ModelDownloader.findModelPath(for: vaeComponent) {
            vaeSize = calculateDirectorySize(at: path)
        } else {
            vaeSize = 0
        }

        memoryStats = MemoryStats.current
    }

    /// Download a transformer variant
    func downloadTransformer(_ variant: ModelRegistry.TransformerVariant) async {
        guard !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0
        downloadMessage = "Starting download of \(variant.rawValue) transformer..."

        do {
            let downloader = Flux2ModelDownloader(
                hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                    ?? UserDefaults.standard.string(forKey: "hfToken")
            )

            let component = ModelRegistry.ModelComponent.transformer(variant)
            _ = try await downloader.download(component) { progress, message in
                Task { @MainActor in
                    self.downloadProgress = progress
                    self.downloadMessage = message
                }
            }

            downloadedTransformers.insert(variant.rawValue)
            refreshDownloadedDiffusionModels()
            downloadMessage = "Download complete!"

        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        isDownloading = false
    }

    /// Delete a transformer variant
    func deleteTransformer(_ variant: ModelRegistry.TransformerVariant) throws {
        let component = ModelRegistry.ModelComponent.transformer(variant)
        try Flux2ModelDownloader.delete(component)
        downloadedTransformers.remove(variant.rawValue)
        transformerSizes.removeValue(forKey: variant.rawValue)
        refreshDownloadedDiffusionModels()
    }

    /// Download VAE
    func downloadVAE() async {
        guard !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0
        downloadMessage = "Starting VAE download..."

        do {
            let downloader = Flux2ModelDownloader(
                hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                    ?? UserDefaults.standard.string(forKey: "hfToken")
            )

            let component = ModelRegistry.ModelComponent.vae(.standard)
            _ = try await downloader.download(component) { progress, message in
                Task { @MainActor in
                    self.downloadProgress = progress
                    self.downloadMessage = message
                }
            }

            isVAEDownloaded = true
            refreshDownloadedDiffusionModels()
            downloadMessage = "VAE download complete!"

        } catch {
            errorMessage = "VAE download failed: \(error.localizedDescription)"
        }

        isDownloading = false
    }

    /// Delete VAE
    func deleteVAE() throws {
        let component = ModelRegistry.ModelComponent.vae(.standard)
        try Flux2ModelDownloader.delete(component)
        isVAEDownloaded = false
        vaeSize = 0
        refreshDownloadedDiffusionModels()
    }

    /// Check if a transformer variant is downloaded
    func isTransformerDownloaded(_ variant: ModelRegistry.TransformerVariant) -> Bool {
        downloadedTransformers.contains(variant.rawValue)
    }

    /// Get transformer display info
    func transformerDisplayInfo(_ variant: ModelRegistry.TransformerVariant) -> (name: String, size: String, modelType: String) {
        let name: String
        let modelType: String

        switch variant {
        case .bf16:
            name = "Dev bf16"
            modelType = "Flux.2 Dev"
        case .qint8:
            name = "Dev qint8"
            modelType = "Flux.2 Dev"
        case .klein4B_bf16:
            name = "Klein 4B bf16"
            modelType = "Flux.2 Klein 4B"
        case .klein4B_8bit:
            name = "Klein 4B 8-bit"
            modelType = "Flux.2 Klein 4B"
        case .klein9B_bf16:
            name = "Klein 9B bf16"
            modelType = "Flux.2 Klein 9B"
        case .klein4B_base_bf16:
            name = "Klein 4B Base (Training)"
            modelType = "Flux.2 Klein 4B"
        case .klein9B_base_bf16:
            name = "Klein 9B Base (Training)"
            modelType = "Flux.2 Klein 9B"
        }

        let size = "\(variant.estimatedSizeGB)GB"
        return (name, size, modelType)
    }

    // MARK: - Load Model

    func loadModel() async {
        guard !isLoading else { return }
        guard let variant = selectedVariant else {
            errorMessage = "No model selected. Please download a model first."
            return
        }

        isLoading = true
        loadingMessage = "Preparing to load model..."
        errorMessage = nil

        do {
            let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? UserDefaults.standard.string(forKey: "hfToken")

            // Always load as VLM (Vision-Language Model) - supports both text and vision
            try await core.loadVLMModel(
                variant: variant,
                hfToken: hfToken
            ) { progress, message in
                Task { @MainActor in
                    self.loadingMessage = "\(message) (\(Int(progress * 100))%)"
                }
            }

            if let model = TextEncoderModelRegistry.shared.model(withVariant: variant) {
                currentLoadedModelId = model.id
            }
            isLoaded = true
            isVLMLoaded = true
            loadingMessage = ""
            refreshDownloadedModels()

        } catch {
            errorMessage = error.localizedDescription
            loadingMessage = ""
        }

        isLoading = false
    }

    func loadModel(from path: String) {
        isLoading = true
        loadingMessage = "Loading model..."
        errorMessage = nil

        Task {
            do {
                // Always load as VLM for unified experience
                try core.loadVLMModel(from: path)
                await MainActor.run {
                    isLoaded = true
                    isVLMLoaded = true
                    isLoading = false
                    loadingMessage = ""
                    refreshDownloadedModels()
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isLoading = false
                    loadingMessage = ""
                }
            }
        }
    }

    func loadModel(_ modelId: String) async {
        guard let model = availableModels.first(where: { $0.id == modelId }) else { return }

        // Unload current model if different
        if isLoaded && currentLoadedModelId != modelId {
            unloadModel()
        }

        selectedVariant = model.variant
        await loadModel()
    }

    // MARK: - Unload Model

    func unloadModel() {
        core.unloadModel()
        isLoaded = false
        isVLMLoaded = false
        currentLoadedModelId = nil
        memoryStats = MemoryStats.current
    }

    // MARK: - Download Model

    func downloadModel(_ modelId: String) async {
        guard let model = availableModels.first(where: { $0.id == modelId }) else { return }
        guard !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0
        downloadMessage = "Starting download..."

        do {
            let downloader = TextEncoderModelDownloader(
                hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                    ?? UserDefaults.standard.string(forKey: "hfToken")
            )

            _ = try await downloader.download(model) { progress, message in
                Task { @MainActor in
                    self.downloadProgress = progress
                    self.downloadMessage = message
                }
            }

            downloadedModels.insert(modelId)
            refreshDownloadedModels()
            downloadMessage = "Download complete!"

        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        isDownloading = false
    }

    // MARK: - Delete Model

    func deleteModel(_ modelId: String) async throws {
        guard let model = availableModels.first(where: { $0.id == modelId }) else { return }

        // Can't delete if currently loaded
        if currentLoadedModelId == modelId {
            throw ModelManagerError.cannotDeleteLoadedModel
        }

        guard let path = TextEncoderModelDownloader.findModelPath(for: model) else { return }

        try FileManager.default.removeItem(at: path)
        downloadedModels.remove(modelId)
        modelSizes.removeValue(forKey: modelId)
        refreshDownloadedModels()
    }

    // MARK: - Memory Management

    func clearCache() {
        Memory.clearCache()
        memoryStats = MemoryStats.current
    }

    func resetPeakMemory() {
        Memory.peakMemory = 0
        memoryStats = MemoryStats.current
    }

    // MARK: - Formatting

    static func formatBytes(_ bytes: Int) -> String {
        let absBytes = abs(bytes)
        if absBytes >= 1024 * 1024 * 1024 {
            return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
        } else if absBytes >= 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        } else if absBytes >= 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        }
        return "\(bytes) B"
    }
}

// MARK: - Errors

enum ModelManagerError: LocalizedError {
    case cannotDeleteLoadedModel

    var errorDescription: String? {
        switch self {
        case .cannotDeleteLoadedModel:
            return "Cannot delete a loaded model. Unload it first."
        }
    }
}
