// ModelViewModel.swift - Model download and management
// Copyright 2025 Vincent Gourbin

import SwiftUI
import Flux2Core

/// ViewModel for model management
@MainActor
class ModelViewModel: ObservableObject {
    // MARK: - Download State

    @Published var downloadProgress: [ModelRegistry.ModelComponent: Double] = [:]
    @Published var downloadingModels: Set<ModelRegistry.ModelComponent> = []
    @Published var downloadErrors: [ModelRegistry.ModelComponent: String] = [:]

    // MARK: - Model Status

    /// Check if a model is downloaded
    func isDownloaded(_ component: ModelRegistry.ModelComponent) -> Bool {
        ModelRegistry.isDownloaded(component)
    }

    /// Get download progress for a model (0.0 to 1.0)
    func progress(_ component: ModelRegistry.ModelComponent) -> Double {
        downloadProgress[component] ?? 0.0
    }

    /// Check if a model is currently downloading
    func isDownloading(_ component: ModelRegistry.ModelComponent) -> Bool {
        downloadingModels.contains(component)
    }

    // MARK: - Download Actions

    /// Download a model component
    func download(_ component: ModelRegistry.ModelComponent) async {
        guard !downloadingModels.contains(component) else { return }

        downloadingModels.insert(component)
        downloadErrors[component] = nil
        downloadProgress[component] = 0.0

        do {
            // Simulated download - in real implementation, use ModelDownloader
            for i in 1...100 {
                try await Task.sleep(nanoseconds: 50_000_000)  // 50ms
                downloadProgress[component] = Double(i) / 100.0
            }

            // In real implementation:
            // try await ModelDownloader.download(component) { progress in
            //     Task { @MainActor in
            //         self.downloadProgress[component] = progress
            //     }
            // }

        } catch {
            downloadErrors[component] = error.localizedDescription
        }

        downloadingModels.remove(component)
    }

    /// Delete a downloaded model
    func delete(_ component: ModelRegistry.ModelComponent) {
        let path = ModelRegistry.localPath(for: component)
        try? FileManager.default.removeItem(at: path)
        downloadProgress[component] = 0.0
    }

    // MARK: - Batch Operations

    /// Download all models for a quantization config
    func downloadAll(for config: Flux2QuantizationConfig) async {
        // Download text encoder
        let textVariant = ModelRegistry.TextEncoderVariant(rawValue: config.textEncoder.rawValue)!
        await download(.textEncoder(textVariant))

        // Download transformer
        let transVariant = ModelRegistry.TransformerVariant(rawValue: config.transformer.rawValue)!
        await download(.transformer(transVariant))

        // Download VAE
        await download(.vae(.standard))
    }

    /// Delete all models
    func deleteAll() {
        for variant in ModelRegistry.TransformerVariant.allCases {
            delete(.transformer(variant))
        }
        for variant in ModelRegistry.TextEncoderVariant.allCases {
            delete(.textEncoder(variant))
        }
        delete(.vae(.standard))
    }

    // MARK: - Storage Info

    /// Total size of downloaded models in GB
    var totalDownloadedSizeGB: Int {
        var total = 0
        for variant in ModelRegistry.TransformerVariant.allCases {
            if isDownloaded(.transformer(variant)) {
                total += variant.estimatedSizeGB
            }
        }
        for variant in ModelRegistry.TextEncoderVariant.allCases {
            if isDownloaded(.textEncoder(variant)) {
                total += variant.estimatedSizeGB
            }
        }
        if isDownloaded(.vae(.standard)) {
            total += 3
        }
        return total
    }

    /// Available disk space in GB
    var availableDiskSpaceGB: Int {
        let fileManager = FileManager.default
        guard let attrs = try? fileManager.attributesOfFileSystem(forPath: NSHomeDirectory()),
              let space = attrs[.systemFreeSize] as? Int64 else {
            return 0
        }
        return Int(space / 1_073_741_824)
    }
}

// MARK: - Model Info

extension ModelViewModel {
    /// Get display info for a model component
    func info(for component: ModelRegistry.ModelComponent) -> ModelInfo {
        ModelInfo(
            name: component.displayName,
            sizeGB: component.estimatedSizeGB,
            isDownloaded: isDownloaded(component),
            isDownloading: isDownloading(component),
            progress: progress(component),
            error: downloadErrors[component]
        )
    }
}

struct ModelInfo {
    let name: String
    let sizeGB: Int
    let isDownloaded: Bool
    let isDownloading: Bool
    let progress: Double
    let error: String?
}
