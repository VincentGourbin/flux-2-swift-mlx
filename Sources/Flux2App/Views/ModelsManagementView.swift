/**
 * ModelsManagementView.swift
 * Models management tab: model rows, available-model cards, and the per-component
 * (diffusion / transformer / VAE) sections. Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Models Management View

struct ModelsManagementView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var modelToDelete: ModelInfo?
    @State private var qwen3ModelToDelete: Qwen3ModelInfo?
    @State private var showDeleteConfirmation = false
    @State private var showQwen3DeleteConfirmation = false
    @State private var memoryRefreshTrigger = false

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Memory status bar
                HStack(spacing: 16) {
                    Label("MLX Memory", systemImage: "memorychip")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)

                    HStack(spacing: 8) {
                        Text("Active: \(ModelManager.formatBytes(modelManager.memoryStats.active))")
                        Text("Cache: \(ModelManager.formatBytes(modelManager.memoryStats.cache))")
                            .foregroundStyle(modelManager.memoryStats.cache > 0 ? .orange : .secondary)
                        Text("Peak: \(ModelManager.formatBytes(modelManager.memoryStats.peak))")
                            .foregroundStyle(.blue)
                    }
                    .font(.caption.monospaced())

                    Spacer()

                    Button(action: {
                        modelManager.clearCache()
                        modelManager.resetPeakMemory()
                        memoryRefreshTrigger.toggle()
                    }) {
                        Label("Clear Cache", systemImage: "trash.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(modelManager.memoryStats.cache == 0)
                    .help("Clear MLX recyclable cache")

                    Button(action: {
                        modelManager.unloadModel()
                        modelManager.unloadQwen3Model()
                        memoryRefreshTrigger.toggle()
                    }) {
                        Label("Unload All", systemImage: "xmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(.orange)
                    .disabled(!modelManager.isLoaded && !modelManager.isQwen3Loaded)
                    .help("Unload all models to free GPU memory")
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .id(memoryRefreshTrigger)

                Divider()

                // Download progress bar
                if modelManager.isDownloading {
                    VStack(spacing: 4) {
                        ProgressView(value: modelManager.downloadProgress)
                            .progressViewStyle(.linear)
                        Text(modelManager.downloadMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)

                    Divider()
                }

                if let error = modelManager.errorMessage {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.primary)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                        Button("Dismiss") {
                            modelManager.errorMessage = nil
                        }
                        .buttonStyle(.borderless)
                        .font(.caption)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(Color.orange.opacity(0.12))

                    Divider()
                }

                // ===== MISTRAL MODELS SECTION =====
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Label("Mistral Small 3.2", systemImage: "brain.filled.head.profile")
                            .font(.headline)

                        Spacer()

                        Button(action: {
                            NSWorkspace.shared.open(ModelManager.modelsCacheDirectory)
                        }) {
                            Label("Open Cache Folder", systemImage: "folder")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Open models cache folder in Finder")

                        Button(action: {
                            modelManager.refreshDownloadedModels()
                            modelManager.refreshDownloadedQwen3Models()
                        }) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .help("Refresh")
                    }
                    .padding(.horizontal)
                    .padding(.top)

                    Text("24B VLM for text, vision, chat, and FLUX.2-dev embeddings")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)

                    // Downloaded Mistral models
                    if !modelManager.downloadedModels.isEmpty {
                        ForEach(modelManager.availableModels.filter { modelManager.downloadedModels.contains($0.id) }, id: \.id) { model in
                            ModelRowView(
                                model: model,
                                size: modelManager.modelSizes[model.id],
                                isLoaded: modelManager.currentLoadedModelId == model.id,
                                onDelete: {
                                    modelToDelete = model
                                    showDeleteConfirmation = true
                                },
                                onLoad: {
                                    Task { await modelManager.loadModel(model.id) }
                                }
                            )
                            .padding(.horizontal)
                        }
                    }

                    // Available Mistral models to download
                    let availableMistral = modelManager.availableModels.filter { !modelManager.downloadedModels.contains($0.id) }
                    if !availableMistral.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(availableMistral, id: \.id) { model in
                                    AvailableModelCard(model: model, modelManager: modelManager)
                                }
                            }
                            .padding(.horizontal)
                        }
                    } else if modelManager.downloadedModels.isEmpty {
                        Text("No Mistral models downloaded. Download one to get started.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.bottom)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))

                Divider()

                // ===== QWEN3 MODELS SECTION =====
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Label("Qwen3 (FLUX.2 Klein)", systemImage: "cube.fill")
                            .font(.headline)
                            .foregroundStyle(.orange)

                        if modelManager.isQwen3Loaded {
                            Text(modelManager.loadedQwen3Variant?.displayName ?? "Loaded")
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.green.opacity(0.2))
                                .foregroundStyle(.green)
                                .cornerRadius(4)
                        }

                        Spacer()

                        if modelManager.isQwen3Loaded {
                            Button(action: { modelManager.unloadQwen3Model() }) {
                                Label("Unload", systemImage: "xmark.circle")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .tint(.orange)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top)

                    Text("4B/8B models for Klein 4B (Apache 2.0) and Klein 9B (non-commercial) embeddings")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)

                    // Qwen3 loading progress
                    if modelManager.isQwen3Loading {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text(modelManager.qwen3LoadingMessage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                    }

                    // Downloaded Qwen3 models
                    if !modelManager.downloadedQwen3Models.isEmpty {
                        ForEach(modelManager.availableQwen3Models.filter { modelManager.downloadedQwen3Models.contains($0.id) }, id: \.id) { model in
                            Qwen3ModelRowView(
                                model: model,
                                size: modelManager.qwen3ModelSizes[model.id],
                                isLoaded: modelManager.loadedQwen3Variant == model.variant,
                                onDelete: {
                                    qwen3ModelToDelete = model
                                    showQwen3DeleteConfirmation = true
                                },
                                onLoad: {
                                    Task { await modelManager.loadQwen3Model(model.id) }
                                }
                            )
                            .padding(.horizontal)
                        }
                    }

                    // Available Qwen3 models to download
                    let availableQwen3 = modelManager.availableQwen3Models.filter { !modelManager.downloadedQwen3Models.contains($0.id) }
                    if !availableQwen3.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(availableQwen3, id: \.id) { model in
                                    AvailableQwen3ModelCard(model: model, modelManager: modelManager)
                                }
                            }
                            .padding(.horizontal)
                        }
                    } else if modelManager.downloadedQwen3Models.isEmpty {
                        Text("No Qwen3 models downloaded. Download one for FLUX.2 Klein embeddings.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.bottom)
                .background(Color.orange.opacity(0.05))

                Divider()

                // ===== DIFFUSION MODELS SECTION =====
                DiffusionModelsSection()
                    .environmentObject(modelManager)
            }
        }
        .alert("Delete Model", isPresented: $showDeleteConfirmation, presenting: modelToDelete) { model in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                Task {
                    try? await modelManager.deleteModel(model.id)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete \(model.name)? This cannot be undone.")
        }
        .alert("Delete Qwen3 Model", isPresented: $showQwen3DeleteConfirmation, presenting: qwen3ModelToDelete) { model in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                Task {
                    try? await modelManager.deleteQwen3Model(model.id)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete \(model.name)? This cannot be undone.")
        }
    }
}

// MARK: - Model Row View

struct ModelRowView: View {
    let model: ModelInfo
    let size: Int64?
    let isLoaded: Bool
    let onDelete: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .cornerRadius(4)
                    }
                }
                HStack(spacing: 8) {
                    Text(model.variant.displayName)
                    Text("•")
                    Text(model.parameters)
                    if let size = size {
                        Text("•")
                        Text(TextEncoderModelDownloader.formatSize(size))
                            .foregroundStyle(.blue)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if !isLoaded {
                Button("Load") {
                    onLoad()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .disabled(isLoaded)
            .help(isLoaded ? "Unload model first" : "Delete model")
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Available Model Card

struct AvailableModelCard: View {
    let model: ModelInfo
    @ObservedObject var modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.name)
                .font(.caption.bold())
                .lineLimit(1)

            HStack(spacing: 4) {
                Text(model.variant.estimatedSize)
                Text("•")
                Text(model.parameters)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Button(action: {
                Task { await modelManager.downloadModel(model.id) }
            }) {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download")
                }
                .font(.caption)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(modelManager.isDownloading)
        }
        .padding(10)
        .frame(width: 160)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Qwen3 Model Row View

struct Qwen3ModelRowView: View {
    let model: Qwen3ModelInfo
    let size: Int64?
    let isLoaded: Bool
    let onDelete: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .cornerRadius(4)
                    }
                    Text(model.variant.kleinVariant.displayName)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.orange.opacity(0.2))
                        .foregroundStyle(.orange)
                        .cornerRadius(4)
                }
                HStack(spacing: 8) {
                    Text(model.variant.displayName)
                    Text("•")
                    Text(model.parameters)
                    if let size = size {
                        Text("•")
                        Text(TextEncoderModelDownloader.formatSize(size))
                            .foregroundStyle(.blue)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if !isLoaded {
                Button("Load") {
                    onLoad()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .disabled(isLoaded)
            .help(isLoaded ? "Unload model first" : "Delete model")
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Available Qwen3 Model Card

struct AvailableQwen3ModelCard: View {
    let model: Qwen3ModelInfo
    @ObservedObject var modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.name)
                .font(.caption.bold())
                .lineLimit(1)

            HStack(spacing: 4) {
                Text(model.variant.estimatedSize)
                Text("•")
                Text(model.parameters)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Text(model.variant.kleinVariant.displayName)
                .font(.caption2)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(.orange.opacity(0.2))
                .foregroundStyle(.orange)
                .cornerRadius(4)

            Button(action: {
                Task { await modelManager.downloadQwen3Model(model.id) }
            }) {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download")
                }
                .font(.caption)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .controlSize(.small)
            .disabled(modelManager.isDownloading)
        }
        .padding(10)
        .frame(width: 160)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }
}

// MARK: - Diffusion Models Section

struct DiffusionModelsSection: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var transformerToDelete: ModelRegistry.TransformerVariant?
    @State private var showDeleteAlert = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Diffusion Models (Flux2Core)", systemImage: "photo.stack.fill")
                    .font(.headline)
                    .foregroundStyle(.purple)

                Spacer()

                Button(action: {
                    modelManager.refreshDownloadedDiffusionModels()
                }) {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh")
            }
            .padding(.horizontal)
            .padding(.top)

            Text("Transformer and VAE models for image generation")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal)

            // Transformers grouped by model type
            TransformerSection(
                title: "Flux.2 Dev (32B)",
                variants: [.bf16, .qint8],
                transformerToDelete: $transformerToDelete,
                showDeleteAlert: $showDeleteAlert
            )
            .environmentObject(modelManager)

            TransformerSection(
                title: "Flux.2 Klein 4B",
                variants: [.klein4B_bf16, .klein4B_8bit],
                transformerToDelete: $transformerToDelete,
                showDeleteAlert: $showDeleteAlert
            )
            .environmentObject(modelManager)

            TransformerSection(
                title: "Flux.2 Klein 9B",
                variants: [.klein9B_bf16],
                transformerToDelete: $transformerToDelete,
                showDeleteAlert: $showDeleteAlert
            )
            .environmentObject(modelManager)

            // VAE Section
            VAESection()
                .environmentObject(modelManager)
        }
        .padding(.bottom)
        .background(Color.purple.opacity(0.05))
        .alert("Delete Transformer", isPresented: $showDeleteAlert, presenting: transformerToDelete) { variant in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                try? modelManager.deleteTransformer(variant)
            }
        } message: { variant in
            let info = modelManager.transformerDisplayInfo(variant)
            Text("Are you sure you want to delete \(info.name)? This cannot be undone.")
        }
    }
}

struct TransformerSection: View {
    @EnvironmentObject var modelManager: ModelManager
    let title: String
    let variants: [ModelRegistry.TransformerVariant]
    @Binding var transformerToDelete: ModelRegistry.TransformerVariant?
    @Binding var showDeleteAlert: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
                .padding(.horizontal)

            ForEach(variants, id: \.self) { variant in
                let info = modelManager.transformerDisplayInfo(variant)
                let isDownloaded = modelManager.isTransformerDownloaded(variant)
                let size = modelManager.transformerSizes[variant.rawValue]

                HStack {
                    // Status indicator
                    Circle()
                        .fill(isDownloaded ? Color.green : Color.gray.opacity(0.3))
                        .frame(width: 8, height: 8)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(info.name)
                            .font(.caption.bold())
                        HStack(spacing: 4) {
                            Text("~\(info.size)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            if variant.isGated {
                                Label("Gated", systemImage: "lock.fill")
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                                    .help("Requires a Hugging Face token and accepted license")
                            }
                            if let size = size {
                                Text("(\(ModelManager.formatBytes(Int(size))))")
                                    .font(.caption2)
                                    .foregroundStyle(.blue)
                            }
                        }
                    }

                    Spacer()

                    if isDownloaded {
                        Button(action: {
                            transformerToDelete = variant
                            showDeleteAlert = true
                        }) {
                            Image(systemName: "trash")
                                .foregroundStyle(.red)
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button(action: {
                            Task { await modelManager.downloadTransformer(variant) }
                        }) {
                            Label("Download", systemImage: "arrow.down.circle")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(modelManager.isDownloading)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 4)
            }
        }
    }
}

struct VAESection: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var showDeleteAlert = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("VAE")
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
                .padding(.horizontal)

            HStack {
                Circle()
                    .fill(modelManager.isVAEDownloaded ? Color.green : Color.gray.opacity(0.3))
                    .frame(width: 8, height: 8)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Standard VAE")
                        .font(.caption.bold())
                    HStack(spacing: 4) {
                        Text("~3GB")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if modelManager.isVAEDownloaded && modelManager.vaeSize > 0 {
                            Text("(\(ModelManager.formatBytes(Int(modelManager.vaeSize))))")
                                .font(.caption2)
                                .foregroundStyle(.blue)
                        }
                    }
                }

                Spacer()

                if modelManager.isVAEDownloaded {
                    Button(action: { showDeleteAlert = true }) {
                        Image(systemName: "trash")
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                } else {
                    Button(action: {
                        Task { await modelManager.downloadVAE() }
                    }) {
                        Label("Download", systemImage: "arrow.down.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(modelManager.isDownloading)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 4)
        }
        .alert("Delete VAE", isPresented: $showDeleteAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                try? modelManager.deleteVAE()
            }
        } message: {
            Text("Are you sure you want to delete the VAE? This cannot be undone.")
        }
    }
}

