/**
 * ModelStatusBar.swift
 * Contextual model status bar shown above the workspace. Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Model Status Bar (Contextual)

struct ModelStatusBar: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false
    @FocusedValue(\.generationUnloadModels) private var unloadModels
    @FocusedValue(\.generationModelConfiguration) private var generationViewModel
    let selectedTab: Int

    /// Is this a Qwen3-focused tab?
    private var isQwen3Tab: Bool {
        selectedTab == 3  // Qwen3 Chat
    }

    /// Is this an Image Generation tab?
    private var isImageGenerationTab: Bool {
        selectedTab == 4 || selectedTab == 5  // T2I or I2I
    }

    /// Is this a Tools tab (shows both models)?
    private var isToolsTab: Bool {
        selectedTab == 6 || selectedTab == 7  // FLUX.2 Tools or Models
    }

    var body: some View {
        HStack {
            if isImageGenerationTab {
                // === IMAGE GENERATION STATUS BAR ===
                imageGenerationStatusBar
            } else if isQwen3Tab {
                // === QWEN3 STATUS BAR ===
                qwen3StatusBar
            } else if isToolsTab {
                // === TOOLS TAB: Show both models status ===
                toolsStatusBar
            } else {
                // === MISTRAL STATUS BAR ===
                mistralStatusBar
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(
            isImageGenerationTab ? Color.purple.opacity(0.05) :
            isQwen3Tab ? Color.orange.opacity(0.05) :
            Color(NSColor.controlBackgroundColor)
        )
        .onAppear {
            if detailedProfiling { FluxProfiler.shared.enable() } else { FluxProfiler.shared.disable() }
        }
    }

    // MARK: - Mistral Status Bar

    @ViewBuilder
    private var mistralStatusBar: some View {
        // Model status indicator
        Circle()
            .fill(modelManager.isLoaded ? Color.green : Color.red)
            .frame(width: 8, height: 8)

        Text(modelManager.isLoaded ? "Mistral Loaded" : "Mistral Not Loaded")
            .font(.caption)
            .foregroundColor(.secondary)

        // VLM indicator
        if modelManager.isVLMLoaded {
            Text("VLM")
                .font(.caption.bold())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.blue.opacity(0.2))
                .foregroundColor(.blue)
                .cornerRadius(4)
        }

        if modelManager.isLoading {
            ProgressView()
                .scaleEffect(0.6)
            Text(modelManager.loadingMessage)
                .font(.caption)
                .foregroundColor(.secondary)
        }

        Spacer()

        // Detailed profiling toggle
        Toggle("Detailed Profiling", isOn: $detailedProfiling)
            .toggleStyle(.checkbox)
            .font(.caption)
            .onChange(of: detailedProfiling) { _, newValue in
                if newValue { FluxProfiler.shared.enable() } else { FluxProfiler.shared.disable() }
            }
            .help("Enable detailed profiling and memory logging")

        Divider()
            .frame(height: 20)
            .padding(.horizontal, 8)

        // Model variant picker
        HStack(spacing: 0) {
            Text("Model")
                .foregroundColor(.secondary)
                .padding(.trailing, 8)

            ForEach(ModelVariant.allCases, id: \.self) { variant in
                let isDownloaded = isVariantDownloaded(variant)
                let isSelected = modelManager.selectedVariant == variant
                Button(action: {
                    if isDownloaded {
                        modelManager.selectedVariant = variant
                    }
                }) {
                    Text(variant.shortName)
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            isSelected
                                ? Color.accentColor
                                : (isDownloaded ? Color.gray.opacity(0.3) : Color.gray.opacity(0.1))
                        )
                        .foregroundColor(
                            isSelected
                                ? .white
                                : (isDownloaded ? .primary : .secondary.opacity(0.5))
                        )
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(!isDownloaded)
                .help(isDownloaded ? variant.displayName : "\(variant.displayName) - Not downloaded")
            }
        }

        // Load/Unload button
        Button(action: {
            Task {
                if modelManager.isLoaded {
                    modelManager.unloadModel()
                } else {
                    await modelManager.loadModel()
                }
            }
        }) {
            Text(modelManager.isLoaded ? "Unload" : "Load Mistral")
        }
        .disabled(modelManager.isLoading || (!modelManager.isLoaded && modelManager.selectedVariant == nil))
    }

    // MARK: - Qwen3 Status Bar

    @ViewBuilder
    private var qwen3StatusBar: some View {
        // Model status indicator
        Circle()
            .fill(modelManager.isQwen3Loaded ? Color.green : Color.red)
            .frame(width: 8, height: 8)

        Text(modelManager.isQwen3Loaded ? "Qwen3 Loaded" : "Qwen3 Not Loaded")
            .font(.caption)
            .foregroundColor(.secondary)

        // Loaded variant indicator
        if let variant = modelManager.loadedQwen3Variant {
            Text(variant.displayName)
                .font(.caption.bold())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.orange.opacity(0.2))
                .foregroundColor(.orange)
                .cornerRadius(4)
        }

        if modelManager.isQwen3Loading {
            ProgressView()
                .scaleEffect(0.6)
            Text(modelManager.qwen3LoadingMessage)
                .font(.caption)
                .foregroundColor(.secondary)
        }

        Spacer()

        // Qwen3 variant picker
        HStack(spacing: 0) {
            Text("Model")
                .foregroundColor(.secondary)
                .padding(.trailing, 8)

            ForEach(Qwen3Variant.allCases, id: \.self) { variant in
                let modelInfo = TextEncoderModelRegistry.shared.qwen3Model(withVariant: variant)
                let modelId = modelInfo?.id ?? ""
                let isDownloaded = modelManager.downloadedQwen3Models.contains(modelId)
                let isSelected = modelManager.loadedQwen3Variant == variant
                Button(action: {
                    if isDownloaded && !isSelected {
                        Task { await modelManager.loadQwen3Model(modelId) }
                    }
                }) {
                    Text(variant.shortName)
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            isSelected
                                ? Color.orange
                                : (isDownloaded ? Color.gray.opacity(0.3) : Color.gray.opacity(0.1))
                        )
                        .foregroundColor(
                            isSelected
                                ? .white
                                : (isDownloaded ? .primary : .secondary.opacity(0.5))
                        )
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(!isDownloaded || modelManager.isQwen3Loading)
                .help(isDownloaded ? variant.displayName : "\(variant.displayName) - Not downloaded")
            }
        }

        // Load/Unload button
        Button(action: {
            if modelManager.isQwen3Loaded {
                modelManager.unloadQwen3Model()
            }
        }) {
            Text(modelManager.isQwen3Loaded ? "Unload Qwen3" : "Select Model")
        }
        .disabled(!modelManager.isQwen3Loaded || modelManager.isQwen3Loading)
    }

    // MARK: - Tools Status Bar (Both Models)

    @ViewBuilder
    private var toolsStatusBar: some View {
        // Mistral status
        HStack(spacing: 4) {
            Circle()
                .fill(modelManager.isLoaded ? Color.green : Color.gray)
                .frame(width: 6, height: 6)
            Text("Mistral")
                .font(.caption)
                .foregroundColor(modelManager.isLoaded ? .primary : .secondary)
            if modelManager.isVLMLoaded {
                Text("VLM")
                    .font(.caption2.bold())
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Color.blue.opacity(0.2))
                    .foregroundColor(.blue)
                    .cornerRadius(3)
            }
        }

        Divider()
            .frame(height: 16)
            .padding(.horizontal, 8)

        // Qwen3 status
        HStack(spacing: 4) {
            Circle()
                .fill(modelManager.isQwen3Loaded ? Color.green : Color.gray)
                .frame(width: 6, height: 6)
            Text("Qwen3")
                .font(.caption)
                .foregroundColor(modelManager.isQwen3Loaded ? .orange : .secondary)
            if let variant = modelManager.loadedQwen3Variant {
                Text(variant.shortName)
                    .font(.caption2.bold())
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Color.orange.opacity(0.2))
                    .foregroundColor(.orange)
                    .cornerRadius(3)
            }
        }

        Spacer()

        // Detailed profiling toggle
        Toggle("Detailed Profiling", isOn: $detailedProfiling)
            .toggleStyle(.checkbox)
            .font(.caption)
            .onChange(of: detailedProfiling) { _, newValue in
                if newValue { FluxProfiler.shared.enable() } else { FluxProfiler.shared.disable() }
            }

        Text("Manage models in Models tab")
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    // MARK: - Image Generation Status Bar

    @ViewBuilder
    private var imageGenerationStatusBar: some View {
        ImageGenerationHeaderLeftStatus()

        Spacer(minLength: 12)

        if let generationViewModel {
            ImageGenerationModelHeaderControls(
                viewModel: generationViewModel,
                onUnload: { unloadModels?() },
                unloadEnabled: unloadModels != nil
            )
        }
    }

    // MARK: - Helpers

    private func isVariantDownloaded(_ variant: ModelVariant) -> Bool {
        guard let model = TextEncoderModelRegistry.shared.model(withVariant: variant) else { return false }
        return modelManager.downloadedModels.contains(model.id)
    }
}

