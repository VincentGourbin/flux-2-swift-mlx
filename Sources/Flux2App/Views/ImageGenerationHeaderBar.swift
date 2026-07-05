/**
 * ImageGenerationHeaderBar.swift
 * Inline model configuration for the image-generation status bar.
 */

import SwiftUI
import Flux2Core
import FluxTextEncoders

#if canImport(AppKit)
import AppKit
#endif

private struct GenerationModelConfigurationKey: FocusedValueKey {
    typealias Value = ImageGenerationViewModel
}

extension FocusedValues {
    /// Active Text to Image / Image to Image view model, for the purple header bar.
    var generationModelConfiguration: ImageGenerationViewModel? {
        get { self[GenerationModelConfigurationKey.self] }
        set { self[GenerationModelConfigurationKey.self] = newValue }
    }
}

/// Left-side download / memory indicators for image generation tabs.
struct ImageGenerationHeaderLeftStatus: View {
    @EnvironmentObject var modelManager: ModelManager

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "photo.stack.fill")
                .foregroundStyle(.purple)
            Text("Image Generation")
                .font(.caption.bold())
                .foregroundStyle(.purple)
        }

        Divider()
            .frame(height: 16)
            .padding(.horizontal, 4)

        HStack(spacing: 4) {
            Circle()
                .fill(modelManager.isVAEDownloaded ? Color.green : Color.gray)
                .frame(width: 6, height: 6)
            Text("VAE")
                .font(.caption)
                .foregroundStyle(modelManager.isVAEDownloaded ? .primary : .secondary)
        }

        Text("MLX: \(ModelManager.formatBytes(modelManager.memoryStats.active))")
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)

        if modelManager.memoryStats.cache > 0 {
            Button(action: { modelManager.clearCache() }) {
                Label("Clear \(ModelManager.formatBytes(modelManager.memoryStats.cache))", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
        }
    }
}

/// Flush-right model pickers + memory estimate + ready indicator + unload.
struct ImageGenerationModelHeaderControls: View {
    @ObservedObject var viewModel: ImageGenerationViewModel
    @EnvironmentObject var modelManager: ModelManager
    var onUnload: () -> Void
    var unloadEnabled: Bool

    private var selectedVariant: ModelRegistry.TransformerVariant {
        viewModel.selectedTransformerVariant
    }

    private var isReady: Bool {
        modelManager.isTransformerDownloaded(selectedVariant) && modelManager.isVAEDownloaded
    }

    private var familyPickerWidth: CGFloat {
        headerMenuWidth(for: ModelFamily.allCases.map(\.displayName))
    }

    private var modelPickerWidth: CGFloat {
        headerMenuWidth(
            for: viewModel.selectableModels.map {
                ImageGenerationHeaderTitles.model($0, modelManager: modelManager)
            }
        )
    }

    private var encoderPickerWidth: CGFloat {
        headerMenuWidth(
            for: viewModel.downloadedTextQuantizations(in: modelManager.downloadedModels).map(\.displayName)
        )
    }

    private var transformerPickerWidth: CGFloat {
        headerMenuWidth(
            for: viewModel.compatibleTransformerQuantizations.map {
                ImageGenerationHeaderTitles.transformer($0, model: viewModel.selectedModel, modelManager: modelManager)
            }
        )
    }

    var body: some View {
        HStack(spacing: 10) {
            inlineField("Family") {
                Picker("", selection: $viewModel.selectedFamily) {
                    ForEach(ModelFamily.allCases) { family in
                        Text(family.displayName).tag(family as ModelFamily?)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: familyPickerWidth)
            }

            if let family = viewModel.selectedFamily {
                inlineField("Pixel factor") {
                    Text("\(family.pixelAlignment) px")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            inlineField("Model") {
                Picker("", selection: $viewModel.selectedModel) {
                    ForEach(viewModel.selectableModels, id: \.self) { model in
                        Text(ImageGenerationHeaderTitles.model(model, modelManager: modelManager)).tag(model)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: modelPickerWidth)
                .disabled(!viewModel.isFamilySelected)
            }

            if viewModel.selectedModel == .dev {
                inlineField("Encoder") {
                    Picker("", selection: $viewModel.textQuantization) {
                        ForEach(
                            viewModel.downloadedTextQuantizations(in: modelManager.downloadedModels),
                            id: \.self
                        ) { quant in
                            Text(quant.displayName).tag(quant)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(width: encoderPickerWidth)
                    .disabled(!viewModel.isFamilySelected)
                }
            }

            inlineField("Transformer") {
                Picker("", selection: $viewModel.transformerQuantization) {
                    ForEach(viewModel.compatibleTransformerQuantizations, id: \.self) { quant in
                        Text(
                            ImageGenerationHeaderTitles.transformer(
                                quant,
                                model: viewModel.selectedModel,
                                modelManager: modelManager
                            )
                        ).tag(quant)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: transformerPickerWidth)
                .disabled(!viewModel.isFamilySelected)
            }

            Button("Unload Models", action: onUnload)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!unloadEnabled)
                .help("Unload diffusion models from memory. The next Generate reloads them.")

            if modelManager.isDownloading {
                ProgressView(value: modelManager.downloadProgress)
                    .frame(width: 56)
            }

            HStack(spacing: 4) {
                Image(systemName: "memorychip")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("~\(viewModel.estimatedPeakMemoryGB)GB")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .help("Estimated peak memory for the current model configuration")

            readyIndicator

            ImageGenerationHeaderGenerateControls(viewModel: viewModel)
        }
    }

    @ViewBuilder
    private var readyIndicator: some View {
        if isReady {
            HStack(spacing: 4) {
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
                Text("Ready")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
            .help("Transformer and VAE are downloaded for the current selection")
        } else {
            Menu {
                if !modelManager.isTransformerDownloaded(selectedVariant) {
                    Button("Download Transformer") {
                        Task { await modelManager.downloadTransformer(selectedVariant) }
                    }
                }
                if !modelManager.isVAEDownloaded {
                    Button("Download VAE") {
                        Task { await modelManager.downloadVAE() }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Circle()
                        .fill(Color.orange)
                        .frame(width: 8, height: 8)
                    Text("Download")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Required weights are missing — click to download")
            .disabled(modelManager.isDownloading)
        }
    }

    private func headerMenuWidth(for strings: [String], minimum: CGFloat = 80) -> CGFloat {
        #if canImport(AppKit)
        let font = NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)
        let widest = strings.map { text in
            (text as NSString).size(withAttributes: [.font: font]).width
        }.max() ?? minimum
        return max(minimum, ceil(widest) + 36)
        #else
        return minimum
        #endif
    }

    private func inlineField<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            content()
        }
    }
}

/// Generate / cancel / progress — flush right in the purple header after the Ready divider.
struct ImageGenerationHeaderGenerateControls: View {
    @ObservedObject var viewModel: ImageGenerationViewModel
    @EnvironmentObject var modelManager: ModelManager

    private var generateEnabled: Bool {
        guard viewModel.canGenerate,
              modelManager.isTransformerDownloaded(viewModel.selectedTransformerVariant),
              modelManager.isVAEDownloaded else {
            return false
        }
        if viewModel.requiresReferenceImages {
            return viewModel.hasPrimaryReference
        }
        return true
    }

    var body: some View {
        HStack(spacing: 8) {
            if let error = viewModel.errorMessage, !viewModel.isGenerating {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .help(error)
            }

            if viewModel.isResetting {
                ProgressView()
                    .controlSize(.small)
            } else if viewModel.isGenerating {
                ProgressView(value: viewModel.progress)
                    .frame(width: 64)

                Text(viewModel.statusMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .frame(maxWidth: 140, alignment: .leading)

                Button("Cancel", role: .cancel) {
                    viewModel.cancel()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.red)
            } else {
                Divider()
                    .frame(height: 16)

                Button {
                    viewModel.startGeneration()
                } label: {
                    Text("Generate")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(!generateEnabled)
            }
        }
    }
}

@MainActor
enum ImageGenerationHeaderTitles {
    static func model(_ model: Flux2Model, modelManager: ModelManager) -> String {
        let hasDownloadedTransformer = ImageGenerationViewModel.compatibleTransformerQuantizations(for: model)
            .contains { quantization in
                let variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization)
                return modelManager.isTransformerDownloaded(variant)
            }
        return hasDownloadedTransformer ? model.displayName : "\(model.displayName) (not downloaded)"
    }

    static func transformer(
        _ quantization: TransformerQuantization,
        model: Flux2Model,
        modelManager: ModelManager
    ) -> String {
        let variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization)
        let suffix = modelManager.isTransformerDownloaded(variant) ? "" : " (not downloaded)"
        return "\(quantization.displayName)\(suffix)"
    }
}
