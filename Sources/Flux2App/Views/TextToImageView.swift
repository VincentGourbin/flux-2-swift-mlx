/**
 * TextToImageView.swift
 * Text-to-Image generation interface for Flux.2
 */

import SwiftUI
import Flux2Core
import FluxTextEncoders

#if canImport(AppKit)
import AppKit
#endif

struct TextToImageView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = ImageGenerationViewModel()

    var body: some View {
        HSplitView {
            // Left panel: Controls
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Model Selection Section
                    modelSelectionSection

                    Divider()

                    // Prompt Section
                    promptSection

                    Divider()

                    // Parameters Section
                    parametersSection

                    Divider()

                    // Generate Button
                    generateSection
                }
                .padding()
            }
            .frame(minWidth: 350, idealWidth: 400, maxWidth: 500)

            // Right panel: Output
            outputSection
        }
        .onAppear {
            // Refresh diffusion model status
            modelManager.refreshDownloadedDiffusionModels()
        }
        .onChange(of: viewModel.selectedModel) { _, newModel in
            // Apply recommended defaults when model changes
            viewModel.applyRecommendedDefaults(for: newModel)
        }
    }

    // MARK: - Model Selection Section

    @ViewBuilder
    private var modelSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Model Configuration", systemImage: "cpu")
                .font(.headline)

            // Model type picker
            HStack {
                Text("Model:")
                    .frame(width: 100, alignment: .leading)
                Picker("", selection: $viewModel.selectedModel) {
                    ForEach(Flux2Model.allCases, id: \.self) { model in
                        Text(model.displayName).tag(model)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }

            // Text encoder quantization (only for Dev model)
            if viewModel.selectedModel == .dev {
                HStack {
                    Text("Text Encoder:")
                        .frame(width: 100, alignment: .leading)
                    Picker("", selection: $viewModel.textQuantization) {
                        ForEach(MistralQuantization.allCases, id: \.self) { quant in
                            Text(quant.displayName).tag(quant)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity)
                }
            }

            // Transformer quantization
            HStack {
                Text("Transformer:")
                    .frame(width: 100, alignment: .leading)
                Picker("", selection: $viewModel.transformerQuantization) {
                    ForEach(TransformerQuantization.allCases, id: \.self) { quant in
                        // Klein 9B only supports bf16
                        if viewModel.selectedModel != .klein9B || quant == .bf16 {
                            Text(quant.displayName).tag(quant)
                        }
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }

            // Memory estimate
            HStack {
                Image(systemName: "memorychip")
                    .foregroundStyle(.secondary)
                Text("Estimated peak: ~\(viewModel.estimatedPeakMemoryGB)GB")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                // Model download status
                let variant = viewModel.selectedTransformerVariant
                if modelManager.isTransformerDownloaded(variant) {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
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

            // VAE status
            HStack {
                Text("VAE:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if modelManager.isVAEDownloaded {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                } else {
                    Button(action: {
                        Task { await modelManager.downloadVAE() }
                    }) {
                        Label("Download VAE", systemImage: "arrow.down.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(modelManager.isDownloading)
                }
                Spacer()
            }

            // Download progress
            if modelManager.isDownloading {
                VStack(spacing: 4) {
                    ProgressView(value: modelManager.downloadProgress)
                    Text(modelManager.downloadMessage)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Prompt Section

    @ViewBuilder
    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Prompt", systemImage: "text.cursor")
                .font(.headline)

            TextEditor(text: $viewModel.prompt)
                .font(.body)
                .scrollContentBackground(.hidden)
                .background(Color(nsColor: .textBackgroundColor))
                .cornerRadius(8)
                .frame(minHeight: 100, maxHeight: 200)

            Toggle("Upsample prompt", isOn: $viewModel.upsamplePrompt)
                .font(.caption)
                .help("Enhance prompt with visual details using Mistral")
        }
    }

    // MARK: - Parameters Section

    @ViewBuilder
    private var parametersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Generation Parameters", systemImage: "slider.horizontal.3")
                .font(.headline)

            // Dimensions
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Width: \(viewModel.width)")
                        .font(.caption)
                    Slider(value: Binding(
                        get: { Double(viewModel.width) },
                        set: { viewModel.width = Int($0) }
                    ), in: 256...2048, step: 64)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Height: \(viewModel.height)")
                        .font(.caption)
                    Slider(value: Binding(
                        get: { Double(viewModel.height) },
                        set: { viewModel.height = Int($0) }
                    ), in: 256...2048, step: 64)
                }
            }

            // Quick dimension presets
            HStack {
                Text("Presets:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("512") {
                    viewModel.width = 512
                    viewModel.height = 512
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                Button("1024") {
                    viewModel.width = 1024
                    viewModel.height = 1024
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                Button("Portrait") {
                    viewModel.width = 768
                    viewModel.height = 1024
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                Button("Landscape") {
                    viewModel.width = 1024
                    viewModel.height = 768
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }

            // Steps and Guidance
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Steps: \(viewModel.steps)")
                        .font(.caption)
                    Slider(value: Binding(
                        get: { Double(viewModel.steps) },
                        set: { viewModel.steps = Int($0) }
                    ), in: 4...100, step: 1)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Guidance: \(String(format: "%.1f", viewModel.guidance))")
                        .font(.caption)
                    Slider(value: Binding(
                        get: { Double(viewModel.guidance) },
                        set: { viewModel.guidance = Float($0) }
                    ), in: 1...10, step: 0.5)
                }
            }

            // Seed
            HStack {
                Text("Seed:")
                    .font(.caption)
                TextField("Random", text: $viewModel.seed)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                Button(action: {
                    viewModel.seed = String(UInt64.random(in: 0...UInt64.max))
                }) {
                    Image(systemName: "dice")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Generate random seed")
            }
        }
    }

    // MARK: - Generate Section

    @ViewBuilder
    private var generateSection: some View {
        VStack(spacing: 12) {
            // Generate button
            Button(action: {
                Task { await viewModel.generate() }
            }) {
                HStack {
                    if viewModel.isGenerating {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    Text(viewModel.isGenerating ? "Generating..." : "Generate Image")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!viewModel.canGenerate || !modelManager.isTransformerDownloaded(viewModel.selectedTransformerVariant) || !modelManager.isVAEDownloaded)

            // Progress
            if viewModel.isGenerating {
                VStack(spacing: 4) {
                    ProgressView(value: viewModel.progress)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Error message
            if let error = viewModel.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                .padding(8)
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }
        }
    }

    // MARK: - Output Section

    @ViewBuilder
    private var outputSection: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("Generated Image", systemImage: "photo")
                    .font(.headline)

                Spacer()

                if viewModel.generatedImage != nil {
                    Button(action: { viewModel.saveImage() }) {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Button(action: {
                    Task { await viewModel.clearPipeline() }
                }) {
                    Label("Clear Memory", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Clear pipeline to free GPU memory")
            }
            .padding()

            Divider()

            // Checkpoints row (if available)
            if viewModel.showCheckpoints && !viewModel.checkpointImages.isEmpty {
                checkpointsSection
                Divider()
            }

            // Main image display
            GeometryReader { geometry in
                if let cgImage = viewModel.generatedImage {
                    ScrollView([.horizontal, .vertical]) {
                        Image(decorative: cgImage, scale: 1.0)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(
                                maxWidth: geometry.size.width,
                                maxHeight: geometry.size.height
                            )
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .background(Color(nsColor: .windowBackgroundColor))
                } else {
                    VStack {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 64))
                            .foregroundStyle(.secondary.opacity(0.5))
                        Text("Generated image will appear here")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        if !viewModel.statusMessage.isEmpty && !viewModel.isGenerating {
                            Text(viewModel.statusMessage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.top, 8)
                        }
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .background(Color(nsColor: .windowBackgroundColor))
                }
            }
        }
    }

    // MARK: - Checkpoints Section

    @ViewBuilder
    private var checkpointsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Checkpoints", systemImage: "clock.arrow.circlepath")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)

                Spacer()

                Button(action: { viewModel.clearCheckpoints() }) {
                    Text("Clear")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
            .padding(.horizontal)
            .padding(.top, 8)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.checkpointImages) { checkpoint in
                        VStack(spacing: 2) {
                            Image(decorative: checkpoint.image, scale: 1.0)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 80, height: 80)
                                .cornerRadius(4)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 4)
                                        .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                                )

                            Text("Step \(checkpoint.step)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.horizontal)
            }
            .frame(height: 110)
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }
}

#Preview {
    TextToImageView()
        .environmentObject(ModelManager())
        .frame(width: 1200, height: 800)
}
