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
    @StateObject private var viewModel = ImageGenerationViewModel(workflow: .textToImage)
    @StateObject private var paletteCoordinator = PaletteDetachCoordinator()
    @AppStorage("imageSaveUpscaleBy") private var imageSaveUpscaleBy = 1.0

    var body: some View {
        ZStack(alignment: .topLeading) {
            HSplitView {
                PaletteColumn {
                    PalettePanel(
                        storageKey: "t2i.parameters",
                        title: "Generation Parameters",
                        systemImage: "slider.horizontal.3",
                        coordinator: paletteCoordinator,
                        content: { parametersPaletteContent }
                    )

                    PalettePanel(
                        storageKey: "t2i.outputOptions",
                        title: "Output Options",
                        systemImage: "slider.horizontal.below.rectangle",
                        coordinator: paletteCoordinator,
                        content: { outputOptionsPaletteContent }
                    )
                }
                .frame(minWidth: 350, idealWidth: 400, maxWidth: 500)

                outputSection
            }

            floatingPalettes
        }
        .onAppear {
            // Refresh diffusion model status
            modelManager.refreshDownloadedModels()
            modelManager.refreshDownloadedDiffusionModels()
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: modelManager.downloadedTransformers,
                downloadedTextModels: modelManager.downloadedModels
            )
        }
        .onChange(of: viewModel.selectedModel) { _, newModel in
            // Apply recommended defaults when model changes
            if viewModel.shouldApplyDefaultsForModelChange() {
                viewModel.applyRecommendedDefaults(for: newModel)
            }
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: modelManager.downloadedTransformers,
                downloadedTextModels: modelManager.downloadedModels
            )
        }
        .onChange(of: modelManager.downloadedTransformers) { _, downloaded in
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: downloaded,
                downloadedTextModels: modelManager.downloadedModels
            )
        }
        .onChange(of: modelManager.downloadedModels) { _, downloaded in
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: modelManager.downloadedTransformers,
                downloadedTextModels: downloaded
            )
        }
        .focusedSceneValue(\.generationProjectCommands, GenerationProjectCommands(
            newProject: { viewModel.newProject() },
            openProject: { viewModel.openProject() },
            saveProject: { viewModel.saveProject() },
            saveProjectAs: { viewModel.saveProjectAs() }
        ))
        .focusedSceneValue(\.generationProjectName, viewModel.projectDisplayName)
        .focusedSceneValue(\.generationModelConfiguration, viewModel)
        .focusedSceneValue(\.generationUnloadModels) {
            Task { await viewModel.clearPipeline() }
        }
        .onDisappear {
            viewModel.persistSessionState()
        }
        .onReceive(NotificationCenter.default.publisher(for: .flux2PersistSession)) { _ in
            viewModel.persistSessionState()
        }
    }

    // MARK: - Floating palettes

    @ViewBuilder
    private var floatingPalettes: some View {
        if paletteCoordinator.isDetached("t2i.parameters") {
            PaletteFloatingPanel(
                storageKey: "t2i.parameters",
                title: "Generation Parameters",
                systemImage: "slider.horizontal.3",
                position: paletteCoordinator.positionBinding(for: "t2i.parameters"),
                coordinator: paletteCoordinator,
                content: { parametersPaletteContent }
            )
        }
        if paletteCoordinator.isDetached("t2i.outputOptions") {
            PaletteFloatingPanel(
                storageKey: "t2i.outputOptions",
                title: "Output Options",
                systemImage: "slider.horizontal.below.rectangle",
                position: paletteCoordinator.positionBinding(for: "t2i.outputOptions"),
                coordinator: paletteCoordinator,
                content: { outputOptionsPaletteContent }
            )
        }
    }

    // MARK: - Parameters Section

    @ViewBuilder
    private var parametersPaletteContent: some View {
        VStack(alignment: .leading, spacing: 12) {
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
                    HStack {
                        Text("Guidance: \(String(format: "%.1f", viewModel.guidance))")
                            .font(.caption)
                        Spacer()
                        Button("Default") {
                            viewModel.resetGuidanceToModelDefault()
                        }
                        .controlSize(.mini)
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.guidance) },
                        set: { viewModel.guidance = Float($0) }
                    ), in: 1...10, step: 0.5)
                }
            }

            Divider()

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

    // MARK: - Output Options

    @ViewBuilder
    private var outputOptionsPaletteContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                if viewModel.generatedImage != nil {
                    Button(action: { viewModel.saveImage() }) {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Button(action: { viewModel.clearPreview() }) {
                    Label("Clear Preview", systemImage: "xmark.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!viewModel.hasPreviewContent)
                .help("Clear the generated image from the preview pane")
            }

            if viewModel.generatedImage != nil {
                LanczosUpscaleField(factor: $imageSaveUpscaleBy)

                Button(action: { viewModel.openOutputFolder() }) {
                    Label("Open Folder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Open the image output folder in Finder")
            }
        }
    }

    // MARK: - Output Section

    @ViewBuilder
    private var outputSection: some View {
        VStack(spacing: 0) {
            ImageGenerationPromptSection(viewModel: viewModel)

            Divider()

            // Checkpoints row (if available)
            if viewModel.showCheckpoints && !viewModel.checkpointImages.isEmpty {
                checkpointsSection
                Divider()
            }

            // Main image display
            GeometryReader { geometry in
                if let cgImage = viewModel.generatedImage {
                    PreviewZoomableImageView(
                        image: cgImage,
                        zoomScale: Binding(
                            get: { CGFloat(viewModel.previewZoomScale) },
                            set: { viewModel.previewZoomScale = Double($0) }
                        )
                    )
                    .frame(width: geometry.size.width, height: geometry.size.height)
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
