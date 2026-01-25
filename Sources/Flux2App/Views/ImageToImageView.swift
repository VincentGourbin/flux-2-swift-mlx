/**
 * ImageToImageView.swift
 * Image-to-Image generation interface for Flux.2
 */

import SwiftUI
import Flux2Core
import FluxTextEncoders
import UniformTypeIdentifiers

#if canImport(AppKit)
import AppKit
#endif

struct ImageToImageView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = ImageGenerationViewModel()
    @State private var isTargetedForDrop = false

    var body: some View {
        HSplitView {
            // Left panel: Controls
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Reference Images Section
                    referenceImagesSection

                    Divider()

                    // Model Selection Section
                    modelSelectionSection

                    Divider()

                    // Prompt Section
                    promptSection

                    Divider()

                    // I2I Parameters Section
                    i2iParametersSection

                    Divider()

                    // Standard Parameters Section
                    parametersSection

                    Divider()

                    // Optional: Interpret Images Section
                    interpretImagesSection

                    Divider()

                    // Generate Button
                    generateSection
                }
                .padding()
            }
            .frame(minWidth: 380, idealWidth: 450, maxWidth: 550)

            // Right panel: Output
            outputSection
        }
        .onAppear {
            modelManager.refreshDownloadedDiffusionModels()
        }
        .onChange(of: viewModel.selectedModel) { _, newModel in
            viewModel.applyRecommendedDefaults(for: newModel)
        }
    }

    // MARK: - Reference Images Section

    @ViewBuilder
    private var referenceImagesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Reference Images (1-3)", systemImage: "photo.stack")
                    .font(.headline)

                Spacer()

                if !viewModel.referenceImages.isEmpty {
                    Button(action: { viewModel.clearReferenceImages() }) {
                        Label("Clear All", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            Text("Drop images here or click to add. The model will use these as reference for generation.")
                .font(.caption)
                .foregroundStyle(.secondary)

            // Drop zone with image slots
            HStack(spacing: 12) {
                ForEach(0..<3, id: \.self) { index in
                    if index < viewModel.referenceImages.count {
                        // Show existing image
                        ReferenceImageSlot(
                            image: viewModel.referenceImages[index],
                            onRemove: {
                                viewModel.removeReferenceImage(viewModel.referenceImages[index].id)
                            }
                        )
                    } else if index == viewModel.referenceImages.count {
                        // Show add button for next slot
                        AddImageSlot(onAdd: { selectImage() })
                            .onDrop(of: [.image], isTargeted: $isTargetedForDrop) { providers in
                                handleImageDrop(providers)
                                return true
                            }
                    } else {
                        // Empty placeholder
                        EmptyImageSlot()
                    }
                }
            }
            .frame(height: 120)
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

            // Text encoder quantization (only for Dev)
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
                        if viewModel.selectedModel != .klein9B || quant == .bf16 {
                            Text(quant.displayName).tag(quant)
                        }
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }

            // Memory and download status
            HStack {
                Image(systemName: "memorychip")
                    .foregroundStyle(.secondary)
                Text("~\(viewModel.estimatedPeakMemoryGB)GB")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                let variant = viewModel.selectedTransformerVariant
                if modelManager.isTransformerDownloaded(variant) && modelManager.isVAEDownloaded {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                } else {
                    VStack(alignment: .trailing, spacing: 4) {
                        if !modelManager.isTransformerDownloaded(variant) {
                            Button("Download Transformer") {
                                Task { await modelManager.downloadTransformer(variant) }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                        if !modelManager.isVAEDownloaded {
                            Button("Download VAE") {
                                Task { await modelManager.downloadVAE() }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                    }
                    .disabled(modelManager.isDownloading)
                }
            }

            if modelManager.isDownloading {
                ProgressView(value: modelManager.downloadProgress)
                Text(modelManager.downloadMessage)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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
                .frame(minHeight: 80, maxHeight: 150)

            Toggle("Upsample prompt", isOn: $viewModel.upsamplePrompt)
                .font(.caption)
                .help("Enhance prompt using VLM to analyze reference images")
        }
    }

    // MARK: - I2I Parameters Section

    @ViewBuilder
    private var i2iParametersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Image-to-Image Settings", systemImage: "photo.on.rectangle")
                .font(.headline)

            // Strength slider
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Strength: \(String(format: "%.2f", viewModel.strength))")
                        .font(.caption)
                    Spacer()
                    Text("Preserve")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("<-->")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("Transform")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Slider(value: $viewModel.strength, in: 0.1...1.0, step: 0.05)
            }

            Text("Lower strength preserves more of the reference image structure. Higher strength allows more creative transformation.")
                .font(.caption2)
                .foregroundStyle(.secondary)
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

            // Match reference button
            if let firstRef = viewModel.referenceImages.first {
                Button(action: {
                    viewModel.width = firstRef.image.width
                    viewModel.height = firstRef.image.height
                }) {
                    Label("Match Reference Size", systemImage: "arrow.up.left.and.arrow.down.right")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
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
            }
        }
    }

    // MARK: - Interpret Images Section

    @ViewBuilder
    private var interpretImagesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Interpret Images (Optional)", systemImage: "eye")
                    .font(.subheadline.bold())

                Spacer()

                Button(action: selectInterpretImage) {
                    Label("Add", systemImage: "plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Text("VLM will analyze these images and inject descriptions into the prompt. Different from reference images - these provide semantic context only.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if !viewModel.interpretImageURLs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(viewModel.interpretImageURLs, id: \.self) { url in
                            InterpretImageThumbnail(url: url) {
                                viewModel.interpretImageURLs.removeAll { $0 == url }
                            }
                        }
                    }
                }
                .frame(height: 60)
            }
        }
    }

    // MARK: - Generate Section

    @ViewBuilder
    private var generateSection: some View {
        VStack(spacing: 12) {
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
            .disabled(!canGenerate)

            if viewModel.isGenerating {
                VStack(spacing: 4) {
                    ProgressView(value: viewModel.progress)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

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

                    Button(action: { useAsReference() }) {
                        Label("Use as Reference", systemImage: "arrow.uturn.left")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.referenceImages.count >= 3)
                }

                Button(action: {
                    Task { await viewModel.clearPipeline() }
                }) {
                    Label("Clear Memory", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
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

                        if viewModel.referenceImages.isEmpty {
                            Text("Add at least one reference image to start")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .padding(.top, 4)
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

    // MARK: - Helpers

    private var canGenerate: Bool {
        !viewModel.prompt.isEmpty &&
        !viewModel.referenceImages.isEmpty &&
        !viewModel.isGenerating &&
        modelManager.isTransformerDownloaded(viewModel.selectedTransformerVariant) &&
        modelManager.isVAEDownloaded
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK {
            for url in panel.urls.prefix(3 - viewModel.referenceImages.count) {
                viewModel.addReferenceImage(from: url)
            }
        }
    }

    private func selectInterpretImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true

        if panel.runModal() == .OK {
            viewModel.interpretImageURLs.append(contentsOf: panel.urls)
        }
    }

    private func handleImageDrop(_ providers: [NSItemProvider]) {
        for provider in providers {
            if provider.canLoadObject(ofClass: NSImage.self) {
                _ = provider.loadObject(ofClass: NSImage.self) { image, _ in
                    if let nsImage = image as? NSImage {
                        DispatchQueue.main.async {
                            viewModel.addReferenceImage(from: nsImage)
                        }
                    }
                }
            }
        }
    }

    private func useAsReference() {
        guard let cgImage = viewModel.generatedImage else { return }

        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        viewModel.addReferenceImage(from: nsImage)
    }
}

// MARK: - Supporting Views

struct ReferenceImageSlot: View {
    let image: ReferenceImage
    let onRemove: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Image(nsImage: image.thumbnail)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 100, height: 100)
                .clipped()
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.accentColor, lineWidth: 2)
                )

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.white)
                    .background(Circle().fill(.red))
            }
            .buttonStyle(.plain)
            .offset(x: 8, y: -8)
        }
    }
}

struct AddImageSlot: View {
    let onAdd: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: onAdd) {
            VStack {
                Image(systemName: "plus.circle.fill")
                    .font(.title)
                Text("Add")
                    .font(.caption)
            }
            .foregroundStyle(isHovering ? Color.accentColor : .secondary)
            .frame(width: 100, height: 100)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [5]))
                    .foregroundStyle(isHovering ? Color.accentColor : .secondary)
            )
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
    }
}

struct EmptyImageSlot: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5]))
            .foregroundStyle(.secondary.opacity(0.3))
            .frame(width: 100, height: 100)
    }
}

struct InterpretImageThumbnail: View {
    let url: URL
    let onRemove: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            if let nsImage = NSImage(contentsOf: url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 50, height: 50)
                    .clipped()
                    .cornerRadius(4)
            } else {
                Rectangle()
                    .fill(.secondary.opacity(0.2))
                    .frame(width: 50, height: 50)
                    .cornerRadius(4)
            }

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .background(Circle().fill(.red).frame(width: 14, height: 14))
            }
            .buttonStyle(.plain)
            .offset(x: 4, y: -4)
        }
    }
}

#Preview {
    ImageToImageView()
        .environmentObject(ModelManager())
        .frame(width: 1200, height: 900)
}
