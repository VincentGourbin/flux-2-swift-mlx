// GenerationView.swift - Main generation interface
// Copyright 2025 Vincent Gourbin

import SwiftUI
import Flux2Core

struct GenerationView: View {
    @StateObject private var viewModel = GenerationViewModel()
    @State private var selectedMode = 0  // 0 = T2I, 1 = I2I

    var body: some View {
        HSplitView {
            // Left panel: Controls
            controlsPanel
                .frame(minWidth: 320, maxWidth: 400)

            // Right panel: Preview
            ImagePreviewView(image: viewModel.generatedImage)
                .frame(minWidth: 400)
        }
    }

    // MARK: - Controls Panel

    private var controlsPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Mode selector
                modeSelector

                Divider()

                // Prompt input
                promptSection

                // Reference images (I2I only)
                if selectedMode == 1 {
                    referenceImagesSection
                }

                Divider()

                // Generation parameters
                parametersSection

                Divider()

                // Seed section
                seedSection

                Spacer()

                // Generate button and progress
                generateSection
            }
            .padding()
        }
    }

    // MARK: - Mode Selector

    private var modeSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Generation Mode")
                .font(.headline)

            Picker("Mode", selection: $selectedMode) {
                Text("Text to Image").tag(0)
                Text("Image to Image").tag(1)
            }
            .pickerStyle(.segmented)
        }
    }

    // MARK: - Prompt Section

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.headline)

            TextEditor(text: $viewModel.prompt)
                .frame(height: 80)
                .font(.body)
                .border(Color.secondary.opacity(0.3))
                .cornerRadius(4)

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
    }

    // MARK: - Reference Images Section

    private var referenceImagesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Reference Images")
                    .font(.headline)
                Text("(1-3)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 8) {
                ForEach(0..<3, id: \.self) { index in
                    if index < viewModel.referenceImages.count {
                        ReferenceImageThumbnail(
                            image: viewModel.referenceImages[index],
                            onRemove: {
                                viewModel.referenceImages.remove(at: index)
                            }
                        )
                    } else if index == viewModel.referenceImages.count {
                        AddImageButton {
                            selectReferenceImage()
                        }
                    }
                }
            }
        }
    }

    // MARK: - Parameters Section

    private var parametersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Parameters")
                .font(.headline)

            // Dimensions (T2I only)
            if selectedMode == 0 {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Width: \(viewModel.width)")
                            .font(.caption)
                        Stepper("", value: $viewModel.width, in: 256...2048, step: 64)
                            .labelsHidden()
                    }

                    VStack(alignment: .leading) {
                        Text("Height: \(viewModel.height)")
                            .font(.caption)
                        Stepper("", value: $viewModel.height, in: 256...2048, step: 64)
                            .labelsHidden()
                    }
                }

                // Presets
                HStack {
                    Text("Presets:")
                        .font(.caption)
                    ForEach(GenerationPreset.allPresets, id: \.name) { preset in
                        Button(preset.name) {
                            viewModel.applyPreset(preset)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }

            // Steps
            HStack {
                Text("Steps: \(viewModel.steps)")
                Slider(value: Binding(
                    get: { Double(viewModel.steps) },
                    set: { viewModel.steps = Int($0) }
                ), in: 10...100, step: 1)
            }

            // Guidance
            HStack {
                Text("Guidance: \(viewModel.guidance, specifier: "%.1f")")
                Slider(value: $viewModel.guidance, in: 1...20, step: 0.5)
            }
        }
    }

    // MARK: - Seed Section

    private var seedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Toggle("Use Seed", isOn: $viewModel.useSeed)

                if viewModel.useSeed {
                    TextField("Seed", value: $viewModel.seed, format: .number)
                        .frame(width: 120)
                        .textFieldStyle(.roundedBorder)

                    Button("Random") {
                        viewModel.seed = UInt64.random(in: 0...UInt64.max)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
        }
    }

    // MARK: - Generate Section

    private var generateSection: some View {
        VStack(spacing: 12) {
            Button(action: {
                Task {
                    await viewModel.generate(mode: selectedMode)
                }
            }) {
                HStack {
                    if viewModel.isGenerating {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    Text(viewModel.isGenerating ? "Generating..." : "Generate")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(viewModel.isGenerating || viewModel.prompt.isEmpty)

            if viewModel.isGenerating {
                VStack(spacing: 4) {
                    ProgressView(value: viewModel.progress, total: 1.0)
                    Text("Step \(viewModel.currentStep)/\(viewModel.steps)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    // MARK: - Helpers

    private func selectReferenceImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            if let image = NSImage(contentsOf: url) {
                viewModel.addReferenceImage(image)
            }
        }
    }
}

// MARK: - Supporting Views

struct ReferenceImageThumbnail: View {
    let image: NSImage
    let onRemove: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 80, height: 80)
                .clipped()
                .cornerRadius(8)

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.white)
                    .background(Circle().fill(.black.opacity(0.5)))
            }
            .buttonStyle(.plain)
            .offset(x: 4, y: -4)
        }
    }
}

struct AddImageButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack {
                Image(systemName: "plus")
                    .font(.title2)
                Text("Add")
                    .font(.caption)
            }
            .frame(width: 80, height: 80)
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    GenerationView()
}
