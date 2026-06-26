/**
 * VisionView.swift
 * Vision (VLM) inspection tab. Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Vision View

struct VisionView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var selectedImage: NSImage?
    @State private var prompt = "Describe this image in detail."
    @State private var output = ""
    @State private var isProcessing = false
    @State private var currentTokenCount = 0
    @State private var generationStats: GenerationStats?
    @State private var maxTokens: Double = 1024
    @State private var temperature = 0.7

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                // Image drop zone
                GroupBox("Image") {
                    ZStack {
                        if let image = selectedImage {
                            Image(nsImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        } else {
                            VStack {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.largeTitle)
                                    .foregroundColor(.secondary)
                                Text("Drop image here or click to select")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .frame(minWidth: 300, minHeight: 300)
                    .onDrop(of: [.image], isTargeted: nil) { providers in
                        loadImage(from: providers)
                        return true
                    }
                    .onTapGesture {
                        selectImage()
                    }
                }

                // Prompt and output
                VStack(spacing: 16) {
                    GroupBox("Prompt") {
                        TextField("What do you want to know about this image?", text: $prompt)
                            .textFieldStyle(.plain)
                    }

                    // Parameters
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Tokens: \(Int(maxTokens))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: $maxTokens, in: 128...4096, step: 128)
                                .frame(width: 150)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Temperature: \(String(format: "%.1f", temperature))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: $temperature, in: 0...2, step: 0.1)
                                .frame(width: 120)
                        }
                    }

                    HStack {
                        // VLM status - model is now loaded with VLM by default
                        if !modelManager.isVLMLoaded {
                            Label("Load model from top bar", systemImage: "arrow.up.circle")
                                .foregroundStyle(.secondary)
                                .font(.caption)
                        } else {
                            Label("VLM Ready", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.caption)
                        }

                        Spacer()

                        Button(action: processImage) {
                            HStack {
                                if isProcessing {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                }
                                Text(isProcessing ? "Analyzing..." : "Analyze Image")
                            }
                        }
                        .disabled(selectedImage == nil || isProcessing || !modelManager.isVLMLoaded)
                    }

                    GroupBox("Response") {
                        ScrollView {
                            Text(output.isEmpty ? "Load a model from the top bar, select an image, and click Analyze" : output)
                                .font(.body)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .foregroundColor(output.isEmpty ? .secondary : .primary)
                                .textSelection(.enabled)
                        }
                        .frame(minHeight: 200)
                    }

                    // Stats bar
                    if isProcessing {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text("Generating (\(currentTokenCount) tokens)...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                    } else if let stats = generationStats {
                        HStack {
                            Label("\(stats.tokenCount) tokens", systemImage: "number")
                            Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
                            Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")
                            Spacer()
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                    }
                }
            }

            Spacer()
        }
        .padding()
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK, let url = panel.url {
            selectedImage = NSImage(contentsOf: url)
        }
    }

    private func loadImage(from providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        provider.loadObject(ofClass: NSImage.self) { image, _ in
            if let image = image as? NSImage {
                DispatchQueue.main.async {
                    selectedImage = image
                }
            }
        }
    }

    private func processImage() {
        guard let image = selectedImage else { return }

        isProcessing = true
        output = ""
        currentTokenCount = 0
        generationStats = nil

        let startTime = Date()
        let params = GenerateParameters(
            maxTokens: Int(maxTokens),
            temperature: Float(temperature)
        )
        let userPrompt = prompt

        // Run inference on background thread to keep UI responsive
        Task.detached(priority: .userInitiated) {
            do {
                let result = try FluxTextEncoders.shared.analyzeImage(
                    image: image,
                    prompt: userPrompt,
                    parameters: params
                ) { token in
                    // Stream tokens to UI
                    Task { @MainActor in
                        output += token
                        currentTokenCount += 1
                    }
                    return true
                }

                await MainActor.run {
                    // Don't overwrite streamed output, just update stats
                    isProcessing = false
                    generationStats = GenerationStats(
                        tokenCount: result.generatedTokens,
                        duration: Date().timeIntervalSince(startTime)
                    )
                }
            } catch {
                await MainActor.run {
                    output = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

