/**
 * GenerateView.swift
 * Text-to-image Generate tab. Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Generate View

struct GenerateView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false
    @State private var prompt = ""
    @State private var output = ""
    @State private var profilingInfo = ""
    @State private var isGenerating = false
    @State private var maxTokens: Double = 512
    @State private var temperature = 0.7

    // Mistral Small 3.2 supports up to 131K context, but we limit generation to 8K for practical use
    private let maxGenerationTokens: Double = 8192

    var body: some View {
        VStack(spacing: 16) {
            // Prompt input
            GroupBox("Prompt") {
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 100)
            }

            // Parameters
            HStack {
                GroupBox("Max Tokens: \(Int(maxTokens))") {
                    Slider(value: $maxTokens, in: 64...maxGenerationTokens, step: 64)
                    HStack {
                        Text("64")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(maxGenerationTokens))")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                GroupBox("Temperature: \(String(format: "%.1f", temperature))") {
                    Slider(value: $temperature, in: 0...2, step: 0.1)
                    HStack {
                        Text("0")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("2")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Generate button
            Button(action: generate) {
                HStack {
                    if isGenerating {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    Text(isGenerating ? "Generating..." : "Generate")
                }
            }
            .disabled(prompt.isEmpty || isGenerating || !modelManager.isLoaded)

            // Output
            GroupBox("Output") {
                ScrollView {
                    Text(output)
                        .font(.body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 150)
            }

            // Profiling info (shown when enabled)
            if detailedProfiling && !profilingInfo.isEmpty {
                GroupBox("Profiling") {
                    ScrollView {
                        Text(profilingInfo)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 100)
                }
            }

            Spacer()
        }
        .padding()
    }

    private func generate() {
        guard !prompt.isEmpty else { return }
        isGenerating = true
        output = ""
        profilingInfo = ""

        // Reset profiler
        FluxProfiler.shared.reset()

        Task {
            do {
                let params = GenerateParameters(
                    maxTokens: Int(maxTokens),
                    temperature: Float(temperature)
                )

                let result = try FluxTextEncoders.shared.generate(
                    prompt: prompt,
                    parameters: params
                ) { token in
                    Task { @MainActor in
                        output += token
                    }
                    return true
                }

                await MainActor.run {
                    output = result.text
                    isGenerating = false

                    // Get profiling info if enabled
                    if detailedProfiling {
                        let metrics = FluxProfiler.shared.getMetrics()
                        profilingInfo = metrics.compactSummary
                    }
                }
            } catch {
                await MainActor.run {
                    output = "Error: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}

