// SettingsView.swift - Application settings
// Copyright 2025 Vincent Gourbin

import SwiftUI
import Flux2Core

struct SettingsView: View {
    @AppStorage("textEncoderQuantization") private var textQuantization = "8bit"
    @AppStorage("transformerQuantization") private var transformerQuantization = "qint8"

    var body: some View {
        TabView {
            quantizationSettings
                .tabItem {
                    Label("Quantization", systemImage: "cpu")
                }

            memorySettings
                .tabItem {
                    Label("Memory", systemImage: "memorychip")
                }
        }
        .frame(width: 500, height: 400)
        .padding()
    }

    // MARK: - Quantization Settings

    private var quantizationSettings: some View {
        Form {
            Section {
                Picker("Text Encoder (Mistral)", selection: $textQuantization) {
                    Text("Full Precision (bf16) - ~48GB").tag("bf16")
                    Text("8-bit - ~25GB").tag("8bit")
                    Text("6-bit - ~19GB").tag("6bit")
                    Text("4-bit - ~14GB").tag("4bit")
                }

                Text("Higher precision = better prompt understanding")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section {
                Picker("Diffusion Transformer", selection: $transformerQuantization) {
                    Text("Full Precision (bf16) - ~64GB").tag("bf16")
                    Text("8-bit (qint8) - ~32GB").tag("qint8")
                    Text("4-bit (qint4) - ~16GB (experimental)").tag("qint4")
                }

                Text("Higher precision = better image quality")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Section("Presets") {
                HStack(spacing: 12) {
                    PresetButton(
                        title: "High Quality",
                        subtitle: "~90GB",
                        action: {
                            textQuantization = "bf16"
                            transformerQuantization = "bf16"
                        }
                    )

                    PresetButton(
                        title: "Balanced",
                        subtitle: "~60GB",
                        isRecommended: true,
                        action: {
                            textQuantization = "8bit"
                            transformerQuantization = "qint8"
                        }
                    )

                    PresetButton(
                        title: "Memory Efficient",
                        subtitle: "~50GB",
                        action: {
                            textQuantization = "4bit"
                            transformerQuantization = "qint8"
                        }
                    )

                    PresetButton(
                        title: "Minimal",
                        subtitle: "~35GB",
                        action: {
                            textQuantization = "4bit"
                            transformerQuantization = "qint4"
                        }
                    )
                }
            }

            Section("Current Configuration") {
                let config = currentConfig
                VStack(alignment: .leading, spacing: 4) {
                    Text("Estimated peak memory: ~\(config.estimatedTotalMemoryGB)GB")
                    Text("Text encoding phase: ~\(config.textEncodingPhaseMemoryGB)GB")
                    Text("Image generation phase: ~\(config.imageGenerationPhaseMemoryGB)GB")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Memory Settings

    private var memorySettings: some View {
        Form {
            Section("System Information") {
                LabeledContent("System RAM") {
                    Text("\(ModelRegistry.systemRAMGB)GB")
                }

                LabeledContent("Recommended Config") {
                    Text(ModelRegistry.defaultConfig.description)
                        .font(.caption)
                }
            }

            Section("Memory Management") {
                Text("Flux.2 uses a two-phase approach to manage memory:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top) {
                        Text("1.")
                            .fontWeight(.bold)
                        Text("Text Encoding: Mistral processes the prompt, then is unloaded")
                    }

                    HStack(alignment: .top) {
                        Text("2.")
                            .fontWeight(.bold)
                        Text("Image Generation: Transformer + VAE generate the image")
                    }
                }
                .font(.caption)
            }

            Section("Tips") {
                VStack(alignment: .leading, spacing: 8) {
                    tip("Close other apps to free RAM before generation")
                    tip("Lower quantization if you experience memory issues")
                    tip("Smaller images use less memory during generation")
                    tip("qint4 transformer may show some quality degradation")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    // MARK: - Helpers

    private var currentConfig: Flux2QuantizationConfig {
        let textQuant = MistralQuantization(rawValue: textQuantization) ?? .mlx8bit
        let transQuant = TransformerQuantization(rawValue: transformerQuantization) ?? .qint8
        return Flux2QuantizationConfig(textEncoder: textQuant, transformer: transQuant)
    }

    private func tip(_ text: String) -> some View {
        HStack(alignment: .top) {
            Image(systemName: "lightbulb")
                .foregroundColor(.yellow)
            Text(text)
        }
    }
}

// MARK: - Preset Button

struct PresetButton: View {
    let title: String
    let subtitle: String
    var isRecommended: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                if isRecommended {
                    Text("Recommended")
                        .font(.caption2)
                        .foregroundColor(.green)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
        }
        .buttonStyle(.bordered)
    }
}

#Preview {
    SettingsView()
}
