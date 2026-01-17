// ModelManagerView.swift - Model download and management UI
// Copyright 2025 Vincent Gourbin

import SwiftUI
import Flux2Core

struct ModelManagerView: View {
    @EnvironmentObject var modelManager: ModelViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                headerSection

                // Text Encoder Models
                modelSection(
                    title: "Text Encoder (Mistral Small 3.2)",
                    description: "Processes text prompts to guide image generation",
                    models: ModelRegistry.TextEncoderVariant.allCases.map { .textEncoder($0) }
                )

                // Transformer Models
                modelSection(
                    title: "Diffusion Transformer",
                    description: "Core image generation model (~32B parameters)",
                    models: ModelRegistry.TransformerVariant.allCases.map { .transformer($0) }
                )

                // VAE
                modelSection(
                    title: "VAE",
                    description: "Encodes/decodes images to/from latent space",
                    models: [.vae(.standard)]
                )

                Spacer()
            }
            .padding()
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model Manager")
                .font(.title)
                .bold()

            HStack {
                Text("Downloaded: \(modelManager.totalDownloadedSizeGB)GB")
                Text("•")
                Text("Available: \(modelManager.availableDiskSpaceGB)GB")
            }
            .font(.subheadline)
            .foregroundColor(.secondary)
        }
    }

    // MARK: - Model Section

    private func modelSection(
        title: String,
        description: String,
        models: [ModelRegistry.ModelComponent]
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            VStack(spacing: 8) {
                ForEach(models, id: \.self) { model in
                    ModelRow(
                        component: model,
                        info: modelManager.info(for: model),
                        onDownload: {
                            Task {
                                await modelManager.download(model)
                            }
                        },
                        onDelete: {
                            modelManager.delete(model)
                        }
                    )
                }
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Model Row

struct ModelRow: View {
    let component: ModelRegistry.ModelComponent
    let info: ModelInfo
    let onDownload: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack {
            // Status icon
            statusIcon

            // Info
            VStack(alignment: .leading, spacing: 2) {
                Text(info.name)
                    .font(.body)

                HStack {
                    Text("~\(info.sizeGB)GB")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    if let error = info.error {
                        Text("• Error: \(error)")
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
            }

            Spacer()

            // Progress or Actions
            if info.isDownloading {
                ProgressView(value: info.progress, total: 1.0)
                    .frame(width: 100)
                Text("\(Int(info.progress * 100))%")
                    .font(.caption)
                    .frame(width: 40)
            } else if info.isDownloaded {
                Button("Delete") {
                    onDelete()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button("Download") {
                    onDownload()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(6)
    }

    private var statusIcon: some View {
        Group {
            if info.isDownloading {
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(width: 20, height: 20)
            } else if info.isDownloaded {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else {
                Image(systemName: "circle")
                    .foregroundColor(.secondary)
            }
        }
        .frame(width: 24)
    }
}

#Preview {
    ModelManagerView()
        .environmentObject(ModelViewModel())
}
