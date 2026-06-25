/**
 * ImageGenerationPromptSection.swift
 * Wide prompt bar above the image preview (editor + stacked options).
 */

import SwiftUI
import Flux2Core

struct ImageGenerationPromptSection: View {
    @ObservedObject var viewModel: ImageGenerationViewModel

    private var fieldLabel: String {
        if viewModel.requiresReferenceImages {
            if viewModel.hasLocalFillSelection {
                return viewModel.enrichInpaintPromptWithVLM ? "Optional hint" : "Fill prompt"
            }
            return "AI Prompt"
        }
        return "AI Prompt"
    }

    private var contextualHint: String? {
        guard viewModel.requiresReferenceImages, viewModel.hasLocalFillSelection else { return nil }
        if viewModel.enrichInpaintPromptWithVLM {
            return "Leave empty for Qwen to write the Flux prompt, or add a short hint."
        }
        return "Describe what should appear inside the selection."
    }

    private var upsampledTitle: String {
        if viewModel.requiresReferenceImages, viewModel.hasLocalFillSelection {
            return "VLM prompt"
        }
        return "Upsampled prompt"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(fieldLabel)
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)

                    if let contextualHint {
                        Text(contextualHint)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    TextEditor(text: $viewModel.prompt)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .padding(8)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.black.opacity(0.22), lineWidth: 1)
                        )
                        .frame(height: 72)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                promptOptionsColumn
            }

            if let upsampled = viewModel.upsampledPrompt {
                upsampledPromptRow(upsampled)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.35))
    }

    private var promptOptionsColumn: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Upsample prompt", isOn: $viewModel.upsamplePrompt)
                .disabled(!viewModel.isUpsamplePromptApplicable)
                .opacity(viewModel.isUpsamplePromptApplicable ? 1 : 0.45)
                .help(
                    viewModel.requiresReferenceImages
                        ? "Enhance prompt using Mistral VLM to analyze reference images"
                        : "Enhance prompt with visual details using Mistral"
                )

            if viewModel.isEnrichInpaintPromptApplicable {
                Toggle("Enrich prompt with Qwen3.5 VLM", isOn: $viewModel.enrichInpaintPromptWithVLM)
                    .help("Write the inpaint prompt from the image and selection using Qwen3.5 VLM")
            }

            Toggle("Clear prompt after generation", isOn: $viewModel.clearPromptAfterGeneration)
                .help("Empty the prompt automatically once a run finishes successfully")
        }
        .font(.caption)
        .toggleStyle(.checkbox)
        .fixedSize(horizontal: true, vertical: true)
        .padding(.horizontal, 12)
    }

    @ViewBuilder
    private func upsampledPromptRow(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Label(upsampledTitle, systemImage: "sparkles")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .frame(width: 120, alignment: .leading)

            ScrollView(.horizontal, showsIndicators: false) {
                Text(text)
                    .font(.caption)
                    .textSelection(.enabled)
                    .lineLimit(2)
            }
        }
        .padding(8)
        .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(Color.accentColor.opacity(0.3), lineWidth: 1)
        }
    }
}
