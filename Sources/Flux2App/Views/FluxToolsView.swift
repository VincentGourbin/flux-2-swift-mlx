/**
 * FluxToolsView.swift
 * FLUX.2 Tools tab: the tool-mode picker and the tool surfaces. Extracted from
 * ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - FLUX.2 Tools View

enum FluxToolMode: String, CaseIterable {
    case embeddings = "FLUX.2 Embeddings"
    case klein4B = "Klein 4B"
    case klein9B = "Klein 9B"
    case upsamplingT2I = "Upsampling T2I"
    case upsamplingI2I = "Upsampling I2I"
    case kleinUpT2I = "Klein Upsampling T2I"
    case kleinUpI2I = "Klein Upsampling I2I"

    var icon: String {
        switch self {
        case .embeddings: return "cube.transparent"
        case .klein4B: return "cube.fill"
        case .klein9B: return "cube.fill"
        case .upsamplingT2I: return "wand.and.stars"
        case .upsamplingI2I: return "photo.on.rectangle"
        case .kleinUpT2I: return "wand.and.stars"
        case .kleinUpI2I: return "photo.on.rectangle"
        }
    }

    var description: String {
        switch self {
        case .embeddings: return "Mistral → FLUX.2 (512×15360)"
        case .klein4B: return "Qwen3-4B → Klein 4B (512×7680)"
        case .klein9B: return "Qwen3-8B → Klein 9B (512×12288)"
        case .upsamplingT2I: return "Enhance text prompts for image generation"
        case .upsamplingI2I: return "Generate image editing instructions"
        case .kleinUpT2I: return "Enhance prompts with Qwen3 → Klein embeddings"
        case .kleinUpI2I: return "Edit instructions with Qwen3 → Klein embeddings"
        }
    }

    var isKlein: Bool {
        switch self {
        case .klein4B, .klein9B, .kleinUpT2I, .kleinUpI2I:
            return true
        default:
            return false
        }
    }

    var kleinVariant: KleinVariant? {
        switch self {
        case .klein4B, .kleinUpT2I, .kleinUpI2I: return .klein4B  // Default to 4B for upsampling
        case .klein9B: return .klein9B
        default: return nil
        }
    }

    var isUpsampling: Bool {
        switch self {
        case .upsamplingT2I, .upsamplingI2I, .kleinUpT2I, .kleinUpI2I:
            return true
        default:
            return false
        }
    }
}

struct FluxToolsView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var selectedMode: FluxToolMode = .embeddings
    @State private var selectedKleinUpVariant: KleinVariant = .klein4B  // For Klein upsampling mode
    @State private var prompt = ""
    @State private var imagePath: String?
    @State private var outputText = ""
    @State private var isProcessing = false
    @State private var isLoadingKlein = false
    @State private var kleinLoadingMessage = ""
    @State private var lastEmbeddings: MLXArray?

    var body: some View {
        VStack(spacing: 16) {
            // Mode selector
            HStack {
                Picker("Mode", selection: $selectedMode) {
                    Section("Mistral (FLUX.2-dev)") {
                        Label(FluxToolMode.embeddings.rawValue, systemImage: FluxToolMode.embeddings.icon)
                            .tag(FluxToolMode.embeddings)
                    }
                    Section("Qwen3 (FLUX.2 Klein)") {
                        Label(FluxToolMode.klein4B.rawValue, systemImage: FluxToolMode.klein4B.icon)
                            .tag(FluxToolMode.klein4B)
                        Label(FluxToolMode.klein9B.rawValue, systemImage: FluxToolMode.klein9B.icon)
                            .tag(FluxToolMode.klein9B)
                    }
                    Section("Upsampling (Mistral)") {
                        Label(FluxToolMode.upsamplingT2I.rawValue, systemImage: FluxToolMode.upsamplingT2I.icon)
                            .tag(FluxToolMode.upsamplingT2I)
                        Label(FluxToolMode.upsamplingI2I.rawValue, systemImage: FluxToolMode.upsamplingI2I.icon)
                            .tag(FluxToolMode.upsamplingI2I)
                    }
                    Section("Upsampling (Klein)") {
                        Label(FluxToolMode.kleinUpT2I.rawValue, systemImage: FluxToolMode.kleinUpT2I.icon)
                            .tag(FluxToolMode.kleinUpT2I)
                        Label(FluxToolMode.kleinUpI2I.rawValue, systemImage: FluxToolMode.kleinUpI2I.icon)
                            .tag(FluxToolMode.kleinUpI2I)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 220)

                Spacer()

                Text(selectedMode.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(selectedMode.isKlein ? Color.orange.opacity(0.1) : Color.purple.opacity(0.1))
                    .cornerRadius(4)
                
                // Klein model variant picker (for Klein upsampling modes)
                if selectedMode == .kleinUpT2I || selectedMode == .kleinUpI2I {
                    Picker("Model", selection: $selectedKleinUpVariant) {
                        Text("4B (Apache)").tag(KleinVariant.klein4B)
                        Text("9B (NC)").tag(KleinVariant.klein9B)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 140)
                }

                // Klein model status
                if selectedMode.isKlein {
                    let targetVariant: KleinVariant = (selectedMode == .kleinUpT2I || selectedMode == .kleinUpI2I)
                        ? selectedKleinUpVariant
                        : (selectedMode.kleinVariant ?? .klein4B)

                    if FluxTextEncoders.shared.isKleinLoaded {
                        if let loadedVariant = FluxTextEncoders.shared.kleinVariant,
                           loadedVariant == targetVariant {
                            Label("Ready", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.caption)
                        } else {
                            Label("Different model loaded", systemImage: "exclamationmark.triangle")
                                .foregroundStyle(.orange)
                                .font(.caption)
                        }
                    } else {
                        Label("Click to load", systemImage: "arrow.down.circle")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                }
            }

            // Klein loading progress
            if isLoadingKlein {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text(kleinLoadingMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }

            // Image picker (for I2I modes - placeholder for future VLM integration)
            if selectedMode == .upsamplingI2I || selectedMode == .kleinUpI2I {
                GroupBox("Reference Image (optional)") {
                    HStack {
                        if let path = imagePath {
                            Text(URL(fileURLWithPath: path).lastPathComponent)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("Clear") { imagePath = nil }
                        } else {
                            Text("No image selected")
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("Select...") { selectImage() }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            // Prompt input
            GroupBox(selectedMode == .embeddings || selectedMode == .klein4B || selectedMode == .klein9B ? "Text to Embed" : "Input Prompt") {
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 80)
            }

            // System prompt info (for upsampling modes)
            if selectedMode.isUpsampling {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("System Prompt (BFL Official)")
                            .font(.caption.bold())
                        let systemPrompt: String = {
                            switch selectedMode {
                            case .upsamplingT2I:
                                return FluxConfig.systemMessage(for: .upsamplingT2I)
                            case .upsamplingI2I:
                                return FluxConfig.systemMessage(for: .upsamplingI2I)
                            case .kleinUpT2I:
                                return KleinConfig.systemMessage(for: .upsamplingT2I)
                            case .kleinUpI2I:
                                return KleinConfig.systemMessage(for: .upsamplingI2I)
                            default:
                                return ""
                            }
                        }()
                        Text(systemPrompt.prefix(150) + "...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            // Action buttons
            HStack {
                Button(action: processAction) {
                    HStack {
                        if isProcessing {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isProcessing ? "Processing..." : actionButtonTitle)
                    }
                }
                .disabled(prompt.isEmpty || isProcessing || isLoadingKlein || (!selectedMode.isKlein && !modelManager.isLoaded && !modelManager.isVLMLoaded))

                Spacer()

                Button("Copy Text") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(outputText, forType: .string)
                }
                .disabled(outputText.isEmpty)

                Button("Export Embeddings...") { exportEmbeddings() }
                    .disabled(lastEmbeddings == nil)
            }

            // Output
            GroupBox(selectedMode == .embeddings ? "Embeddings Info" : "Output") {
                ScrollView {
                    Text(outputText.isEmpty ? placeholderText : outputText)
                        .font(selectedMode == .embeddings ? .system(.body, design: .monospaced) : .body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(outputText.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 180)
            }

            Spacer()
        }
        .padding()
        .onChange(of: selectedMode) { _, _ in
            outputText = ""
            lastEmbeddings = nil
        }
    }

    private var actionButtonTitle: String {
        switch selectedMode {
        case .embeddings: return "Extract FLUX.2 Embeddings"
        case .klein4B: return "Extract Klein 4B Embeddings"
        case .klein9B: return "Extract Klein 9B Embeddings"
        case .upsamplingT2I, .upsamplingI2I: return "Upsample & Extract"
        case .kleinUpT2I, .kleinUpI2I: return "Klein Upsample & Extract"
        }
    }

    private var placeholderText: String {
        switch selectedMode {
        case .embeddings: return "Enter text and click Extract for FLUX.2-dev embeddings (Mistral)"
        case .klein4B: return "Enter text for Klein 4B embeddings (Qwen3-4B, Apache 2.0)"
        case .klein9B: return "Enter text for Klein 9B embeddings (Qwen3-8B, non-commercial)"
        case .upsamplingT2I: return "Enter a simple prompt to enhance it for FLUX.2"
        case .upsamplingI2I: return "Describe the edit you want (e.g., 'make the sky more dramatic')"
        case .kleinUpT2I: return "Enter a simple prompt to enhance with Qwen3 for Klein"
        case .kleinUpI2I: return "Describe the edit for Klein (e.g., 'change the background')"
        }
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK {
            imagePath = panel.url?.path
        }
    }

    private func processAction() {
        isProcessing = true
        outputText = ""

        Task {
            do {
                switch selectedMode {
                case .embeddings:
                    let embeddings = try FluxTextEncoders.shared.extractFluxEmbeddings(prompt: prompt)
                    await MainActor.run {
                        lastEmbeddings = embeddings
                        let flatEmbeddings = embeddings.reshaped([-1])
                        let firstValues = flatEmbeddings[0..<min(10, flatEmbeddings.size)].asArray(Float.self)
                        outputText = """
                        === FLUX.2 Embeddings (Mistral) ===

                        Shape: \(embeddings.shape)
                        Dtype: \(embeddings.dtype)
                        Total: \(embeddings.shape.reduce(1, *)) elements

                        Format:
                        • Model: Mistral Small 3.2
                        • Layers: 10, 20, 30 (concatenated)
                        • Sequence: 512 tokens (LEFT-padded)
                        • Dims: 5,120 × 3 = 15,360

                        First 10 values:
                        \(firstValues.map { String(format: "%.6f", $0) }.joined(separator: ", "))

                        ✅ Ready for FLUX.2-dev
                        """
                        isProcessing = false
                    }
                    
                case .klein4B, .klein9B:
                    guard let kleinVariant = selectedMode.kleinVariant else { return }
                    
                    // Load Klein model if needed
                    if !FluxTextEncoders.shared.isKleinLoaded || FluxTextEncoders.shared.kleinVariant != kleinVariant {
                        await MainActor.run {
                            isLoadingKlein = true
                            kleinLoadingMessage = "Loading Qwen3 for \(kleinVariant.displayName)..."
                        }
                        
                        try await FluxTextEncoders.shared.loadKleinModel(
                            variant: kleinVariant,
                            hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                        ) { progress, message in
                            Task { @MainActor in
                                kleinLoadingMessage = message
                            }
                        }
                        
                        await MainActor.run {
                            isLoadingKlein = false
                        }
                    }
                    
                    // Extract Klein embeddings
                    let embeddings = try FluxTextEncoders.shared.extractKleinEmbeddings(prompt: prompt)
                    await MainActor.run {
                        lastEmbeddings = embeddings
                        let flatEmbeddings = embeddings.reshaped([-1])
                        let firstValues = flatEmbeddings[0..<min(10, flatEmbeddings.size)].asArray(Float.self)
                        outputText = """
                        === FLUX.2 Klein Embeddings (\(kleinVariant.displayName)) ===

                        Shape: \(embeddings.shape)
                        Dtype: \(embeddings.dtype)
                        Total: \(embeddings.shape.reduce(1, *)) elements

                        Format:
                        • Model: Qwen3-\(kleinVariant == .klein4B ? "4B" : "8B")
                        • Layers: 9, 18, 27 (concatenated)
                        • Sequence: 512 tokens (LEFT-padded)
                        • Dims: \(kleinVariant.hiddenSize) × 3 = \(kleinVariant.outputDimension)

                        First 10 values:
                        \(firstValues.map { String(format: "%.6f", $0) }.joined(separator: ", "))

                        ✅ Ready for FLUX.2 Klein \(kleinVariant == .klein4B ? "4B" : "9B")
                        """
                        isProcessing = false
                    }

                case .upsamplingT2I, .upsamplingI2I:
                    let fluxMode: FluxConfig.Mode = selectedMode == .upsamplingT2I ? .upsamplingT2I : .upsamplingI2I
                    let hasImage = selectedMode == .upsamplingI2I && imagePath != nil

                    // Step 1: Generate enhanced prompt
                    await MainActor.run {
                        outputText = "Generating enhanced prompt..."
                    }

                    var enhancedPrompt = ""

                    if hasImage, let path = imagePath {
                        // I2I with image: use VLM to analyze image with I2I system prompt
                        // This allows the model to actually SEE the image
                        _ = try FluxTextEncoders.shared.analyzeImage(
                            path: path,
                            prompt: prompt,
                            systemPrompt: FluxConfig.systemMessage(for: .upsamplingI2I),
                            parameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
                        ) { token in
                            enhancedPrompt += token
                            return true
                        }
                    } else {
                        // T2I or I2I without image: use text-only chat
                        let messages = FluxConfig.buildMessages(prompt: prompt, mode: fluxMode)
                        _ = try FluxTextEncoders.shared.chat(
                            messages: messages,
                            parameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
                        ) { token in
                            enhancedPrompt += token
                            return true
                        }
                    }

                    // Step 2: Extract embeddings
                    await MainActor.run {
                        if hasImage {
                            outputText = "Enhanced prompt:\n\(enhancedPrompt)\n\nExtracting embeddings with image..."
                        } else {
                            outputText = "Enhanced prompt:\n\(enhancedPrompt)\n\nExtracting embeddings..."
                        }
                    }

                    let embeddings: MLXArray
                    let embeddingType: String

                    if hasImage, let path = imagePath {
                        // I2I with image: extract embeddings including image tokens
                        embeddings = try FluxTextEncoders.shared.extractFluxEmbeddingsWithImage(
                            imagePath: path,
                            prompt: enhancedPrompt
                        )
                        embeddingType = "Image + Text"
                    } else {
                        // T2I or I2I without image: text-only embeddings
                        embeddings = try FluxTextEncoders.shared.extractFluxEmbeddings(prompt: enhancedPrompt)
                        embeddingType = "Text only"
                    }

                    await MainActor.run {
                        lastEmbeddings = embeddings
                        let flatEmbeddings = embeddings.reshaped([-1])
                        let firstValues = flatEmbeddings[0..<min(5, flatEmbeddings.size)].asArray(Float.self)

                        outputText = """
                        === Enhanced Prompt ===
                        \(enhancedPrompt)

                        === FLUX.2 Embeddings (\(embeddingType)) ===
                        Shape: \(embeddings.shape)
                        First values: \(firstValues.map { String(format: "%.4f", $0) }.joined(separator: ", "))...

                        ✅ Ready to export for FLUX.2 diffusion
                        """
                        isProcessing = false
                    }

                case .kleinUpT2I, .kleinUpI2I:
                    let kleinMode: KleinConfig.Mode = selectedMode == .kleinUpT2I ? .upsamplingT2I : .upsamplingI2I
                    let kleinVariant = selectedKleinUpVariant  // Use selected variant (4B or 9B)

                    // Load Klein model if needed
                    if !FluxTextEncoders.shared.isKleinLoaded || FluxTextEncoders.shared.kleinVariant != kleinVariant {
                        await MainActor.run {
                            isLoadingKlein = true
                            kleinLoadingMessage = "Loading Qwen3 for \(kleinVariant.displayName)..."
                        }

                        try await FluxTextEncoders.shared.loadKleinModel(
                            variant: kleinVariant,
                            hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                        ) { progress, message in
                            Task { @MainActor in
                                kleinLoadingMessage = message
                            }
                        }

                        await MainActor.run {
                            isLoadingKlein = false
                        }
                    }

                    // Step 1: Generate enhanced prompt using Qwen3
                    await MainActor.run {
                        outputText = "Generating enhanced prompt with Qwen3..."
                    }

                    let messages = KleinConfig.buildMessages(prompt: prompt, mode: kleinMode)

                    // Generate enhanced prompt - use result.text which has thinking tags stripped
                    let result = try FluxTextEncoders.shared.chatQwen3(
                        messages: messages,
                        parameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
                    )
                    let enhancedPrompt = result.text

                    // Step 2: Extract Klein embeddings from enhanced prompt
                    await MainActor.run {
                        outputText = "Enhanced prompt:\n\(enhancedPrompt)\n\nExtracting Klein embeddings..."
                    }

                    let embeddings = try FluxTextEncoders.shared.extractKleinEmbeddings(prompt: enhancedPrompt)

                    await MainActor.run {
                        lastEmbeddings = embeddings
                        let flatEmbeddings = embeddings.reshaped([-1])
                        let firstValues = flatEmbeddings[0..<min(5, flatEmbeddings.size)].asArray(Float.self)
                        let qwenModel = kleinVariant == .klein4B ? "Qwen3-4B" : "Qwen3-8B"
                        let kleinModel = kleinVariant == .klein4B ? "Klein 4B" : "Klein 9B"

                        outputText = """
                        === Enhanced Prompt (\(qwenModel)) ===
                        \(enhancedPrompt)

                        === Klein Embeddings ===
                        Shape: \(embeddings.shape)
                        Model: \(qwenModel) → \(kleinModel)
                        Dims: \(kleinVariant.hiddenSize) × 3 = \(kleinVariant.outputDimension)
                        First values: \(firstValues.map { String(format: "%.4f", $0) }.joined(separator: ", "))...

                        ✅ Ready for FLUX.2 \(kleinModel) diffusion
                        """
                        isProcessing = false
                    }
                }
            } catch {
                await MainActor.run {
                    outputText = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }

    private func exportEmbeddings() {
        guard let embeddings = lastEmbeddings else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.data]
        panel.nameFieldStringValue = "flux_embeddings.bin"

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try FluxTextEncoders.shared.exportEmbeddings(embeddings, to: url.path, format: .binary)
                outputText += "\n\n✅ Exported to: \(url.lastPathComponent)"
            } catch {
                outputText += "\n\n❌ Export failed: \(error.localizedDescription)"
            }
        }
    }
}

