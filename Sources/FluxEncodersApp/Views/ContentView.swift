/**
 * ContentView.swift
 * Main content view for Mistral App
 */

import SwiftUI
import FluxTextEncoders
import MLX

struct ContentView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var chatViewModel = ChatViewModel()
    @State private var selectedTab = 0

    var body: some View {
        NavigationSplitView {
            // Sidebar
            List(selection: Binding(
                get: { selectedTab },
                set: { selectedTab = $0 }
            )) {
                Section("Mistral 24B") {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                        .tag(0)
                    Label("Generate", systemImage: "text.cursor")
                        .tag(1)
                    Label("Vision", systemImage: "eye")
                        .tag(2)
                }

                Section("Qwen3") {
                    Label("Chat", systemImage: "message.fill")
                        .tag(5)
                        .foregroundStyle(.orange)
                }

                Section("Tools") {
                    Label("FLUX.2 Tools", systemImage: "cube.transparent")
                        .tag(3)
                    Label("Models", systemImage: "square.stack.3d.down.right")
                        .tag(4)
                }
            }
            .listStyle(.sidebar)
            .frame(minWidth: 180)

        } detail: {
            // Main content
            VStack(spacing: 0) {
                // Model status bar - contextual based on selected tab
                ModelStatusBar(selectedTab: selectedTab)
                    .environmentObject(modelManager)

                Divider()

                // Content based on selection
                Group {
                    switch selectedTab {
                    case 0:
                        ChatView(viewModel: chatViewModel)
                    case 1:
                        GenerateView()
                    case 2:
                        VisionView()
                    case 3:
                        FluxToolsView()
                    case 4:
                        ModelsManagementView()
                    case 5:
                        Qwen3ChatView()
                    default:
                        ChatView(viewModel: chatViewModel)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("FLUX.2 Text Encoders")
    }
}

// MARK: - Model Status Bar (Contextual)

struct ModelStatusBar: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false
    let selectedTab: Int

    /// Is this a Qwen3-focused tab?
    private var isQwen3Tab: Bool {
        selectedTab == 5  // Qwen3 Chat
    }

    /// Is this a Tools tab (shows both models)?
    private var isToolsTab: Bool {
        selectedTab == 3 || selectedTab == 4  // FLUX.2 Tools or Models
    }

    var body: some View {
        HStack {
            if isQwen3Tab {
                // === QWEN3 STATUS BAR ===
                qwen3StatusBar
            } else if isToolsTab {
                // === TOOLS TAB: Show both models status ===
                toolsStatusBar
            } else {
                // === MISTRAL STATUS BAR ===
                mistralStatusBar
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(isQwen3Tab ? Color.orange.opacity(0.05) : Color(NSColor.controlBackgroundColor))
        .onAppear {
            FluxProfiler.shared.isEnabled = detailedProfiling
        }
    }

    // MARK: - Mistral Status Bar

    @ViewBuilder
    private var mistralStatusBar: some View {
        // Model status indicator
        Circle()
            .fill(modelManager.isLoaded ? Color.green : Color.red)
            .frame(width: 8, height: 8)

        Text(modelManager.isLoaded ? "Mistral Loaded" : "Mistral Not Loaded")
            .font(.caption)
            .foregroundColor(.secondary)

        // VLM indicator
        if modelManager.isVLMLoaded {
            Text("VLM")
                .font(.caption.bold())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.blue.opacity(0.2))
                .foregroundColor(.blue)
                .cornerRadius(4)
        }

        if modelManager.isLoading {
            ProgressView()
                .scaleEffect(0.6)
            Text(modelManager.loadingMessage)
                .font(.caption)
                .foregroundColor(.secondary)
        }

        Spacer()

        // Detailed profiling toggle
        Toggle("Detailed Profiling", isOn: $detailedProfiling)
            .toggleStyle(.checkbox)
            .font(.caption)
            .onChange(of: detailedProfiling) { _, newValue in
                FluxProfiler.shared.isEnabled = newValue
            }
            .help("Enable detailed profiling and memory logging")

        Divider()
            .frame(height: 20)
            .padding(.horizontal, 8)

        // Model variant picker
        HStack(spacing: 0) {
            Text("Model")
                .foregroundColor(.secondary)
                .padding(.trailing, 8)

            ForEach(ModelVariant.allCases, id: \.self) { variant in
                let isDownloaded = isVariantDownloaded(variant)
                let isSelected = modelManager.selectedVariant == variant
                Button(action: {
                    if isDownloaded {
                        modelManager.selectedVariant = variant
                    }
                }) {
                    Text(variant.shortName)
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            isSelected
                                ? Color.accentColor
                                : (isDownloaded ? Color.gray.opacity(0.3) : Color.gray.opacity(0.1))
                        )
                        .foregroundColor(
                            isSelected
                                ? .white
                                : (isDownloaded ? .primary : .secondary.opacity(0.5))
                        )
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(!isDownloaded)
                .help(isDownloaded ? variant.displayName : "\(variant.displayName) - Not downloaded")
            }
        }

        // Load/Unload button
        Button(action: {
            Task {
                if modelManager.isLoaded {
                    modelManager.unloadModel()
                } else {
                    await modelManager.loadModel()
                }
            }
        }) {
            Text(modelManager.isLoaded ? "Unload" : "Load Mistral")
        }
        .disabled(modelManager.isLoading || (!modelManager.isLoaded && modelManager.selectedVariant == nil))
    }

    // MARK: - Qwen3 Status Bar

    @ViewBuilder
    private var qwen3StatusBar: some View {
        // Model status indicator
        Circle()
            .fill(modelManager.isQwen3Loaded ? Color.green : Color.red)
            .frame(width: 8, height: 8)

        Text(modelManager.isQwen3Loaded ? "Qwen3 Loaded" : "Qwen3 Not Loaded")
            .font(.caption)
            .foregroundColor(.secondary)

        // Loaded variant indicator
        if let variant = modelManager.loadedQwen3Variant {
            Text(variant.displayName)
                .font(.caption.bold())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.orange.opacity(0.2))
                .foregroundColor(.orange)
                .cornerRadius(4)
        }

        if modelManager.isQwen3Loading {
            ProgressView()
                .scaleEffect(0.6)
            Text(modelManager.qwen3LoadingMessage)
                .font(.caption)
                .foregroundColor(.secondary)
        }

        Spacer()

        // Qwen3 variant picker
        HStack(spacing: 0) {
            Text("Model")
                .foregroundColor(.secondary)
                .padding(.trailing, 8)

            ForEach(Qwen3Variant.allCases, id: \.self) { variant in
                let modelInfo = TextEncoderModelRegistry.shared.qwen3Model(withVariant: variant)
                let modelId = modelInfo?.id ?? ""
                let isDownloaded = modelManager.downloadedQwen3Models.contains(modelId)
                let isSelected = modelManager.loadedQwen3Variant == variant
                Button(action: {
                    if isDownloaded && !isSelected {
                        Task { await modelManager.loadQwen3Model(modelId) }
                    }
                }) {
                    Text(variant.shortName)
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            isSelected
                                ? Color.orange
                                : (isDownloaded ? Color.gray.opacity(0.3) : Color.gray.opacity(0.1))
                        )
                        .foregroundColor(
                            isSelected
                                ? .white
                                : (isDownloaded ? .primary : .secondary.opacity(0.5))
                        )
                        .cornerRadius(6)
                }
                .buttonStyle(.plain)
                .disabled(!isDownloaded || modelManager.isQwen3Loading)
                .help(isDownloaded ? variant.displayName : "\(variant.displayName) - Not downloaded")
            }
        }

        // Load/Unload button
        Button(action: {
            if modelManager.isQwen3Loaded {
                modelManager.unloadQwen3Model()
            }
        }) {
            Text(modelManager.isQwen3Loaded ? "Unload Qwen3" : "Select Model")
        }
        .disabled(!modelManager.isQwen3Loaded || modelManager.isQwen3Loading)
    }

    // MARK: - Tools Status Bar (Both Models)

    @ViewBuilder
    private var toolsStatusBar: some View {
        // Mistral status
        HStack(spacing: 4) {
            Circle()
                .fill(modelManager.isLoaded ? Color.green : Color.gray)
                .frame(width: 6, height: 6)
            Text("Mistral")
                .font(.caption)
                .foregroundColor(modelManager.isLoaded ? .primary : .secondary)
            if modelManager.isVLMLoaded {
                Text("VLM")
                    .font(.caption2.bold())
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Color.blue.opacity(0.2))
                    .foregroundColor(.blue)
                    .cornerRadius(3)
            }
        }

        Divider()
            .frame(height: 16)
            .padding(.horizontal, 8)

        // Qwen3 status
        HStack(spacing: 4) {
            Circle()
                .fill(modelManager.isQwen3Loaded ? Color.green : Color.gray)
                .frame(width: 6, height: 6)
            Text("Qwen3")
                .font(.caption)
                .foregroundColor(modelManager.isQwen3Loaded ? .orange : .secondary)
            if let variant = modelManager.loadedQwen3Variant {
                Text(variant.shortName)
                    .font(.caption2.bold())
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(Color.orange.opacity(0.2))
                    .foregroundColor(.orange)
                    .cornerRadius(3)
            }
        }

        Spacer()

        // Detailed profiling toggle
        Toggle("Detailed Profiling", isOn: $detailedProfiling)
            .toggleStyle(.checkbox)
            .font(.caption)
            .onChange(of: detailedProfiling) { _, newValue in
                FluxProfiler.shared.isEnabled = newValue
            }

        Text("Manage models in Models tab")
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    // MARK: - Helpers

    private func isVariantDownloaded(_ variant: ModelVariant) -> Bool {
        guard let model = TextEncoderModelRegistry.shared.model(withVariant: variant) else { return false }
        return modelManager.downloadedModels.contains(model.id)
    }
}

// MARK: - Chat View

struct ChatView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false
    @ObservedObject var viewModel: ChatViewModel
    @State private var showSettings = false

    // Mistral Small 3.2 supports up to 131K context
    private let maxGenerationTokens = 8192

    var body: some View {
        VStack(spacing: 0) {
            // Messages list
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                        }

                        if viewModel.isGenerating {
                            HStack {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Generating...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let lastMessage = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Stats bar - show live during generation, final stats after
            if viewModel.isGenerating {
                LiveStatsBarView(tokenCount: viewModel.currentTokenCount)
            } else if let stats = viewModel.lastGenerationStats {
                StatsBarView(stats: stats, profileSummary: viewModel.lastProfileSummary)
            }

            // Settings bar (collapsible)
            if showSettings {
                VStack(spacing: 8) {
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Tokens: \(viewModel.maxTokens)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: Binding(
                                get: { Double(viewModel.maxTokens) },
                                set: { viewModel.maxTokens = Int($0) }
                            ), in: 64...Double(maxGenerationTokens), step: 64)
                            .frame(width: 200)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Temperature: \(String(format: "%.1f", viewModel.temperature))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: Binding(
                                get: { Double(viewModel.temperature) },
                                set: { viewModel.temperature = Float($0) }
                            ), in: 0...2, step: 0.1)
                            .frame(width: 150)
                        }

                        Spacer()

                        Button("Clear Chat") {
                            viewModel.clearChat()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
            }

            Divider()

            // Input area
            HStack(spacing: 12) {
                Button(action: { withAnimation { showSettings.toggle() } }) {
                    Image(systemName: showSettings ? "gearshape.fill" : "gearshape")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Toggle settings")

                TextField("Type a message...", text: $viewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)
                    .onSubmit {
                        sendMessage()
                    }

                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(viewModel.inputText.isEmpty || viewModel.isGenerating || !modelManager.isLoaded)
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
        }
        .onAppear {
            viewModel.modelManager = modelManager
        }
    }

    private func sendMessage() {
        guard !viewModel.inputText.isEmpty else { return }
        Task {
            await viewModel.sendMessage()
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .assistant {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mistral")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(message.content)
                        .padding(12)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)
                }
                Spacer()
            } else {
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("You")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(message.content)
                        .padding(12)
                        .background(Color.gray.opacity(0.2))
                        .cornerRadius(12)
                }
            }
        }
        .id(message.id)
    }
}

// MARK: - Live Stats Bar View (during generation)

struct LiveStatsBarView: View {
    let tokenCount: Int

    var body: some View {
        HStack(spacing: 16) {
            ProgressView()
                .scaleEffect(0.7)

            Image(systemName: "text.cursor")
                .foregroundStyle(.blue)

            Text("Generating (\(tokenCount) tokens)...")
                .foregroundStyle(.blue)
                .fontWeight(.medium)

            Spacer()
        }
        .font(.caption)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Stats Bar View (final)

struct StatsBarView: View {
    let stats: GenerationStats
    let profileSummary: ProfileSummary?
    @State private var showProfileDetails = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 20) {
                Label("\(stats.tokenCount) tokens", systemImage: "number")
                Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
                Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")

                Spacer()

                if profileSummary != nil {
                    Button(action: { showProfileDetails.toggle() }) {
                        Label(showProfileDetails ? "Hide Profile" : "Show Profile",
                              systemImage: showProfileDetails ? "chevron.up" : "chevron.down")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
            .padding(.vertical, 8)

            if showProfileDetails, let summary = profileSummary {
                ProfileDetailsView(summary: summary)
            }
        }
        .background(.ultraThinMaterial)
    }
}

// MARK: - Profile Details View

struct ProfileDetailsView: View {
    let summary: ProfileSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()

            // Device info
            HStack {
                Text("Device:")
                    .foregroundStyle(.secondary)
                Text(summary.deviceInfo.architecture)
                    .fontWeight(.medium)
                Spacer()
                Text("RAM: \(formatBytesUI(summary.deviceInfo.memorySize))")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)

            Divider()

            // Steps table header
            HStack {
                Text("Step")
                    .frame(width: 140, alignment: .leading)
                Text("Time")
                    .frame(width: 70, alignment: .trailing)
                Text("MLX \u{0394}")
                    .frame(width: 80, alignment: .trailing)
                Text("Process \u{0394}")
                    .frame(width: 80, alignment: .trailing)
            }
            .font(.caption2.bold())
            .foregroundStyle(.secondary)

            // Steps
            ForEach(Array(summary.steps.enumerated()), id: \.offset) { _, step in
                HStack {
                    Text(step.name)
                        .frame(width: 140, alignment: .leading)
                        .lineLimit(1)
                    Text(String(format: "%.3fs", step.duration))
                        .frame(width: 70, alignment: .trailing)
                    Text(formatDeltaBytesUI(step.endMemory.mlxActive - step.startMemory.mlxActive))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.mlxActive > step.startMemory.mlxActive ? .orange : .green)
                    Text(formatDeltaBytesUI(Int(step.endMemory.processFootprint - step.startMemory.processFootprint)))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.processFootprint > step.startMemory.processFootprint ? .orange : .green)
                }
                .font(.caption)
            }

            Divider()

            // Totals
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Peak")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.peakMemoryUsed))
                        .fontWeight(.medium)
                        .foregroundStyle(.orange)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Active")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.finalSnapshot.mlxActive))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Cache")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.finalSnapshot.mlxCache))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("Process")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(Int(summary.finalSnapshot.processFootprint)))
                        .fontWeight(.medium)
                        .foregroundStyle(.blue)
                }

                Spacer()
            }
            .font(.caption)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }
}

// Helper functions for formatting
private func formatBytesUI(_ bytes: Int) -> String {
    let absBytes = abs(bytes)
    if absBytes >= 1024 * 1024 * 1024 {
        return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
    } else if absBytes >= 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    } else if absBytes >= 1024 {
        return String(format: "%.1f KB", Double(bytes) / 1024)
    }
    return "\(bytes) B"
}

private func formatDeltaBytesUI(_ bytes: Int) -> String {
    let sign = bytes >= 0 ? "+" : ""
    return sign + formatBytesUI(bytes)
}

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

// MARK: - Models Management View

struct ModelsManagementView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var modelToDelete: ModelInfo?
    @State private var qwen3ModelToDelete: Qwen3ModelInfo?
    @State private var showDeleteConfirmation = false
    @State private var showQwen3DeleteConfirmation = false
    @State private var memoryRefreshTrigger = false

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                // Memory status bar
                HStack(spacing: 16) {
                    Label("MLX Memory", systemImage: "memorychip")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)

                    HStack(spacing: 8) {
                        Text("Active: \(ModelManager.formatBytes(modelManager.memoryStats.active))")
                        Text("Cache: \(ModelManager.formatBytes(modelManager.memoryStats.cache))")
                            .foregroundStyle(modelManager.memoryStats.cache > 0 ? .orange : .secondary)
                        Text("Peak: \(ModelManager.formatBytes(modelManager.memoryStats.peak))")
                            .foregroundStyle(.blue)
                    }
                    .font(.caption.monospaced())

                    Spacer()

                    Button(action: {
                        modelManager.clearCache()
                        modelManager.resetPeakMemory()
                        memoryRefreshTrigger.toggle()
                    }) {
                        Label("Clear Cache", systemImage: "trash.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(modelManager.memoryStats.cache == 0)
                    .help("Clear MLX recyclable cache")

                    Button(action: {
                        modelManager.unloadModel()
                        modelManager.unloadQwen3Model()
                        memoryRefreshTrigger.toggle()
                    }) {
                        Label("Unload All", systemImage: "xmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(.orange)
                    .disabled(!modelManager.isLoaded && !modelManager.isQwen3Loaded)
                    .help("Unload all models to free GPU memory")
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .id(memoryRefreshTrigger)

                Divider()

                // Download progress bar
                if modelManager.isDownloading {
                    VStack(spacing: 4) {
                        ProgressView(value: modelManager.downloadProgress)
                            .progressViewStyle(.linear)
                        Text(modelManager.downloadMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)

                    Divider()
                }

                // ===== MISTRAL MODELS SECTION =====
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Label("Mistral Small 3.2", systemImage: "brain.filled.head.profile")
                            .font(.headline)

                        Spacer()

                        Button(action: {
                            NSWorkspace.shared.open(ModelManager.modelsCacheDirectory)
                        }) {
                            Label("Open Cache Folder", systemImage: "folder")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Open models cache folder in Finder")

                        Button(action: {
                            modelManager.refreshDownloadedModels()
                            modelManager.refreshDownloadedQwen3Models()
                        }) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .help("Refresh")
                    }
                    .padding(.horizontal)
                    .padding(.top)

                    Text("24B VLM for text, vision, chat, and FLUX.2-dev embeddings")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)

                    // Downloaded Mistral models
                    if !modelManager.downloadedModels.isEmpty {
                        ForEach(modelManager.availableModels.filter { modelManager.downloadedModels.contains($0.id) }, id: \.id) { model in
                            ModelRowView(
                                model: model,
                                size: modelManager.modelSizes[model.id],
                                isLoaded: modelManager.currentLoadedModelId == model.id,
                                onDelete: {
                                    modelToDelete = model
                                    showDeleteConfirmation = true
                                },
                                onLoad: {
                                    Task { await modelManager.loadModel(model.id) }
                                }
                            )
                            .padding(.horizontal)
                        }
                    }

                    // Available Mistral models to download
                    let availableMistral = modelManager.availableModels.filter { !modelManager.downloadedModels.contains($0.id) }
                    if !availableMistral.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(availableMistral, id: \.id) { model in
                                    AvailableModelCard(model: model, modelManager: modelManager)
                                }
                            }
                            .padding(.horizontal)
                        }
                    } else if modelManager.downloadedModels.isEmpty {
                        Text("No Mistral models downloaded. Download one to get started.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.bottom)
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))

                Divider()

                // ===== QWEN3 MODELS SECTION =====
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Label("Qwen3 (FLUX.2 Klein)", systemImage: "cube.fill")
                            .font(.headline)
                            .foregroundStyle(.orange)

                        if modelManager.isQwen3Loaded {
                            Text(modelManager.loadedQwen3Variant?.displayName ?? "Loaded")
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.green.opacity(0.2))
                                .foregroundStyle(.green)
                                .cornerRadius(4)
                        }

                        Spacer()

                        if modelManager.isQwen3Loaded {
                            Button(action: { modelManager.unloadQwen3Model() }) {
                                Label("Unload", systemImage: "xmark.circle")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .tint(.orange)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.top)

                    Text("4B/8B models for Klein 4B (Apache 2.0) and Klein 9B (non-commercial) embeddings")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)

                    // Qwen3 loading progress
                    if modelManager.isQwen3Loading {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text(modelManager.qwen3LoadingMessage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                    }

                    // Downloaded Qwen3 models
                    if !modelManager.downloadedQwen3Models.isEmpty {
                        ForEach(modelManager.availableQwen3Models.filter { modelManager.downloadedQwen3Models.contains($0.id) }, id: \.id) { model in
                            Qwen3ModelRowView(
                                model: model,
                                size: modelManager.qwen3ModelSizes[model.id],
                                isLoaded: modelManager.loadedQwen3Variant == model.variant,
                                onDelete: {
                                    qwen3ModelToDelete = model
                                    showQwen3DeleteConfirmation = true
                                },
                                onLoad: {
                                    Task { await modelManager.loadQwen3Model(model.id) }
                                }
                            )
                            .padding(.horizontal)
                        }
                    }

                    // Available Qwen3 models to download
                    let availableQwen3 = modelManager.availableQwen3Models.filter { !modelManager.downloadedQwen3Models.contains($0.id) }
                    if !availableQwen3.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(availableQwen3, id: \.id) { model in
                                    AvailableQwen3ModelCard(model: model, modelManager: modelManager)
                                }
                            }
                            .padding(.horizontal)
                        }
                    } else if modelManager.downloadedQwen3Models.isEmpty {
                        Text("No Qwen3 models downloaded. Download one for FLUX.2 Klein embeddings.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.bottom)
                .background(Color.orange.opacity(0.05))
            }
        }
        .alert("Delete Model", isPresented: $showDeleteConfirmation, presenting: modelToDelete) { model in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                Task {
                    try? await modelManager.deleteModel(model.id)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete \(model.name)? This cannot be undone.")
        }
        .alert("Delete Qwen3 Model", isPresented: $showQwen3DeleteConfirmation, presenting: qwen3ModelToDelete) { model in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                Task {
                    try? await modelManager.deleteQwen3Model(model.id)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete \(model.name)? This cannot be undone.")
        }
    }
}

// MARK: - Model Row View

struct ModelRowView: View {
    let model: ModelInfo
    let size: Int64?
    let isLoaded: Bool
    let onDelete: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .cornerRadius(4)
                    }
                }
                HStack(spacing: 8) {
                    Text(model.variant.displayName)
                    Text("•")
                    Text(model.parameters)
                    if let size = size {
                        Text("•")
                        Text(TextEncoderModelDownloader.formatSize(size))
                            .foregroundStyle(.blue)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if !isLoaded {
                Button("Load") {
                    onLoad()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .disabled(isLoaded)
            .help(isLoaded ? "Unload model first" : "Delete model")
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Available Model Card

struct AvailableModelCard: View {
    let model: ModelInfo
    @ObservedObject var modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.name)
                .font(.caption.bold())
                .lineLimit(1)

            HStack(spacing: 4) {
                Text(model.variant.estimatedSize)
                Text("•")
                Text(model.parameters)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Button(action: {
                Task { await modelManager.downloadModel(model.id) }
            }) {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download")
                }
                .font(.caption)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(modelManager.isDownloading)
        }
        .padding(10)
        .frame(width: 160)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Qwen3 Model Row View

struct Qwen3ModelRowView: View {
    let model: Qwen3ModelInfo
    let size: Int64?
    let isLoaded: Bool
    let onDelete: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .cornerRadius(4)
                    }
                    Text(model.variant.kleinVariant.displayName)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.orange.opacity(0.2))
                        .foregroundStyle(.orange)
                        .cornerRadius(4)
                }
                HStack(spacing: 8) {
                    Text(model.variant.displayName)
                    Text("•")
                    Text(model.parameters)
                    if let size = size {
                        Text("•")
                        Text(TextEncoderModelDownloader.formatSize(size))
                            .foregroundStyle(.blue)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if !isLoaded {
                Button("Load") {
                    onLoad()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .disabled(isLoaded)
            .help(isLoaded ? "Unload model first" : "Delete model")
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Available Qwen3 Model Card

struct AvailableQwen3ModelCard: View {
    let model: Qwen3ModelInfo
    @ObservedObject var modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.name)
                .font(.caption.bold())
                .lineLimit(1)

            HStack(spacing: 4) {
                Text(model.variant.estimatedSize)
                Text("•")
                Text(model.parameters)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Text(model.variant.kleinVariant.displayName)
                .font(.caption2)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(.orange.opacity(0.2))
                .foregroundStyle(.orange)
                .cornerRadius(4)

            Button(action: {
                Task { await modelManager.downloadQwen3Model(model.id) }
            }) {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download")
                }
                .font(.caption)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .controlSize(.small)
            .disabled(modelManager.isDownloading)
        }
        .padding(10)
        .frame(width: 160)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
    }
}

// MARK: - Qwen3 Chat View

struct Qwen3ChatView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var prompt = ""
    @State private var response = ""
    @State private var isGenerating = false
    @State private var tokensPerSecond: Double = 0
    @State private var promptTokens: Int = 0
    @State private var generatedTokens: Int = 0
    @State private var temperature: Double = 0.7
    @State private var maxTokens: Double = 512

    private let core = FluxTextEncoders.shared

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("Qwen3 Chat", systemImage: "message.fill")
                    .font(.headline)
                    .foregroundStyle(.orange)

                Spacer()

                if modelManager.isQwen3Loaded {
                    Text(modelManager.loadedQwen3Variant?.displayName ?? "Loaded")
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.green.opacity(0.2))
                        .foregroundStyle(.green)
                        .cornerRadius(4)
                } else {
                    Text("Qwen3 not loaded")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding()
            .background(.ultraThinMaterial)

            Divider()

            if !modelManager.isQwen3Loaded {
                VStack(spacing: 16) {
                    Image(systemName: "cube.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.orange.opacity(0.5))
                    Text("Qwen3 Model Not Loaded")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("Load a Qwen3 model from the Models tab to use Qwen3 Chat")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HSplitView {
                    // Input panel
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Prompt")
                            .font(.caption.bold())
                            .foregroundStyle(.secondary)

                        TextEditor(text: $prompt)
                            .font(.body.monospaced())
                            .scrollContentBackground(.hidden)
                            .background(Color(nsColor: .textBackgroundColor))
                            .cornerRadius(8)

                        // Parameters
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Temperature: \(temperature, specifier: "%.2f")")
                                    .font(.caption)
                                Slider(value: $temperature, in: 0...2)
                            }
                            HStack {
                                Text("Max Tokens: \(Int(maxTokens))")
                                    .font(.caption)
                                Slider(value: $maxTokens, in: 64...2048)
                            }
                        }

                        HStack {
                            Button(action: generate) {
                                HStack {
                                    if isGenerating {
                                        ProgressView()
                                            .scaleEffect(0.8)
                                    }
                                    Text(isGenerating ? "Generating..." : "Generate")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.orange)
                            .disabled(prompt.isEmpty || isGenerating || !modelManager.isQwen3Loaded)

                            Button("Clear") {
                                prompt = ""
                                response = ""
                                tokensPerSecond = 0
                                promptTokens = 0
                                generatedTokens = 0
                            }
                            .buttonStyle(.bordered)

                            Spacer()

                            // Stats
                            if tokensPerSecond > 0 {
                                HStack(spacing: 8) {
                                    Text("\(promptTokens) prompt")
                                    Text("•")
                                    Text("\(generatedTokens) generated")
                                    Text("•")
                                    Text("\(tokensPerSecond, specifier: "%.1f") tok/s")
                                }
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding()
                    .frame(minWidth: 300)

                    // Output panel
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Response")
                            .font(.caption.bold())
                            .foregroundStyle(.secondary)

                        ScrollView {
                            Text(response.isEmpty ? "Response will appear here..." : response)
                                .font(.body.monospaced())
                                .foregroundStyle(response.isEmpty ? .secondary : .primary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding()
                        }
                        .background(Color(nsColor: .textBackgroundColor))
                        .cornerRadius(8)
                    }
                    .padding()
                    .frame(minWidth: 300)
                }
            }
        }
    }

    private func generate() {
        guard !prompt.isEmpty, !isGenerating else { return }

        isGenerating = true
        response = ""
        tokensPerSecond = 0
        promptTokens = 0
        generatedTokens = 0

        Task {
            do {
                let parameters = GenerateParameters(
                    maxTokens: Int(maxTokens),
                    temperature: Float(temperature),
                    topP: 0.9
                )

                let result = try core.generateQwen3(
                    prompt: prompt,
                    parameters: parameters
                )

                DispatchQueue.main.async {
                    // Use result.text which has thinking tags stripped
                    response = result.text
                    tokensPerSecond = result.tokensPerSecond
                    promptTokens = result.promptTokens
                    generatedTokens = result.generatedTokens
                    isGenerating = false
                }
            } catch {
                DispatchQueue.main.async {
                    response = "Error: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("hfToken") private var hfToken = ""

    var body: some View {
        Form {
            Section("HuggingFace") {
                SecureField("HF Token", text: $hfToken)
                    .textFieldStyle(.roundedBorder)
            }

            Section("Model") {
                Text("Variant: \(modelManager.selectedVariant?.rawValue ?? "None")")
                Text("Status: \(modelManager.isLoaded ? "Loaded" : "Not Loaded")")
            }
        }
        .padding()
        .frame(width: 400, height: 200)
    }
}

#Preview {
    ContentView()
        .environmentObject(ModelManager())
}
