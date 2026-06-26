/**
 * ChatViews.swift
 * Chat surfaces: the general assistant chat, message bubbles, and the Qwen3 chat
 * panel. Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

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

