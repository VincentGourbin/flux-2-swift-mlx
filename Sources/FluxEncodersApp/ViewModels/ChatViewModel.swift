/**
 * ChatViewModel.swift
 * View model for chat functionality
 */

import SwiftUI
import FluxTextEncoders

// MARK: - Chat Message

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: MessageRole
    var content: String
    let timestamp: Date

    enum MessageRole {
        case user
        case assistant
        case system
    }

    init(role: MessageRole, content: String) {
        self.role = role
        self.content = content
        self.timestamp = Date()
    }
}

// MARK: - Generation Stats

struct GenerationStats {
    let tokenCount: Int
    let duration: Double
    var tokensPerSecond: Double { Double(tokenCount) / max(duration, 0.001) }
}

// MARK: - Chat View Model

@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText = ""
    @Published var isGenerating = false
    @Published var currentTokenCount = 0

    // Stats after generation
    @Published var lastGenerationStats: GenerationStats?
    @Published var lastProfileSummary: ProfileSummary?

    // Generation parameters (configurable from UI)
    @Published var maxTokens: Int = 2048
    @Published var temperature: Float = 0.7

    weak var modelManager: ModelManager?

    func sendMessage() async {
        guard !inputText.isEmpty else { return }
        guard modelManager?.isLoaded == true else { return }

        let userMessage = ChatMessage(role: .user, content: inputText)
        messages.append(userMessage)

        inputText = ""
        isGenerating = true
        currentTokenCount = 0
        lastGenerationStats = nil
        lastProfileSummary = nil

        // Reset profiler for this message
        FluxProfiler.shared.reset()
        let generationStart = Date()

        // Build messages array for chat API
        let chatMessages = messages.map { message -> [String: String] in
            let role: String
            switch message.role {
            case .user: role = "user"
            case .assistant: role = "assistant"
            case .system: role = "system"
            }
            return ["role": role, "content": message.content]
        }

        // Add placeholder for assistant response
        let assistantMessage = ChatMessage(role: .assistant, content: "")
        messages.append(assistantMessage)
        let assistantIndex = messages.count - 1

        // Capture parameters for detached task
        let maxTok = maxTokens
        let temp = temperature

        // Run inference on background thread to keep UI responsive
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let result = try FluxTextEncoders.shared.chat(
                    messages: chatMessages,
                    parameters: GenerateParameters(
                        maxTokens: maxTok,
                        temperature: temp,
                        topP: 0.95,
                        repetitionPenalty: 1.1
                    )
                ) { token in
                    Task { @MainActor [weak self] in
                        self?.messages[assistantIndex].content += token
                        self?.currentTokenCount += 1
                    }
                    return true
                }

                // Don't reassign result.text - streaming callback already built the content
                // This avoids race conditions with pending Task callbacks

                // Capture result values before MainActor hop
                let generatedTokens = result.generatedTokens
                let duration = Date().timeIntervalSince(generationStart)
                let profilerEnabled = FluxProfiler.shared.isEnabled
                let summary = profilerEnabled ? FluxProfiler.shared.summary() : nil

                await MainActor.run { [weak self] in
                    self?.lastGenerationStats = GenerationStats(
                        tokenCount: generatedTokens,
                        duration: duration
                    )

                    if let summary = summary {
                        self?.lastProfileSummary = summary
                    }

                    self?.isGenerating = false
                }

            } catch {
                let errorMessage = error.localizedDescription
                await MainActor.run { [weak self] in
                    self?.messages[assistantIndex].content = "Error: \(errorMessage)"
                    self?.isGenerating = false
                }
            }
        }
    }

    func clearChat() {
        messages.removeAll()
        lastGenerationStats = nil
        lastProfileSummary = nil
        currentTokenCount = 0
    }
}
