/**
 * Qwen3Generator.swift
 * Text generation with streaming support for Qwen3 models
 *
 * Used for both text generation and as part of Klein embedding pipeline
 */

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

/// Text generator for Qwen3 models
public final class Qwen3Generator: @unchecked Sendable {
    private let model: Qwen3ForCausalLM
    private let tokenizer: Tokenizer

    // Qwen3 special tokens
    private let eosTokenId: Int
    private let padTokenId: Int
    private let imEndTokenId: Int

    public init(model: Qwen3ForCausalLM, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer

        // Qwen3 special token IDs (standard for Qwen3 models)
        self.padTokenId = 151643      // <|endoftext|>
        self.eosTokenId = 151645      // <|im_end|>
        self.imEndTokenId = 151645    // <|im_end|>
    }

    /// Generate text from a prompt
    /// - Parameters:
    ///   - prompt: The user's prompt
    ///   - parameters: Generation parameters (temperature, topP, etc.)
    ///   - enableThinking: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)
    ///   - onToken: Optional callback for streaming tokens
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        enableThinking: Bool = false,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()

        // Set random seed if provided
        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Format with Qwen3 chat template (user message only)
        // For FLUX.2 usage, thinking is disabled by default
        let formattedPrompt = formatQwen3ChatTemplate(userMessage: prompt, enableThinking: enableThinking)

        // Tokenize
        let promptTokens = tokenizer.encode(text: formattedPrompt)
        var inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])

        // Create KV cache
        let cache = model.createCache()

        // Prefill
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)

        // Generation loop - optimized with async eval pipeline
        var generatedTokens: [Int] = []
        let hasCallback = onToken != nil

        // Token accumulation for batched streaming
        var pendingTokens: [Int] = []
        let streamBatchSize = 10

        // Sample first token (no history yet for repetition penalty)
        var nextTokenArray = sampleNextToken(
            logits: logits,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize,
            generatedTokens: generatedTokens
        )

        for i in 0..<parameters.maxTokens {
            // Kick off async eval for current token (GPU starts computing)
            MLX.asyncEval(nextTokenArray)

            // Now sync to get the token value
            let nextToken = Int(nextTokenArray.item(Int32.self))

            // Check for EOS tokens
            if nextToken == eosTokenId || nextToken == padTokenId {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming
            if hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let tokenText = tokenizer.decode(tokens: pendingTokens)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            // Forward pass for next token
            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)

            // Sample next token with repetition penalty (lazy - will be async eval'd at start of next iteration)
            nextTokenArray = sampleNextToken(
                logits: logits,
                temperature: parameters.temperature,
                topP: parameters.topP,
                repetitionPenalty: parameters.repetitionPenalty,
                repetitionContextSize: parameters.repetitionContextSize,
                generatedTokens: generatedTokens
            )

            // Periodically clear GPU cache
            if (i + 1) % 20 == 0 {
                Memory.clearCache()
            }
        }

        // Flush remaining tokens
        if hasCallback && !pendingTokens.isEmpty {
            let tokenText = tokenizer.decode(tokens: pendingTokens)
            _ = onToken!(tokenText)
        }

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        var outputText = tokenizer.decode(tokens: generatedTokens)

        // Strip empty thinking tags when thinking is disabled
        if !enableThinking {
            outputText = stripEmptyThinkingTags(outputText)
        }

        // Clear KV cache to free memory
        cache.forEach { $0.clear() }
        MLX.Memory.clearCache()

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// Generate with chat messages (multi-turn conversation)
    /// - Parameters:
    ///   - messages: Array of message dictionaries with "role" and "content" keys
    ///   - parameters: Generation parameters
    ///   - enableThinking: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)
    ///   - stream: If true, call onToken incrementally; if false, call once at end with complete text
    ///   - onToken: Optional callback for streaming tokens
    public func chat(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        enableThinking: Bool = false,
        stream: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()

        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Format with Qwen3 chat template
        let formattedPrompt = formatQwen3ChatMessages(messages: messages, enableThinking: enableThinking)

        // Tokenize
        let promptTokens = tokenizer.encode(text: formattedPrompt)
        var inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])

        let cache = model.createCache()

        // Prefill
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)

        // Generation loop - optimized with async eval pipeline
        var generatedTokens: [Int] = []
        let hasCallback = onToken != nil

        var pendingTokens: [Int] = []
        let streamBatchSize = 10

        // Sample first token (no history yet for repetition penalty)
        var nextTokenArray = sampleNextToken(
            logits: logits,
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize,
            generatedTokens: generatedTokens
        )

        for i in 0..<parameters.maxTokens {
            // Kick off async eval for current token
            MLX.asyncEval(nextTokenArray)

            // Sync to get the token value
            let nextToken = Int(nextTokenArray.item(Int32.self))

            if nextToken == eosTokenId || nextToken == padTokenId {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming (only if stream mode is enabled)
            if stream && hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let tokenText = tokenizer.decode(tokens: pendingTokens)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            // Forward pass for next token
            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)

            // Sample next token with repetition penalty (lazy)
            nextTokenArray = sampleNextToken(
                logits: logits,
                temperature: parameters.temperature,
                topP: parameters.topP,
                repetitionPenalty: parameters.repetitionPenalty,
                repetitionContextSize: parameters.repetitionContextSize,
                generatedTokens: generatedTokens
            )

            if (i + 1) % 20 == 0 {
                Memory.clearCache()
            }
        }

        // Flush remaining pending tokens (streaming mode)
        if stream && hasCallback && !pendingTokens.isEmpty {
            let tokenText = tokenizer.decode(tokens: pendingTokens)
            _ = onToken!(tokenText)
        }

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        var outputText = tokenizer.decode(tokens: generatedTokens)

        // Strip empty thinking tags when thinking is disabled
        if !enableThinking {
            outputText = stripEmptyThinkingTags(outputText)
        }

        // Non-streaming mode: call callback once with complete text
        if !stream && hasCallback {
            _ = onToken!(outputText)
        }

        cache.forEach { $0.clear() }
        MLX.Memory.clearCache()

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// AsyncStream-based generation
    /// - Parameters:
    ///   - prompt: The user's prompt
    ///   - parameters: Generation parameters
    ///   - enableThinking: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)
    public func generateStream(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        enableThinking: Bool = false
    ) -> AsyncStream<String> {
        let generator = self
        let thinking = enableThinking
        return AsyncStream { continuation in
            Task { @Sendable in
                do {
                    _ = try generator.generate(prompt: prompt, parameters: parameters, enableThinking: thinking) { token in
                        continuation.yield(token)
                        return true
                    }
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Private Helpers

    /// Sample next token from logits with repetition penalty (returns lazy MLXArray for async eval)
    private func sampleNextToken(
        logits: MLXArray,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float = 1.0,
        repetitionContextSize: Int = 20,
        generatedTokens: [Int] = []
    ) -> MLXArray {
        var lastLogits = logits[0, -1]

        // Apply repetition penalty to recently generated tokens
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            let contextSize = min(repetitionContextSize, generatedTokens.count)
            let recentTokens = Array(generatedTokens.suffix(contextSize))
            let tokenSet = Set(recentTokens)

            var logitsArray = lastLogits.asArray(Float.self)
            for tokenId in tokenSet {
                if tokenId >= 0 && tokenId < logitsArray.count {
                    if logitsArray[tokenId] > 0 {
                        logitsArray[tokenId] /= repetitionPenalty
                    } else {
                        logitsArray[tokenId] *= repetitionPenalty
                    }
                }
            }
            lastLogits = MLXArray(logitsArray)
        }

        if temperature == 0 {
            return argMax(lastLogits)
        } else {
            return sampleTopPGPU(lastLogits, temperature: temperature, topP: topP)
        }
    }

    /// Format a single user message using Qwen3 chat template
    /// - Parameters:
    ///   - userMessage: The user's message
    ///   - enableThinking: If false, appends /no_think to disable Qwen3 thinking mode
    private func formatQwen3ChatTemplate(userMessage: String, enableThinking: Bool = true) -> String {
        var prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += userMessage
        // Disable thinking mode for FLUX.2 and other direct-response use cases
        if !enableThinking {
            prompt += " /no_think"
        }
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    }

    /// Format multi-turn chat messages using Qwen3 chat template
    /// - Parameters:
    ///   - messages: Array of message dictionaries with "role" and "content" keys
    ///   - enableThinking: If false, appends /no_think to the last user message
    private func formatQwen3ChatMessages(messages: [[String: String]], enableThinking: Bool = true) -> String {
        var prompt = ""

        // Check if system message is included
        let hasSystemMessage = messages.first { $0["role"] == "system" } != nil

        // Add default system message if not provided
        if !hasSystemMessage {
            prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        }

        // Find the last user message index for /no_think insertion
        let lastUserIndex = messages.lastIndex { $0["role"] == "user" }

        for (index, message) in messages.enumerated() {
            guard let role = message["role"], let content = message["content"] else {
                continue
            }

            prompt += "<|im_start|>\(role)\n"
            prompt += content
            // Add /no_think to the last user message if thinking is disabled
            if !enableThinking && role == "user" && index == lastUserIndex {
                prompt += " /no_think"
            }
            prompt += "<|im_end|>\n"
        }

        // Add assistant prompt
        prompt += "<|im_start|>assistant\n"

        return prompt
    }

    /// GPU-optimized top-p (nucleus) sampling using MLX
    /// Based on mlx-swift-lm implementation for compatibility with MLX 0.30+
    private func sampleTopPGPU(_ logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
        // Apply temperature and softmax
        let probs = softmax(logits / temperature, axis: -1)

        // Sort indices by probability (descending order)
        let sortedIndices = argSort(-probs, axis: -1)

        // Gather sorted probabilities using take() for MLX 0.30+ compatibility
        let sortedProbs = MLX.take(probs, sortedIndices, axis: -1)

        // Compute cumulative probabilities
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Create mask for top-p: keep tokens where cumulative prob > (1 - topP)
        let topProbs = MLX.where(
            cumulativeProbs .> (1 - topP),
            sortedProbs,
            MLX.zeros(like: sortedProbs)
        )

        // Sample from categorical distribution using log probabilities
        let sortedToken = MLXRandom.categorical(MLX.log(topProbs + 1e-10))

        // Map back to original vocabulary index
        return sortedIndices[sortedToken]
    }

    /// Strip empty thinking tags from output text
    /// Qwen3 may still output <think></think> even with /no_think flag
    private func stripEmptyThinkingTags(_ text: String) -> String {
        // Remove <think>\n</think>\n pattern (with optional whitespace)
        var result = text
        // Handle various whitespace patterns
        result = result.replacingOccurrences(
            of: "<think>\\s*</think>\\s*",
            with: "",
            options: .regularExpression
        )
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
