/**
 * MistralGenerator.swift
 * Text generation with streaming support for Mistral models
 */

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Parameters for text generation
public struct GenerateParameters: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var repetitionContextSize: Int
    public var seed: UInt64?

    /// Maximum context length supported by Mistral Small 3.2 (131K tokens)
    public static let maxContextLength = 131_072

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.7,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 20,
        seed: UInt64? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.seed = seed
    }

    /// Greedy decoding (temperature = 0)
    public static let greedy = GenerateParameters(
        maxTokens: 2048,
        temperature: 0.0,
        topP: 1.0,
        repetitionPenalty: 1.0
    )

    /// Creative generation
    public static let creative = GenerateParameters(
        maxTokens: 4096,
        temperature: 0.9,
        topP: 0.95,
        repetitionPenalty: 1.2
    )

    /// Balanced generation
    public static let balanced = GenerateParameters(
        maxTokens: 2048,
        temperature: 0.7,
        topP: 0.9,
        repetitionPenalty: 1.1
    )
}

/// Generator result
public struct GenerationResult: Sendable {
    public let text: String
    public let tokens: [Int]
    public let promptTokens: Int
    public let generatedTokens: Int
    public let totalTime: Double
    public let tokensPerSecond: Double

    public func summary() -> String {
        return """
        Prompt: \(promptTokens) tokens
        Generated: \(generatedTokens) tokens
        Speed: \(String(format: "%.1f", tokensPerSecond)) tokens/s
        Time: \(String(format: "%.2f", totalTime))s
        """
    }
}

/// Text generator for Mistral models
public final class MistralGenerator: @unchecked Sendable {
    private let model: MistralForCausalLM
    private let tokenizer: TekkenTokenizer

    public init(model: MistralForCausalLM, tokenizer: TekkenTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Generate text with streaming callback
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()
        let profiler = FluxProfiler.shared

        // Set random seed if provided
        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Tokenization with profiling
        profiler.startTokenization()
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let promptTokens = tokenizer.encodeChatMessages(messages: messages, addGenerationPrompt: true)
        var inputIds = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        profiler.endTokenization(tokenCount: promptTokens.count)

        // Create KV cache
        let cache = model.createCache()

        // Prefill with profiling
        profiler.startPrefill()
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)
        profiler.endPrefill()

        // Generation loop - optimized with async eval pipeline
        // Pattern from mlx-swift-lm: overlap GPU computation with CPU token processing
        profiler.startGeneration()
        var generatedTokens: [Int] = []
        let eosToken = tokenizer.eosToken
        let hasCallback = onToken != nil

        // Token accumulation for batched streaming (reduces I/O overhead)
        var pendingTokens: [Int] = []
        let streamBatchSize = 10  // Accumulate up to 10 tokens before streaming

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

            // While GPU is evaluating, prepare for next iteration
            // This overlaps CPU work with GPU computation

            // Now sync to get the token value
            let nextToken = Int(nextTokenArray.item(Int32.self))

            // Check for EOS
            if nextToken == eosToken {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming: accumulate tokens then flush
            if hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let decodeStart = CFAbsoluteTimeGetCurrent()
                    let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
                    profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            // Forward pass for next token - start GPU computation early
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

            // Periodically clear GPU cache to prevent memory accumulation
            if (i + 1) % 20 == 0 {
                Memory.clearCache()
            }
        }

        // Flush any remaining tokens
        if hasCallback && !pendingTokens.isEmpty {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
            profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
            _ = onToken!(tokenText)
        }

        profiler.endGeneration(tokenCount: generatedTokens.count)

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        let decodeStart = CFAbsoluteTimeGetCurrent()
        let outputText = tokenizer.decode(generatedTokens, skipSpecialTokens: true)
        profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)

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

    /// Generate with chat template
    /// - Parameters:
    ///   - messages: Chat messages
    ///   - parameters: Generation parameters
    ///   - stream: If true, call onToken incrementally; if false, call once at end with complete text
    ///   - onToken: Callback for token output
    public func chat(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        stream: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()
        let profiler = FluxProfiler.shared

        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Tokenization with profiling
        profiler.startTokenization()
        let promptTokens = tokenizer.encodeChatMessages(messages: messages, addGenerationPrompt: true)
        var inputIds = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        profiler.endTokenization(tokenCount: promptTokens.count)

        let cache = model.createCache()

        // Prefill with profiling
        profiler.startPrefill()
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)
        profiler.endPrefill()

        // Generation loop - optimized with async eval pipeline
        profiler.startGeneration()
        var generatedTokens: [Int] = []
        let eosToken = tokenizer.eosToken
        let hasCallback = onToken != nil

        // Token accumulation for batched streaming (reduces I/O overhead)
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

            if nextToken == eosToken {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming (only if stream mode is enabled)
            if stream && hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let decodeStart = CFAbsoluteTimeGetCurrent()
                    let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
                    profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
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

            // Periodically clear GPU cache to prevent memory accumulation
            if (i + 1) % 20 == 0 {
                Memory.clearCache()
            }
        }

        // Flush remaining pending tokens (streaming mode)
        if stream && hasCallback && !pendingTokens.isEmpty {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
            profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
            _ = onToken!(tokenText)
        }

        profiler.endGeneration(tokenCount: generatedTokens.count)

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        let decodeStart = CFAbsoluteTimeGetCurrent()
        let outputText = tokenizer.decode(generatedTokens, skipSpecialTokens: true)
        profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)

        // Non-streaming mode: call callback once with complete text
        if !stream && hasCallback {
            _ = onToken!(outputText)
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

    /// AsyncStream-based generation for async/await usage
    public func generateStream(
        prompt: String,
        parameters: GenerateParameters = .balanced
    ) -> AsyncStream<String> {
        let generator = self
        return AsyncStream { continuation in
            Task { @Sendable in
                do {
                    _ = try generator.generate(prompt: prompt, parameters: parameters) { token in
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

    /// GPU-optimized top-p (nucleus) sampling using MLX
    /// Based on mlx-swift-lm implementation for compatibility with MLX 0.30+
    private func sampleTopPGPU(_ logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
        // Apply temperature and softmax
        let probs = softmax(logits / temperature, axis: -1)

        // Sort indices by probability (descending order)
        let sortedIndices = argSort(-probs, axis: -1)

        // Gather sorted probabilities using take() for MLX 0.30+ compatibility
        // For 1D input, take() returns same shape
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
        // sortedToken is a scalar, use it to index sortedIndices
        return sortedIndices[sortedToken]
    }
}
