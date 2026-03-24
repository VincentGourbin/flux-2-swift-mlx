/**
 * Qwen3VLGenerator.swift
 * Text generation for Qwen3-VL models (language component only)
 *
 * Same generation logic as Qwen3Generator but uses Qwen3VLForCausalLM.
 * Phase 1: text-only generation (no vision input).
 */

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

/// Text generator for Qwen3-VL models
public final class Qwen3VLGenerator: @unchecked Sendable {
    private let model: Qwen3VLForCausalLM
    private let tokenizer: Tokenizer

    private let eosTokenId: Int
    private let padTokenId: Int

    public init(model: Qwen3VLForCausalLM, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        self.padTokenId = 151643
        self.eosTokenId = 151645
    }

    /// Generate text from a prompt
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()

        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Format with Qwen3 chat template
        var formatted = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        formatted += "<|im_start|>user\n\(prompt) /no_think<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"

        let promptTokens = tokenizer.encode(text: formatted)
        var inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])

        let cache = model.createCache()

        // Prefill
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)

        var generatedTokens: [Int] = []
        var pendingTokens: [Int] = []

        // Sample first token
        var nextTokenArray = sampleToken(
            logits: logits, temperature: parameters.temperature, topP: parameters.topP,
            repetitionPenalty: parameters.repetitionPenalty, generatedTokens: generatedTokens
        )

        for i in 0..<parameters.maxTokens {
            MLX.asyncEval(nextTokenArray)
            let nextToken = Int(nextTokenArray.item(Int32.self))

            if nextToken == eosTokenId || nextToken == padTokenId { break }

            generatedTokens.append(nextToken)

            // Stream tokens
            if let callback = onToken {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= 10 {
                    let text = tokenizer.decode(tokens: pendingTokens)
                    if !callback(text) { break }
                    pendingTokens.removeAll()
                }
            }

            // Next token
            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)
            nextTokenArray = sampleToken(
                logits: logits, temperature: parameters.temperature, topP: parameters.topP,
                repetitionPenalty: parameters.repetitionPenalty, generatedTokens: generatedTokens
            )

            if (i + 1) % 20 == 0 { Memory.clearCache() }
        }

        // Flush remaining
        if let callback = onToken, !pendingTokens.isEmpty {
            _ = callback(tokenizer.decode(tokens: pendingTokens))
        }

        let totalTime = Date().timeIntervalSince(startTime)
        var outputText = tokenizer.decode(tokens: generatedTokens)

        // Strip empty thinking tags
        outputText = outputText.replacingOccurrences(
            of: "<think>\\s*</think>\\s*",
            with: "",
            options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        cache.forEach { $0.clear() }
        MLX.Memory.clearCache()

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: Double(generatedTokens.count) / totalTime
        )
    }

    private func sampleToken(
        logits: MLXArray, temperature: Float, topP: Float,
        repetitionPenalty: Float = 1.0, generatedTokens: [Int] = []
    ) -> MLXArray {
        var lastLogits = logits[0, -1]

        // Apply repetition penalty
        if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            let contextSize = min(20, generatedTokens.count)
            let recentTokens = Set(generatedTokens.suffix(contextSize))
            var logitsArray = lastLogits.asArray(Float.self)
            for tokenId in recentTokens {
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
        }
        let probs = softmax(lastLogits / temperature, axis: -1)
        let sortedIndices = argSort(-probs, axis: -1)
        let sortedProbs = MLX.take(probs, sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let topProbs = MLX.where(cumulativeProbs .> (1 - topP), sortedProbs, MLX.zeros(like: sortedProbs))
        let sortedToken = MLXRandom.categorical(MLX.log(topProbs + 1e-10))
        return sortedIndices[sortedToken]
    }
}
