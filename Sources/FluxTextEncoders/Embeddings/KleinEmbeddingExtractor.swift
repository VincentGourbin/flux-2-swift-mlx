/**
 * KleinEmbeddingExtractor.swift
 * Embedding extraction for FLUX.2 Klein using Qwen3 models
 *
 * Klein uses Qwen3 (4B or 8B) as text encoder instead of Mistral
 *
 * IMPORTANT: This implementation matches the official flux2 behavior:
 * - NO system message (unlike Mistral which uses a system message)
 * - RIGHT padding (not left padding)
 * - Includes <think>\n\n</think>\n\n tokens via enable_thinking=False
 */

import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - Klein Embedding Extractor

/// Extracts embeddings from Qwen3 model hidden states for FLUX.2 Klein
public class KleinEmbeddingExtractor {
    private let model: Qwen3ForCausalLM
    private let tokenizer: Tokenizer
    private let variant: KleinVariant

    // Qwen3 special tokens
    private let padTokenId: Int
    private let imStartTokenId: Int   // <|im_start|>
    private let imEndTokenId: Int     // <|im_end|>

    public init(model: Qwen3ForCausalLM, tokenizer: Tokenizer, variant: KleinVariant) {
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant

        // Qwen3 special token IDs
        // These are standard for Qwen3 models
        self.padTokenId = 151643      // <|endoftext|> used as pad
        self.imStartTokenId = 151644  // <|im_start|>
        self.imEndTokenId = 151645    // <|im_end|>
    }

    /// Extract Klein embeddings from a text prompt
    /// - Parameters:
    ///   - prompt: User prompt text
    ///   - maxLength: Maximum sequence length (default: 512)
    /// - Returns: Embeddings tensor with shape [1, maxLength, outputDim]
    ///           Klein 4B: [1, 512, 7680]
    ///           Klein 9B: [1, 512, 12288]
    public func extractKleinEmbeddings(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> MLXArray {
        // Use autoreleasepool to ensure all intermediate objects are released
        // This is critical for preventing memory accumulation during training
        return try autoreleasepool {
            // 1. Clean the prompt (remove [IMG] tokens)
            let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")

            // 2. Apply Qwen3 chat template (NO system message - matches official flux2)
            // Format: <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
            let formattedPrompt = formatQwen3ChatTemplate(
                userMessage: cleanedPrompt,
                addGenerationPrompt: true  // Matches Python behavior with enable_thinking=False
            )

            // 3. Tokenize
            var tokenIds = tokenizer.encode(text: formattedPrompt)

            FluxDebug.log("Klein embeddings: encoded \(tokenIds.count) tokens before padding")

            // 4. Truncate if needed
            if tokenIds.count > maxLength {
                tokenIds = Array(tokenIds.prefix(maxLength))
                FluxDebug.log("Klein embeddings: truncated to \(maxLength) tokens")
            }

            // 5. RIGHT-pad to fixed length (matching official flux2 implementation)
            let originalLength = tokenIds.count
            let padCount: Int
            if tokenIds.count < maxLength {
                padCount = maxLength - tokenIds.count
                let padding = Array(repeating: padTokenId, count: padCount)
                tokenIds = tokenIds + padding  // RIGHT padding
            } else {
                padCount = 0
            }

            FluxDebug.log("Klein embeddings: padded from \(originalLength) to \(tokenIds.count) tokens (pad count: \(padCount))")

            // 6. Create input tensor - use direct Int array to avoid .map overhead
            let inputIds = MLXArray(tokenIds).asType(.int32).reshaped([1, tokenIds.count])

            // 7. Create attention mask using GPU-native operations (avoids CPU-bound Swift Array creation)
            // For RIGHT padding: [1, 1, ..., 1, 0, 0, ..., 0]
            // Use MLXArray.arange and comparison to generate mask on GPU
            let positionIndices = MLXArray.arange(maxLength, dtype: .int32)
            let attentionMask = (positionIndices .< Int32(originalLength)).asType(.int32).reshaped([1, maxLength])

            FluxDebug.log("Klein embeddings: attention mask created with \(padCount) masked positions")

            // 8. Forward pass with SELECTIVE hidden states extraction
            // CRITICAL: Use forwardWithHiddenStates to only extract the 3 layers we need (9, 18, 27)
            // instead of collecting all 37 hidden states which causes massive memory usage
            let extractedStates = model.model.forwardWithHiddenStates(
                inputIds,
                layerIndices: variant.hiddenStateLayers,
                attentionMask: attentionMask
            )

            // 9. Collect hidden states in order and concatenate immediately
            // Don't keep intermediate arrays around
            var orderedStates: [MLXArray] = []
            for layerIdx in variant.hiddenStateLayers {
                guard let hiddenState = extractedStates[layerIdx] else {
                    throw KleinEmbeddingError.invalidLayerIndex(layerIdx, model.model.layers.count)
                }
                orderedStates.append(hiddenState)
            }

            // 10. Concatenate along hidden dimension: [1, seq, hidden] x 3 -> [1, seq, hidden*3]
            let embeddings = concatenated(orderedStates, axis: -1)

            // CRITICAL: Force synchronous evaluation to materialize computation
            eval(embeddings)

            // Clear the GPU cache to release intermediate buffers
            MLX.Memory.clearCache()

            FluxDebug.log("Klein embeddings: shape \(embeddings.shape)")

            return embeddings
        }
    }

    /// Format prompt using Qwen3 chat template (matches official flux2 implementation)
    /// The official flux2 implementation uses NO system message for Klein embeddings
    /// and includes <think>\n\n</think>\n\n tokens via enable_thinking=False
    /// - Parameters:
    ///   - userMessage: User message content
    ///   - addGenerationPrompt: Whether to add assistant prompt with thinking tokens
    /// - Returns: Formatted prompt string
    private func formatQwen3ChatTemplate(
        userMessage: String,
        addGenerationPrompt: Bool
    ) -> String {
        var prompt = ""

        // User message (NO system message - matches official flux2)
        prompt += "<|im_start|>user\n"
        prompt += userMessage
        prompt += "<|im_end|>\n"

        // Assistant prompt with thinking tokens (if requested)
        // This matches the official flux2 behavior with enable_thinking=False
        if addGenerationPrompt {
            prompt += "<|im_start|>assistant\n"
            prompt += "<think>\n\n</think>\n\n"
        }

        return prompt
    }

    /// Get token IDs for Klein format (useful for debugging/comparison)
    public func getKleinTokenIds(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> [Int] {
        let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
        let formattedPrompt = formatQwen3ChatTemplate(
            userMessage: cleanedPrompt,
            addGenerationPrompt: true
        )

        var tokenIds = tokenizer.encode(text: formattedPrompt)

        // Truncate if needed
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        // RIGHT-pad to fixed length (matching official flux2)
        if tokenIds.count < maxLength {
            let padCount = maxLength - tokenIds.count
            let padding = Array(repeating: padTokenId, count: padCount)
            tokenIds = tokenIds + padding  // RIGHT padding
        }

        return tokenIds
    }

    /// Get the variant this extractor is configured for
    public var kleinVariant: KleinVariant {
        return variant
    }

    /// Get embedding dimension for this variant
    public var embeddingDimension: Int {
        return variant.outputDimension
    }
}

// MARK: - Errors

public enum KleinEmbeddingError: LocalizedError {
    case noHiddenStates
    case invalidLayerIndex(Int, Int)
    case tokenizerError(String)
    case modelNotLoaded

    public var errorDescription: String? {
        switch self {
        case .noHiddenStates:
            return "Qwen3 model did not return hidden states"
        case .invalidLayerIndex(let idx, let max):
            return "Invalid layer index \(idx), Qwen3 model has \(max) layers"
        case .tokenizerError(let message):
            return "Qwen3 tokenizer error: \(message)"
        case .modelNotLoaded:
            return "Klein model not loaded"
        }
    }
}
