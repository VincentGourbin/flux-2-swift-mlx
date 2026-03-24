/**
 * KleinVLEmbeddingExtractor.swift
 * Embedding extraction for FLUX.2 Klein using Qwen3-VL model
 *
 * Same extraction logic as KleinEmbeddingExtractor but uses
 * Qwen3-VL (with MRoPE) instead of Qwen3 text-only.
 *
 * Phase 1: Text-only mode — generates text-only position IDs
 * (temporal sequential, spatial zeros). Vision input will be
 * added in Phase 2 when ViT + DeepStack are implemented.
 */

import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - Klein VL Embedding Extractor

/// Extracts embeddings from Qwen3-VL model hidden states for FLUX.2 Klein
public class KleinVLEmbeddingExtractor {
    private let model: Qwen3VLForCausalLM
    private let tokenizer: Tokenizer
    private let variant: KleinVariant

    private let padTokenId: Int
    private let imStartTokenId: Int
    private let imEndTokenId: Int

    public init(model: Qwen3VLForCausalLM, tokenizer: Tokenizer, variant: KleinVariant) {
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant

        // Qwen3 special token IDs (same tokenizer family)
        self.padTokenId = 151643
        self.imStartTokenId = 151644
        self.imEndTokenId = 151645
    }

    /// Extract Klein embeddings from a text prompt using Qwen3-VL
    /// Output shape matches standard Klein: [1, maxLength, outputDim]
    ///   Klein 4B: [1, 512, 7680]
    public func extractKleinEmbeddings(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> MLXArray {
        return try autoreleasepool {
            // 1. Clean the prompt
            let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")

            // 2. Apply Qwen3 chat template (same format as text-only)
            let formattedPrompt = formatQwen3ChatTemplate(
                userMessage: cleanedPrompt,
                addGenerationPrompt: true
            )

            // 3. Tokenize
            var tokenIds = tokenizer.encode(text: formattedPrompt)

            FluxDebug.log("Klein-VL embeddings: encoded \(tokenIds.count) tokens before padding")

            // 4. Truncate if needed
            if tokenIds.count > maxLength {
                tokenIds = Array(tokenIds.prefix(maxLength))
            }

            // 5. RIGHT-pad to fixed length
            let originalLength = tokenIds.count
            if tokenIds.count < maxLength {
                let padCount = maxLength - tokenIds.count
                let padding = Array(repeating: padTokenId, count: padCount)
                tokenIds = tokenIds + padding
            }

            // 6. Create input tensor
            let inputIds = MLXArray(tokenIds).asType(.int32).reshaped([1, tokenIds.count])

            // 7. Create attention mask
            let positionIndices = MLXArray.arange(maxLength, dtype: .int32)
            let attentionMask = (positionIndices .< Int32(originalLength)).asType(.int32).reshaped([1, maxLength])

            // 8. Generate text-only position IDs for MRoPE
            let positionIds = Qwen3VLMRoPE.textOnlyPositionIds(seqLen: maxLength)

            // 9. Forward pass with SELECTIVE hidden states extraction
            let extractedStates = model.model.forwardWithHiddenStates(
                inputIds,
                layerIndices: variant.hiddenStateLayers,
                positionIds: positionIds,
                attentionMask: attentionMask
            )

            // 10. Collect hidden states in order
            var orderedStates: [MLXArray] = []
            for layerIdx in variant.hiddenStateLayers {
                guard let hiddenState = extractedStates[layerIdx] else {
                    throw KleinVLEmbeddingError.invalidLayerIndex(layerIdx, model.model.layers.count)
                }
                orderedStates.append(hiddenState)
            }

            // 11. Concatenate along hidden dimension
            let embeddings = concatenated(orderedStates, axis: -1)

            eval(embeddings)
            MLX.Memory.clearCache()

            FluxDebug.log("Klein-VL embeddings: shape \(embeddings.shape)")

            return embeddings
        }
    }

    /// Format prompt using Qwen3 chat template (same as text-only Klein)
    private func formatQwen3ChatTemplate(
        userMessage: String,
        addGenerationPrompt: Bool
    ) -> String {
        var prompt = ""
        prompt += "<|im_start|>user\n"
        prompt += userMessage
        prompt += "<|im_end|>\n"

        if addGenerationPrompt {
            prompt += "<|im_start|>assistant\n"
            prompt += "<think>\n\n</think>\n\n"
        }

        return prompt
    }

    public var kleinVariant: KleinVariant { variant }
    public var embeddingDimension: Int { variant.outputDimension }
}

// MARK: - Errors

public enum KleinVLEmbeddingError: LocalizedError {
    case invalidLayerIndex(Int, Int)
    case modelNotLoaded

    public var errorDescription: String? {
        switch self {
        case .invalidLayerIndex(let idx, let max):
            return "Invalid layer index \(idx), Qwen3-VL model has \(max) layers"
        case .modelNotLoaded:
            return "Klein-VL model not loaded"
        }
    }
}
