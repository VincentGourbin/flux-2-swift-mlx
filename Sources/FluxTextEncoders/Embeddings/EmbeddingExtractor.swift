/**
 * EmbeddingExtractor.swift
 * Configurable embedding extraction from Mistral hidden states
 */

import Foundation
import MLX
import MLXNN

// MARK: - FLUX.2 Configuration

/// Configuration constants for FLUX.2-compatible embeddings
/// These must match exactly with mflux-gradio Python implementation
public enum FluxConfig {
    /// System message used for text encoding (matches HuggingFace/mflux-gradio)
    /// Used for extracting embeddings from image descriptions
    public static let systemMessage = """
You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.
"""

    /// System message for Text-to-Image prompt upsampling (official BFL)
    /// Rewrites user prompts to be more descriptive for better image generation
    public static let systemMessageUpsamplingT2I = """
You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.

Guidelines:
1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.

Output only the revised prompt and nothing else.
"""

    /// System message for Image-to-Image editing (official BFL)
    /// Converts editing requests into concise instructions
    public static let systemMessageUpsamplingI2I = """
You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).

Rules:
- Single instruction only, no commentary
- Use clear, analytical language (avoid "whimsical," "cascading," etc.)
- Specify what changes AND what stays the same (face, lighting, composition)
- Reference actual image elements
- Turn negatives into positives ("don't change X" → "keep X")
- Make abstractions concrete ("futuristic" → "glowing cyan neon, metallic panels")

Output only the final instruction in plain text and nothing else.
"""

    /// Maximum sequence length for padding (text-only embeddings)
    public static let maxSequenceLength = 512

    /// Maximum image size for I2I upsampling (768² as per BFL reference)
    public static let maxImageSizeUpsampling = 768

    /// Maximum image size for reference images (512² as per BFL reference)
    public static let maxImageSizeReference = 512

    /// Hidden state layers to extract (produces 3 * 5120 = 15360 dimensions)
    public static let hiddenStateLayers = [10, 20, 30]

    /// FLUX.2 operation modes
    public enum Mode {
        /// Extract embeddings for image generation conditioning
        case embeddings
        /// Upsample/enhance text-to-image prompts
        case upsamplingT2I
        /// Convert image editing requests to instructions
        case upsamplingI2I
    }

    /// Get the system message for a given FLUX mode
    public static func systemMessage(for mode: Mode) -> String {
        switch mode {
        case .embeddings:
            return systemMessage
        case .upsamplingT2I:
            return systemMessageUpsamplingT2I
        case .upsamplingI2I:
            return systemMessageUpsamplingI2I
        }
    }

    /// Build chat messages for a given FLUX mode
    /// - Parameters:
    ///   - prompt: User prompt
    ///   - mode: FLUX operation mode
    /// - Returns: Messages array ready for chat template
    public static func buildMessages(prompt: String, mode: Mode) -> [[String: String]] {
        return [
            ["role": "system", "content": systemMessage(for: mode)],
            ["role": "user", "content": prompt]
        ]
    }
}

// MARK: - Embedding Extractor

/// Extracts embeddings from Mistral model hidden states
public class EmbeddingExtractor {
    private let model: MistralForCausalLM
    private let tokenizer: TekkenTokenizer

    public init(model: MistralForCausalLM, tokenizer: TekkenTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Extract embeddings from a text prompt
    /// - Parameters:
    ///   - prompt: Input text
    ///   - config: Configuration specifying which layers to extract
    /// - Returns: Embeddings tensor with shape depending on config
    public func extractEmbeddings(
        prompt: String,
        config: HiddenStatesConfig = .mfluxDefault
    ) throws -> MLXArray {
        // Tokenize input
        let tokenIds = tokenizer.encode(prompt, addSpecialTokens: true)
        let inputIds = MLXArray(tokenIds).reshaped([1, tokenIds.count])

        FluxDebug.log("Extracting embeddings for \(tokenIds.count) tokens from layers: \(config.layerIndices)")

        // Forward pass with hidden states
        let output = model(inputIds, outputHiddenStates: true)

        guard let allHiddenStates = output.hiddenStates else {
            throw EmbeddingError.noHiddenStates
        }

        // Resolve layer indices (handle negative indices)
        let numLayers = allHiddenStates.count
        let resolvedIndices = config.layerIndices.map { idx -> Int in
            if idx < 0 {
                return numLayers + idx
            }
            return idx
        }

        // Validate indices
        for idx in resolvedIndices {
            guard idx >= 0 && idx < numLayers else {
                throw EmbeddingError.invalidLayerIndex(idx, numLayers)
            }
        }

        // Extract hidden states from specified layers
        var extractedStates: [MLXArray] = []
        for idx in resolvedIndices {
            var layerHidden = allHiddenStates[idx]

            // Apply pooling if configured
            layerHidden = applyPooling(layerHidden, strategy: config.pooling)

            extractedStates.append(layerHidden)
        }

        // Combine extracted states
        var embeddings: MLXArray
        if config.concatenate && extractedStates.count > 1 {
            // Concatenate along the hidden dimension
            embeddings = concatenated(extractedStates, axis: -1)
        } else if extractedStates.count == 1 {
            embeddings = extractedStates[0]
        } else {
            // Stack along a new dimension
            embeddings = stacked(extractedStates, axis: 1)
        }

        // Normalize if configured
        if config.normalize {
            embeddings = l2Normalize(embeddings)
        }

        // Evaluate to ensure computation is complete
        eval(embeddings)

        return embeddings
    }

    /// Extract embeddings using mflux-compatible format
    /// Returns shape: [batch, seq_len, 15360] (3 layers * 5120 hidden)
    public func extractMfluxEmbeddings(prompt: String) throws -> MLXArray {
        return try extractEmbeddings(prompt: prompt, config: .mfluxDefault)
    }

    /// Extract embeddings with chat template applied
    public func extractChatEmbeddings(
        messages: [[String: String]],
        config: HiddenStatesConfig = .mfluxDefault
    ) throws -> MLXArray {
        let prompt = tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false)
        return try extractEmbeddings(prompt: prompt, config: config)
    }

    /// Extract FLUX.2-compatible embeddings
    /// This method produces embeddings identical to mflux-gradio Python implementation
    /// - Parameters:
    ///   - prompt: User prompt text
    ///   - maxLength: Maximum sequence length (default: 512)
    /// - Returns: Embeddings tensor with shape [1, maxLength, 15360]
    public func extractFluxEmbeddings(
        prompt: String,
        maxLength: Int = FluxConfig.maxSequenceLength
    ) throws -> MLXArray {
        // 1. Build messages with FLUX system message (matching Python exactly)
        let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
        let messages: [[String: String]] = [
            ["role": "system", "content": FluxConfig.systemMessage],
            ["role": "user", "content": cleanedPrompt]
        ]

        // 2. Encode with chat template (addGenerationPrompt=false matches Python)
        var tokenIds = tokenizer.encodeChatMessages(
            messages: messages,
            addGenerationPrompt: false
        )

        FluxDebug.log("FLUX embeddings: encoded \(tokenIds.count) tokens before padding")

        // 3. Truncate if needed
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
            FluxDebug.log("FLUX embeddings: truncated to \(maxLength) tokens")
        }

        // 4. LEFT-pad to fixed length (matching Python mflux-gradio behavior)
        let padTokenId = tokenizer.padToken
        let originalLength = tokenIds.count
        let padCount: Int
        if tokenIds.count < maxLength {
            padCount = maxLength - tokenIds.count
            let padding = Array(repeating: padTokenId, count: padCount)
            tokenIds = padding + tokenIds
        } else {
            padCount = 0
        }

        FluxDebug.log("FLUX embeddings: padded from \(originalLength) to \(tokenIds.count) tokens (pad count: \(padCount))")

        // 5. Create input tensor
        let inputIds = MLXArray(tokenIds).reshaped([1, tokenIds.count])

        // 6. Create attention mask (1 for real tokens, 0 for padding)
        // This matches HuggingFace attention_mask behavior for proper padding handling
        var attentionMaskValues = Array(repeating: Int32(0), count: padCount)
        attentionMaskValues.append(contentsOf: Array(repeating: Int32(1), count: originalLength))
        let attentionMask = MLXArray(attentionMaskValues).reshaped([1, maxLength])

        FluxDebug.log("FLUX embeddings: attention mask created with \(padCount) masked positions")

        // 7. Forward pass with hidden states and attention mask
        let output = model(inputIds, outputHiddenStates: true, attentionMask: attentionMask)

        guard let allHiddenStates = output.hiddenStates else {
            throw EmbeddingError.noHiddenStates
        }

        // 8. Extract hidden states from FLUX layers (10, 20, 30)
        // Note: hidden_states includes embedding layer at index 0, so layer 10 is at index 10
        var extractedStates: [MLXArray] = []
        for layerIdx in FluxConfig.hiddenStateLayers {
            guard layerIdx >= 0 && layerIdx < allHiddenStates.count else {
                throw EmbeddingError.invalidLayerIndex(layerIdx, allHiddenStates.count)
            }
            extractedStates.append(allHiddenStates[layerIdx])
        }

        // 9. Concatenate along hidden dimension: [1, seq, 5120] x 3 -> [1, seq, 15360]
        let embeddings = concatenated(extractedStates, axis: -1)

        // 10. Evaluate to ensure computation is complete
        eval(embeddings)

        FluxDebug.log("FLUX embeddings: shape \(embeddings.shape)")

        return embeddings
    }

    /// Get token IDs for FLUX format (useful for debugging/comparison)
    public func getFluxTokenIds(
        prompt: String,
        maxLength: Int = FluxConfig.maxSequenceLength
    ) -> [Int] {
        let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
        let messages: [[String: String]] = [
            ["role": "system", "content": FluxConfig.systemMessage],
            ["role": "user", "content": cleanedPrompt]
        ]

        var tokenIds = tokenizer.encodeChatMessages(
            messages: messages,
            addGenerationPrompt: false
        )

        // Truncate if needed
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        // LEFT-pad to fixed length (matching Python mflux-gradio behavior)
        let padTokenId = tokenizer.padToken
        if tokenIds.count < maxLength {
            let padCount = maxLength - tokenIds.count
            let padding = Array(repeating: padTokenId, count: padCount)
            tokenIds = padding + tokenIds
        }

        return tokenIds
    }

    // MARK: - Private Helpers

    private func applyPooling(_ hiddenStates: MLXArray, strategy: PoolingStrategy) -> MLXArray {
        switch strategy {
        case .none:
            return hiddenStates

        case .lastToken:
            // Take last token: [batch, seq, hidden] -> [batch, 1, hidden]
            let seqLen = hiddenStates.shape[1]
            return hiddenStates[0..., (seqLen - 1)..<seqLen, 0...]

        case .mean:
            // Average over sequence dimension
            return mean(hiddenStates, axis: 1, keepDims: true)

        case .max:
            // Max over sequence dimension
            return MLX.max(hiddenStates, axis: 1, keepDims: true)

        case .cls:
            // First token (CLS)
            return hiddenStates[0..., 0..<1, 0...]
        }
    }

    private func l2Normalize(_ x: MLXArray) -> MLXArray {
        let norm = sqrt(sum(x * x, axis: -1, keepDims: true))
        return x / (norm + 1e-8)
    }
}

// MARK: - Errors

public enum EmbeddingError: LocalizedError {
    case noHiddenStates
    case invalidLayerIndex(Int, Int)
    case tokenizationFailed

    public var errorDescription: String? {
        switch self {
        case .noHiddenStates:
            return "Model did not return hidden states"
        case .invalidLayerIndex(let idx, let max):
            return "Invalid layer index \(idx), model has \(max) layers"
        case .tokenizationFailed:
            return "Failed to tokenize input"
        }
    }
}

// MARK: - Convenience Extensions

extension EmbeddingExtractor {
    /// Get embedding dimension for a given configuration
    public func embeddingDimension(config: HiddenStatesConfig) -> Int {
        let hiddenSize = model.config.hiddenSize

        if config.concatenate {
            return hiddenSize * config.layerIndices.count
        } else {
            return hiddenSize
        }
    }

    /// Export embeddings to file (binary format)
    public func exportEmbeddings(
        _ embeddings: MLXArray,
        to path: String,
        format: ExportFormat = .binary
    ) throws {
        switch format {
        case .binary:
            // Export as raw float32 binary
            let floats = embeddings.asArray(Float.self)
            var data = Data()
            for f in floats {
                var value = f
                data.append(Data(bytes: &value, count: MemoryLayout<Float>.size))
            }
            try data.write(to: URL(fileURLWithPath: path))

        case .numpy:
            // Export as simple binary (npy not available)
            try exportEmbeddings(embeddings, to: path, format: .binary)

        case .json:
            // Export as JSON (for debugging, not recommended for large tensors)
            let floats = embeddings.asArray(Float.self)
            let json = try JSONEncoder().encode(floats)
            try json.write(to: URL(fileURLWithPath: path))
        }

        FluxDebug.log("Exported embeddings to \(path) (shape: \(embeddings.shape))")
    }
}

public enum ExportFormat {
    case binary
    case numpy
    case json
}
