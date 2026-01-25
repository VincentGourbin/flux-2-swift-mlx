/**
 * KleinConfig.swift
 * Configuration for FLUX.2 Klein embedding extraction
 *
 * FLUX.2 Klein uses Qwen3 as text encoder instead of Mistral
 */

import Foundation

// MARK: - Klein Variant

/// FLUX.2 Klein model variants
public enum KleinVariant: String, CaseIterable, Sendable {
    /// Klein 4B - uses Qwen3-4B, Apache 2.0 license
    case klein4B = "4b"
    
    /// Klein 9B - uses Qwen3-8B, non-commercial license
    case klein9B = "9b"
    
    public var displayName: String {
        switch self {
        case .klein4B: return "Klein 4B"
        case .klein9B: return "Klein 9B"
        }
    }
    
    /// Hidden state layers to extract (different from Mistral's [10, 20, 30])
    public var hiddenStateLayers: [Int] {
        // Klein uses layers 9, 18, 27 (0-indexed after embedding)
        return [9, 18, 27]
    }
    
    /// Maximum sequence length for padding
    public var maxSequenceLength: Int {
        return 512
    }
    
    /// Output dimension after concatenating hidden states
    /// Klein 4B: 3 × 2560 = 7,680
    /// Klein 9B: 3 × 4096 = 12,288
    public var outputDimension: Int {
        switch self {
        case .klein4B: return 7_680    // 3 × 2560
        case .klein9B: return 12_288   // 3 × 4096
        }
    }
    
    /// Qwen3 model hidden size
    public var hiddenSize: Int {
        switch self {
        case .klein4B: return 2560
        case .klein9B: return 4096
        }
    }
    
    /// Recommended Qwen3 model ID for this Klein variant
    public var qwen3ModelId: String {
        switch self {
        case .klein4B: return "lmstudio-community/Qwen3-4B-MLX-8bit"
        case .klein9B: return "lmstudio-community/Qwen3-8B-MLX-8bit"
        }
    }
    
    /// Alternative 4-bit quantized Qwen3 model ID
    public var qwen3ModelId4bit: String {
        switch self {
        case .klein4B: return "lmstudio-community/Qwen3-4B-MLX-4bit"
        case .klein9B: return "lmstudio-community/Qwen3-8B-MLX-4bit"
        }
    }
    
    /// Qwen3 configuration for this variant
    public var qwen3Config: Qwen3TextConfig {
        switch self {
        case .klein4B: return .qwen3_4B
        case .klein9B: return .qwen3_8B
        }
    }
}

// MARK: - Klein Configuration

/// Configuration constants for FLUX.2 Klein embeddings
public enum KleinConfig {
    
    /// System message for Klein embeddings (same as FLUX.2)
    /// Used for extracting embeddings from image descriptions
    public static let systemMessage = """
You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.
"""
    
    /// System message for Text-to-Image prompt upsampling
    public static let systemMessageUpsamplingT2I = """
You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.

Guidelines:
1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.

Output only the revised prompt and nothing else.
"""
    
    /// System message for Image-to-Image editing
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
    
    /// Klein operation modes
    public enum Mode {
        /// Extract embeddings for image generation conditioning
        case embeddings
        /// Upsample/enhance text-to-image prompts
        case upsamplingT2I
        /// Convert image editing requests to instructions
        case upsamplingI2I
    }
    
    /// Get the system message for a given Klein mode
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
    
    /// Build chat messages for a given Klein mode (Qwen3 format)
    /// Note: Qwen3 uses different chat template than Mistral
    /// - Parameters:
    ///   - prompt: User prompt
    ///   - mode: Klein operation mode
    /// - Returns: Messages array ready for chat template
    public static func buildMessages(prompt: String, mode: Mode) -> [[String: String]] {
        return [
            ["role": "system", "content": systemMessage(for: mode)],
            ["role": "user", "content": prompt]
        ]
    }
}

// MARK: - Hidden States Config for Klein

extension HiddenStatesConfig {
    /// Klein 4B configuration: layers [9, 18, 27], hidden_size=2560 -> 7680 dims
    public static let klein4B = HiddenStatesConfig(
        layerIndices: [9, 18, 27],
        concatenate: true,
        normalize: false,
        pooling: .none
    )
    
    /// Klein 9B configuration: layers [9, 18, 27], hidden_size=4096 -> 12288 dims
    public static let klein9B = HiddenStatesConfig(
        layerIndices: [9, 18, 27],
        concatenate: true,
        normalize: false,
        pooling: .none
    )
    
    /// Get Klein config for a specific variant
    public static func klein(_ variant: KleinVariant) -> HiddenStatesConfig {
        switch variant {
        case .klein4B: return .klein4B
        case .klein9B: return .klein9B
        }
    }
}
