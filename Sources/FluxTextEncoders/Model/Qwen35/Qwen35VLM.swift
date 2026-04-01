/**
 * Qwen35VLM.swift
 * Qwen3.5 Vision-Language Model combining vision encoder + language model
 *
 * Takes an image + text prompt, produces text output.
 * Used as a standalone VLM service (image analysis, captioning).
 */

import Foundation
import MLX
import MLXNN
import Tokenizers
import CoreGraphics

// MARK: - VLM

public class Qwen35VLM {
    public let config: Qwen35Config
    public let languageModel: Qwen35ForConditionalGeneration
    public let visionEncoder: Qwen35VisionEncoder
    public let imageProcessor: Qwen35ImageProcessor
    public let tokenizer: Tokenizer

    public init(
        config: Qwen35Config,
        languageModel: Qwen35ForConditionalGeneration,
        visionEncoder: Qwen35VisionEncoder,
        imageProcessor: Qwen35ImageProcessor,
        tokenizer: Tokenizer
    ) {
        self.config = config
        self.languageModel = languageModel
        self.visionEncoder = visionEncoder
        self.imageProcessor = imageProcessor
        self.tokenizer = tokenizer
    }

    /// Encode a single image to vision embeddings
    public func encodeImage(_ image: CGImage) -> (MLXArray, Int) {
        let pixelValues = imageProcessor.preprocess(image)
        let embeddings = visionEncoder(pixelValues)
        eval(embeddings)
        return (embeddings, embeddings.dim(1))
    }

    /// Encode multiple images to vision embeddings
    public func encodeImages(_ images: [CGImage]) -> [(embeddings: MLXArray, numTokens: Int)] {
        return images.map { encodeImage($0) }
    }

    /// Build input embeddings with vision features merged in
    /// Replaces image token positions with vision embeddings (supports multiple images)
    public func buildInputEmbeddings(
        inputIds: MLXArray,
        imageEmbeddingsList: [(embeddings: MLXArray, numTokens: Int)]
    ) -> MLXArray {
        let textEmbed = languageModel.model.embed_tokens(inputIds)

        if imageEmbeddingsList.isEmpty {
            return textEmbed
        }

        // Concatenate all image embeddings into one flat sequence
        let allImgEmbeds = concatenated(imageEmbeddingsList.map { $0.embeddings }, axis: 1)  // [1, totalTokens, 2560]
        let totalVisionTokens = allImgEmbeds.dim(1)

        // Find image token positions
        let imageTokenId = Int32(config.imageTokenId)
        let seqLen = inputIds.dim(1)
        let ids = inputIds.squeezed(axis: 0)

        // Build merged sequence: replace image_pad tokens with vision embeddings
        var parts: [MLXArray] = []
        var imgIdx = 0
        var rangeStart = 0

        for pos in 0..<seqLen {
            let tokenId = ids[pos].item(Int32.self)
            if tokenId == imageTokenId && imgIdx < totalVisionTokens {
                if rangeStart < pos {
                    parts.append(textEmbed[0..., rangeStart..<pos, 0...])
                }
                parts.append(allImgEmbeds[0..., imgIdx..<(imgIdx + 1), 0...])
                imgIdx += 1
                rangeStart = pos + 1
            }
        }

        if rangeStart < seqLen {
            parts.append(textEmbed[0..., rangeStart..<seqLen, 0...])
        }

        return parts.isEmpty ? textEmbed : concatenated(parts, axis: 1)
    }

    /// Generate text from image + prompt
    /// - Parameters:
    ///   - image: Optional CGImage to analyze
    ///   - prompt: User prompt text
    ///   - systemPrompt: Optional system prompt (default: "You are a helpful assistant.")
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - topP: Top-p sampling
    ///   - onToken: Streaming callback
    public func generate(
        image: CGImage?,
        prompt: String,
        systemPrompt: String? = nil,
        enableThinking: Bool = true,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let images: [CGImage] = image.map { [$0] } ?? []
        return try generateMultiImage(
            images: images, prompt: prompt, systemPrompt: systemPrompt,
            enableThinking: enableThinking,
            maxTokens: maxTokens, temperature: temperature, topP: topP, onToken: onToken
        )
    }

    /// Generate text from multiple images + prompt
    public func generateMultiImage(
        images: [CGImage],
        prompt: String,
        systemPrompt: String? = nil,
        enableThinking: Bool = true,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()

        // Encode all images
        let imageResults = encodeImages(images)
        let tokenCounts = imageResults.map { $0.numTokens }

        // Build token sequence with N vision blocks
        let tokenIds = buildTokenSequence(prompt: prompt, systemPrompt: systemPrompt, enableThinking: enableThinking, imageTokenCounts: tokenCounts)
        var inputIds = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, tokenIds.count])

        // Build merged embeddings (text + vision)
        let mergedEmbed = buildInputEmbeddings(inputIds: inputIds, imageEmbeddingsList: imageResults)

        // Create cache
        let cache = languageModel.createCache()

        // Prefill with merged embeddings
        var logits = languageModel.forward(inputIds, cache: cache, inputEmbeddings: mergedEmbed)
        eval(logits)

        // Generation loop
        var generatedTokens: [Int] = []
        var pendingTokens: [Int] = []
        let eosTokenId = config.textConfig.eosTokenId

        for i in 0..<maxTokens {
            // Sample
            let nextTokenArray = sampleToken(logits: logits, temperature: temperature, topP: topP,
                                              generatedTokens: generatedTokens)
            MLX.asyncEval(nextTokenArray)
            let nextToken = Int(nextTokenArray.item(Int32.self))

            if nextToken == eosTokenId { break }
            generatedTokens.append(nextToken)

            // Stream
            if let callback = onToken {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= 10 {
                    if !callback(tokenizer.decode(tokens: pendingTokens)) { break }
                    pendingTokens.removeAll()
                }
            }

            // Next forward
            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = languageModel.forward(inputIds, cache: cache)

            if (i + 1) % 20 == 0 { Memory.clearCache() }
        }

        // Flush
        if let callback = onToken, !pendingTokens.isEmpty {
            _ = callback(tokenizer.decode(tokens: pendingTokens))
        }

        let totalTime = Date().timeIntervalSince(startTime)
        let rawText = tokenizer.decode(tokens: generatedTokens)
        var outputText = rawText

        // Strip thinking tags — but preserve JSON content
        // If the response contains a JSON object, extract it regardless of position
        if let jsonStart = rawText.lastIndex(of: "{"), let jsonEnd = rawText.lastIndex(of: "}"), jsonStart < jsonEnd {
            // There's a JSON object — check if it's a score JSON
            let jsonCandidate = String(rawText[jsonStart...jsonEnd])
            if jsonCandidate.contains("score") {
                // It's a score JSON — use it directly (don't strip it away)
                outputText = jsonCandidate
            } else if let thinkEnd = rawText.range(of: "</think>") {
                outputText = String(rawText[thinkEnd.upperBound...])
            }
        } else if let thinkEnd = rawText.range(of: "</think>") {
            // No JSON found — strip thinking as before
            outputText = String(rawText[thinkEnd.upperBound...])
        }
        // Strip end-of-turn token
        outputText = outputText.replacingOccurrences(of: "<|im_end|>", with: "")
        outputText = outputText.trimmingCharacters(in: .whitespacesAndNewlines)

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: tokenIds.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: Double(generatedTokens.count) / totalTime
        )
    }

    // MARK: - Private

    /// Build token sequence for N images + text prompt
    private func buildTokenSequence(prompt: String, systemPrompt: String? = nil, enableThinking: Bool = true, imageTokenCounts: [Int]) -> [Int] {
        let sysMsg = systemPrompt ?? "You are a helpful assistant."
        var formatted = "<|im_start|>system\n\(sysMsg)<|im_end|>\n"
        formatted += "<|im_start|>user\n"

        // Insert one vision block per image
        for count in imageTokenCounts {
            formatted += "<|vision_start|>"
            formatted += String(repeating: "<|image_pad|>", count: count)
            formatted += "<|vision_end|>"
        }

        formatted += prompt
        formatted += "<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"

        // Qwen3.5 thinking control via chat template (NOT /no_think which is Qwen3 only)
        // enable_thinking=false: add empty <think></think> to skip reasoning
        // enable_thinking=true: add opening <think> to let model reason
        if !enableThinking {
            formatted += "<think>\n\n</think>\n\n"
        } else {
            formatted += "<think>\n"
        }

        return tokenizer.encode(text: formatted)
    }

    private func sampleToken(
        logits: MLXArray, temperature: Float, topP: Float,
        generatedTokens: [Int]
    ) -> MLXArray {
        var lastLogits = logits[0, -1]

        // Repetition penalty
        if !generatedTokens.isEmpty {
            let recent = Set(generatedTokens.suffix(20))
            var arr = lastLogits.asArray(Float.self)
            for id in recent where id >= 0 && id < arr.count {
                arr[id] = arr[id] > 0 ? arr[id] / 1.1 : arr[id] * 1.1
            }
            lastLogits = MLXArray(arr)
        }

        if temperature == 0 { return argMax(lastLogits) }

        let probs = softmax(lastLogits / temperature, axis: -1)
        let sortedIndices = argSort(-probs, axis: -1)
        let sortedProbs = MLX.take(probs, sortedIndices, axis: -1)
        let cumProbs = cumsum(sortedProbs, axis: -1)
        let topProbs = MLX.where(cumProbs .> (1 - topP), sortedProbs, MLX.zeros(like: sortedProbs))
        let sortedToken = MLXRandom.categorical(MLX.log(topProbs + 1e-10))
        return sortedIndices[sortedToken]
    }
}

// MARK: - Loading

extension Qwen35VLM {
    /// Load complete VLM from a model path
    public static func load(from modelPath: String) async throws -> Qwen35VLM {
        let config = try Qwen35Config.load(from: "\(modelPath)/config.json")

        // Load language model (skips vision weights)
        FluxDebug.log("Loading Qwen3.5 language model...")
        let langModel = try Qwen35ForConditionalGeneration.load(from: modelPath)

        // Load vision encoder weights
        FluxDebug.log("Loading Qwen3.5 vision encoder...")
        let visionEncoder = Qwen35VisionEncoder(config: config.visionConfig)

        // NOTE: Vision encoder weights are NOT quantized in MLX VLM models
        // Only the language model is quantized. Do NOT call quantize() on visionEncoder.

        // Load vision weights
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        var visionWeights: [String: MLXArray] = [:]
        for filename in safetensorFiles {
            let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/\(filename)"))
            for (key, value) in weights {
                if key.hasPrefix("vision_tower.") {
                    // Remap: "vision_tower.X" → direct module path
                    let mappedKey = key
                    visionWeights[mappedKey] = value
                }
            }
        }

        if !visionWeights.isEmpty {
            FluxDebug.log("Applying \(visionWeights.count) vision weights...")
            // Remap keys: "vision_tower.blocks.0..." → "blocks.0..."
            var remapped: [String: MLXArray] = [:]
            for (key, value) in visionWeights {
                let stripped = key.hasPrefix("vision_tower.") ? String(key.dropFirst("vision_tower.".count)) : key
                remapped[stripped] = value
            }

            // Handle pos_embed specially — it's a raw tensor, not a Module parameter
            // In quantized models, it may be stored as "pos_embed.weight" instead of "pos_embed"
            let posEmbed = remapped.removeValue(forKey: "pos_embed")
                ?? remapped.removeValue(forKey: "pos_embed.weight")
            if let posEmbed = posEmbed {
                visionEncoder.posEmbedStorage = posEmbed
            } else {
                print("[Qwen3.5] WARNING: pos_embed not found in vision weights")
            }

            // Handle class_token if present
            remapped.removeValue(forKey: "class_token")

            // patch_embed.proj is now a native Conv3d — weights [1024, 2, 16, 16, 3] load directly
            // No reshape needed

            let visionParams = ModuleParameters.unflattened(remapped)
            do {
                try visionEncoder.update(parameters: visionParams, verify: .noUnusedKeys)
            } catch {
                FluxDebug.log("[Qwen3.5] Vision weight loading warning: \(error)")
                try visionEncoder.update(parameters: visionParams, verify: .none)
            }
            eval(visionEncoder)
        }

        // Load tokenizer
        FluxDebug.log("Loading tokenizer...")
        let modelFolderURL = URL(fileURLWithPath: modelPath)
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolderURL)

        // Create image processor
        let imageProcessor = Qwen35ImageProcessor(
            patchSize: config.visionConfig.patchSize,
            spatialMergeSize: config.visionConfig.spatialMergeSize
        )

        FluxDebug.log("Qwen3.5 VLM loaded successfully")

        return Qwen35VLM(
            config: config,
            languageModel: langModel,
            visionEncoder: visionEncoder,
            imageProcessor: imageProcessor,
            tokenizer: tokenizer
        )
    }
}

private struct Qwen35VisionQuantConfig: Decodable {
    let quantization: QuantDetail?
    struct QuantDetail: Decodable {
        let groupSize: Int
        let bits: Int
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }
}
