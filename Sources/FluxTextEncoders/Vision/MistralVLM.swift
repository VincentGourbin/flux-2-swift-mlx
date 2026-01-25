/**
 * MistralVLM.swift
 * Vision-Language Model combining Pixtral vision encoder with Mistral language model
 */

import Foundation
import MLX
import MLXNN

/// Debug flag - uses FluxProfiler.shared.isEnabled or VLM_DEBUG=1 environment variable
private var vlmDebug: Bool {
    FluxProfiler.shared.isEnabled || ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil
}

private func debugPrint(_ message: @autoclosure () -> String) {
    if vlmDebug {
        print(message())
        fflush(stdout)
    }
}

/// Configuration for the full VLM
public struct MistralVLMConfig: Codable, Sendable {
    public let visionConfig: PixtralVisionConfig
    public let projectorConfig: MultiModalProjectorConfig
    public let textConfig: MistralTextConfig
    public let imageTokenIndex: Int
    public let visionFeatureLayer: Int  // -1 means last layer

    public static func load(from path: String) throws -> MistralVLMConfig {
        let configPath = path.hasSuffix("config.json") ? path : "\(path)/config.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]

        // Parse vision config
        let visionJson = json["vision_config"] as? [String: Any] ?? [:]
        let visionConfig = PixtralVisionConfig(
            hiddenSize: visionJson["hidden_size"] as? Int ?? 1024,
            intermediateSize: visionJson["intermediate_size"] as? Int ?? 4096,
            numHiddenLayers: visionJson["num_hidden_layers"] as? Int ?? 24,
            numAttentionHeads: visionJson["num_attention_heads"] as? Int ?? 16,
            headDim: visionJson["head_dim"] as? Int ?? 64,
            patchSize: visionJson["patch_size"] as? Int ?? 14,
            imageSize: visionJson["image_size"] as? Int ?? 1540,
            numChannels: visionJson["num_channels"] as? Int ?? 3,
            hiddenAct: visionJson["hidden_act"] as? String ?? "silu",
            ropeTheta: (visionJson["rope_theta"] as? NSNumber)?.floatValue ?? 10000.0,
            attentionDropout: (visionJson["attention_dropout"] as? NSNumber)?.floatValue ?? 0.0
        )

        // Parse text config
        let textJson = json["text_config"] as? [String: Any] ?? [:]
        let textConfig = MistralTextConfig(
            vocabSize: textJson["vocab_size"] as? Int ?? 131072,
            hiddenSize: textJson["hidden_size"] as? Int ?? 5120,
            intermediateSize: textJson["intermediate_size"] as? Int ?? 32768,
            numHiddenLayers: textJson["num_hidden_layers"] as? Int ?? 40,
            numAttentionHeads: textJson["num_attention_heads"] as? Int ?? 32,
            numKeyValueHeads: textJson["num_key_value_heads"] as? Int ?? 8,
            maxPositionEmbeddings: textJson["max_position_embeddings"] as? Int ?? 131072,
            rmsNormEps: (textJson["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5,
            ropeTheta: (textJson["rope_theta"] as? NSNumber)?.floatValue ?? 1_000_000_000.0,
            tieWordEmbeddings: json["tie_word_embeddings"] as? Bool ?? true,
            headDim: textJson["head_dim"] as? Int ?? 128
        )

        // Parse projector config
        let spatialMergeSize = json["spatial_merge_size"] as? Int ?? 2
        let projectorHiddenAct = json["projector_hidden_act"] as? String ?? "gelu"

        let projectorConfig = MultiModalProjectorConfig(
            visionHiddenSize: visionConfig.hiddenSize,
            textHiddenSize: textConfig.hiddenSize,
            spatialMergeSize: spatialMergeSize,
            projectorHiddenAct: projectorHiddenAct
        )

        let imageTokenIndex = json["image_token_index"] as? Int ?? 10
        let visionFeatureLayer = json["vision_feature_layer"] as? Int ?? -1

        return MistralVLMConfig(
            visionConfig: visionConfig,
            projectorConfig: projectorConfig,
            textConfig: textConfig,
            imageTokenIndex: imageTokenIndex,
            visionFeatureLayer: visionFeatureLayer
        )
    }

    public init(
        visionConfig: PixtralVisionConfig,
        projectorConfig: MultiModalProjectorConfig,
        textConfig: MistralTextConfig,
        imageTokenIndex: Int,
        visionFeatureLayer: Int
    ) {
        self.visionConfig = visionConfig
        self.projectorConfig = projectorConfig
        self.textConfig = textConfig
        self.imageTokenIndex = imageTokenIndex
        self.visionFeatureLayer = visionFeatureLayer
    }
}

// MARK: - VLM Model

/// Full Vision-Language Model
public class MistralVLM: Module {
    public let config: MistralVLMConfig

    @ModuleInfo(key: "vision_tower") var visionTower: VisionModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: MultiModalProjector
    @ModuleInfo(key: "language_model") var languageModel: MistralForCausalLM

    // Vision embedding cache - avoids re-computing vision tower for same image
    // This provides ~40% speedup on VLM inference for repeated images
    private var cachedVisionEmbeddings: MLXArray?
    private var cachedPixelHash: Int?
    private var cachedPatchesH: Int = 0
    private var cachedPatchesW: Int = 0

    public init(config: MistralVLMConfig) {
        self.config = config

        self._visionTower.wrappedValue = VisionModel(config: config.visionConfig)
        self._multiModalProjector.wrappedValue = MultiModalProjector(config: config.projectorConfig)
        self._languageModel.wrappedValue = MistralForCausalLM(config: config.textConfig)
    }

    /// Clear the vision embedding cache
    public func clearVisionCache() {
        cachedVisionEmbeddings = nil
        cachedPixelHash = nil
        cachedPatchesH = 0
        cachedPatchesW = 0
        debugPrint("[Vision] Cache cleared")
    }

    /// Check if vision embeddings are cached for the given pixel values
    public var hasVisionCache: Bool {
        return cachedVisionEmbeddings != nil
    }

    /// Process image through vision encoder and projector
    /// - Parameter pixelValues: Preprocessed image [batch, H, W, 3] (NHWC format)
    /// - Parameter useCache: Whether to use cached embeddings if available (default: true)
    /// - Returns: Image embeddings [batch, num_image_tokens, hidden_size]
    public func encodeImage(_ pixelValues: MLXArray, useCache: Bool = true) -> (embeddings: MLXArray, patchesH: Int, patchesW: Int) {
        // Compute a simple hash of pixel values shape and a sample for cache checking
        // Note: Full data hash would be expensive, so we use shape + sample values
        let pixelHash = computePixelHash(pixelValues)

        // Check if we have cached embeddings for this image
        if useCache, let cached = cachedVisionEmbeddings, cachedPixelHash == pixelHash {
            debugPrint("[Vision] Using cached embeddings (hash: \(pixelHash))")
            return (cached, cachedPatchesH, cachedPatchesW)
        }

        // Debug: Check input pixel values (only compute stats if debugging)
        if vlmDebug {
            print("[Vision] Input pixel values shape: \(pixelValues.shape), dtype: \(pixelValues.dtype)")
            let pixelMean = MLX.mean(pixelValues).item(Float.self)
            let pixelStd = MLX.std(pixelValues).item(Float.self)
            print("[Vision] Pixel values - mean: \(pixelMean), std: \(pixelStd)")
            fflush(stdout)
        }

        // Get vision features
        let visionOutput = visionTower(pixelValues)

        // Debug: Check vision encoder output (only compute stats if debugging)
        if vlmDebug {
            print("[Vision] Vision encoder output shape: \(visionOutput.shape), dtype: \(visionOutput.dtype)")
            let visionMean = MLX.mean(visionOutput).item(Float.self)
            let visionStd = MLX.std(visionOutput).item(Float.self)
            let visionMin = MLX.min(visionOutput).item(Float.self)
            let visionMax = MLX.max(visionOutput).item(Float.self)
            print("[Vision] Vision output - mean: \(visionMean), std: \(visionStd), min: \(visionMin), max: \(visionMax)")
            fflush(stdout)
        }

        // Get patch dimensions (NHWC format)
        let imageH = pixelValues.shape[1]
        let imageW = pixelValues.shape[2]
        let patchSize = config.visionConfig.patchSize
        let patchesH = imageH / patchSize
        let patchesW = imageW / patchSize

        // Project to text space with patch merging
        let imageEmbeddings = multiModalProjector(visionOutput, patchesH: patchesH, patchesW: patchesW)

        // Debug: Check projector output (only compute stats if debugging)
        if vlmDebug {
            print("[Vision] Projector output shape: \(imageEmbeddings.shape), dtype: \(imageEmbeddings.dtype)")
            let projMean = MLX.mean(imageEmbeddings).item(Float.self)
            let projStd = MLX.std(imageEmbeddings).item(Float.self)
            let projMin = MLX.min(imageEmbeddings).item(Float.self)
            let projMax = MLX.max(imageEmbeddings).item(Float.self)
            print("[Vision] Projector output - mean: \(projMean), std: \(projStd), min: \(projMin), max: \(projMax)")
            fflush(stdout)
        }

        // Cache the embeddings
        if useCache {
            cachedVisionEmbeddings = imageEmbeddings
            cachedPixelHash = pixelHash
            cachedPatchesH = patchesH
            cachedPatchesW = patchesW
            debugPrint("[Vision] Cached new embeddings (hash: \(pixelHash))")
        }

        return (imageEmbeddings, patchesH, patchesW)
    }

    /// Compute a simple hash for cache key based on pixel values shape and sample
    private func computePixelHash(_ pixelValues: MLXArray) -> Int {
        // Use shape as primary key (different sizes = different images)
        var hasher = Hasher()
        hasher.combine(pixelValues.shape[0])
        hasher.combine(pixelValues.shape[1])
        hasher.combine(pixelValues.shape[2])
        hasher.combine(pixelValues.shape[3])

        // Sample a few pixels for additional discrimination
        // This avoids computing full array hash which would be expensive
        eval(pixelValues)  // Ensure computed
        let sample = pixelValues[0, 0, 0, 0].item(Float.self)
        hasher.combine(sample.bitPattern)

        return hasher.finalize()
    }

    /// Forward pass for multimodal input
    /// - Parameters:
    ///   - inputIds: Token IDs with image token placeholders
    ///   - pixelValues: Optional preprocessed image
    ///   - cache: KV cache for generation
    /// - Returns: Logits
    public func callAsFunction(
        _ inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        // If no image, use text-only forward
        guard let pixels = pixelValues else {
            return languageModel.forward(inputIds, cache: cache)
        }

        // Encode image
        let (imageEmbeddings, _, _) = encodeImage(pixels)

        // Get text embeddings
        let textEmbeddings = languageModel.model.embed_tokens(inputIds)

        // Debug: Check text embedding stats for non-image tokens (only if debugging)
        if vlmDebug {
            let seqLen = inputIds.shape[1]
            print("[Debug] Input IDs shape: \(inputIds.shape)")
            print("[Debug] First 5 tokens: \(inputIds[0, 0..<min(5, seqLen)].asArray(Int32.self))")

            // Check embedding for token at position 0 (should be BOS)
            let firstTokenEmbed = textEmbeddings[0, 0, 0...]
            let firstMean = MLX.mean(firstTokenEmbed).item(Float.self)
            let firstStd = MLX.std(firstTokenEmbed).item(Float.self)
            print("[Debug] Token 0 (BOS) embedding: mean=\(firstMean), std=\(firstStd)")

            // Check embedding for token at position 1 (should be [INST])
            let secondTokenEmbed = textEmbeddings[0, 1, 0...]
            let secondMean = MLX.mean(secondTokenEmbed).item(Float.self)
            let secondStd = MLX.std(secondTokenEmbed).item(Float.self)
            print("[Debug] Token 1 ([INST]) embedding: mean=\(secondMean), std=\(secondStd)")

            // Check embedding for image token at position 2
            let imgTokenEmbed = textEmbeddings[0, 2, 0...]
            let imgTokenMean = MLX.mean(imgTokenEmbed).item(Float.self)
            let imgTokenStd = MLX.std(imgTokenEmbed).item(Float.self)
            print("[Debug] Token 2 (IMG) embedding: mean=\(imgTokenMean), std=\(imgTokenStd)")

            // Check last non-image token
            let lastIdx = seqLen - 1
            let lastTokenEmbed = textEmbeddings[0, lastIdx, 0...]
            let lastMean = MLX.mean(lastTokenEmbed).item(Float.self)
            let lastStd = MLX.std(lastTokenEmbed).item(Float.self)
            print("[Debug] Token \(lastIdx) (last) embedding: mean=\(lastMean), std=\(lastStd)")
            fflush(stdout)
        }

        // Merge image and text embeddings
        let mergedEmbeddings = mergeEmbeddings(
            textEmbeddings: textEmbeddings,
            imageEmbeddings: imageEmbeddings,
            inputIds: inputIds
        )

        // Forward through language model (skip embedding layer)
        return forwardFromEmbeddings(mergedEmbeddings, cache: cache)
    }

    /// Merge text and image embeddings by replacing image token positions
    /// Delegates to vectorized implementation for efficiency
    private func mergeEmbeddings(
        textEmbeddings: MLXArray,
        imageEmbeddings: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        return mergeEmbeddingsMLX(textEmbeddings, imageEmbeddings, inputIds)
    }

    /// Vectorized embedding merge using mx.where (matches Python mlx-vlm exactly)
    /// This replaces image token embeddings in-place rather than concatenating
    private func mergeEmbeddingsMLX(
        _ textEmbeddings: MLXArray,
        _ imageEmbeddings: MLXArray,
        _ inputIds: MLXArray
    ) -> MLXArray {
        let numImageFeatures = imageEmbeddings.shape[1]

        if vlmDebug {
            print("[Merge] Text embeddings shape: \(textEmbeddings.shape)")
            print("[Merge] Image embeddings shape: \(imageEmbeddings.shape)")
            print("[Merge] Using mx.where approach (matches Python)")

            // Compare scales (expensive - only in debug)
            let textStd = MLX.std(textEmbeddings).item(Float.self)
            let imageStd = MLX.std(imageEmbeddings).item(Float.self)
            let scaleRatio = imageStd / textStd
            print("[Merge] Text std: \(textStd), Image std: \(imageStd), ratio: \(scaleRatio)x")
            fflush(stdout)
        }

        // Use image embeddings directly (experimental scaling removed - not needed after sampling fix)
        let imageFeatsSource = imageEmbeddings

        // Create boolean mask where input_ids == imageTokenIndex
        let imageTokenId = MLXArray(Int32(config.imageTokenIndex))
        let imageMask = inputIds .== imageTokenId  // [batch, seqLen]

        // Debug: Count image tokens (only sync to CPU if debugging)
        if vlmDebug {
            let numImagePositions = MLX.sum(imageMask).item(Int.self)
            print("[Merge] Found \(numImagePositions) image token positions in input_ids")
            print("[Merge] Image features count: \(numImageFeatures)")
            fflush(stdout)

            // Validate counts match - this is CRITICAL
            if numImagePositions != numImageFeatures {
                print("[Merge] FATAL ERROR: Mismatch! \(numImagePositions) positions vs \(numImageFeatures) features")
                print("[Merge] This indicates a bug in getNumImageTokens or projector output!")
                fflush(stdout)
            }
        }

        // Remove batch dim from image embeddings for indexing: [1, N, H] -> [N, H]
        var imageFeats = imageFeatsSource
        if imageFeats.ndim == 3 && imageFeats.shape[0] == 1 {
            imageFeats = imageFeats.squeezed(axis: 0)  // [numImageFeatures, hiddenSize]
        }

        // For single batch, use mask to replace embeddings
        // Get mask for batch 0: [seqLen]
        let batchMask = imageMask[0]

        // Create cumulative sum to get indices into image features
        // cumsum[i] = number of image tokens up to and including position i
        let cumsum = MLX.cumsum(batchMask.asType(.int32), axis: 0)  // [seqLen]

        // feature_indices: at image positions, use cumsum-1; at text positions, use 0
        // This gives the index into imageFeats for each position
        let featureIndices = MLX.where(batchMask, cumsum - 1, MLXArray(0))  // [seqLen]

        // Gather image features for all positions (at non-image positions, gets feature 0 but will be masked out)
        // gatheredFeatures[i] = imageFeats[featureIndices[i]]
        let gatheredFeatures = imageFeats[featureIndices]  // [seqLen, hiddenSize]

        // Expand mask for broadcasting: [seqLen] -> [seqLen, 1]
        let maskExpanded = batchMask.expandedDimensions(axis: -1)

        // Get text embeddings for batch 0: [seqLen, hiddenSize]
        let textEmb = textEmbeddings[0]

        // Use mx.where to blend: at image positions use gatheredFeatures, else use textEmb
        let merged = MLX.where(maskExpanded, gatheredFeatures, textEmb)

        // Add back batch dimension: [seqLen, hiddenSize] -> [1, seqLen, hiddenSize]
        let result = merged.expandedDimensions(axis: 0)

        if vlmDebug {
            print("[Merge] Result shape: \(result.shape)")
            let mergedMean = MLX.mean(result).item(Float.self)
            let mergedStd = MLX.std(result).item(Float.self)
            print("[Merge] Result stats - mean: \(mergedMean), std: \(mergedStd)")
            fflush(stdout)
        }

        return result
    }

    /// Forward from embeddings (skip embedding lookup)
    private func forwardFromEmbeddings(_ hiddenStates: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = hiddenStates

        // Create causal mask
        let seqLen = h.shape[1]
        let mask = createCausalMask(seqLen: seqLen, offset: cache?.first?.length ?? 0)

        // Pass through transformer layers
        for (i, layer) in languageModel.model.layers.enumerated() {
            let layerCache = cache?[i]
            h = layer(h, mask: mask, cache: layerCache)
        }

        // Final norm
        h = languageModel.model.norm(h)

        // LM head
        return languageModel.lm_head(h)
    }

    /// Forward from embeddings with hidden states collection
    /// - Parameters:
    ///   - hiddenStates: Input embeddings [batch, seq, hidden]
    ///   - layerIndices: Which layer indices to collect hidden states from
    /// - Returns: Dictionary mapping layer index to hidden states
    public func forwardWithHiddenStates(
        _ inputEmbeddings: MLXArray,
        layerIndices: [Int]
    ) -> [Int: MLXArray] {
        var h = inputEmbeddings
        var collectedStates: [Int: MLXArray] = [:]

        // Create causal mask
        let seqLen = h.shape[1]
        let mask = createCausalMask(seqLen: seqLen, offset: 0)

        // Convert negative indices to positive
        let numLayers = languageModel.model.layers.count
        let resolvedIndices = Set(layerIndices.map { idx -> Int in
            if idx < 0 { return numLayers + idx }
            return idx
        })

        // Collect embedding layer output (index 0)
        if resolvedIndices.contains(0) {
            collectedStates[0] = h
        }

        // Pass through transformer layers
        for (i, layer) in languageModel.model.layers.enumerated() {
            h = layer(h, mask: mask, cache: nil)

            // Collect after layer i (layer index is i+1 because 0 is embedding)
            let stateIdx = i + 1
            if resolvedIndices.contains(stateIdx) {
                collectedStates[stateIdx] = h
            }
        }

        return collectedStates
    }

    /// Extract FLUX.2-compatible embeddings from image + text
    /// This method produces embeddings that include both image and text features
    /// - Parameters:
    ///   - pixelValues: Preprocessed image [batch, H, W, 3]
    ///   - inputIds: Token IDs with image token placeholders
    /// - Returns: Embeddings tensor with shape [1, seq, 15360]
    public func extractFluxEmbeddingsWithImage(
        pixelValues: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        // 1. Encode image
        let (imageEmbeddings, _, _) = encodeImage(pixelValues)

        // 2. Get text embeddings
        let textEmbeddings = languageModel.model.embed_tokens(inputIds)

        // 3. Merge image and text embeddings
        let mergedEmbeddings = mergeEmbeddings(
            textEmbeddings: textEmbeddings,
            imageEmbeddings: imageEmbeddings,
            inputIds: inputIds
        )

        // 4. Forward through transformer with hidden states collection
        // FLUX.2 uses layers 10, 20, 30
        let layerIndices = [10, 20, 30]
        let hiddenStates = forwardWithHiddenStates(mergedEmbeddings, layerIndices: layerIndices)

        // 5. Concatenate hidden states from specified layers
        var extractedStates: [MLXArray] = []
        for idx in layerIndices {
            guard let layerHidden = hiddenStates[idx] else {
                fatalError("Missing hidden state for layer \(idx)")
            }
            extractedStates.append(layerHidden)
        }

        // 6. Concatenate along hidden dimension: [1, seq, 5120] x 3 -> [1, seq, 15360]
        let embeddings = concatenated(extractedStates, axis: -1)

        // 7. Evaluate to ensure computation is complete
        eval(embeddings)

        if vlmDebug {
            print("[VLM] FLUX I2I embeddings: shape \(embeddings.shape)")
            fflush(stdout)
        }

        return embeddings
    }

    private func createCausalMask(seqLen: Int, offset: Int) -> MLXArray? {
        if seqLen == 1 {
            return nil
        }

        let totalLen = seqLen + offset

        // GPU-based causal mask creation (replaces O(seqLen * totalLen) CPU loop)
        // Row indices: [seqLen, 1] - each row i
        let rowIndices = MLXArray(Array(0..<seqLen).map { Float($0) }).expandedDimensions(axis: 1)
        // Column indices: [1, totalLen] - each column j
        let colIndices = MLXArray(Array(0..<totalLen).map { Float($0) }).expandedDimensions(axis: 0)

        // Causal mask: allow position j if j <= i + offset
        // mask[i,j] = 0 if j <= i + offset, else -inf
        let mask = MLX.where(
            colIndices .<= (rowIndices + Float(offset)),
            MLXArray(Float(0.0)),
            MLXArray(-Float.infinity)
        )

        return mask.reshaped([1, 1, seqLen, totalLen])
    }

    /// Create KV cache for generation
    public func createCache() -> [KVCache] {
        return languageModel.createCache()
    }

    /// Get number of image tokens after processing
    public func getNumImageTokens(imageHeight: Int, imageWidth: Int) -> Int {
        let patchSize = config.visionConfig.patchSize
        let patchesH = imageHeight / patchSize
        let patchesW = imageWidth / patchSize
        let mergeSize = config.projectorConfig.spatialMergeSize

        let mergedH = patchesH / mergeSize
        let mergedW = patchesW / mergeSize

        return mergedH * mergedW
    }
}

// MARK: - Memory Debugging

/// Helper to log memory usage at key points (only when VLM_DEBUG is set)
private func logMemory(_ label: String) {
    guard vlmDebug else { return }
    let snapshot = FluxProfiler.snapshot()
    let mlxMB = Double(snapshot.mlxActive) / (1024 * 1024)
    let procMB = Double(snapshot.processFootprint) / (1024 * 1024)
    print("[VLM-MEM] \(label): MLX=\(String(format: "%.1f", mlxMB))MB, Process=\(String(format: "%.1f", procMB))MB")
    fflush(stdout)
}

// MARK: - Model Loading

extension MistralVLM {
    /// Load VLM from model path
    public static func load(from modelPath: String) throws -> MistralVLM {
        debugPrint("[VLM] Loading from \(modelPath)"); fflush(stdout)
        logMemory("START")

        // Load VLM config
        debugPrint("[VLM] Loading config..."); fflush(stdout)
        let config = try MistralVLMConfig.load(from: modelPath)
        debugPrint("[VLM] Config loaded. Creating model..."); fflush(stdout)
        logMemory("After config load")

        let model = MistralVLM(config: config)
        debugPrint("[VLM] Model created."); fflush(stdout)
        logMemory("After model creation (random weights)")

        // Find and load safetensors files FIRST to check which components are quantized
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        if safetensorFiles.isEmpty {
            throw MistralModelError.noWeightsFound
        }

        // Load weights from safetensors files
        debugPrint("[VLM] Loading weights from \(safetensorFiles.count) safetensor files..."); fflush(stdout)

        var allWeights: [String: MLXArray] = [:]
        for filename in safetensorFiles {
            debugPrint("[VLM] Loading \(filename)..."); fflush(stdout)
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))
            for (key, value) in weights {
                allWeights[key] = value
            }
            debugPrint("[VLM] Loaded \(weights.count) tensors from \(filename)"); fflush(stdout)
            logMemory("After loading \(filename)")
        }
        logMemory("After loading ALL safetensors (weights in dict)")

        // Check which components are actually quantized by looking for .scales keys
        let visionQuantized = allWeights.keys.contains { $0.contains("vision_tower") && $0.hasSuffix(".scales") }
        let projectorQuantized = allWeights.keys.contains { $0.contains("multi_modal_projector") && $0.hasSuffix(".scales") }
        let languageQuantized = allWeights.keys.contains { $0.contains("language_model") && $0.hasSuffix(".scales") }

        debugPrint("[VLM] Quantization detection:"); fflush(stdout)
        debugPrint("[VLM]   Vision tower quantized: \(visionQuantized)"); fflush(stdout)
        debugPrint("[VLM]   Projector quantized: \(projectorQuantized)"); fflush(stdout)
        debugPrint("[VLM]   Language model quantized: \(languageQuantized)"); fflush(stdout)

        // Check for quantization config
        let configPath = "\(modelPath)/config.json"
        let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
        if let json = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let quantJson = json["quantization"] as? [String: Any],
           let groupSize = quantJson["group_size"] as? Int,
           let bits = quantJson["bits"] as? Int {
            debugPrint("[VLM] Quantizing components (groupSize=\(groupSize), bits=\(bits))"); fflush(stdout)

            // Only quantize components that have quantized weights in safetensors
            if visionQuantized {
                debugPrint("[VLM] Quantizing vision tower..."); fflush(stdout)
                quantize(model: model.visionTower, groupSize: groupSize, bits: bits)
            } else {
                debugPrint("[VLM] Skipping vision tower quantization (weights are bfloat16)"); fflush(stdout)
            }

            if projectorQuantized {
                debugPrint("[VLM] Quantizing projector..."); fflush(stdout)
                quantize(model: model.multiModalProjector, groupSize: groupSize, bits: bits)
            } else {
                debugPrint("[VLM] Skipping projector quantization (weights are bfloat16)"); fflush(stdout)
            }

            if languageQuantized {
                debugPrint("[VLM] Quantizing language model..."); fflush(stdout)
                quantize(model: model.languageModel, groupSize: groupSize, bits: bits)
            } else {
                debugPrint("[VLM] Skipping language model quantization (weights are bfloat16)"); fflush(stdout)
            }

            debugPrint("[VLM] Quantization complete."); fflush(stdout)
            logMemory("After quantization (model structure quantized)")
        }

        // Debug: Check vision tower type BEFORE loading weights
        debugPrint("[VLM] BEFORE loadWeights:"); fflush(stdout)
        debugPrint("[VLM]   vision_tower.layers[0].attention.q_proj type: \(type(of: model.visionTower.visionModel.transformer.layers[0].attention.qProj))"); fflush(stdout)

        // Apply weights to model
        debugPrint("[VLM] Applying weights to model..."); fflush(stdout)
        try model.loadWeights(allWeights)
        logMemory("After loadWeights (weights applied)")

        // Debug: Check vision tower type AFTER loading weights
        debugPrint("[VLM] AFTER loadWeights:"); fflush(stdout)
        debugPrint("[VLM]   vision_tower.layers[0].attention.q_proj type: \(type(of: model.visionTower.visionModel.transformer.layers[0].attention.qProj))"); fflush(stdout)

        // Clear the weights dictionary to free memory
        let weightCount = allWeights.count
        allWeights.removeAll()
        logMemory("After clearing allWeights dictionary")

        // Clear MLX cache to release any temporary buffers
        Memory.clearCache()
        logMemory("After GPU cache clear")

        debugPrint("[VLM] VLM loaded successfully with \(weightCount) tensors"); fflush(stdout)

        return model
    }

    private func loadWeights(_ weights: [String: MLXArray]) throws {
        var convertedWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            let swiftKey = convertKeyName(key)
            var tensor = value

            // Convert Conv2d weights to MLX NHWC format if needed
            // MLX expects: [out_channels, H, W, in_channels] = [1024, 14, 14, 3]
            // Possible source formats:
            // - Already correct MLX: [out, H, W, in] = [1024, 14, 14, 3] - no transpose needed
            // - Standard PyTorch NCHW: [out, in, H, W] = [1024, 3, 14, 14] - transpose [0,2,3,1]
            // - Alternative format: [out, H, in, W] = [1024, 14, 3, 14] - transpose [0,1,3,2]
            if swiftKey.contains("patch_conv.weight") && tensor.ndim == 4 {
                let shape = tensor.shape
                debugPrint("[VLM] patch_conv.weight original shape: \(shape)")

                if shape[3] == 3 {
                    // Already in correct MLX format: [1024, 14, 14, 3] (lmstudio MLX models)
                    debugPrint("[VLM] Already in MLX NHWC format, no transpose needed")
                } else if shape[1] == 3 {
                    // Standard PyTorch NCHW: [1024, 3, 14, 14] -> [1024, 14, 14, 3]
                    debugPrint("[VLM] Detected NCHW format, transposing [0,2,3,1]")
                    tensor = tensor.transposed(axes: [0, 2, 3, 1])
                } else if shape[2] == 3 {
                    // Alternative format: [1024, 14, 3, 14] -> [1024, 14, 14, 3]
                    debugPrint("[VLM] Detected [out,H,in,W] format, transposing [0,1,3,2]")
                    tensor = tensor.transposed(axes: [0, 1, 3, 2])
                } else {
                    debugPrint("[VLM] WARNING: Unexpected patch_conv.weight shape \(shape)")
                }

                debugPrint("[VLM] patch_conv.weight final shape: \(tensor.shape)")
            }

            convertedWeights[swiftKey] = tensor
        }

        debugPrint("[VLM] Converting \(convertedWeights.count) weight tensors..."); fflush(stdout)

        // Debug: Print first few vision tower keys from weights
        let visionKeys = convertedWeights.keys.filter { $0.hasPrefix("vision_tower") }.sorted().prefix(10)
        debugPrint("[VLM] Sample vision tower keys from safetensors:"); fflush(stdout)
        for key in visionKeys {
            if let arr = convertedWeights[key] {
                debugPrint("[VLM]   \(key): shape=\(arr.shape), dtype=\(arr.dtype)"); fflush(stdout)
            }
        }

        // Debug: Print model's expected parameter keys
        debugPrint("[VLM] Getting model parameters..."); fflush(stdout)
        let modelParams = self.parameters()
        let flattenedParams = modelParams.flattened()
        let modelVisionKeys = flattenedParams.filter { $0.0.hasPrefix("vision_tower") }.sorted { $0.0 < $1.0 }.prefix(10)
        debugPrint("[VLM] Sample vision tower keys expected by model:"); fflush(stdout)
        for (key, arr) in modelVisionKeys {
            debugPrint("[VLM]   \(key): shape=\(arr.shape), dtype=\(arr.dtype)"); fflush(stdout)
        }

        // Debug: Check embedding keys specifically
        let embedKeys = convertedWeights.keys.filter { $0.contains("embed_tokens") }.sorted()
        debugPrint("[VLM] Embedding keys in safetensors:"); fflush(stdout)
        for key in embedKeys {
            if let arr = convertedWeights[key] {
                let mean = MLX.mean(arr).item(Float.self)
                let std = MLX.std(arr).item(Float.self)
                debugPrint("[VLM]   \(key): shape=\(arr.shape), dtype=\(arr.dtype), mean=\(mean), std=\(std)"); fflush(stdout)
            }
        }
        let modelEmbedKeys = flattenedParams.filter { $0.0.contains("embed_tokens") }.sorted { $0.0 < $1.0 }
        debugPrint("[VLM] Embedding keys expected by model:"); fflush(stdout)
        for (key, arr) in modelEmbedKeys {
            debugPrint("[VLM]   \(key): shape=\(arr.shape), dtype=\(arr.dtype)"); fflush(stdout)
        }

        debugPrint("[VLM] Unflattening parameters..."); fflush(stdout)
        let parameters = ModuleParameters.unflattened(convertedWeights)
        debugPrint("[VLM] Calling update(parameters:)..."); fflush(stdout)
        try update(parameters: parameters, verify: .none)
        debugPrint("[VLM] Calling eval(self)..."); fflush(stdout)
        eval(self)

        // Debug: Verify weights are loaded correctly (only runs when VLM_DEBUG is set)
        if vlmDebug {
            // Check language model embedding table
            let embedWeight = languageModel.model.embed_tokens.weight
            let embedMean = MLX.mean(embedWeight).item(Float.self)
            let embedStd = MLX.std(embedWeight).item(Float.self)
            print("[VLM] embed_tokens.weight - shape: \(embedWeight.shape), mean: \(embedMean), std: \(embedStd)")

            // Check if it's QuantizedEmbedding and verify scales/biases
            if let qemb = languageModel.model.embed_tokens as? QuantizedEmbedding {
                print("[VLM] embed_tokens.scales - shape: \(qemb.scales.shape), mean: \(MLX.mean(qemb.scales).item(Float.self)), std: \(MLX.std(qemb.scales).item(Float.self))")
                if let biases = qemb.biases {
                    print("[VLM] embed_tokens.biases - shape: \(biases.shape), mean: \(MLX.mean(biases).item(Float.self)), std: \(MLX.std(biases).item(Float.self))")
                }
                print("[VLM] embed_tokens.groupSize: \(qemb.groupSize), bits: \(qemb.bits)")

                // Test dequantization of first embedding
                let testIdx = MLXArray([Int32(1)])  // BOS token
                let testEmbed = qemb(testIdx)
                print("[VLM] Test embedding for token 1 (BOS): shape=\(testEmbed.shape), mean=\(MLX.mean(testEmbed).item(Float.self)), std=\(MLX.std(testEmbed).item(Float.self))")
            }

            // Verify vision encoder weights are loaded
            let patchConvWeight = visionTower.visionModel.patchConv.weight
            let patchConvMean = MLX.mean(patchConvWeight).item(Float.self)
            let patchConvStd = MLX.std(patchConvWeight).item(Float.self)
            print("[VLM] patch_conv.weight - shape: \(patchConvWeight.shape), mean: \(patchConvMean), std: \(patchConvStd)")

            let lnPreWeight = visionTower.visionModel.lnPre.weight
            let lnPreMean = MLX.mean(lnPreWeight).item(Float.self)
            print("[VLM] ln_pre.weight - shape: \(lnPreWeight.shape), mean: \(lnPreMean)")

            // Check first transformer layer - verify it's QuantizedLinear
            let qProjLayer = visionTower.visionModel.transformer.layers[0].attention.qProj
            print("[VLM] q_proj layer type: \(type(of: qProjLayer))")
            let layer0AttnQWeight = qProjLayer.weight
            let qMean = MLX.mean(layer0AttnQWeight).item(Float.self)
            print("[VLM] transformer.layers.0.attention.q_proj.weight - shape: \(layer0AttnQWeight.shape), mean: \(qMean)")

            // Compare layer 0 vs layer 23 weights
            let layer0QProj = visionTower.visionModel.transformer.layers[0].attention.qProj
            let layer23QProj = visionTower.visionModel.transformer.layers[23].attention.qProj
            print("[VLM] Layer 0 q_proj.weight: shape=\(layer0QProj.weight.shape), mean=\(MLX.mean(layer0QProj.weight).item(Float.self))")
            print("[VLM] Layer 23 q_proj.weight: shape=\(layer23QProj.weight.shape), mean=\(MLX.mean(layer23QProj.weight).item(Float.self))")
            if let ql0 = layer0QProj as? QuantizedLinear, let ql23 = layer23QProj as? QuantizedLinear {
                print("[VLM] Layer 0 q_proj.scales: shape=\(ql0.scales.shape), mean=\(MLX.mean(ql0.scales).item(Float.self))")
                print("[VLM] Layer 23 q_proj.scales: shape=\(ql23.scales.shape), mean=\(MLX.mean(ql23.scales).item(Float.self))")
                if let biases0 = ql0.biases, let biases23 = ql23.biases {
                    print("[VLM] Layer 0 q_proj.biases: shape=\(biases0.shape), mean=\(MLX.mean(biases0).item(Float.self))")
                    print("[VLM] Layer 23 q_proj.biases: shape=\(biases23.shape), mean=\(MLX.mean(biases23).item(Float.self))")
                }
            }

            // Check projector weights
            let projLinear1Layer = multiModalProjector.linear1
            print("[VLM] projector.linear_1 layer type: \(type(of: projLinear1Layer))")
            let projLinear1Weight = projLinear1Layer.weight
            let projMean = MLX.mean(projLinear1Weight).item(Float.self)
            let projStd = MLX.std(projLinear1Weight).item(Float.self)
            print("[VLM] projector.linear_1.weight - shape: \(projLinear1Weight.shape), mean: \(projMean), std: \(projStd)")

            // Check projector norm weights
            let projNorm = multiModalProjector.norm
            print("[VLM] projector.norm.weight - shape: \(projNorm.weight.shape), mean: \(MLX.mean(projNorm.weight).item(Float.self))")

            // Check patch merger
            let mergerWeight = multiModalProjector.patchMerger.mergingLayer.weight
            print("[VLM] projector.patch_merger.merging_layer.weight - shape: \(mergerWeight.shape), mean: \(MLX.mean(mergerWeight).item(Float.self))")

            fflush(stdout)
        }

        debugPrint("[VLM] Weights applied successfully")
    }

    private func convertKeyName(_ key: String) -> String {
        // VLM weight key mapping:
        // Safetensor keys use "vision_tower.*" but our model expects "vision_tower.vision_model.*"
        // because VisionModel has a nested visionModel property (VisionEncoder)
        //
        // Mapping:
        // - vision_tower.* -> vision_tower.vision_model.* (add .vision_model.)
        // - multi_modal_projector.* -> multi_modal_projector.* (no change)
        // - language_model.model.* -> language_model.model.* (no change)
        // - language_model.lm_head.* -> language_model.lm_head.* (no change)

        // Add .vision_model. to vision tower keys (if not already present)
        if key.hasPrefix("vision_tower.") && !key.hasPrefix("vision_tower.vision_model.") {
            return key.replacingOccurrences(of: "vision_tower.", with: "vision_tower.vision_model.")
        }

        return key
    }
}
