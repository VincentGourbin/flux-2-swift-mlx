// LoRATrainer.swift - Main LoRA training loop
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import ImageIO
import UniformTypeIdentifiers

/// Main trainer class for LoRA fine-tuning
public final class LoRATrainer: @unchecked Sendable {
    
    // MARK: - Properties
    
    /// Training configuration
    public let config: LoRATrainingConfig
    
    /// Target model type
    public let modelType: Flux2Model
    
    /// Training dataset
    private var dataset: TrainingDataset?

    /// Validation dataset (for validation loss computation)
    private var validationDataset: TrainingDataset?

    /// Latent cache
    private var latentCache: LatentCache?

    /// Validation latent cache
    private var validationLatentCache: LatentCache?

    /// Text embedding cache
    private var textEmbeddingCache: TextEmbeddingCache?

    /// Validation text embedding cache
    private var validationTextEmbeddingCache: TextEmbeddingCache?
    
    /// Reference to transformer with injected LoRA (for saving)
    private weak var loraTransformer: Flux2Transformer2DModel?
    
    /// Checkpoint manager
    private var checkpointManager: CheckpointManager?
    
    /// Training state
    private var state: TrainingState?
    
    /// Learning rate scheduler
    private var lrScheduler: LearningRateScheduler?

    /// AdamW optimizer for LoRA parameters
    private var optimizer: AdamW?

    /// Event handler
    private weak var eventHandler: TrainingEventHandler?
    
    /// Progress callback
    private var progressCallback: TrainingProgressCallback?
    
    /// Whether training is running
    private var isRunning: Bool = false
    
    /// Whether training should stop
    private var shouldStop: Bool = false

    /// Early stopping state (loss plateau detection)
    private var bestLoss: Float = Float.infinity
    private var epochsWithoutImprovement: Int = 0
    private var earlyStopReason: String?

    /// Overfitting detection state (train/val gap)
    private var lastValGap: Float = 0
    private var consecutiveGapIncreases: Int = 0
    private var bestValGap: Float = Float.infinity

    /// Val loss stagnation detection state (per epoch)
    private var bestValLossThisEpoch: Float = Float.infinity
    private var bestValLossPreviousEpoch: Float = Float.infinity
    private var consecutiveValStagnationEpochs: Int = 0

    /// Cached valueAndGrad function to avoid reconstructing gradient graph per step
    /// Type: (Model, [MLXArray]) -> ([MLXArray], ModuleParameters)
    private var cachedLossAndGrad: ((Flux2Transformer2DModel, [MLXArray]) -> ([MLXArray], ModuleParameters))?

    /// VAE reference for validation image generation (kept even when latents are cached)
    private var validationVAE: AutoencoderKLFlux2?

    /// Text encoder reference for validation image generation
    private var validationTextEncoder: ((String) async throws -> MLXArray)?

    /// EMA manager for weight averaging (optional, based on config)
    private var emaManager: EMAManager?

    /// Pre-computed position IDs per resolution bucket (key: "WxH")
    private var positionIdCache: [String: (txtIds: MLXArray, imgIds: MLXArray)] = [:]

    // MARK: - Initialization
    
    /// Initialize LoRA trainer
    /// - Parameters:
    ///   - config: Training configuration
    ///   - modelType: Target model type
    public init(config: LoRATrainingConfig, modelType: Flux2Model) {
        self.config = config
        self.modelType = modelType
    }
    
    /// Set event handler
    public func setEventHandler(_ handler: TrainingEventHandler) {
        self.eventHandler = handler
    }
    
    /// Set progress callback
    public func setProgressCallback(_ callback: @escaping TrainingProgressCallback) {
        self.progressCallback = callback
    }
    
    // MARK: - Training Setup
    
    /// Prepare for training
    public func prepare() async throws {
        Flux2Debug.log("[LoRATrainer] Preparing for training...")
        
        // Validate configuration
        try config.validate()
        
        // Initialize dataset
        dataset = try TrainingDataset(config: config)
        guard let dataset = dataset else {
            throw LoRATrainerError.datasetLoadFailed
        }
        
        let validation = dataset.validate()
        if !validation.isValid {
            throw LoRATrainerError.invalidDataset(validation.errors.joined(separator: ", "))
        }
        
        Flux2Debug.log("[LoRATrainer] Dataset loaded: \(dataset.count) samples")

        // Initialize validation dataset if provided
        if let valPath = config.validationDatasetPath {
            var valConfig = config
            valConfig.datasetPath = valPath
            validationDataset = try TrainingDataset(config: valConfig)
            if let valDataset = validationDataset {
                let valValidation = valDataset.validate()
                if !valValidation.isValid {
                    Flux2Debug.log("[LoRATrainer] Warning: Validation dataset invalid, skipping: \(valValidation.errors.joined(separator: ", "))")
                    validationDataset = nil
                } else {
                    Flux2Debug.log("[LoRATrainer] Validation dataset loaded: \(valDataset.count) samples")
                }
            }
        }

        // Initialize caches
        if config.cacheLatents {
            latentCache = LatentCache(config: config)
            // Also create cache for validation dataset
            if validationDataset != nil, let valPath = config.validationDatasetPath {
                var valConfig = config
                valConfig.datasetPath = valPath
                validationLatentCache = LatentCache(config: valConfig)
            }
        }

        // Always initialize text embedding cache (required for pre-caching optimization)
        // This dramatically reduces memory usage by caching embeddings before training
        let textCacheDir = config.datasetPath.appendingPathComponent(".text_cache")
        textEmbeddingCache = TextEmbeddingCache(cacheDirectory: textCacheDir)
        // Also create cache for validation dataset
        if validationDataset != nil, let valPath = config.validationDatasetPath {
            let valTextCacheDir = valPath.appendingPathComponent(".text_cache")
            validationTextEmbeddingCache = TextEmbeddingCache(cacheDirectory: valTextCacheDir)
        }

        // Initialize checkpoint manager
        let outputDir = config.outputPath.deletingLastPathComponent()
        checkpointManager = CheckpointManager(
            outputDirectory: outputDir,
            maxCheckpoints: config.keepOnlyLastNCheckpoints
        )
        
        // Calculate total steps
        let stepsPerEpoch = dataset.batchesPerEpoch
        let totalSteps = config.maxSteps ?? (stepsPerEpoch * config.epochs)
        
        // Initialize training state
        state = TrainingState(
            totalSteps: totalSteps,
            totalEpochs: config.epochs
        )
        
        // Initialize learning rate scheduler
        lrScheduler = LRSchedulerFactory.create(
            type: config.lrScheduler,
            baseLR: config.learningRate,
            warmupSteps: config.warmupSteps,
            totalSteps: totalSteps
        )
        
        Flux2Debug.log("[LoRATrainer] Preparation complete")
        Flux2Debug.log("  Total steps: \(totalSteps)")
        Flux2Debug.log("  Steps per epoch: \(stepsPerEpoch)")
        
        // Memory estimation
        let estimatedMemory = config.estimateMemoryGB(for: modelType)
        Flux2Debug.log("  Estimated memory: \(String(format: "%.1f", estimatedMemory)) GB")
    }
    
    // MARK: - Pre-caching
    
    /// Pre-cache latents using VAE
    public func preCacheLatents(vae: AutoencoderKLFlux2) async throws {
        guard let dataset = dataset, let cache = latentCache else {
            throw LoRATrainerError.notPrepared
        }

        Flux2Debug.log("[LoRATrainer] Pre-caching training latents...")

        try await cache.preEncodeDataset(dataset, vae: vae) { current, total in
            let progress = Float(current) / Float(total) * 100
            Flux2Debug.log("  Pre-caching: \(current)/\(total) (\(Int(progress))%)")
        }

        let stats = cache.getStatistics()
        Flux2Debug.log("[LoRATrainer] Training latent caching complete:")
        Flux2Debug.log(stats.summary)

        // Also cache validation latents if validation dataset exists
        if let valDataset = validationDataset, let valCache = validationLatentCache {
            Flux2Debug.log("[LoRATrainer] Pre-caching validation latents...")

            try await valCache.preEncodeDataset(valDataset, vae: vae) { current, total in
                let progress = Float(current) / Float(total) * 100
                Flux2Debug.log("  Pre-caching validation: \(current)/\(total) (\(Int(progress))%)")
            }

            let valStats = valCache.getStatistics()
            Flux2Debug.log("[LoRATrainer] Validation latent caching complete:")
            Flux2Debug.log(valStats.summary)
        }
    }
    
    /// Pre-cache all text embeddings before training
    /// This caches: all training captions, validation captions, empty caption (for dropout), and validation prompt
    public func preCacheTextEmbeddings(
        textEncoder: @escaping (String) async throws -> MLXArray
    ) async throws {
        guard let dataset = dataset, let cache = textEmbeddingCache else {
            throw LoRATrainerError.notPrepared
        }
        
        Flux2Debug.log("[LoRATrainer] Pre-caching text embeddings...")
        
        // Collect all unique captions to cache
        var captionsToCache: Set<String> = []
        
        // 1. Training dataset captions
        for caption in dataset.allCaptions {
            captionsToCache.insert(caption)
        }
        
        // 2. Validation dataset captions
        if let valDataset = validationDataset {
            for caption in valDataset.allCaptions {
                captionsToCache.insert(caption)
            }
        }
        
        // 3. Empty caption for caption dropout
        captionsToCache.insert("")
        
        // 4. Validation prompt
        if let valPrompt = config.validationPrompt {
            captionsToCache.insert(valPrompt)
        }
        
        let total = captionsToCache.count
        var cached = 0
        
        Flux2Debug.log("[LoRATrainer] Caching \(total) unique text embeddings...")
        
        for caption in captionsToCache {
            // Skip if already cached
            if cache.isCached(caption: caption) {
                cached += 1
                continue
            }
            
            // Encode and cache
            let embedding = try await textEncoder(caption)
            // Squeeze batch dimension if present: [1, seq, dim] -> [seq, dim]
            let squeezedEmbedding = embedding.shape.count > 2
                ? embedding.squeezed(axis: 0)
                : embedding
            eval(squeezedEmbedding)
            
            try cache.saveEmbeddings(
                pooled: MLXArray.zeros([1]),  // Placeholder for Flux2
                hidden: squeezedEmbedding,
                for: caption
            )
            
            cached += 1
            
            if cached % 10 == 0 || cached == total {
                let progress = Float(cached) / Float(total) * 100
                Flux2Debug.log("  Text embeddings: \(cached)/\(total) (\(Int(progress))%)")
            }
        }
        
        // Also cache to validation text embedding cache if separate
        if let valCache = validationTextEmbeddingCache, valCache !== cache {
            Flux2Debug.log("[LoRATrainer] Copying embeddings to validation cache...")
            if let valDataset = validationDataset {
                for caption in valDataset.allCaptions {
                    if !valCache.isCached(caption: caption),
                       let embeddings = try cache.getEmbeddings(for: caption) {
                        try valCache.saveEmbeddings(
                            pooled: embeddings.pooled,
                            hidden: embeddings.hidden,
                            for: caption
                        )
                    }
                }
            }
            // Also cache validation prompt and empty caption
            if let valPrompt = config.validationPrompt,
               !valCache.isCached(caption: valPrompt),
               let embeddings = try cache.getEmbeddings(for: valPrompt) {
                try valCache.saveEmbeddings(
                    pooled: embeddings.pooled,
                    hidden: embeddings.hidden,
                    for: valPrompt
                )
            }
            if !valCache.isCached(caption: ""),
               let embeddings = try cache.getEmbeddings(for: "") {
                try valCache.saveEmbeddings(
                    pooled: embeddings.pooled,
                    hidden: embeddings.hidden,
                    for: ""
                )
            }
        }
        
        MLX.Memory.clearCache()
        Flux2Debug.log("[LoRATrainer] Text embedding caching complete: \(cache.count) entries in memory")
    }
    
    // MARK: - Main Training Loop
    
    /// Run the training loop
    /// - Parameters:
    ///   - transformer: The transformer model (will have LoRA injected)
    ///   - vae: VAE encoder (if not using cached latents)
    ///   - textEncoder: Function to encode text prompts to hidden states
    public func train(
        transformer: Flux2Transformer2DModel,
        vae: AutoencoderKLFlux2?,
        textEncoder: @escaping (String) async throws -> MLXArray
    ) async throws {
        guard let dataset = dataset,
              var state = state,
              let lrScheduler = lrScheduler,
              let _ = checkpointManager else {
            throw LoRATrainerError.notPrepared
        }

        isRunning = true
        shouldStop = false

        // Store references for validation image generation
        self.validationVAE = vae
        self.validationTextEncoder = textEncoder

        // Pre-cache all text embeddings BEFORE training starts
        // This avoids running the text encoder during training, saving ~4-6 GB VRAM
        try await preCacheTextEmbeddings(textEncoder: textEncoder)
        
        // CRITICAL: Release text encoder reference to free ~4-6 GB VRAM
        // generateValidationImage now uses cached embeddings, so text encoder is no longer needed
        self.validationTextEncoder = nil
        Flux2Debug.log("[LoRATrainer] Text encoder released, using cached embeddings only")
        MLX.Memory.clearCache()

        // Inject LoRA directly into transformer (replaces Linear with LoRAInjectedLinear)
        // This is the correct approach - LoRA becomes part of the forward pass
        transformer.applyLoRA(
            rank: config.rank,
            alpha: config.alpha,
            targetBlocks: config.targetLayers.toTargetBlocks()
        )
        loraTransformer = transformer

        Flux2Debug.log("[LoRATrainer] Injected LoRA into transformer with \(transformer.loraParameterCount) trainable parameters")

        // CRITICAL: Freeze base model weights so valueAndGrad only computes gradients for LoRA params
        // This dramatically speeds up training by not computing gradients for frozen weights
        transformer.freeze(recursive: true)

        // Unfreeze only LoRA parameters so they receive gradients
        transformer.unfreezeLoRAParameters()

        Flux2Debug.log("[LoRATrainer] Base model frozen, LoRA parameters unfrozen")

        // Initialize EMA if enabled
        if config.useEMA {
            emaManager = EMAManager(decay: config.emaDecay)
            emaManager?.initialize(from: transformer)
            Flux2Debug.log("[LoRATrainer] EMA initialized with decay=\(config.emaDecay)")
        }

        // Create AdamW optimizer for the transformer's trainable parameters (LoRA only)
        self.optimizer = AdamW(
            learningRate: config.learningRate,
            betas: (config.adamBeta1, config.adamBeta2),
            eps: config.adamEpsilon,
            weightDecay: config.weightDecay
        )
        Flux2Debug.log("[LoRATrainer] Created AdamW optimizer (lr=\(config.learningRate), wd=\(config.weightDecay))")

        // Create cached valueAndGrad function ONCE - this is critical for performance
        // Creating it inside trainStep() forces gradient graph reconstruction per call
        let usesGuidance = modelType.usesGuidanceEmbeds
        let lossWeightingMode = config.lossWeighting
        func lossFunction(model: Flux2Transformer2DModel, arrays: [MLXArray]) -> [MLXArray] {
            // Input arrays order: [packedLatents, batchedHidden, timesteps, imgIds, txtIds, velocityTarget, guidance]
            let packedLatentsIn = arrays[0]
            let batchedHiddenIn = arrays[1]
            let timestepsIn = arrays[2]
            let imgIdsIn = arrays[3]
            let txtIdsIn = arrays[4]
            let velocityTargetIn = arrays[5]
            let guidanceIn: MLXArray? = usesGuidance ? arrays[6] : nil

            let modelOutput = model(
                hiddenStates: packedLatentsIn,
                encoderHiddenStates: batchedHiddenIn,
                timestep: timestepsIn,
                guidance: guidanceIn,
                imgIds: imgIdsIn,
                txtIds: txtIdsIn
            )

            // Compute loss with optional weighting
            let loss: MLXArray
            switch lossWeightingMode {
            case .none, .uniform:
                // Standard MSE loss
                loss = mseLoss(predictions: modelOutput, targets: velocityTargetIn, reduction: .mean)

            case .bellShaped:
                // Bell-shaped weighting centered at t=500 (Ostris "weighted")
                // weight(t) = exp(-2 * ((t - 500) / 1000)^2)
                // Normalized so mean weight ≈ 1
                let t = timestepsIn  // Shape: [B]
                let centered = (t - 500.0) / 1000.0
                let weights = MLX.exp(-2.0 * centered * centered)
                // Normalize weights so mean = 1 (preserves loss scale)
                let normalizedWeights = weights / MLX.mean(weights)

                // Compute per-sample MSE, then weight and average
                // modelOutput and velocityTargetIn are [B, seq, features]
                let squaredError = (modelOutput - velocityTargetIn) * (modelOutput - velocityTargetIn)
                let perSampleMSE = MLX.mean(squaredError, axes: [1, 2])  // [B]
                let weightedMSE = perSampleMSE * normalizedWeights
                loss = MLX.mean(weightedMSE)

            case .snr, .minSNR:
                // SNR-based weighting: weight = 1 / (sigma^2 + 1)
                // Higher weight for cleaner samples (low sigma)
                let sigma = timestepsIn / 1000.0  // [B]
                let weights = 1.0 / (sigma * sigma + 1.0)
                let normalizedWeights = weights / MLX.mean(weights)

                let squaredError = (modelOutput - velocityTargetIn) * (modelOutput - velocityTargetIn)
                let perSampleMSE = MLX.mean(squaredError, axes: [1, 2])
                let weightedMSE = perSampleMSE * normalizedWeights
                loss = MLX.mean(weightedMSE)

            case .cosine:
                // Cosine weighting: cos(t * pi / 2)^2
                let t = timestepsIn / 1000.0  // [B] normalized to [0, 1]
                let cosWeights = MLX.cos(t * Float.pi / 2)
                let weights = cosWeights * cosWeights
                let normalizedWeights = weights / MLX.mean(weights)

                let squaredError = (modelOutput - velocityTargetIn) * (modelOutput - velocityTargetIn)
                let perSampleMSE = MLX.mean(squaredError, axes: [1, 2])
                let weightedMSE = perSampleMSE * normalizedWeights
                loss = MLX.mean(weightedMSE)

            case .sigmoid:
                // Sigmoid weighting: inverted sigmoid for higher weight at early timesteps
                let t = (timestepsIn / 1000.0 - 0.5) * 10  // [B] normalized to [-5, 5]
                let weights = MLX.sigmoid(-t)
                let normalizedWeights = weights / MLX.mean(weights)

                let squaredError = (modelOutput - velocityTargetIn) * (modelOutput - velocityTargetIn)
                let perSampleMSE = MLX.mean(squaredError, axes: [1, 2])
                let weightedMSE = perSampleMSE * normalizedWeights
                loss = MLX.mean(weightedMSE)
            }

            return [loss]
        }
        self.cachedLossAndGrad = valueAndGrad(model: transformer, lossFunction)
        Flux2Debug.log("[LoRATrainer] Created cached valueAndGrad function for efficient gradient computation")

        // Pre-compute position IDs per bucket (these only depend on resolution)
        Flux2Debug.log("[LoRATrainer] Pre-computing position IDs per bucket...")
        positionIdCache.removeAll()
        let patchSize = 2  // Standard patch size for Flux
        let txtLen = 512   // Fixed sequence length for Klein
        
        if config.enableBucketing {
            // Pre-compute for each unique resolution in the dataset's active buckets
            for bucket in dataset.buckets {
                let key = "\(bucket.width)x\(bucket.height)"
                if positionIdCache[key] == nil {
                    let latentH = bucket.height / 8
                    let latentW = bucket.width / 8
                    let imgH = latentH / patchSize
                    let imgW = latentW / patchSize
                    
                    let txtIds = generateTextPositionIDs(length: txtLen)
                    let imgIds = generateImagePositionIDs(height: imgH, width: imgW)
                    eval(txtIds, imgIds)
                    positionIdCache[key] = (txtIds: txtIds, imgIds: imgIds)
                }
            }
            // Also cache validation dataset buckets if present
            if let valDataset = validationDataset {
                for bucket in valDataset.buckets {
                    let key = "\(bucket.width)x\(bucket.height)"
                    if positionIdCache[key] == nil {
                        let latentH = bucket.height / 8
                        let latentW = bucket.width / 8
                        let imgH = latentH / patchSize
                        let imgW = latentW / patchSize
                        
                        let txtIds = generateTextPositionIDs(length: txtLen)
                        let imgIds = generateImagePositionIDs(height: imgH, width: imgW)
                        eval(txtIds, imgIds)
                        positionIdCache[key] = (txtIds: txtIds, imgIds: imgIds)
                    }
                }
            }
        } else {
            // Single resolution mode
            let key = "\(config.imageSize)x\(config.imageSize)"
            let latentH = config.imageSize / 8
            let latentW = config.imageSize / 8
            let imgH = latentH / patchSize
            let imgW = latentW / patchSize
            
            let txtIds = generateTextPositionIDs(length: txtLen)
            let imgIds = generateImagePositionIDs(height: imgH, width: imgW)
            eval(txtIds, imgIds)
            positionIdCache[key] = (txtIds: txtIds, imgIds: imgIds)
        }
        Flux2Debug.log("[LoRATrainer] Position IDs cached for \(positionIdCache.count) resolution(s)")

        // Set training mode
        TrainingMode.shared.isTraining = true
        
        // Emit start event
        eventHandler?.handleEvent(.started)
        
        // Resume from checkpoint if specified
        // TODO: Update checkpoint loading for LoRATrainingModel
        if config.resumeFromCheckpoint != nil {
            Flux2Debug.log("[LoRATrainer] Warning: Checkpoint resume not yet supported with new training model")
        }
        
        self.state = state
        
        do {
            // Generate reference image at step 0 (before any LoRA training)
            // This shows the baseline model output for comparison
            if config.validationPrompt != nil && config.validationEveryNSteps > 0 {
                Flux2Debug.log("[LoRATrainer] Generating reference image (step 0, no LoRA applied)...")
                if let prompt = config.validationPrompt {
                    do {
                        if let imageURL = try await generateValidationImage(
                            transformer: transformer,
                            prompt: prompt,
                            step: 0
                        ) {
                            Flux2Debug.log("[LoRATrainer] Reference image saved: \(imageURL.lastPathComponent)")
                            eventHandler?.handleEvent(.validationImageGenerated(
                                path: imageURL.path,
                                step: 0
                            ))
                        }
                    } catch {
                        Flux2Debug.log("[LoRATrainer] Warning: Failed to generate reference image: \(error.localizedDescription)")
                    }
                }
            }
            
            // Training loop
            while !state.isComplete && !shouldStop {
                // Start new epoch if needed
                if state.epochStep == 0 {
                    state.startEpoch()
                    dataset.startEpoch()
                    eventHandler?.handleEvent(.epochStarted(epoch: state.epoch))
                }

                // Process batches
                while let batch = try dataset.nextBatch(), !shouldStop {
                    // Get current learning rate and update optimizer
                    let currentLR = lrScheduler.getLearningRate(step: state.globalStep)
                    optimizer?.learningRate = currentLR

                    // Training step
                    let loss = try await trainStep(
                        batch: batch,
                        transformer: transformer,
                        vae: vae,
                        textEncoder: textEncoder,
                        learningRate: currentLR
                    )
                    
                    // Update state
                    state.update(loss: loss, batchSize: batch.count)
                    self.state = state
                    
                    // Emit step event
                    eventHandler?.handleEvent(.stepCompleted(step: state.globalStep, loss: loss))
                    progressCallback?(state)
                    
                    // Logging
                    if state.globalStep % config.logEveryNSteps == 0 {
                        Flux2Debug.log(state.progressSummary)
                    }

                    // Validation loss computation (at validation intervals)
                    if config.validationEveryNSteps > 0 &&
                       validationDataset != nil &&
                       state.globalStep % config.validationEveryNSteps == 0 {
                        do {
                            if let valLoss = try await computeValidationLoss(
                                transformer: transformer,
                                vae: vae,
                                textEncoder: textEncoder
                            ) {
                                state.lastValidationLoss = valLoss
                                self.state = state
                                
                                // Calculate gap between validation and training loss
                                let valGap = valLoss - state.currentLoss
                                Flux2Debug.log("[Validation] Step \(state.globalStep) - Val Loss: \(String(format: "%.4f", valLoss)) | Train Loss: \(String(format: "%.4f", state.currentLoss)) | Gap: \(String(format: "%+.4f", valGap))")

                                // Emit validation loss event
                                eventHandler?.handleEvent(.validationLossComputed(
                                    step: state.globalStep,
                                    trainLoss: state.currentLoss,
                                    valLoss: valLoss
                                ))
                                
                                // Overfitting detection based on train/val gap
                                if config.earlyStoppingOnOverfit {
                                    // Check if gap exceeds maximum allowed
                                    if valGap > config.earlyStoppingMaxValGap {
                                        earlyStopReason = "Overfitting detected: val-train gap (\(String(format: "%.4f", valGap))) exceeds max (\(String(format: "%.4f", config.earlyStoppingMaxValGap)))"
                                        Flux2Debug.log("[Training] Early stopping: \(earlyStopReason!)")
                                        shouldStop = true
                                    }
                                    // Check if gap is consistently increasing
                                    else if valGap > lastValGap + 0.01 {  // Gap increased significantly
                                        consecutiveGapIncreases += 1
                                        Flux2Debug.log("[Training] Overfitting warning: gap increasing (\(consecutiveGapIncreases)/\(config.earlyStoppingGapPatience))")
                                        
                                        if consecutiveGapIncreases >= config.earlyStoppingGapPatience {
                                            earlyStopReason = "Overfitting detected: gap increased \(consecutiveGapIncreases) times (was \(String(format: "%.4f", bestValGap)), now \(String(format: "%.4f", valGap)))"
                                            Flux2Debug.log("[Training] Early stopping: \(earlyStopReason!)")
                                            shouldStop = true
                                        }
                                    } else {
                                        // Gap stable or decreasing
                                        consecutiveGapIncreases = 0
                                        if valGap < bestValGap {
                                            bestValGap = valGap
                                        }
                                    }
                                    lastValGap = valGap
                                }

                                // Track best val loss this epoch (for epoch-based stagnation detection)
                                if config.earlyStoppingOnValStagnation {
                                    if valLoss < bestValLossThisEpoch {
                                        bestValLossThisEpoch = valLoss
                                    }
                                }
                            }
                        } catch {
                            Flux2Debug.log("[Validation] Failed to compute validation loss: \(error.localizedDescription)")
                        }
                    }
                    
                    // Checkpointing - save intermediate LoRA weights
                    if config.saveEveryNSteps > 0 &&
                       state.globalStep % config.saveEveryNSteps == 0 {
                        let outputDir = config.outputPath.deletingLastPathComponent()
                        let checkpointName = "checkpoint_\(state.globalStep).safetensors"
                        let checkpointURL = outputDir.appendingPathComponent(checkpointName)

                        do {
                            try saveLoRAWeights(from: transformer, to: checkpointURL)
                            eventHandler?.handleEvent(.checkpointSaved(
                                path: checkpointURL.path,
                                step: state.globalStep
                            ))

                            // Clean up old checkpoints if keepOnlyLastNCheckpoints > 0
                            if config.keepOnlyLastNCheckpoints > 0 {
                                cleanupOldCheckpoints(
                                    in: outputDir,
                                    keepLast: config.keepOnlyLastNCheckpoints
                                )
                            }
                        } catch {
                            Flux2Debug.log("[Checkpoint] Failed to save: \(error.localizedDescription)")
                        }

                        state.lastCheckpointTime = Date()
                    }
                    
                    // Validation image generation
                    if config.validationEveryNSteps > 0 &&
                       config.validationPrompt != nil &&
                       state.globalStep % config.validationEveryNSteps == 0 {
                        // Generate validation image with current LoRA weights
                        if let prompt = config.validationPrompt {
                            do {
                                if let imageURL = try await generateValidationImage(
                                    transformer: transformer,
                                    prompt: prompt,
                                    step: state.globalStep
                                ) {
                                    eventHandler?.handleEvent(.validationImageGenerated(
                                        path: imageURL.path,
                                        step: state.globalStep
                                    ))
                                }
                            } catch {
                                Flux2Debug.log("[Validation] Failed to generate image: \(error.localizedDescription)")
                            }
                        }
                        state.lastValidationTime = Date()
                    }
                    
                    // Check completion
                    if state.isComplete {
                        break
                    }
                }
                
                // Epoch complete
                eventHandler?.handleEvent(.epochCompleted(
                    epoch: state.epoch,
                    avgLoss: state.averageLoss
                ))

                // Early stopping check (train loss plateau)
                if config.enableEarlyStopping {
                    let currentLoss = state.averageLoss
                    let improvement = bestLoss - currentLoss

                    if improvement > config.earlyStoppingMinDelta {
                        // Loss improved
                        bestLoss = currentLoss
                        epochsWithoutImprovement = 0
                        Flux2Debug.log("[LoRATrainer] Early stopping: loss improved to \(String(format: "%.4f", currentLoss))")
                    } else {
                        // No significant improvement
                        epochsWithoutImprovement += 1
                        Flux2Debug.log("[LoRATrainer] Early stopping: no improvement for \(epochsWithoutImprovement)/\(config.earlyStoppingPatience) epochs")

                        if epochsWithoutImprovement >= config.earlyStoppingPatience {
                            earlyStopReason = "Loss plateau detected (best: \(String(format: "%.4f", bestLoss)), current: \(String(format: "%.4f", currentLoss)))"
                            Flux2Debug.log("[LoRATrainer] Early stopping triggered: \(earlyStopReason!)")
                            shouldStop = true
                        }
                    }
                }

                // Val loss stagnation check (epoch-based)
                if config.earlyStoppingOnValStagnation && !shouldStop && bestValLossThisEpoch < Float.infinity {
                    let valImprovement = bestValLossPreviousEpoch - bestValLossThisEpoch

                    if bestValLossPreviousEpoch < Float.infinity {
                        // We have a previous epoch to compare to
                        if valImprovement < config.earlyStoppingMinValImprovement {
                            consecutiveValStagnationEpochs += 1
                            Flux2Debug.log("[Training] Val loss stagnation: Δval=\(String(format: "%.4f", valImprovement)) < \(String(format: "%.4f", config.earlyStoppingMinValImprovement)) (\(consecutiveValStagnationEpochs)/\(config.earlyStoppingValStagnationPatience) epochs)")

                            if consecutiveValStagnationEpochs >= config.earlyStoppingValStagnationPatience {
                                earlyStopReason = "Val loss stagnation: no significant improvement for \(consecutiveValStagnationEpochs) epochs (previous: \(String(format: "%.4f", bestValLossPreviousEpoch)), current: \(String(format: "%.4f", bestValLossThisEpoch)))"
                                Flux2Debug.log("[Training] Early stopping: \(earlyStopReason!)")
                                shouldStop = true
                            }
                        } else {
                            // Good improvement, reset counter
                            consecutiveValStagnationEpochs = 0
                            Flux2Debug.log("[Training] Val loss improved this epoch: Δval=\(String(format: "%.4f", valImprovement))")
                        }
                    }

                    // Update for next epoch
                    bestValLossPreviousEpoch = bestValLossThisEpoch
                    bestValLossThisEpoch = Float.infinity  // Reset for next epoch
                }

                // Reset epoch step for next epoch
                if !state.isComplete && !shouldStop {
                    state.epochStep = 0
                    self.state = state
                }
            }
            
            // Training complete
            TrainingMode.shared.isTraining = false
            isRunning = false
            
            // Save final LoRA weights from transformer
            try saveLoRAWeights(from: transformer, to: config.outputPath)
            
            eventHandler?.handleEvent(.completed(
                finalLoss: state.averageLoss,
                totalSteps: state.globalStep
            ))
            
            Flux2Debug.log("[LoRATrainer] Training complete!")
            Flux2Debug.log(state.detailedStatus)
            
        } catch {
            TrainingMode.shared.isTraining = false
            isRunning = false
            eventHandler?.handleEvent(.error(error))
            throw error
        }
    }
    
    // MARK: - Single Training Step

    /// Struct to hold all inputs for the loss function
    private struct TrainingInputs {
        let packedLatents: MLXArray
        let batchedHidden: MLXArray
        let timesteps: MLXArray
        let guidance: MLXArray?
        let imgIds: MLXArray
        let txtIds: MLXArray
        /// Velocity target for flow matching: v = noise - original_latents
        let packedVelocityTarget: MLXArray
    }

    /// Execute a single training step with real gradient computation
    private func trainStep(
        batch: TrainingBatch,
        transformer: Flux2Transformer2DModel,
        vae: AutoencoderKLFlux2?,
        textEncoder: (String) async throws -> MLXArray,
        learningRate: Float
    ) async throws -> Float {
        let stepNum = (state?.globalStep ?? 0) + 1
        Flux2Debug.log("[Step \(stepNum)] Starting trainStep...")
        
        // Get latents (from cache or encode)
        Flux2Debug.log("[Step \(stepNum)] Loading latents...")
        let latents: MLXArray
        if let cache = latentCache {
            latents = try cache.getLatents(for: batch, vae: vae)
        } else if let vae = vae {
            let normalizedImages = batch.images * 2.0 - 1.0
            let nchwImages = normalizedImages.transposed(0, 3, 1, 2)
            latents = vae.encode(nchwImages)
        } else {
            throw LoRATrainerError.noVAEProvided
        }
        // NO eval here - let lazy evaluation build the full graph
        Flux2Debug.log("[Step \(stepNum)] Latents loaded: \(latents.shape)")

        // Get text embeddings from cache
        Flux2Debug.log("[Step \(stepNum)] Loading text embeddings...")
        guard let cache = textEmbeddingCache else {
            throw LoRATrainerError.trainingFailed("Text embedding cache not initialized")
        }
        
        var hiddenStates: [MLXArray] = []
        for originalCaption in batch.captions {
            let caption: String
            if config.captionDropoutRate > 0 && Float.random(in: 0..<1) < config.captionDropoutRate {
                caption = ""
            } else {
                caption = originalCaption
            }

            guard let cached = try cache.getEmbeddings(for: caption) else {
                throw LoRATrainerError.trainingFailed("Text embedding not cached for: '\(caption.prefix(50))...'")
            }
            
            let embedding = cached.hidden.shape.count > 2
                ? cached.hidden.squeezed(axis: 0)
                : cached.hidden
            hiddenStates.append(embedding)
        }

        let batchedHidden = MLX.stacked(hiddenStates, axis: 0)
        // NO eval here - let lazy evaluation build the full graph
        hiddenStates.removeAll()  // Release references
        Flux2Debug.log("[Step \(stepNum)] Text embeddings loaded: \(batchedHidden.shape)")

        // Get position IDs
        let batchSize = latents.shape[0]
        let imageWidth = latents.shape[3] * 8
        let imageHeight = latents.shape[2] * 8
        let resKey = "\(imageWidth)x\(imageHeight)"
        
        guard let posIds = positionIdCache[resKey] else {
            throw LoRATrainerError.trainingFailed("Position IDs not cached for resolution: \(resKey)")
        }
        let txtIds = posIds.txtIds
        let imgIds = posIds.imgIds

        // Sample timesteps and noise
        Flux2Debug.log("[Step \(stepNum)] Preparing noise...")
        let timesteps = sampleTimesteps(batchSize: batchSize)
        let sigmas = timesteps.asType(.float32) / 1000.0
        let noise = MLXRandom.normal(latents.shape)
        // NO eval here - let lazy evaluation build the full graph

        let sigmasExpanded = sigmas.reshaped([batchSize, 1, 1, 1])
        let noisyLatents = (1 - sigmasExpanded) * latents + sigmasExpanded * noise
        // NO eval here - let lazy evaluation build the full graph

        let packedLatents = packLatentsForTransformer(noisyLatents, patchSize: 2)
        // NO eval here - let lazy evaluation build the full graph

        let guidance: MLXArray? = modelType.usesGuidanceEmbeds ?
            MLXArray.full([batchSize], values: MLXArray(Float(4.0))) : nil

        let velocityTarget = noise - latents
        let packedVelocityTarget = packLatentsForTransformer(velocityTarget, patchSize: 2)
        // NO eval here - let lazy evaluation build the full graph
        Flux2Debug.log("[Step \(stepNum)] Inputs prepared")

        // Prepare inputs array
        var inputArrays: [MLXArray] = [
            packedLatents,
            batchedHidden,
            timesteps.asType(DType.float32),
            imgIds.asType(DType.int32),
            txtIds.asType(DType.int32),
            packedVelocityTarget
        ]
        if let g = guidance {
            inputArrays.append(g)
        } else {
            inputArrays.append(MLXArray(0.0))
        }

        // Forward + backward pass
        Flux2Debug.log("[Step \(stepNum)] Computing forward+backward...")
        guard let lossAndGrad = self.cachedLossAndGrad else {
            throw LoRATrainerError.trainingFailed("cachedLossAndGrad not initialized")
        }

        let (losses, grads) = lossAndGrad(transformer, inputArrays)
        let loss = losses[0]
        
        // Filter gradients to keep only LoRA parameters
        Flux2Debug.log("[Step \(stepNum)] Processing gradients...")
        let filteredGrads = filterLoRAGradients(grads)

        // Apply gradient clipping if configured
        var clippedGrads = filteredGrads
        if config.maxGradNorm > 0 {
            clippedGrads = clipGradNorm(filteredGrads, maxNorm: config.maxGradNorm)
        }

        // Optimizer update
        Flux2Debug.log("[Step \(stepNum)] Optimizer update...")
        guard let optimizer = self.optimizer else {
            throw LoRATrainerError.trainingFailed("Optimizer not initialized")
        }
        optimizer.update(model: transformer, gradients: clippedGrads)

        // EMA update
        emaManager?.update(from: transformer)

        // SINGLE eval at the end following MLX documentation pattern:
        // mx.eval(model.parameters(), optimizer.state)
        // This evaluates the full computation graph (forward, backward, optimizer update) at once
        Flux2Debug.log("[Step \(stepNum)] Final eval...")
        eval(transformer.trainableParameters())
        eval(loss)
        let lossValue = loss.item(Float.self)

        // Clear GPU cache at EVERY step to prevent memory accumulation
        MLX.Memory.clearCache()

        return lossValue
    }

    // MARK: - Validation Loss

    /// Compute validation loss on the validation dataset (no gradient computation)
    /// Returns the average loss across all validation samples, or nil if no validation dataset
    private func computeValidationLoss(
        transformer: Flux2Transformer2DModel,
        vae: AutoencoderKLFlux2?,
        textEncoder: (String) async throws -> MLXArray
    ) async throws -> Float? {
        guard let valDataset = validationDataset else {
            return nil
        }

        var totalLoss: Float = 0.0
        var sampleCount: Int = 0

        // Process all validation samples
        valDataset.startEpoch()

        while let batch = try valDataset.nextBatch() {
            let loss = try await computeBatchLoss(
                batch: batch,
                transformer: transformer,
                vae: vae,
                textEncoder: textEncoder,
                useValidationCache: true
            )
            totalLoss += loss * Float(batch.count)
            sampleCount += batch.count
        }

        guard sampleCount > 0 else { return nil }

        let avgLoss = totalLoss / Float(sampleCount)

        // Clear GPU cache after validation
        MLX.Memory.clearCache()

        return avgLoss
    }

    /// Compute loss for a batch without gradient computation (for validation)
    private func computeBatchLoss(
        batch: TrainingBatch,
        transformer: Flux2Transformer2DModel,
        vae: AutoencoderKLFlux2?,
        textEncoder: (String) async throws -> MLXArray,
        useValidationCache: Bool
    ) async throws -> Float {
        // Get latents (from validation cache or encode)
        let latents: MLXArray
        let cacheToUse = useValidationCache ? validationLatentCache : latentCache

        if let cache = cacheToUse {
            latents = try cache.getLatents(for: batch, vae: vae)
        } else if let vae = vae {
            let normalizedImages = batch.images * 2.0 - 1.0
            let nchwImages = normalizedImages.transposed(0, 3, 1, 2)
            latents = vae.encode(nchwImages)
        } else {
            throw LoRATrainerError.noVAEProvided
        }

        // Get text embeddings from cache (MUST be pre-cached)
        let textCacheToUse = useValidationCache ? validationTextEmbeddingCache : textEmbeddingCache
        guard let cache = textCacheToUse else {
            throw LoRATrainerError.trainingFailed("Text embedding cache not initialized for validation")
        }
        
        var hiddenStates: [MLXArray] = []

        for caption in batch.captions {
            // Text embeddings MUST be pre-cached
            guard let cached = try cache.getEmbeddings(for: caption) else {
                throw LoRATrainerError.trainingFailed("Text embedding not cached for validation: '\(caption.prefix(50))...'")
            }
            let embedding = cached.hidden.shape.count > 2
                ? cached.hidden.squeezed(axis: 0)
                : cached.hidden
            hiddenStates.append(embedding)
        }

        // Stack embeddings
        let batchedHidden = MLX.stacked(hiddenStates, axis: 0)
        eval(batchedHidden)

        // Get pre-computed position IDs from cache
        let batchSize = latents.shape[0]
        let imageWidth = latents.shape[3] * 8
        let imageHeight = latents.shape[2] * 8
        let resKey = "\(imageWidth)x\(imageHeight)"
        
        guard let posIds = positionIdCache[resKey] else {
            throw LoRATrainerError.trainingFailed("Position IDs not cached for validation resolution: \(resKey)")
        }
        let txtIds = posIds.txtIds
        let imgIds = posIds.imgIds

        // Sample timesteps (use fixed seed for reproducible validation)
        let timesteps = MLXRandom.randInt(low: 0, high: 1000, [batchSize])
        let sigmas = timesteps.asType(.float32) / 1000.0

        // Sample noise
        let noise = MLXRandom.normal(latents.shape)

        // Add noise to latents
        let sigmasExpanded = sigmas.reshaped([batchSize, 1, 1, 1])
        let noisyLatents = (1 - sigmasExpanded) * latents + sigmasExpanded * noise

        // Pack latents
        let packedLatents = packLatentsForTransformer(noisyLatents, patchSize: 2)

        // Prepare guidance
        let guidance: MLXArray? = modelType.usesGuidanceEmbeds ?
            MLXArray.full([batchSize], values: MLXArray(Float(4.0))) : nil

        // Compute velocity target
        let velocityTarget = noise - latents
        let packedVelocityTarget = packLatentsForTransformer(velocityTarget, patchSize: 2)

        // Forward pass (no gradients needed)
        let modelOutput = transformer(
            hiddenStates: packedLatents,
            encoderHiddenStates: batchedHidden,
            timestep: timesteps.asType(DType.float32),
            guidance: guidance,
            imgIds: imgIds.asType(DType.int32),
            txtIds: txtIds.asType(DType.int32)
        )

        // Compute MSE loss
        let loss = mseLoss(predictions: modelOutput, targets: packedVelocityTarget, reduction: .mean)

        // Evaluate to get the loss value
        eval(loss)
        return loss.item(Float.self)
    }

    /// Flatten ModuleParameters gradients into a simple [path: MLXArray] dictionary
    private func flattenGradients(_ grads: ModuleParameters, prefix: String = "") -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        func flatten(_ item: NestedItem<String, MLXArray>, path: String) {
            switch item {
            case .none:
                break
            case .value(let arr):
                result[path] = arr
            case .array(let items):
                for (idx, subItem) in items.enumerated() {
                    flatten(subItem, path: "\(path)[\(idx)]")
                }
            case .dictionary(let dict):
                for (key, subItem) in dict {
                    let newPath = path.isEmpty ? key : "\(path).\(key)"
                    flatten(subItem, path: newPath)
                }
            }
        }

        for (key, item) in grads {
            flatten(item, path: key)
        }

        return result
    }

    /// Clip gradient norm using flattened gradient dictionary
    private func clipFlatGradNorm(_ grads: [String: MLXArray], maxNorm: Float) -> [String: MLXArray] {
        guard !grads.isEmpty else { return grads }

        // Compute total norm
        var totalNormSq = MLXArray(0.0)
        for (_, grad) in grads {
            totalNormSq = totalNormSq + (grad * grad).sum()
        }
        let totalNorm = sqrt(totalNormSq)

        // Compute clip coefficient
        let maxNormArr = MLXArray(maxNorm)
        let clipCoef = minimum(maxNormArr / (totalNorm + MLXArray(1e-6)), MLXArray(1.0))

        // Apply clipping
        var result: [String: MLXArray] = [:]
        for (key, grad) in grads {
            result[key] = grad * clipCoef
        }

        return result
    }

    /// Clip gradient norm to prevent exploding gradients (legacy, kept for reference)
    private func clipGradNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        // OPTIMIZED: Use batched operations instead of sequential loops
        // The previous version created a huge sequential computation graph

        // Collect all gradients
        var allGrads: [MLXArray] = []

        func collectGrads(_ item: NestedItem<String, MLXArray>) {
            switch item {
            case .none:
                break
            case .value(let arr):
                allGrads.append(arr)
            case .array(let items):
                for subItem in items {
                    collectGrads(subItem)
                }
            case .dictionary(let dict):
                for (_, subItem) in dict {
                    collectGrads(subItem)
                }
            }
        }

        for (_, item) in grads {
            collectGrads(item)
        }

        // Compute total norm - BATCHED operation
        guard !allGrads.isEmpty else { return grads }

        // Compute squared norms in parallel (each is independent)
        let squaredNorms = allGrads.map { ($0 * $0).sum() }

        // Stack and sum in ONE operation instead of sequential loop
        let totalNormSq = MLX.stacked(squaredNorms).sum()
        let totalNorm = sqrt(totalNormSq)

        // Compute clip coefficient (single scalar operation)
        let maxNormArr = MLXArray(maxNorm)
        let clipCoef = minimum(maxNormArr / (totalNorm + MLXArray(1e-6)), MLXArray(1.0))

        // Force eval of clipCoef ONCE before applying to all grads
        // This prevents recomputation for each gradient
        eval(clipCoef)

        // Apply clipping - each multiplication is now independent
        func clipItem(_ item: NestedItem<String, MLXArray>) -> NestedItem<String, MLXArray> {
            switch item {
            case .none:
                return .none
            case .value(let arr):
                return .value(arr * clipCoef)
            case .array(let items):
                return .array(items.map { clipItem($0) })
            case .dictionary(let dict):
                return .dictionary(dict.mapValues { clipItem($0) })
            }
        }

        var result = ModuleParameters()
        for (key, item) in grads {
            result[key] = clipItem(item)
        }

        return result
    }

    /// Debug: Print gradient paths containing "lora"
    private func printGradientPaths(_ grads: ModuleParameters, prefix: String, limit: Int) {
        var count = 0
        var loraCount = 0

        func printRecursive(_ item: NestedItem<String, MLXArray>, path: String) {
            switch item {
            case .none:
                break
            case .value(let arr):
                // Print if path contains "lora" or if we haven't printed many yet
                if path.lowercased().contains("lora") {
                    print("[DEBUG PATHS LORA] \(path): shape=\(arr.shape)")
                    loraCount += 1
                } else if count < limit {
                    print("[DEBUG PATHS] \(path): shape=\(arr.shape)")
                    count += 1
                }
            case .array(let items):
                for (idx, subItem) in items.enumerated() {
                    printRecursive(subItem, path: "\(path)[\(idx)]")
                }
            case .dictionary(let dict):
                for (key, subItem) in dict {
                    let newPath = path.isEmpty ? key : "\(path).\(key)"
                    printRecursive(subItem, path: newPath)
                }
            }
        }

        for (key, item) in grads {
            printRecursive(item, path: key)
        }
        print("[DEBUG PATHS] Total LoRA paths found: \(loraCount)")
        fflush(stdout)
    }

    /// Filter gradients to keep only LoRA parameters (loraA and loraB)
    /// Non-LoRA gradients are removed (.none) to prevent updates to base model weights
    /// This is much faster than zeroing out - we simply skip non-LoRA parameters
    private func filterLoRAGradients(_ grads: ModuleParameters) -> ModuleParameters {
        // Recursive function that properly tracks path and filters
        func filterRecursive(_ item: NestedItem<String, MLXArray>, path: [String]) -> NestedItem<String, MLXArray> {
            switch item {
            case .none:
                return .none
            case .value(let arr):
                // Check if the last path component is loraA or loraB
                let lastKey = path.last ?? ""
                if lastKey == "loraA" || lastKey == "loraB" {
                    return .value(arr)
                } else {
                    // Skip non-LoRA gradients entirely (much faster than creating zeros)
                    return .none
                }
            case .array(let items):
                let filteredItems = items.enumerated().map { (idx, subItem) in
                    filterRecursive(subItem, path: path + ["[\(idx)]"])
                }
                // Check if all items are .none, if so return .none
                let hasNonNone = filteredItems.contains { item in
                    if case .none = item { return false }
                    return true
                }
                return hasNonNone ? .array(filteredItems) : .none
            case .dictionary(let dict):
                var newDict: [String: NestedItem<String, MLXArray>] = [:]
                for (key, subItem) in dict {
                    let filtered = filterRecursive(subItem, path: path + [key])
                    // Only include non-.none items
                    if case .none = filtered {
                        continue
                    }
                    newDict[key] = filtered
                }
                return newDict.isEmpty ? .none : .dictionary(newDict)
            }
        }

        var result = ModuleParameters()
        for (key, item) in grads {
            let filtered = filterRecursive(item, path: [key])
            // Only include non-.none items
            if case .none = filtered {
                continue
            }
            result[key] = filtered
        }

        return result
    }

    // MARK: - LoRA Parameter Helpers

    /// Apply gradients to LoRA parameters in the transformer
    /// Uses simple SGD: param = param - lr * grad
    private func applyLoRAGradients(
        transformer: Flux2Transformer2DModel,
        gradients: ModuleParameters,
        learningRate: Float
    ) {
        // Flatten gradients for easier access
        let flatGrads = flattenGradients(gradients)

        // Debug: log gradient paths on first step
        if state?.globalStep == 1 {
            Flux2Debug.log("[ApplyLoRAGradients] Available gradient paths (\(flatGrads.count) total):")
            for path in flatGrads.keys.sorted().prefix(20) {
                Flux2Debug.log("  - \(path)")
            }
            if flatGrads.count > 20 {
                Flux2Debug.log("  ... and \(flatGrads.count - 20) more")
            }
        }

        // Update double-stream blocks
        for (idx, block) in transformer.transformerBlocks.enumerated() {
            // Update each attention projection if it's a LoRAInjectedLinear
            updateLoRAIfPresent(block.attn.toQ, gradPath: "transformerBlocks[\(idx)].attn.toQ", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toK, gradPath: "transformerBlocks[\(idx)].attn.toK", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toV, gradPath: "transformerBlocks[\(idx)].attn.toV", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.addQProj, gradPath: "transformerBlocks[\(idx)].attn.addQProj", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.addKProj, gradPath: "transformerBlocks[\(idx)].attn.addKProj", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.addVProj, gradPath: "transformerBlocks[\(idx)].attn.addVProj", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toOut, gradPath: "transformerBlocks[\(idx)].attn.toOut", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toAddOut, gradPath: "transformerBlocks[\(idx)].attn.toAddOut", flatGrads: flatGrads, lr: learningRate)
        }

        // Update single-stream blocks
        for (idx, block) in transformer.singleTransformerBlocks.enumerated() {
            updateLoRAIfPresent(block.attn.toQkvMlp, gradPath: "singleTransformerBlocks[\(idx)].attn.toQkvMlp", flatGrads: flatGrads, lr: learningRate)
            updateLoRAIfPresent(block.attn.toOut, gradPath: "singleTransformerBlocks[\(idx)].attn.toOut", flatGrads: flatGrads, lr: learningRate)
        }
    }

    /// Update a single LoRA layer if it exists and has gradients
    private func updateLoRAIfPresent(_ linear: Linear, gradPath: String, flatGrads: [String: MLXArray], lr: Float) {
        guard let lora = linear as? LoRAInjectedLinear else { return }

        // Convert bracket notation to dot notation for alternative matching
        // e.g., "transformerBlocks[0]" -> "transformerBlocks.0"
        let dotPath = gradPath.replacingOccurrences(of: "[", with: ".").replacingOccurrences(of: "]", with: "")

        // Try different possible gradient paths (bracket and dot notation)
        let possiblePaths = [
            "\(gradPath).lora_a",
            "\(dotPath).lora_a",
            "\(gradPath).loraA",
            "\(dotPath).loraA"
        ]

        var gradA: MLXArray? = nil
        var gradB: MLXArray? = nil

        for basePath in possiblePaths {
            if gradA == nil, let g = flatGrads[basePath] {
                gradA = g
            }
            let bPath = basePath.replacingOccurrences(of: "lora_a", with: "lora_b")
                                .replacingOccurrences(of: "loraA", with: "loraB")
            if gradB == nil, let g = flatGrads[bPath] {
                gradB = g
            }
        }

        // Apply SGD update
        if let gA = gradA {
            lora.loraA = lora.loraA - lr * gA
        }
        if let gB = gradB {
            lora.loraB = lora.loraB - lr * gB
        }

        // Debug warning if no gradients found
        if gradA == nil && gradB == nil && state?.globalStep == 1 {
            Flux2Debug.log("[ApplyLoRAGradients] WARNING: No gradients found for path \(gradPath)")
        }
    }

    /// Evaluate all LoRA parameters to synchronize GPU
    private func evalLoRAParameters(transformer: Flux2Transformer2DModel) {
        // Double-stream blocks
        for block in transformer.transformerBlocks {
            if let lora = block.attn.toQ as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toK as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toV as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.addQProj as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.addKProj as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.addVProj as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toAddOut as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
        }

        // Single-stream blocks
        for block in transformer.singleTransformerBlocks {
            if let lora = block.attn.toQkvMlp as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
            if let lora = block.attn.toOut as? LoRAInjectedLinear { eval(lora.loraA, lora.loraB) }
        }
    }

    // MARK: - Save Weights

    /// Save LoRA weights from transformer to safetensors file
    /// - Parameters:
    ///   - transformer: Transformer with LoRA layers
    ///   - url: Destination path for weights
    ///   - useEMA: If true and EMA is enabled, saves EMA weights instead of current weights
    private func saveLoRAWeights(from transformer: Flux2Transformer2DModel, to url: URL, useEMA: Bool = true) throws {
        // Get LoRA weights - prefer EMA if available and requested
        let weights: [String: MLXArray]
        let isEMA: Bool

        if useEMA, let ema = emaManager, ema.isInitialized {
            weights = ema.getEMAWeights()
            isEMA = true
            Flux2Debug.log("[LoRATrainer] Using EMA weights for saving")
        } else {
            weights = transformer.getLoRAParameters()
            isEMA = false
        }

        guard !weights.isEmpty else {
            throw LoRATrainerError.trainingFailed("No LoRA weights found in transformer")
        }

        // Save to safetensors format
        try save(arrays: weights, url: url)

        // Also save metadata alongside
        let metadataPath = url.deletingPathExtension().appendingPathExtension("json")
        let metadata = LoRAWeightsMetadata(
            rank: config.rank,
            alpha: config.alpha,
            targetLayers: config.targetLayers.rawValue,
            triggerWord: config.triggerWord,
            trainedOn: Date(),
            usedEMA: isEMA,
            emaDecay: isEMA ? config.emaDecay : nil
        )
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: metadataPath)

        let emaLabel = isEMA ? " (EMA)" : ""
        Flux2Debug.log("[LoRATrainer] Saved LoRA weights\(emaLabel) to \(url.path) (\(weights.count) tensors)")
    }

    // MARK: - Validation Image Generation

    /// Generate a validation image using the current LoRA weights
    /// - Parameters:
    ///   - transformer: Transformer with LoRA injected
    ///   - prompt: Validation prompt
    ///   - step: Current training step (for filename)
    /// - Returns: URL of saved image, or nil if generation failed
    private func generateValidationImage(
        transformer: Flux2Transformer2DModel,
        prompt: String,
        step: Int
    ) async throws -> URL? {
        guard let vae = validationVAE else {
            Flux2Debug.log("[Validation] Skipping - VAE not available")
            return nil
        }

        let size = 512  // Fixed 512x512 for validation previews
        let seed = config.validationSeed ?? 42
        let validationSteps = 15  // Quick preview, 15 steps is enough

        Flux2Debug.log("[Validation] Generating preview image for step \(step)...")

        // Set seed for reproducibility
        MLXRandom.seed(seed)

        // 1. Get text embeddings from cache (MUST be pre-cached)
        let textEmbeddings: MLXArray
        if let cache = textEmbeddingCache,
           let cached = try cache.getEmbeddings(for: prompt) {
            // Add batch dimension if needed: [seq, dim] -> [1, seq, dim]
            textEmbeddings = cached.hidden.shape.count == 2
                ? cached.hidden.expandedDimensions(axis: 0)
                : cached.hidden
        } else {
            Flux2Debug.log("[Validation] Skipping - prompt not in embedding cache")
            return nil
        }
        eval(textEmbeddings)

        // 2. Generate random patchified latents
        let patchifiedLatents = LatentUtils.generatePatchifiedLatents(
            height: size,
            width: size,
            seed: seed
        )
        var packedLatents = LatentUtils.packPatchifiedToSequence(patchifiedLatents)
        eval(packedLatents)

        // 3. Setup position IDs
        let textLength = textEmbeddings.shape[1]
        let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(
            textLength: textLength,
            height: size,
            width: size
        )

        // 4. Setup scheduler
        let scheduler = FlowMatchEulerScheduler()
        let imageSeqLen = packedLatents.shape[1]
        scheduler.setTimesteps(numInferenceSteps: validationSteps, imageSeqLen: imageSeqLen, strength: 1.0)

        // 5. Denoising loop
        let guidanceTensor: MLXArray? = modelType.usesGuidanceEmbeds ? MLXArray([4.0]) : nil

        for stepIdx in 0..<(scheduler.sigmas.count - 1) {
            let sigma = scheduler.sigmas[stepIdx]
            let t = MLXArray([sigma])

            let noisePred = transformer.callAsFunction(
                hiddenStates: packedLatents,
                encoderHiddenStates: textEmbeddings,
                timestep: t,
                guidance: guidanceTensor,
                imgIds: imageIds,
                txtIds: textIds
            )

            packedLatents = scheduler.step(
                modelOutput: noisePred,
                timestep: sigma,
                sample: packedLatents
            )
            eval(packedLatents)
        }

        // 6. Unpack and denormalize latents
        var finalPatchified = LatentUtils.unpackSequenceToPatchified(
            packedLatents,
            height: size,
            width: size
        )
        finalPatchified = LatentUtils.denormalizeLatentsWithBatchNorm(
            finalPatchified,
            runningMean: vae.batchNormRunningMean,
            runningVar: vae.batchNormRunningVar
        )
        let finalLatents = LatentUtils.unpatchifyLatents(finalPatchified)
        eval(finalLatents)

        // 7. Decode with VAE
        let decoded = vae.decode(finalLatents)
        eval(decoded)

        // 8. Convert to image
        guard let image = postprocessVAEOutput(decoded) else {
            Flux2Debug.log("[Validation] Failed to convert VAE output to image")
            return nil
        }

        // 9. Save to disk
        let outputDir = config.outputPath.deletingLastPathComponent()
        let filename = "preview_\(step).png"
        let outputURL = outputDir.appendingPathComponent(filename)

        try saveImage(image, to: outputURL)

        // Also save as preview_latest.png
        let latestURL = outputDir.appendingPathComponent("preview_latest.png")
        try saveImage(image, to: latestURL)

        Flux2Debug.log("[Validation] Saved preview image: \(outputURL.lastPathComponent)")

        return outputURL
    }

    /// Save a CGImage to disk as PNG
    private func saveImage(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw LoRATrainerError.trainingFailed("Failed to create image destination")
        }

        CGImageDestinationAddImage(destination, image, nil)

        guard CGImageDestinationFinalize(destination) else {
            throw LoRATrainerError.trainingFailed("Failed to write image to disk")
        }
    }

    /// Clean up old checkpoint files, keeping only the most recent N
    private func cleanupOldCheckpoints(in directory: URL, keepLast n: Int) {
        let fm = FileManager.default

        do {
            let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.creationDateKey])
            let checkpoints = files.filter { $0.lastPathComponent.hasPrefix("checkpoint_") && $0.pathExtension == "safetensors" }

            // Sort by step number (descending)
            let sorted = checkpoints.sorted { url1, url2 in
                let step1 = extractStepNumber(from: url1.lastPathComponent) ?? 0
                let step2 = extractStepNumber(from: url2.lastPathComponent) ?? 0
                return step1 > step2
            }

            // Delete all but the last N
            if sorted.count > n {
                for checkpoint in sorted.dropFirst(n) {
                    try fm.removeItem(at: checkpoint)
                    // Also remove the .json metadata file
                    let metadataURL = checkpoint.deletingPathExtension().appendingPathExtension("json")
                    try? fm.removeItem(at: metadataURL)
                    Flux2Debug.log("[Checkpoint] Cleaned up old checkpoint: \(checkpoint.lastPathComponent)")
                }
            }
        } catch {
            Flux2Debug.log("[Checkpoint] Error cleaning up old checkpoints: \(error.localizedDescription)")
        }
    }

    /// Extract step number from checkpoint filename (e.g., "checkpoint_500.safetensors" -> 500)
    private func extractStepNumber(from filename: String) -> Int? {
        let pattern = "checkpoint_(\\d+)"
        if let range = filename.range(of: pattern, options: .regularExpression),
           let stepStr = filename[range].split(separator: "_").last,
           let step = Int(stepStr) {
            return step
        }
        return nil
    }

    // MARK: - Control

    /// Stop training gracefully
    public func stop() {
        shouldStop = true
        Flux2Debug.log("[LoRATrainer] Stopping training...")
    }
    
    /// Get current training state
    public var currentState: TrainingState? {
        state
    }
    
    /// Whether training is currently running
    public var running: Bool {
        isRunning
    }

    // MARK: - Helpers

    /// Pack latents into patches for transformer input
    /// - Parameters:
    ///   - latents: Latents in [B, C, H, W] format
    ///   - patchSize: Size of each patch (typically 2)
    /// - Returns: Packed latents in [B, seq_len, patch_features] format
    private func packLatentsForTransformer(_ latents: MLXArray, patchSize: Int) -> MLXArray {
        let shape = latents.shape
        let batchSize = shape[0]
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]

        let patchH = height / patchSize
        let patchW = width / patchSize
        let patchFeatures = channels * patchSize * patchSize

        // Reshape: [B, C, H, W] -> [B, C, patchH, patchSize, patchW, patchSize]
        let reshaped = latents.reshaped([batchSize, channels, patchH, patchSize, patchW, patchSize])

        // Transpose to group spatial patches: [B, patchH, patchW, C, patchSize, patchSize]
        let transposed = reshaped.transposed(0, 2, 4, 1, 3, 5)

        // Flatten to sequence: [B, patchH * patchW, C * patchSize * patchSize]
        let packed = transposed.reshaped([batchSize, patchH * patchW, patchFeatures])

        return packed
    }

    /// Sample timesteps according to the configured strategy
    /// - Parameter batchSize: Number of timesteps to sample
    /// - Returns: Timesteps in range [0, 1000) as integer array
    private func sampleTimesteps(batchSize: Int) -> MLXArray {
        switch config.timestepSampling {
        case .uniform:
            // Standard uniform sampling [0, 1000)
            return MLXRandom.randInt(low: 0, high: 1000, [batchSize])

        case .logitNormal:
            // Sample from normal distribution, apply sigmoid, scale to [0, 1000)
            // This focuses training on medium noise levels (t ≈ 0.5)
            let u = MLXRandom.normal([batchSize]) * config.logitNormalStd + config.logitNormalMean
            let sigmoided = MLX.sigmoid(u)  // Maps to [0, 1]
            let scaled = sigmoided * 1000.0  // Scale to [0, 1000]
            // Clamp and convert to int
            return MLX.clip(scaled, min: 0, max: 999).asType(.int32)

        case .fluxShift:
            // Sample uniform, then add shift to bias toward higher timesteps
            // This helps with learning overall composition
            let base = MLXRandom.uniform(low: Float(0), high: Float(1000), [batchSize])
            let shifted = base + config.fluxShiftValue * 100  // Shift is in units of 100 timesteps
            // Clamp to valid range and convert to int
            return MLX.clip(shifted, min: 0, max: 999).asType(.int32)

        case .content:
            // Ostris content mode: t^3 distribution favors LOW timesteps
            // Good for learning specific subjects (faces, objects)
            // Low timesteps = late denoising = fine details
            let u = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize])
            let cubic = u * u * u  // t^3 concentrates near 0
            let scaled = cubic * 1000.0
            return MLX.clip(scaled, min: 0, max: 999).asType(.int32)

        case .style:
            // Ostris style mode: (1-t^3) distribution favors HIGH timesteps
            // Good for learning artistic styles, compositions
            // High timesteps = early denoising = global structure
            let u = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize])
            let cubic = u * u * u
            let inverted = 1.0 - cubic  // (1-t^3) concentrates near 1
            let scaled = inverted * 1000.0
            return MLX.clip(scaled, min: 0, max: 999).asType(.int32)
        }
    }
}

// MARK: - Errors

public enum LoRATrainerError: Error, LocalizedError {
    case notPrepared
    case datasetLoadFailed
    case invalidDataset(String)
    case noVAEProvided
    case trainingFailed(String)
    case checkpointLoadFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .notPrepared:
            return "Trainer not prepared. Call prepare() first."
        case .datasetLoadFailed:
            return "Failed to load training dataset"
        case .invalidDataset(let reason):
            return "Invalid dataset: \(reason)"
        case .noVAEProvided:
            return "No VAE provided and latents not cached"
        case .trainingFailed(let reason):
            return "Training failed: \(reason)"
        case .checkpointLoadFailed(let reason):
            return "Failed to load checkpoint: \(reason)"
        }
    }
}
