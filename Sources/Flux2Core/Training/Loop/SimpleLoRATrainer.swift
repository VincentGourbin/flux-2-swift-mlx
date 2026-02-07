// SimpleLoRATrainer.swift
// Clean, minimal LoRA trainer following Ostris/mflux reference implementation
// No EMA, no complex DOP, no bells and whistles - just working flow matching training

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXOptimizers
import CoreGraphics
import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Minimal LoRA training configuration (Ostris-compatible defaults)
public struct SimpleLoRAConfig: Sendable {
    // LoRA settings
    public var rank: Int = 32
    public var alpha: Float = 32.0
    
    // Training settings (Ostris defaults)
    public var learningRate: Float = 1e-4
    public var weightDecay: Float = 0.0001
    public var batchSize: Int = 1
    public var maxSteps: Int = 3000
    
    // Timestep/loss settings
    public var timestepSampling: TimestepSamplingMode = .balanced
    public var lossWeighting: LossWeightingMode = .bellShaped
    
    // Differential Output Preservation (DOP) - Ostris regularization
    // Prevents LoRA from affecting outputs when trigger word is not present
    public var dopEnabled: Bool = true  // Ostris default: true
    public var dopMultiplier: Float = 1.0  // Ostris default: 1.0
    public var dopPreservationClass: String = "person"  // Class word to replace trigger
    public var triggerWord: String?  // Must be set for DOP to work
    public var dopEveryNSteps: Int = 1  // OPTIMIZATION: Do DOP every N steps (1=every step, 2=every 2nd, etc.)
    
    // Gradient accumulation (simulate larger batch size)
    public var gradientAccumulationSteps: Int = 1  // 1 = no accumulation, 2+ = accumulate

    // Memory optimization
    public var gradientCheckpointing: Bool = false  // Wrap forward pass with checkpoint() to save memory

    // Checkpointing
    public var saveEveryNSteps: Int = 250
    public var logEveryNSteps: Int = 1

    // Learning curve visualization
    public var generateLearningCurve: Bool = true  // Generate SVG learning curve
    public var learningCurveSmoothingWindow: Int = 20  // Window size for moving average smoothing

    // Output
    public var outputDir: URL
    
    // Validation (optional)
    public var validationPrompts: [ValidationPromptConfig] = []
    public var validationEveryNSteps: Int = 250
    public var validationSeed: UInt64 = 42
    public var validationSteps: Int = 4  // Default 4 for distilled model

    /// Configuration for a single validation prompt
    public struct ValidationPromptConfig: Sendable {
        /// The prompt text (WITHOUT trigger word)
        public var prompt: String

        /// Generate at 512x512 resolution
        public var is512: Bool

        /// Generate at 1024x1024 resolution
        public var is1024: Bool

        /// If true, prepend trigger_word + ", " to the prompt
        public var applyTrigger: Bool

        /// Optional seed for this specific prompt (nil = use global seed)
        public var seed: UInt64?

        public init(prompt: String, is512: Bool = true, is1024: Bool = false, applyTrigger: Bool = true, seed: UInt64? = nil) {
            self.prompt = prompt
            self.is512 = is512
            self.is1024 = is1024
            self.applyTrigger = applyTrigger
            self.seed = seed
        }
    }
    
    public enum TimestepSamplingMode: String, Sendable {
        case uniform    // Standard uniform [0, 1000)
        case content    // t^3 - favors low timesteps (details)
        case style      // (1-t^3) - favors high timesteps (structure)
        case balanced   // 50/50 mix of content and style (Ostris default)
    }
    
    public enum LossWeightingMode: String, Sendable {
        case none       // Standard MSE
        case bellShaped // Ostris "weighted" - bell curve centered at t=500
    }
    
    public init(outputDir: URL) {
        self.outputDir = outputDir
    }
}

/// Cached latent entry
public struct CachedLatentEntry {
    public let filename: String
    public let latent: MLXArray
    public let width: Int
    public let height: Int
    
    public init(filename: String, latent: MLXArray, width: Int, height: Int) {
        self.filename = filename
        self.latent = latent
        self.width = width
        self.height = height
    }
}

/// Cached text embedding entry
public struct CachedEmbeddingEntry {
    public let caption: String
    public let embedding: MLXArray
    
    public init(caption: String, embedding: MLXArray) {
        self.caption = caption
        self.embedding = embedding
    }
}

/// Simple, clean LoRA trainer following Ostris/mflux implementation
/// Key differences from our LoRATrainer:
/// - NO EMA (Ostris has use_ema: false)
/// - DOP (Differential Output Preservation) supported - Ostris regularization
/// - Correct loss weighting normalization (sum, not mean)
/// - Simpler training loop
/// - Support for pause/resume via TrainingController
/// - Checkpoint resume with optimizer state
public final class SimpleLoRATrainer {

    // MARK: - Properties

    public let config: SimpleLoRAConfig
    private let modelType: Flux2Model

    // Training controller (optional, for pause/resume/stop)
    public var controller: TrainingController?

    // Training state for persistence
    private var trainingState: TrainingState?

    // Training state
    private var currentStep: Int = 0
    private var isRunning: Bool = false
    private var shouldStop: Bool = false

    // Cached valueAndGrad functions
    private var cachedLossAndGrad: ((Flux2Transformer2DModel, [MLXArray]) -> ([MLXArray], NestedDictionary<String, MLXArray>))?
    private var cachedDOPLossAndGrad: ((Flux2Transformer2DModel, [MLXArray]) -> ([MLXArray], NestedDictionary<String, MLXArray>))?

    // DOP (Differential Output Preservation) caches
    private var preservationEmbeddings: [String: CachedEmbeddingEntry] = [:]  // caption -> preservation embedding
    private var captionToPreservationCaption: [String: String] = [:]  // original caption -> preservation caption

    // Position ID cache per resolution
    private var positionIdCache: [String: (txtIds: MLXArray, imgIds: MLXArray)] = [:]

    // Loss history for learning curve (step, loss)
    private var lossHistory: [(step: Int, loss: Float)] = []

    // Progress callback
    public typealias ProgressCallback = (Int, Int, Float) -> Void
    private var progressCallback: ProgressCallback?

    // MARK: - Initialization

    public init(config: SimpleLoRAConfig, modelType: Flux2Model) {
        self.config = config
        self.modelType = modelType
    }

    /// Initialize with controller for pause/resume support
    public init(config: SimpleLoRAConfig, modelType: Flux2Model, controller: TrainingController) {
        self.config = config
        self.modelType = modelType
        self.controller = controller
    }

    public func setProgressCallback(_ callback: @escaping ProgressCallback) {
        self.progressCallback = callback
    }
    
    // MARK: - Main Training Entry Point

    /// Train with pre-cached latents and embeddings
    /// This is the cleanest way to train - everything is pre-computed
    /// - Parameters:
    ///   - transformer: The transformer model to train
    ///   - cachedLatents: Pre-encoded latents
    ///   - cachedEmbeddings: Pre-encoded text embeddings
    ///   - vae: Optional VAE for validation image generation
    ///   - textEncoder: Optional text encoder for DOP
    ///   - startStep: Step to start from (for resume, default 0)
    ///   - optimizerState: URL to optimizer state file (for resume)
    public func train(
        transformer: Flux2Transformer2DModel,
        cachedLatents: [CachedLatentEntry],
        cachedEmbeddings: [String: CachedEmbeddingEntry],
        vae: AutoencoderKLFlux2? = nil,
        textEncoder: ((String) async throws -> MLXArray)? = nil,
        startStep: Int = 0,
        optimizerState: URL? = nil
    ) async throws {
        guard !cachedLatents.isEmpty else {
            throw SimpleLoRATrainerError.noLatents
        }
        guard !cachedEmbeddings.isEmpty else {
            throw SimpleLoRATrainerError.noEmbeddings
        }
        
        isRunning = true
        shouldStop = false
        currentStep = startStep

        // Create output directory if needed
        try FileManager.default.createDirectory(at: config.outputDir, withIntermediateDirectories: true)

        // Initialize training state
        let configHash = TrainingState.hashConfig(
            modelType: modelType.rawValue,
            rank: config.rank,
            alpha: config.alpha,
            learningRate: config.learningRate,
            datasetPath: config.outputDir.path
        )
        trainingState = TrainingState(
            currentStep: startStep,
            totalSteps: config.maxSteps,
            rngSeed: config.validationSeed,
            configHash: configHash,
            modelType: modelType.rawValue,
            loraRank: config.rank,
            loraAlpha: config.alpha
        )

        print("=".repeating(60))
        print("SimpleLoRATrainer - Ostris-compatible training")
        print("=".repeating(60))
        print()
        if startStep > 0 {
            print("RESUMING from step \(startStep)")
            print()
        }
        print("Configuration:")
        print("  Rank: \(config.rank), Alpha: \(config.alpha)")
        print("  Learning rate: \(config.learningRate)")
        print("  Weight decay: \(config.weightDecay)")
        print("  Max steps: \(config.maxSteps)")
        print("  Timestep sampling: \(config.timestepSampling.rawValue)")
        print("  Loss weighting: \(config.lossWeighting.rawValue)")
        print("  EMA: DISABLED (like Ostris)")
        if config.dopEnabled && config.triggerWord != nil {
            print("  DOP: ENABLED (trigger='\(config.triggerWord!)', class='\(config.dopPreservationClass)', multiplier=\(config.dopMultiplier))")
        } else {
            print("  DOP: DISABLED")
        }
        print()
        print("Dataset:")
        print("  Latents: \(cachedLatents.count)")
        print("  Unique captions: \(cachedEmbeddings.count)")
        print()
        
        // Pre-cache position IDs for all resolutions
        print("Pre-caching position IDs...")
        let uniqueResolutions = Set(cachedLatents.map { "\($0.width)x\($0.height)" })
        for res in uniqueResolutions {
            let parts = res.split(separator: "x")
            let width = Int(parts[0])!
            let height = Int(parts[1])!
            let textLength = cachedEmbeddings.values.first!.embedding.shape[1]  // [B, SeqLen, HiddenDim]
            let (txtIds, imgIds, _) = LatentUtils.combinePositionIDs(
                textLength: textLength,
                height: height,
                width: width
            )
            positionIdCache[res] = (txtIds: txtIds, imgIds: imgIds)
        }
        print("  Cached \(uniqueResolutions.count) resolutions")
        print()

        // Generate baseline images (WITHOUT LoRA) for comparison
        // Only on fresh start (not resume)
        if startStep == 0 && !config.validationPrompts.isEmpty {
            try await generateBaselineImages()
        }

        // Inject LoRA into transformer
        print("Injecting LoRA...")
        transformer.applyLoRA(
            rank: config.rank,
            alpha: config.alpha,
            targetBlocks: .all  // Ostris trains attention + FFN layers
        )

        // Load checkpoint weights if resuming
        if startStep > 0, let optimizerStatePath = optimizerState {
            // Use the optimizer state path to find the checkpoint directory
            // This is more robust than reconstructing from startStep
            let checkpointDir = optimizerStatePath.deletingLastPathComponent()
            let loraPath = checkpointDir.appendingPathComponent("lora.safetensors")
            print("Loading LoRA weights from checkpoint...")
            try loadLoRACheckpoint(transformer: transformer, from: loraPath)
            print("  Loaded from: \(checkpointDir.lastPathComponent)/\(loraPath.lastPathComponent)")
        } else if startStep > 0 {
            // Fallback: construct path from step number (for backwards compatibility)
            let checkpointDir = config.outputDir.appendingPathComponent("checkpoint_\(String(format: "%06d", startStep))")
            let loraPath = checkpointDir.appendingPathComponent("lora.safetensors")
            print("Loading LoRA weights from checkpoint...")
            try loadLoRACheckpoint(transformer: transformer, from: loraPath)
            print("  Loaded from: \(loraPath.lastPathComponent)")
        }
        let loraParamCount = transformer.loraParameterCount
        print("  LoRA parameters: \(loraParamCount)")
        print()

        // Freeze base model, unfreeze LoRA
        transformer.freeze(recursive: true)
        transformer.unfreezeLoRAParameters()

        // Create optimizer (ResumableAdamW for checkpoint support, no EMA!)
        let optimizer = ResumableAdamW(
            learningRate: config.learningRate,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weightDecay: config.weightDecay
        )

        // Load optimizer state if resuming
        if let optimizerStateURL = optimizerState {
            print("Loading optimizer state from checkpoint...")
            try loadOptimizerState(optimizer: optimizer, from: optimizerStateURL)
            print("  Optimizer state restored")
            print()
        }
        
        // Create cached valueAndGrad function
        let usesGuidance = modelType.usesGuidanceEmbeds
        let lossWeightingMode = config.lossWeighting
        let useGradientCheckpointing = config.gradientCheckpointing

        // Note: gradient_checkpointing config exists but is NOT YET IMPLEMENTED
        // The naive approach of wrapping forward pass breaks gradient flow.
        // Proper implementation requires layer-wise checkpointing in the transformer itself.
        if useGradientCheckpointing {
            print("  ⚠️  Gradient checkpointing: NOT YET IMPLEMENTED (config ignored)")
            print("      Klein 9B + DOP requires layer-wise checkpointing in transformer")
        }

        func lossFunction(model: Flux2Transformer2DModel, arrays: [MLXArray]) -> [MLXArray] {
            // Input arrays: [packedLatents, hidden, timesteps, imgIds, txtIds, velocityTarget, guidance]
            let packedLatentsIn = arrays[0]
            let hiddenIn = arrays[1]
            let timestepsIn = arrays[2]
            let imgIdsIn = arrays[3]
            let txtIdsIn = arrays[4]
            let velocityTargetIn = arrays[5]
            let guidanceIn: MLXArray? = usesGuidance ? arrays[6] : nil

            let modelOutput = model(
                hiddenStates: packedLatentsIn,
                encoderHiddenStates: hiddenIn,
                timestep: timestepsIn,
                guidance: guidanceIn,
                imgIds: imgIdsIn,
                txtIds: txtIdsIn
            )
            
            // Compute loss with optional weighting
            let loss: MLXArray
            switch lossWeightingMode {
            case .none:
                loss = mseLoss(predictions: modelOutput, targets: velocityTargetIn, reduction: .mean)
                
            case .bellShaped:
                // Ostris "weighted" mode - bell curve centered at t=500
                // weight(t) = exp(-2 * ((t - 500) / 1000)^2)
                // Peak at t=500 (weight=1.0), tapers at extremes (weight≈0.61 at t=0 and t=1000)
                
                // timestepsIn are in [0, 1] range (sigmas), convert to [0, 1000] for weighting
                let tScaled = timestepsIn * 1000.0  // [0, 1] -> [0, 1000]
                let centered = (tScaled - 500.0) / 1000.0  // Center at 500
                let weights = MLX.exp(-2.0 * centered * centered)  // Bell curve
                
                // Compute element-wise squared error
                let sqError = (modelOutput - velocityTargetIn) * (modelOutput - velocityTargetIn)
                
                // Reshape weights for broadcasting: [B] -> [B, 1, 1] to match [B, SeqLen, Channels]
                let weightsExpanded = weights.reshaped([weights.shape[0], 1, 1])
                
                // Weighted MSE: sum(weights * sqError) / sum(weights)
                // This normalizes by sum of weights (Ostris convention)
                let weightedError = weightsExpanded * sqError
                loss = MLX.sum(weightedError) / (MLX.sum(weightsExpanded) * Float(sqError.shape[1] * sqError.shape[2]))
            }
            
            return [loss]
        }
        
        cachedLossAndGrad = valueAndGrad(model: transformer, lossFunction)
        
        // DOP (Differential Output Preservation) setup
        // Prevents LoRA from affecting outputs when trigger word is not present
        let dopEnabled = config.dopEnabled && config.triggerWord != nil
        if dopEnabled {
            print("Setting up DOP (Differential Output Preservation)...")
            print("  Trigger word: '\(config.triggerWord!)'")
            print("  Preservation class: '\(config.dopPreservationClass)'")
            print("  Multiplier: \(config.dopMultiplier)")
            
            // Create preservation captions (trigger word → preservation class)
            for (caption, _) in cachedEmbeddings {
                let preservationCaption = caption.replacingOccurrences(
                    of: config.triggerWord!,
                    with: config.dopPreservationClass
                )
                captionToPreservationCaption[caption] = preservationCaption
            }
            
            // Encode preservation embeddings
            if let textEncoder = textEncoder {
                print("  Encoding \(captionToPreservationCaption.count) preservation captions...")
                for (originalCaption, preservationCaption) in captionToPreservationCaption {
                    // Skip if already encoded (same caption)
                    if preservationEmbeddings[originalCaption] == nil {
                        do {
                            let embedding = try await textEncoder(preservationCaption)
                            preservationEmbeddings[originalCaption] = CachedEmbeddingEntry(
                                caption: preservationCaption,
                                embedding: embedding
                            )
                        } catch {
                            print("  Warning: Failed to encode preservation caption: \(error)")
                        }
                    }
                }
                print("  Encoded \(preservationEmbeddings.count) preservation embeddings")
            } else {
                print("  Warning: textEncoder not provided, DOP disabled")
            }
            
            // Create DOP loss function
            // DOP loss = MSE(LoRA output, Base output) for preservation embeddings
            func dopLossFunction(model: Flux2Transformer2DModel, arrays: [MLXArray]) -> [MLXArray] {
                // arrays: [packedLatents, preservationEmbedding, timesteps, imgIds, txtIds, baseOutput, guidance]
                let packedLatentsIn = arrays[0]
                let preservationEmbeddingIn = arrays[1]
                let timestepsIn = arrays[2]
                let imgIdsIn = arrays[3]
                let txtIdsIn = arrays[4]
                let baseOutputIn = arrays[5]  // Pre-computed base model output
                let guidanceIn: MLXArray? = usesGuidance ? arrays[6] : nil

                // Forward pass with LoRA enabled
                let loraOutput = model(
                    hiddenStates: packedLatentsIn,
                    encoderHiddenStates: preservationEmbeddingIn,
                    timestep: timestepsIn,
                    guidance: guidanceIn,
                    imgIds: imgIdsIn,
                    txtIds: txtIdsIn
                )

                // DOP loss: LoRA should match base when trigger is not present
                let dopLoss = mseLoss(predictions: loraOutput, targets: baseOutputIn, reduction: .mean)

                return [dopLoss]
            }
            
            cachedDOPLossAndGrad = valueAndGrad(model: transformer, dopLossFunction)
            print("  DOP ready ✓")
            print()
        } else if config.dopEnabled && config.triggerWord == nil {
            print("Warning: DOP enabled but triggerWord not set, DOP disabled")
            print()
        }
        
        print("Starting training...")
        if config.gradientAccumulationSteps > 1 {
            print("  Gradient accumulation: \(config.gradientAccumulationSteps) steps (effective batch = \(config.batchSize * config.gradientAccumulationSteps))")
        }
        print("-".repeating(60))

        // Gradient accumulation state
        var accumulatedGrads: NestedDictionary<String, MLXArray>? = nil

        // Track timing for ETA
        let trainingStartTime = Date()

        // Training loop (start from startStep + 1 if resuming)
        let firstStep = startStep + 1
        for step in firstStep...config.maxSteps {
            // Update current step FIRST (before stop/pause checks so checkpoint has correct step)
            currentStep = step
            trainingState?.currentStep = step

            // Check for stop request (from controller or internal flag)
            if shouldStop { break }
            if let ctrl = controller {
                if ctrl.shouldForceStop() {
                    print("\n⛔ Force stop requested")
                    break
                }
                if ctrl.shouldStop() {
                    print("\n🛑 Stop requested, saving checkpoint...")
                    // Save final checkpoint before stopping
                    try await saveCheckpoint(step: step, transformer: transformer, optimizer: optimizer)
                    break
                }
                // Check for pause - save checkpoint BEFORE entering pause
                if ctrl.shouldPause() {
                    print("\n⏸️  Pause requested, saving checkpoint first...")
                    let pauseCheckpointDir = config.outputDir.appendingPathComponent("checkpoint_\(String(format: "%06d", step))")
                    try await saveCheckpoint(step: step, transformer: transformer, optimizer: optimizer, isPauseCheckpoint: true)
                    print("   Checkpoint saved. You can safely quit or wait for resume.")

                    // Now wait while paused
                    if !ctrl.waitWhilePaused() {
                        // Stop was requested while paused - keep the checkpoint
                        break
                    }

                    // Resumed from pause - delete the pause checkpoint (only kept if stopped)
                    print("▶️  Resuming training, cleaning up pause checkpoint...")
                    try? FileManager.default.removeItem(at: pauseCheckpointDir)
                }
            }
            
            // Sample random batch
            let idx = Int.random(in: 0..<cachedLatents.count)
            let latentEntry = cachedLatents[idx]
            
            // Find matching embedding (by caption from latent filename)
            // Assume caption matches filename pattern
            let caption = findCaption(for: latentEntry.filename, in: cachedEmbeddings)
            guard let embeddingEntry = cachedEmbeddings[caption] else {
                print("Warning: No embedding for caption '\(caption)', skipping")
                continue
            }
            
            // Get preservation embedding for DOP (if enabled)
            let preservationEmb: MLXArray? = preservationEmbeddings[caption]?.embedding

            // Gradient accumulation: only update optimizer every N steps
            let shouldUpdate = step % config.gradientAccumulationSteps == 0 || step == config.maxSteps

            // Run training step
            let (mainLoss, dopLoss) = try trainStep(
                latent: latentEntry.latent,
                embedding: embeddingEntry.embedding,
                preservationEmbedding: preservationEmb,
                width: latentEntry.width,
                height: latentEntry.height,
                transformer: transformer,
                optimizer: optimizer,
                accumulatedGrads: &accumulatedGrads,
                shouldUpdate: shouldUpdate
            )
            
            // Update training state
            trainingState?.recordLoss(mainLoss)
            trainingState?.totalTrainingTime = Date().timeIntervalSince(trainingStartTime)

            // Log progress
            if step % config.logEveryNSteps == 0 || step == firstStep {
                var logLine = "Step \(step)/\(config.maxSteps) | Loss: \(String(format: "%.4f", mainLoss))"
                if let dop = dopLoss {
                    logLine += " | DOP: \(String(format: "%.4f", dop))"
                }
                // Add ETA
                if let state = trainingState, state.currentStep > firstStep {
                    let eta = state.estimatedTimeRemaining
                    if eta > 0 {
                        let etaMinutes = Int(eta / 60)
                        let etaHours = etaMinutes / 60
                        if etaHours > 0 {
                            logLine += " | ETA: \(etaHours)h\(etaMinutes % 60)m"
                        } else {
                            logLine += " | ETA: \(etaMinutes)m"
                        }
                    }
                }
                print(logLine)
                fflush(stdout)  // Force immediate output for non-interactive mode

                // Record loss for learning curve
                lossHistory.append((step: step, loss: mainLoss))

                // Generate learning curve SVG
                if config.generateLearningCurve && lossHistory.count >= 2 {
                    generateLearningCurveSVG()
                }
            }

            progressCallback?(step, config.maxSteps, mainLoss)
            controller?.notifyStepCompleted(step: step, totalSteps: config.maxSteps, loss: mainLoss)
            controller?.updateState(trainingState!)

            // Checkpoint (regular or on-demand)
            let isManualCheckpoint = controller?.shouldCheckpoint() ?? false
            let isScheduledCheckpoint = step % config.saveEveryNSteps == 0
            let shouldCheckpoint = isScheduledCheckpoint || isManualCheckpoint
            if shouldCheckpoint {
                try await saveCheckpoint(step: step, transformer: transformer, optimizer: optimizer)

                // Generate validation images using DISTILLED model (much better quality with few steps)
                // Also generate on manual checkpoint (not just scheduled validation intervals)
                let isScheduledValidation = step % config.validationEveryNSteps == 0
                if !config.validationPrompts.isEmpty,
                   (isScheduledValidation || isManualCheckpoint) {
                    let checkpointDir = config.outputDir.appendingPathComponent("checkpoint_\(String(format: "%06d", step))")
                    let checkpointPath = checkpointDir.appendingPathComponent("lora.safetensors")
                    try await generateValidationImages(
                        loraCheckpointPath: checkpointPath,
                        checkpointDir: checkpointDir,
                        step: step
                    )
                }
            }
        }
        
        // Save final weights
        let finalPath = config.outputDir.appendingPathComponent("lora_final.safetensors")
        try saveLoRAWeights(from: transformer, to: finalPath)
        print()
        print("Training complete!")
        print("Final weights saved to: \(finalPath.path)")
        
        // Analyze final weight magnitudes
        let loraParams = transformer.getLoRAParameters()
        var totalMagnitude: Float = 0
        var maxMagnitude: Float = 0
        var count = 0
        for (_, param) in loraParams {
            let absParam = MLX.abs(param)
            let mean = MLX.mean(absParam).item(Float.self)
            let max = MLX.max(absParam).item(Float.self)
            totalMagnitude += mean
            maxMagnitude = Swift.max(maxMagnitude, max)
            count += 1
        }
        let avgMagnitude = count > 0 ? totalMagnitude / Float(count) : 0
        print()
        print("Weight Analysis:")
        print("  Mean |delta_W|: \(String(format: "%.6f", avgMagnitude))")
        print("  Max |delta_W|: \(String(format: "%.6f", maxMagnitude))")
        print("  LoRA layers: \(count)")
        
        isRunning = false
    }
    
    // MARK: - Single Training Step
    
    private func trainStep(
        latent: MLXArray,
        embedding: MLXArray,
        preservationEmbedding: MLXArray?,  // DOP: preservation class embedding (trigger → class)
        width: Int,
        height: Int,
        transformer: Flux2Transformer2DModel,
        optimizer: AdamW,
        accumulatedGrads: inout NestedDictionary<String, MLXArray>?,
        shouldUpdate: Bool = true
    ) throws -> (mainLoss: Float, dopLoss: Float?) {
        // Get position IDs
        let resKey = "\(width)x\(height)"
        guard let posIds = positionIdCache[resKey] else {
            throw SimpleLoRATrainerError.noPositionIds(resKey)
        }
        
        // Ensure batch dimension - latent comes as [C, H, W] from cache
        var batchedLatent = latent
        if batchedLatent.shape.count == 3 {
            batchedLatent = batchedLatent.expandedDimensions(axis: 0)  // [1, C, H, W]
        }
        var batchedEmbedding = embedding
        if batchedEmbedding.shape.count == 2 {
            batchedEmbedding = batchedEmbedding.expandedDimensions(axis: 0)
        }
        
        let batchSize = batchedLatent.shape[0]
        
        // Sample timestep (balanced = 50/50 content/style)
        let timesteps = sampleTimesteps(batchSize: batchSize)
        let sigmas = timesteps.asType(.float32) / 1000.0
        
        // Create noisy latent (flow matching interpolation)
        let noise = MLXRandom.normal(batchedLatent.shape)
        let sigmasExpanded = sigmas.reshaped([batchSize, 1, 1, 1])
        let noisyLatent = (1 - sigmasExpanded) * batchedLatent + sigmasExpanded * noise
        
        // Pack for transformer
        let packedLatent = packLatentsForTransformer(noisyLatent, patchSize: 2)
        
        // Velocity target: v = noise - latent (CRITICAL for flow matching!)
        let velocityTarget = noise - batchedLatent
        let packedVelocityTarget = packLatentsForTransformer(velocityTarget, patchSize: 2)
        
        // Guidance embedding (Dev model uses guidance, Klein models do NOT)
        // IMPORTANT: For training, use guidance_scale=1.0 (not 4.0 used in inference)
        // This is because Dev is guidance-distilled - training at scale=1.0 keeps the LoRA
        // from interfering with the baked-in guidance behavior (ref: Ostris ai-toolkit)
        let guidance: MLXArray? = modelType.usesGuidanceEmbeds ?
            MLXArray.full([batchSize], values: MLXArray(Float(1.0))) : nil
        
        // Prepare input arrays
        // CRITICAL: Pass sigmas (in [0, 1] range), NOT timesteps (in [0, 1000] range)!
        // The transformer internally scales timesteps by 1000, so we need [0, 1] input.
        var inputArrays: [MLXArray] = [
            packedLatent,
            batchedEmbedding,
            sigmas,  // Pass sigma [0, 1], NOT timesteps [0, 1000]!
            posIds.imgIds.asType(.int32),
            posIds.txtIds.asType(.int32),
            packedVelocityTarget
        ]
        inputArrays.append(guidance ?? MLXArray(0.0))
        
        // Forward + backward
        guard let lossAndGrad = cachedLossAndGrad else {
            throw SimpleLoRATrainerError.notInitialized
        }
        
        // Timing: Main forward + backward
        let t0 = CFAbsoluteTimeGetCurrent()
        let (losses, grads) = lossAndGrad(transformer, inputArrays)
        let mainLoss = losses[0]
        let t1 = CFAbsoluteTimeGetCurrent()
        
        // Filter to LoRA gradients only
        var filteredGrads = filterLoRAGradients(grads)
        let t2 = CFAbsoluteTimeGetCurrent()
        
        // Evaluate main loss and gradients to free backward graph before DOP
        let mainLossValue = mainLoss.item(Float.self)
        eval(filteredGrads)
        let t3 = CFAbsoluteTimeGetCurrent()
        MLX.Memory.clearCache()
        let t4 = CFAbsoluteTimeGetCurrent()
        
        // Log timing for first 5 steps
        if currentStep <= 5 {
            Flux2Debug.log("[Timing] Step \(currentStep): fwd+bwd=\(String(format: "%.2f", (t1-t0)*1000))ms, filter=\(String(format: "%.2f", (t2-t1)*1000))ms, eval=\(String(format: "%.2f", (t3-t2)*1000))ms, clear=\(String(format: "%.2f", (t4-t3)*1000))ms")
        }
        
        // DOP (Differential Output Preservation) - Ostris regularization
        // OPTIMIZATION: Only do DOP every N steps to reduce overhead
        var dopLossValue: Float? = nil
        var dopTiming: (base: Double, fwdBwd: Double, combine: Double)? = nil
        let shouldDoDOP = currentStep % config.dopEveryNSteps == 0
        if shouldDoDOP,
           let preservationEmb = preservationEmbedding,
           let dopLossAndGrad = cachedDOPLossAndGrad {
            
            let dopT0 = CFAbsoluteTimeGetCurrent()
            
            // Ensure batch dimension for preservation embedding
            var batchedPreservationEmb = preservationEmb
            if batchedPreservationEmb.shape.count == 2 {
                batchedPreservationEmb = batchedPreservationEmb.expandedDimensions(axis: 0)
            }
            
            // 1. Get base model output (LoRA disabled)
            // OPTIMIZATION: Use stopGradient to prevent unnecessary gradient tracking
            transformer.disableLoRA()  // Disable LoRA (sets all scales to 0)
            let baseOutputRaw = transformer(
                hiddenStates: packedLatent,
                encoderHiddenStates: batchedPreservationEmb,
                timestep: sigmas,
                guidance: guidance,
                imgIds: posIds.imgIds.asType(.int32),
                txtIds: posIds.txtIds.asType(.int32)
            )
            // Stop gradient tracking - base output is just a target, no backprop needed
            let baseOutput = stopGradient(baseOutputRaw)
            eval(baseOutput)
            transformer.restoreLoRAScales()  // Re-enable LoRA
            let dopT1 = CFAbsoluteTimeGetCurrent()
            
            // 2. Compute DOP loss and gradients
            var dopInputArrays: [MLXArray] = [
                packedLatent,
                batchedPreservationEmb,
                sigmas,
                posIds.imgIds.asType(.int32),
                posIds.txtIds.asType(.int32),
                baseOutput
            ]
            dopInputArrays.append(guidance ?? MLXArray(0.0))
            
            let (dopLosses, dopGrads) = dopLossAndGrad(transformer, dopInputArrays)
            let dopLoss = dopLosses[0]
            dopLossValue = dopLoss.item(Float.self)
            
            // Filter DOP gradients
            let filteredDOPGrads = filterLoRAGradients(dopGrads)
            eval(filteredDOPGrads)
            MLX.Memory.clearCache()
            let dopT2 = CFAbsoluteTimeGetCurrent()
            
            // 3. Combine gradients: total = main + multiplier * dop
            let dopMultiplier = config.dopMultiplier
            filteredGrads = combineGradients(filteredGrads, filteredDOPGrads, multiplier: dopMultiplier)
            let dopT3 = CFAbsoluteTimeGetCurrent()
            
            dopTiming = (base: (dopT1-dopT0)*1000, fwdBwd: (dopT2-dopT1)*1000, combine: (dopT3-dopT2)*1000)
        }
        
        // Timing: Optimizer update
        let optT0 = CFAbsoluteTimeGetCurrent()

        // Gradient accumulation: add to accumulated gradients
        if accumulatedGrads == nil {
            accumulatedGrads = filteredGrads
        } else {
            // Add current gradients to accumulated
            accumulatedGrads = addGradients(accumulatedGrads!, filteredGrads)
        }

        var optT1 = optT0
        var optT2 = optT0
        var optT3 = optT0
        var optT4 = optT0

        // Only update optimizer if shouldUpdate is true
        if shouldUpdate {
            // Debug: compute gradient norm before clipping (every 50 steps)
            if currentStep % 50 == 1 || currentStep <= 5 {
                let gradNormBeforeClip = computeGradNorm(accumulatedGrads!)
                Flux2Debug.log("[Grad Debug] Step \(currentStep): grad_norm_before_clip = \(String(format: "%.4f", gradNormBeforeClip))")
            }

            // Scale gradients by accumulation steps (average)
            let scaledGrads = scaleGradients(accumulatedGrads!, scale: 1.0 / Float(config.gradientAccumulationSteps))

            // Gradient clipping to prevent NaN (max norm = 1.0)
            let clippedGrads = clipGradNorm(scaledGrads, maxNorm: 1.0)
            optT1 = CFAbsoluteTimeGetCurrent()

            // Optimizer update
            optimizer.update(model: transformer, gradients: clippedGrads)
            optT2 = CFAbsoluteTimeGetCurrent()

            // CRITICAL: Evaluate LoRA parameters to prevent lazy graph accumulation
            eval(transformer.trainableParameters())
            optT3 = CFAbsoluteTimeGetCurrent()

            // Clear GPU cache
            MLX.Memory.clearCache()
            optT4 = CFAbsoluteTimeGetCurrent()

            // Reset accumulated gradients
            accumulatedGrads = nil
        }

        // Log timing for first 5 steps
        if currentStep <= 5 {
            var timingMsg = "[Timing] Step \(currentStep): clip=\(String(format: "%.1f", (optT1-optT0)*1000))ms, opt=\(String(format: "%.1f", (optT2-optT1)*1000))ms, eval=\(String(format: "%.1f", (optT3-optT2)*1000))ms, clear=\(String(format: "%.1f", (optT4-optT3)*1000))ms"
            if let dop = dopTiming {
                timingMsg += ", DOP[base=\(String(format: "%.1f", dop.base))ms, fwd=\(String(format: "%.1f", dop.fwdBwd))ms, comb=\(String(format: "%.1f", dop.combine))ms]"
            }
            if !shouldUpdate {
                timingMsg += " (accumulating)"
            }
            Flux2Debug.log(timingMsg)
        }

        return (mainLoss: mainLossValue, dopLoss: dopLossValue)
    }
    
    // MARK: - Timestep Sampling
    
    private func sampleTimesteps(batchSize: Int) -> MLXArray {
        switch config.timestepSampling {
        case .uniform:
            return MLXRandom.randInt(low: 0, high: 1000, [batchSize])
            
        case .content:
            // t^3 - favors low timesteps (fine details)
            let u = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize])
            let cubic = u * u * u
            let scaled = cubic * 1000.0
            return MLX.clip(scaled, min: 0, max: 999).asType(.int32)
            
        case .style:
            // (1-t^3) - favors high timesteps (global structure)
            let u = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize])
            let cubic = u * u * u
            let inverted = 1.0 - cubic
            let scaled = inverted * 1000.0
            return MLX.clip(scaled, min: 0, max: 999).asType(.int32)
            
        case .balanced:
            // 50/50 mix of content and style (Ostris default)
            let u = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize])
            let cubic = u * u * u
            let styleMask = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize]) .> 0.5
            let contentTimesteps = cubic * 1000.0
            let styleTimesteps = (1.0 - cubic) * 1000.0
            let selected = MLX.where(styleMask, styleTimesteps, contentTimesteps)
            return MLX.clip(selected, min: 0, max: 999).asType(.int32)
        }
    }
    
    // MARK: - Helpers
    
    private func packLatentsForTransformer(_ latents: MLXArray, patchSize: Int) -> MLXArray {
        let shape = latents.shape
        let batchSize = shape[0]
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        
        let patchH = height / patchSize
        let patchW = width / patchSize
        let patchFeatures = channels * patchSize * patchSize
        
        let reshaped = latents.reshaped([batchSize, channels, patchH, patchSize, patchW, patchSize])
        let transposed = reshaped.transposed(0, 2, 4, 1, 3, 5)
        let packed = transposed.reshaped([batchSize, patchH * patchW, patchFeatures])
        
        return packed
    }
    
    /// Compute total gradient norm (for debugging)
    private func computeGradNorm(_ grads: NestedDictionary<String, MLXArray>) -> Float {
        var allGrads: [MLXArray] = []
        func collectGrads(_ item: NestedItem<String, MLXArray>) {
            switch item {
            case .none: break
            case .value(let arr): allGrads.append(arr)
            case .array(let items): items.forEach { collectGrads($0) }
            case .dictionary(let dict): dict.values.forEach { collectGrads($0) }
            }
        }
        for (_, item) in grads { collectGrads(item) }
        
        guard !allGrads.isEmpty else { return 0.0 }
        
        var totalNormSq = MLXArray(0.0)
        for grad in allGrads {
            totalNormSq = totalNormSq + MLX.sum(grad * grad)
        }
        let totalNorm = MLX.sqrt(totalNormSq)
        eval(totalNorm)
        return totalNorm.item(Float.self)
    }
    
    /// Clip gradient norm to prevent exploding gradients
    private func clipGradNorm(_ grads: NestedDictionary<String, MLXArray>, maxNorm: Float) -> NestedDictionary<String, MLXArray> {
        // Collect all gradients
        var allGrads: [MLXArray] = []
        func collectGrads(_ item: NestedItem<String, MLXArray>) {
            switch item {
            case .none: break
            case .value(let arr): allGrads.append(arr)
            case .array(let items): items.forEach { collectGrads($0) }
            case .dictionary(let dict): dict.values.forEach { collectGrads($0) }
            }
        }
        for (_, item) in grads { collectGrads(item) }
        
        guard !allGrads.isEmpty else { return grads }
        
        // Compute total norm
        var totalNormSq = MLXArray(0.0)
        for grad in allGrads {
            totalNormSq = totalNormSq + MLX.sum(grad * grad)
        }
        let totalNorm = MLX.sqrt(totalNormSq)
        
        // Compute clip coefficient
        let maxNormArr = MLXArray(maxNorm)
        let clipCoef = MLX.minimum(maxNormArr / (totalNorm + MLXArray(1e-6)), MLXArray(1.0))
        eval(clipCoef)
        
        // Apply clipping
        func clipItem(_ item: NestedItem<String, MLXArray>) -> NestedItem<String, MLXArray> {
            switch item {
            case .none: return .none
            case .value(let arr): return .value(arr * clipCoef)
            case .array(let items): return .array(items.map { clipItem($0) })
            case .dictionary(let dict):
                var newDict: [String: NestedItem<String, MLXArray>] = [:]
                for (k, v) in dict { newDict[k] = clipItem(v) }
                return .dictionary(newDict)
            }
        }
        
        var clipped = NestedDictionary<String, MLXArray>()
        for (key, item) in grads {
            clipped[key] = clipItem(item)
        }
        return clipped
    }
    
    private func filterLoRAGradients(_ grads: NestedDictionary<String, MLXArray>) -> NestedDictionary<String, MLXArray> {
        var filtered = NestedDictionary<String, MLXArray>()
        
        func filterRecursively(_ item: NestedItem<String, MLXArray>, path: String) -> NestedItem<String, MLXArray>? {
            switch item {
            case .none:
                return nil
            case .value(let arr):
                // Only keep LoRA parameters
                if path.contains("lora") {
                    return .value(arr)
                }
                return nil
            case .array(let items):
                let filteredItems = items.enumerated().compactMap { idx, subItem in
                    filterRecursively(subItem, path: "\(path)[\(idx)]")
                }
                return filteredItems.isEmpty ? nil : .array(filteredItems)
            case .dictionary(let dict):
                var filteredDict: [String: NestedItem<String, MLXArray>] = [:]
                for (key, subItem) in dict {
                    if let filtered = filterRecursively(subItem, path: "\(path)/\(key)") {
                        filteredDict[key] = filtered
                    }
                }
                return filteredDict.isEmpty ? nil : .dictionary(filteredDict)
            }
        }
        
        for (key, item) in grads {
            if let filteredItem = filterRecursively(item, path: key) {
                filtered[key] = filteredItem
            }
        }
        
        return filtered
    }
    
    /// Combine main gradients with DOP gradients: total = main + multiplier * dop
    private func combineGradients(
        _ mainGrads: NestedDictionary<String, MLXArray>,
        _ dopGrads: NestedDictionary<String, MLXArray>,
        multiplier: Float
    ) -> NestedDictionary<String, MLXArray> {
        var combined = NestedDictionary<String, MLXArray>()
        
        func combineRecursively(
            _ mainItem: NestedItem<String, MLXArray>?,
            _ dopItem: NestedItem<String, MLXArray>?
        ) -> NestedItem<String, MLXArray>? {
            switch (mainItem, dopItem) {
            case (.none, .none):
                return nil
            case (.some(let main), .none):
                return main
            case (.none, .some(let dop)):
                // Scale DOP gradient
                if case .value(let dopArr) = dop {
                    return .value(multiplier * dopArr)
                }
                return dop
            case (.some(.value(let mainArr)), .some(.value(let dopArr))):
                // Combine: main + multiplier * dop
                return .value(mainArr + multiplier * dopArr)
            case (.some(.array(let mainItems)), .some(.array(let dopItems))):
                var combinedItems: [NestedItem<String, MLXArray>] = []
                let maxLen = max(mainItems.count, dopItems.count)
                for i in 0..<maxLen {
                    let mainSub = i < mainItems.count ? mainItems[i] : nil
                    let dopSub = i < dopItems.count ? dopItems[i] : nil
                    if let combined = combineRecursively(mainSub, dopSub) {
                        combinedItems.append(combined)
                    }
                }
                return combinedItems.isEmpty ? nil : .array(combinedItems)
            case (.some(.dictionary(let mainDict)), .some(.dictionary(let dopDict))):
                var combinedDict: [String: NestedItem<String, MLXArray>] = [:]
                let allKeys = Set(mainDict.keys).union(dopDict.keys)
                for key in allKeys {
                    if let combined = combineRecursively(mainDict[key], dopDict[key]) {
                        combinedDict[key] = combined
                    }
                }
                return combinedDict.isEmpty ? nil : .dictionary(combinedDict)
            default:
                // Type mismatch - just return main
                return mainItem
            }
        }
        
        let allKeys = Set(mainGrads.keys).union(dopGrads.keys)
        for key in allKeys {
            if let combinedItem = combineRecursively(mainGrads[key], dopGrads[key]) {
                combined[key] = combinedItem
            }
        }
        
        return combined
    }

    /// Add two gradient dictionaries element-wise
    private func addGradients(
        _ grads1: NestedDictionary<String, MLXArray>,
        _ grads2: NestedDictionary<String, MLXArray>
    ) -> NestedDictionary<String, MLXArray> {
        var result = NestedDictionary<String, MLXArray>()

        func addRecursively(
            _ item1: NestedItem<String, MLXArray>?,
            _ item2: NestedItem<String, MLXArray>?
        ) -> NestedItem<String, MLXArray>? {
            switch (item1, item2) {
            case (.none, .none):
                return nil
            case (.some(let i1), .none):
                return i1
            case (.none, .some(let i2)):
                return i2
            case (.some(.value(let arr1)), .some(.value(let arr2))):
                return .value(arr1 + arr2)
            case (.some(.array(let items1)), .some(.array(let items2))):
                var addedItems: [NestedItem<String, MLXArray>] = []
                let maxLen = max(items1.count, items2.count)
                for i in 0..<maxLen {
                    let sub1 = i < items1.count ? items1[i] : nil
                    let sub2 = i < items2.count ? items2[i] : nil
                    if let added = addRecursively(sub1, sub2) {
                        addedItems.append(added)
                    }
                }
                return addedItems.isEmpty ? nil : .array(addedItems)
            case (.some(.dictionary(let dict1)), .some(.dictionary(let dict2))):
                var addedDict: [String: NestedItem<String, MLXArray>] = [:]
                let allKeys = Set(dict1.keys).union(dict2.keys)
                for key in allKeys {
                    if let added = addRecursively(dict1[key], dict2[key]) {
                        addedDict[key] = added
                    }
                }
                return addedDict.isEmpty ? nil : .dictionary(addedDict)
            default:
                return item1
            }
        }

        let allKeys = Set(grads1.keys).union(grads2.keys)
        for key in allKeys {
            if let addedItem = addRecursively(grads1[key], grads2[key]) {
                result[key] = addedItem
            }
        }

        return result
    }

    /// Scale all gradients by a constant factor
    private func scaleGradients(
        _ grads: NestedDictionary<String, MLXArray>,
        scale: Float
    ) -> NestedDictionary<String, MLXArray> {
        let scaleArr = MLXArray(scale)

        func scaleItem(_ item: NestedItem<String, MLXArray>) -> NestedItem<String, MLXArray> {
            switch item {
            case .none: return .none
            case .value(let arr): return .value(arr * scaleArr)
            case .array(let items): return .array(items.map { scaleItem($0) })
            case .dictionary(let dict):
                var scaledDict: [String: NestedItem<String, MLXArray>] = [:]
                for (k, v) in dict { scaledDict[k] = scaleItem(v) }
                return .dictionary(scaledDict)
            }
        }

        var scaled = NestedDictionary<String, MLXArray>()
        for (key, item) in grads {
            scaled[key] = scaleItem(item)
        }
        return scaled
    }

    private func findCaption(for filename: String, in embeddings: [String: CachedEmbeddingEntry]) -> String {
        // Try exact match first
        if embeddings[filename] != nil { return filename }
        
        // Try without extension
        let baseName = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent
        if embeddings[baseName] != nil { return baseName }
        
        // Return first key as fallback (single caption dataset)
        return embeddings.keys.first ?? ""
    }
    
    // MARK: - Save/Load
    
    private func saveLoRAWeights(from transformer: Flux2Transformer2DModel, to path: URL) throws {
        // Use the transformer's built-in method to get LoRA parameters
        // This properly transposes weights for compatibility with other tools
        let loraWeights = transformer.getLoRAParameters()
        
        guard !loraWeights.isEmpty else {
            throw SimpleLoRATrainerError.notInitialized
        }
        
        // Create metadata with LoRA configuration
        // This allows loaders to compute the correct scale: scale = alpha / rank
        let metadata: [String: String] = [
            "lora_rank": "\(config.rank)",
            "lora_alpha": "\(config.alpha)",
            "format": "diffusers",
            "target_model": modelType.rawValue,
            "software": "flux2-swift-mlx"
        ]
        
        // Save using MLX safetensors with metadata
        try save(arrays: loraWeights, metadata: metadata, url: path)
        
        Flux2Debug.log("[SimpleLoRATrainer] Saved \(loraWeights.count) LoRA weights to \(path.lastPathComponent)")
    }

    // MARK: - Checkpoint Management

    /// Save a complete checkpoint (LoRA weights, optimizer state, training state)
    /// - Parameters:
    ///   - step: Current training step
    ///   - transformer: The transformer model with LoRA weights
    ///   - optimizer: The optimizer with momentum/variance state
    ///   - isPauseCheckpoint: If true, marks this as a temporary pause checkpoint (will be deleted on resume)
    private func saveCheckpoint(
        step: Int,
        transformer: Flux2Transformer2DModel,
        optimizer: ResumableAdamW,
        isPauseCheckpoint: Bool = false
    ) async throws {
        // Create checkpoint subdirectory
        let checkpointDir = config.outputDir.appendingPathComponent(
            "checkpoint_\(String(format: "%06d", step))"
        )
        try FileManager.default.createDirectory(at: checkpointDir, withIntermediateDirectories: true)

        // 1. Save LoRA weights
        let loraPath = checkpointDir.appendingPathComponent("lora.safetensors")
        try saveLoRAWeights(from: transformer, to: loraPath)

        // 2. Save optimizer state
        let optimizerPath = checkpointDir.appendingPathComponent("optimizer_state.safetensors")
        try saveOptimizerState(optimizer: optimizer, to: optimizerPath)

        // 3. Save training state
        trainingState?.recordCheckpoint(step: step)
        trainingState?.optimizerStatePath = "optimizer_state.safetensors"
        let statePath = checkpointDir.appendingPathComponent("training_state.json")
        try trainingState?.save(to: statePath)

        // 4. Mark as pause checkpoint if applicable (will be deleted on resume)
        if isPauseCheckpoint {
            let markerPath = checkpointDir.appendingPathComponent(".pause_checkpoint")
            FileManager.default.createFile(atPath: markerPath.path, contents: nil)
        }

        let checkpointType = isPauseCheckpoint ? "pause " : ""
        print("  💾 \(checkpointType.capitalized)Checkpoint saved: checkpoint_\(String(format: "%06d", step))/")
        fflush(stdout)

        // Notify controller
        controller?.notifyCheckpointSaved(step: step, path: checkpointDir)
    }

    /// Save optimizer state (AdamW momentum and variance)
    private func saveOptimizerState(optimizer: ResumableAdamW, to path: URL) throws {
        // Use ResumableAdamW's saveState() to get all momentum/variance tensors
        let stateArrays = optimizer.saveState()

        // Evaluate before saving to ensure all arrays are computed
        eval(Array(stateArrays.values))

        try save(arrays: stateArrays, url: path)
        Flux2Debug.log("[SimpleLoRATrainer] Saved optimizer state (\(stateArrays.count) arrays) to \(path.lastPathComponent)")
    }

    /// Load optimizer state
    private func loadOptimizerState(optimizer: ResumableAdamW, from path: URL) throws {
        // Load state arrays
        let stateArrays = try loadArrays(url: path)

        // Restore optimizer state
        try optimizer.restoreState(from: stateArrays)
        Flux2Debug.log("[SimpleLoRATrainer] Loaded optimizer state from checkpoint")
    }

    /// Load LoRA weights from a checkpoint file into the transformer
    /// Note: Weights are stored transposed (inference format), need to transpose back for training
    private func loadLoRACheckpoint(transformer: Flux2Transformer2DModel, from path: URL) throws {
        let weights = try loadArrays(url: path)

        // Iterate through saved weights and update the transformer's LoRA layers
        // The saved format uses keys like: transformer_blocks.0.attn.to_q.lora_A.weight
        // We need to find the corresponding LoRAInjectedLinear and update its loraA/loraB

        for (key, array) in weights {
            // Skip non-weight keys
            guard key.contains("lora_") else { continue }

            // Transpose back from inference format to training format
            let transposedArray = array.T

            // Parse the key to find the corresponding layer
            // Example: transformer_blocks.0.attn.to_q.lora_A.weight
            let isLoraA = key.contains("lora_A")

            // Find and update the layer using the transformer's update mechanism
            // The LoRA layers are already created by applyLoRA, we just need to update their weights
            if let loraLinear = findLoRALayer(in: transformer, forKey: key) {
                if isLoraA {
                    loraLinear.loraA = transposedArray
                } else {
                    loraLinear.loraB = transposedArray
                }
            }
        }

        // Evaluate to ensure weights are loaded
        eval(transformer.trainableParameters())
        Flux2Debug.log("[SimpleLoRATrainer] Loaded \(weights.count) LoRA weights from checkpoint")
    }

    /// Find the LoRAInjectedLinear layer corresponding to a weight key
    private func findLoRALayer(in transformer: Flux2Transformer2DModel, forKey key: String) -> LoRAInjectedLinear? {
        // Parse key like: transformer_blocks.0.attn.to_q.lora_A.weight
        let components = key.split(separator: ".")

        // Handle transformer_blocks
        if key.hasPrefix("transformer_blocks") {
            guard components.count >= 4,
                  let blockIdx = Int(components[1]) else { return nil }
            guard blockIdx < transformer.transformerBlocks.count else { return nil }

            let block = transformer.transformerBlocks[blockIdx]
            let layerName = String(components[3])

            // Attention layers
            switch layerName {
            case "to_q": return block.attn.toQ as? LoRAInjectedLinear
            case "to_k": return block.attn.toK as? LoRAInjectedLinear
            case "to_v": return block.attn.toV as? LoRAInjectedLinear
            case "add_q_proj": return block.attn.addQProj as? LoRAInjectedLinear
            case "add_k_proj": return block.attn.addKProj as? LoRAInjectedLinear
            case "add_v_proj": return block.attn.addVProj as? LoRAInjectedLinear
            case "to_out": return block.attn.toOut as? LoRAInjectedLinear
            case "to_add_out": return block.attn.toAddOut as? LoRAInjectedLinear
            default: break
            }

            // FFN layers
            if key.contains("ff.activation.proj") {
                return block.ff.activation.proj as? LoRAInjectedLinear
            }
            if key.contains("ff.linear_out") {
                return block.ff.linearOut as? LoRAInjectedLinear
            }
            if key.contains("ff_context.activation.proj") {
                return block.ffContext.activation.proj as? LoRAInjectedLinear
            }
            if key.contains("ff_context.linear_out") {
                return block.ffContext.linearOut as? LoRAInjectedLinear
            }
        }

        // Handle single_transformer_blocks
        if key.hasPrefix("single_transformer_blocks") {
            guard components.count >= 4,
                  let blockIdx = Int(components[1]) else { return nil }
            guard blockIdx < transformer.singleTransformerBlocks.count else { return nil }

            let block = transformer.singleTransformerBlocks[blockIdx]
            let layerName = String(components[3])

            switch layerName {
            case "to_qkv_mlp": return block.attn.toQkvMlp as? LoRAInjectedLinear
            case "to_out": return block.attn.toOut as? LoRAInjectedLinear
            default: break
            }
        }

        return nil
    }

    // MARK: - Baseline Image Generation (no LoRA)

    /// Generate baseline images WITHOUT LoRA for comparison
    /// Called at the beginning of training before any LoRA weights are applied
    private func generateBaselineImages() async throws {
        print("Generating baseline images (no LoRA)...")
        fflush(stdout)

        // Create baseline subdirectory
        let baselineDir = config.outputDir.appendingPathComponent("baseline")
        try FileManager.default.createDirectory(at: baselineDir, withIntermediateDirectories: true)

        // Create pipeline with distilled model
        let pipeline = Flux2Pipeline(
            model: modelType,
            quantization: .balanced
        )

        // Generate images for each validation prompt
        for (index, promptConfig) in config.validationPrompts.enumerated() {
            // Build the actual prompt (with or without trigger)
            let prompt: String
            if promptConfig.applyTrigger, let trigger = config.triggerWord {
                prompt = "\(trigger), \(promptConfig.prompt)"
            } else {
                prompt = promptConfig.prompt
            }

            let triggerSuffix = promptConfig.applyTrigger ? "_trigger" : "_notrigger"
            // Use per-prompt seed if available, otherwise global seed
            let seed = promptConfig.seed ?? config.validationSeed

            // Generate 512x512 if requested
            if promptConfig.is512 {
                print("  [\(index + 1)/\(config.validationPrompts.count)] 512x512\(triggerSuffix): \(promptConfig.prompt.prefix(40))...")
                fflush(stdout)
                let result = try await pipeline.generateTextToImageWithResult(
                    prompt: prompt,
                    height: 512,
                    width: 512,
                    steps: config.validationSteps,
                    guidance: 1.0,
                    seed: seed
                )
                let imagePath = baselineDir.appendingPathComponent("prompt_\(index)\(triggerSuffix)_512x512.png")
                try saveImage(result.image, to: imagePath)
                print("    Saved: baseline/\(imagePath.lastPathComponent)")
            }

            // Generate 1024x1024 if requested
            if promptConfig.is1024 {
                print("  [\(index + 1)/\(config.validationPrompts.count)] 1024x1024\(triggerSuffix): \(promptConfig.prompt.prefix(40))...")
                fflush(stdout)
                let result = try await pipeline.generateTextToImageWithResult(
                    prompt: prompt,
                    height: 1024,
                    width: 1024,
                    steps: config.validationSteps,
                    guidance: 1.0,
                    seed: seed
                )
                let imagePath = baselineDir.appendingPathComponent("prompt_\(index)\(triggerSuffix)_1024x1024.png")
                try saveImage(result.image, to: imagePath)
                print("    Saved: baseline/\(imagePath.lastPathComponent)")
            }
        }

        print()
        MLX.Memory.clearCache()
        fflush(stdout)
    }

    // MARK: - Validation Image Generation (using DISTILLED model)

    /// Generate validation images using DISTILLED model via Flux2Pipeline
    /// This produces better results with few steps since distilled model is optimized for fast inference
    /// WARNING: This loads a second copy of the model - may cause OOM on large models (Klein 9B, Dev)
    private func generateValidationImages(
        loraCheckpointPath: URL,
        checkpointDir: URL,
        step: Int
    ) async throws {
        print("  Generating validation images...")
        fflush(stdout)

        // Create pipeline - this loads a SECOND copy of the model!
        // For large models, consider disabling validation (validation.prompts: [])
        var pipeline: Flux2Pipeline? = Flux2Pipeline(
            model: modelType,
            quantization: .balanced
        )

        // Load LoRA from checkpoint we just saved
        let loraConfig = LoRAConfig(
            filePath: loraCheckpointPath.path,
            scale: 1.0
        )
        try pipeline!.loadLoRA(loraConfig)

        // Generate images for each validation prompt
        for (index, promptConfig) in config.validationPrompts.enumerated() {
            // Build the actual prompt (with or without trigger)
            let prompt: String
            if promptConfig.applyTrigger, let trigger = config.triggerWord {
                prompt = "\(trigger), \(promptConfig.prompt)"
            } else {
                prompt = promptConfig.prompt
            }

            let triggerSuffix = promptConfig.applyTrigger ? "_trigger" : "_notrigger"
            // Use per-prompt seed if available, otherwise global seed
            let seed = promptConfig.seed ?? config.validationSeed

            // Generate 512x512 if requested
            if promptConfig.is512 {
                print("    [\(index + 1)/\(config.validationPrompts.count)] 512x512\(triggerSuffix)...")
                fflush(stdout)
                let result = try await pipeline!.generateTextToImageWithResult(
                    prompt: prompt,
                    height: 512,
                    width: 512,
                    steps: config.validationSteps,
                    guidance: 1.0,
                    seed: seed
                )
                // Save to checkpoint subdirectory
                let imagePath = checkpointDir.appendingPathComponent("prompt_\(index)\(triggerSuffix)_512x512.png")
                try saveImage(result.image, to: imagePath)
                print("      Saved: \(imagePath.lastPathComponent)")
            }

            // Generate 1024x1024 if requested
            if promptConfig.is1024 {
                print("    [\(index + 1)/\(config.validationPrompts.count)] 1024x1024\(triggerSuffix)...")
                fflush(stdout)
                let result = try await pipeline!.generateTextToImageWithResult(
                    prompt: prompt,
                    height: 1024,
                    width: 1024,
                    steps: config.validationSteps,
                    guidance: 1.0,
                    seed: seed
                )
                // Save to checkpoint subdirectory
                let imagePath = checkpointDir.appendingPathComponent("prompt_\(index)\(triggerSuffix)_1024x1024.png")
                try saveImage(result.image, to: imagePath)
                print("      Saved: \(imagePath.lastPathComponent)")
            }
        }

        // CRITICAL: Explicitly unload pipeline and force memory cleanup
        // Without this, the second model copy stays in memory!
        print("    Unloading validation pipeline...")
        await pipeline!.clearAll()
        pipeline = nil

        // Force GPU sync and memory cleanup
        eval([])
        MLX.Memory.clearCache()
        eval([])
        MLX.Memory.clearCache()

        // Give the system time to reclaim memory
        try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds
        MLX.Memory.clearCache()

        print("    Validation complete, memory released")
        fflush(stdout)
    }
    
    private func saveImage(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw SimpleLoRATrainerError.imageCreationFailed
        }
        
        CGImageDestinationAddImage(destination, image, nil)
        
        guard CGImageDestinationFinalize(destination) else {
            throw SimpleLoRATrainerError.imageCreationFailed
        }
    }
    
    // MARK: - Control
    
    public func stop() {
        shouldStop = true
    }

    public var running: Bool {
        isRunning
    }

    // MARK: - Learning Curve Generation

    /// Generate an SVG learning curve from loss history
    /// Overwrites the file each time (no history needed)
    private func generateLearningCurveSVG() {
        guard lossHistory.count >= 2 else { return }

        let svgPath = config.outputDir.appendingPathComponent("learning_curve.svg")

        // SVG dimensions
        let width: Float = 800
        let height: Float = 400
        let padding: Float = 60
        let plotWidth = width - 2 * padding
        let plotHeight = height - 2 * padding

        // Compute smoothed loss using moving average
        let windowSize = min(config.learningCurveSmoothingWindow, lossHistory.count)
        var smoothedLoss: [(step: Int, loss: Float)] = []

        for i in 0..<lossHistory.count {
            let start = max(0, i - windowSize + 1)
            let window = lossHistory[start...i]
            let avgLoss = window.map { $0.loss }.reduce(0, +) / Float(window.count)
            smoothedLoss.append((step: lossHistory[i].step, loss: avgLoss))
        }

        // Find min/max for scaling
        let minStep = Float(lossHistory.first!.step)
        let maxStep = Float(lossHistory.last!.step)
        let stepRange = max(maxStep - minStep, 1)

        // Use raw loss values for scaling (they have larger range than smoothed)
        // This ensures raw data points don't go outside the plot area
        let rawLosses = lossHistory.map { $0.loss }
        let minLoss = rawLosses.min() ?? 0
        let maxLoss = rawLosses.max() ?? 1
        let lossRange = max(maxLoss - minLoss, 0.001)

        // Add 10% padding to loss range
        let paddedMinLoss = minLoss - lossRange * 0.1
        let paddedMaxLoss = maxLoss + lossRange * 0.1
        let paddedLossRange = paddedMaxLoss - paddedMinLoss

        // Helper to convert data to SVG coordinates
        func toSVG(step: Int, loss: Float) -> (x: Float, y: Float) {
            let x = padding + (Float(step) - minStep) / stepRange * plotWidth
            let y = padding + (1 - (loss - paddedMinLoss) / paddedLossRange) * plotHeight
            return (x, y)
        }

        // Build SVG path for smoothed curve
        var pathData = ""
        for (i, point) in smoothedLoss.enumerated() {
            let (x, y) = toSVG(step: point.step, loss: point.loss)
            if i == 0 {
                pathData += "M \(x) \(y)"
            } else {
                pathData += " L \(x) \(y)"
            }
        }

        // Build raw data points (lighter, for reference)
        var rawPathData = ""
        for (i, point) in lossHistory.enumerated() {
            let (x, y) = toSVG(step: point.step, loss: point.loss)
            if i == 0 {
                rawPathData += "M \(x) \(y)"
            } else {
                rawPathData += " L \(x) \(y)"
            }
        }

        // Generate axis labels
        let stepLabels = generateAxisLabels(min: minStep, max: maxStep, count: 5)
        let lossLabels = generateAxisLabels(min: paddedMinLoss, max: paddedMaxLoss, count: 5)

        // Build SVG
        var svg = """
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 \(width) \(height)" width="\(Int(width))" height="\(Int(height))">
          <style>
            .title { font: bold 16px sans-serif; }
            .label { font: 12px sans-serif; }
            .axis-label { font: 10px sans-serif; fill: #666; }
            .grid { stroke: #e0e0e0; stroke-width: 1; }
            .raw-line { fill: none; stroke: #ccc; stroke-width: 1; }
            .smooth-line { fill: none; stroke: #2196F3; stroke-width: 2; }
            .current-loss { font: bold 14px sans-serif; fill: #2196F3; }
          </style>

          <!-- Background -->
          <rect width="\(width)" height="\(height)" fill="white"/>

          <!-- Title -->
          <text x="\(width/2)" y="25" text-anchor="middle" class="title">Learning Curve</text>

          <!-- Grid lines -->
        """

        // Horizontal grid lines (loss)
        for label in lossLabels {
            let (_, y) = toSVG(step: Int(minStep), loss: label)
            svg += """
              <line x1="\(padding)" y1="\(y)" x2="\(width - padding)" y2="\(y)" class="grid"/>
              <text x="\(padding - 5)" y="\(y + 4)" text-anchor="end" class="axis-label">\(String(format: "%.3f", label))</text>
            """
        }

        // Vertical grid lines (step)
        for label in stepLabels {
            let (x, _) = toSVG(step: Int(label), loss: paddedMinLoss)
            svg += """
              <line x1="\(x)" y1="\(padding)" x2="\(x)" y2="\(height - padding)" class="grid"/>
              <text x="\(x)" y="\(height - padding + 15)" text-anchor="middle" class="axis-label">\(Int(label))</text>
            """
        }

        // Axis labels
        svg += """
          <text x="\(width/2)" y="\(height - 10)" text-anchor="middle" class="label">Step</text>
          <text x="15" y="\(height/2)" text-anchor="middle" class="label" transform="rotate(-90, 15, \(height/2))">Loss</text>
        """

        // Raw data line (light gray)
        svg += """
          <path d="\(rawPathData)" class="raw-line"/>
        """

        // Smoothed line (blue)
        svg += """
          <path d="\(pathData)" class="smooth-line"/>
        """

        // Current loss indicator
        if let last = smoothedLoss.last {
            let (x, y) = toSVG(step: last.step, loss: last.loss)
            svg += """
              <circle cx="\(x)" cy="\(y)" r="4" fill="#2196F3"/>
              <text x="\(x + 10)" y="\(y + 5)" class="current-loss">\(String(format: "%.4f", last.loss))</text>
            """
        }

        // Stats box
        let currentLoss = lossHistory.last?.loss ?? 0
        let avgLoss = smoothedLoss.last?.loss ?? 0
        svg += """
          <rect x="\(width - 150)" y="40" width="140" height="60" fill="white" stroke="#ddd" rx="4"/>
          <text x="\(width - 145)" y="58" class="axis-label">Step: \(lossHistory.last?.step ?? 0)</text>
          <text x="\(width - 145)" y="73" class="axis-label">Loss: \(String(format: "%.4f", currentLoss))</text>
          <text x="\(width - 145)" y="88" class="axis-label">Smoothed: \(String(format: "%.4f", avgLoss))</text>
        """

        svg += """
        </svg>
        """

        // Write to file
        do {
            try svg.write(to: svgPath, atomically: true, encoding: .utf8)
        } catch {
            Flux2Debug.log("[LearningCurve] Failed to write SVG: \(error)")
        }
    }

    /// Generate evenly spaced axis labels
    private func generateAxisLabels(min: Float, max: Float, count: Int) -> [Float] {
        guard count > 1 else { return [min] }
        let step = (max - min) / Float(count - 1)
        return (0..<count).map { min + Float($0) * step }
    }
}

// MARK: - Errors

public enum SimpleLoRATrainerError: LocalizedError {
    case noLatents
    case noEmbeddings
    case noPositionIds(String)
    case notInitialized
    case imageCreationFailed
    
    public var errorDescription: String? {
        switch self {
        case .noLatents:
            return "No cached latents provided"
        case .noEmbeddings:
            return "No cached embeddings provided"
        case .noPositionIds(let res):
            return "No position IDs cached for resolution: \(res)"
        case .notInitialized:
            return "Trainer not properly initialized"
        case .imageCreationFailed:
            return "Failed to create validation image"
        }
    }
}

// MARK: - String Extension

extension String {
    func repeating(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
