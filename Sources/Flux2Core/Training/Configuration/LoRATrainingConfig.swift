// LoRATrainingConfig.swift - Configuration for LoRA training
// Copyright 2025 Vincent Gourbin

import Foundation

/// Quantization options for training (affects memory usage)
public enum TrainingQuantization: String, Codable, Sendable, CaseIterable {
    /// Full BF16 precision (~30GB for Dev, ~10GB for Klein 4B)
    case bf16 = "bf16"
    
    /// 8-bit quantization (~18GB for Dev, ~6GB for Klein 4B)
    case int8 = "int8"
    
    /// 4-bit quantization (~13GB for Dev, ~4GB for Klein 4B)
    case int4 = "int4"
    
    /// NF4 quantization (~9GB for Dev, ~3GB for Klein 4B)
    case nf4 = "nf4"
    
    public var displayName: String {
        switch self {
        case .bf16: return "BFloat16 (full precision)"
        case .int8: return "INT8 quantized"
        case .int4: return "INT4 quantized"
        case .nf4: return "NF4 quantized (QLoRA)"
        }
    }
    
    /// Bits per weight
    public var bits: Int {
        switch self {
        case .bf16: return 16
        case .int8: return 8
        case .int4, .nf4: return 4
        }
    }
}

/// Target layers for LoRA training
public enum LoRATargetLayers: String, Codable, Sendable, CaseIterable {
    /// Attention projections only (Q, K, V, O) - most economical
    case attention = "attention"
    
    /// Attention + output projections
    case attentionOutput = "attention_output"
    
    /// Attention + feedforward layers - most comprehensive
    case attentionFFN = "attention_ffn"
    
    /// All linear layers in transformer
    case all = "all"
    
    public var displayName: String {
        switch self {
        case .attention: return "Attention only (Q, K, V)"
        case .attentionOutput: return "Attention + Output"
        case .attentionFFN: return "Attention + FFN"
        case .all: return "All linear layers"
        }
    }
    
    /// Whether to include double block attention layers
    public var includesDoubleAttention: Bool { true }
    
    /// Whether to include single block attention layers
    public var includesSingleAttention: Bool { true }
    
    /// Whether to include output projections
    public var includesOutputProjections: Bool {
        switch self {
        case .attention: return false
        case .attentionOutput, .attentionFFN, .all: return true
        }
    }
    
    /// Whether to include FFN layers
    public var includesFFN: Bool {
        switch self {
        case .attention, .attentionOutput: return false
        case .attentionFFN, .all: return true
        }
    }
}

/// Learning rate scheduler type
public enum LRSchedulerType: String, Codable, Sendable, CaseIterable {
    /// Constant learning rate after warmup
    case constant = "constant"
    
    /// Linear decay to 0
    case linear = "linear"
    
    /// Cosine annealing
    case cosine = "cosine"
    
    /// Cosine with restarts
    case cosineWithRestarts = "cosine_with_restarts"
    
    public var displayName: String {
        switch self {
        case .constant: return "Constant"
        case .linear: return "Linear decay"
        case .cosine: return "Cosine annealing"
        case .cosineWithRestarts: return "Cosine with restarts"
        }
    }
}

/// Configuration for LoRA training
public struct LoRATrainingConfig: Codable, Sendable {
    
    // MARK: - Dataset Configuration

    /// Path to the dataset directory
    public var datasetPath: URL

    /// Path to validation dataset directory (optional, for validation loss)
    public var validationDatasetPath: URL?

    /// Caption file extension ("txt" or "jsonl")
    public var captionExtension: String

    /// Trigger word to replace [trigger] in captions
    public var triggerWord: String?

    /// Target image size (will resize/crop images) - used when bucketing is disabled
    public var imageSize: Int

    /// Enable aspect ratio bucketing (multiple resolutions)
    public var enableBucketing: Bool

    /// Available resolutions for bucketing (e.g., [512, 768, 1024])
    /// Only used when enableBucketing is true
    public var bucketResolutions: [Int]

    /// Shuffle training data
    public var shuffleDataset: Bool

    /// Caption dropout rate for generalization (0.0-1.0, 0 = disabled)
    /// When triggered, replaces caption with empty string to help model generalize
    public var captionDropoutRate: Float
    
    // MARK: - LoRA Configuration
    
    /// LoRA rank (typically 8-64)
    public var rank: Int
    
    /// LoRA alpha for scaling (scale = alpha / rank)
    public var alpha: Float
    
    /// Dropout rate for LoRA layers (0.0 to disable)
    public var dropout: Float
    
    /// Target layers to apply LoRA
    public var targetLayers: LoRATargetLayers
    
    // MARK: - Training Parameters
    
    /// Base learning rate
    public var learningRate: Float
    
    /// Batch size per step
    public var batchSize: Int
    
    /// Number of training epochs
    public var epochs: Int
    
    /// Maximum training steps (nil = use epochs)
    public var maxSteps: Int?
    
    /// Number of warmup steps
    public var warmupSteps: Int
    
    /// Learning rate scheduler type
    public var lrScheduler: LRSchedulerType
    
    /// AdamW weight decay
    public var weightDecay: Float
    
    /// AdamW beta1
    public var adamBeta1: Float
    
    /// AdamW beta2
    public var adamBeta2: Float
    
    /// AdamW epsilon
    public var adamEpsilon: Float
    
    /// Gradient clipping max norm (0 to disable)
    public var maxGradNorm: Float
    
    /// Gradient accumulation steps (simulates larger batch)
    public var gradientAccumulationSteps: Int
    
    // MARK: - Memory Optimization
    
    /// Quantization level for base model
    public var quantization: TrainingQuantization
    
    /// Enable gradient checkpointing (saves ~30-40% VRAM)
    public var gradientCheckpointing: Bool
    
    /// Cache latents to disk (pre-encode with VAE)
    public var cacheLatents: Bool
    
    /// Cache text embeddings to disk
    public var cacheTextEmbeddings: Bool
    
    /// Offload text encoder to CPU after encoding
    public var cpuOffloadTextEncoder: Bool
    
    /// Mixed precision training
    public var mixedPrecision: Bool
    
    // MARK: - Output Configuration
    
    /// Output path for saved LoRA
    public var outputPath: URL
    
    /// Save checkpoint every N steps (0 to disable)
    public var saveEveryNSteps: Int
    
    /// Keep only the last N checkpoints (0 to keep all) - default is 0 to let user manage checkpoints manually
    public var keepOnlyLastNCheckpoints: Int
    
    /// Validation prompt for preview generation
    public var validationPrompt: String?
    
    /// Generate validation image every N steps (0 to disable)
    public var validationEveryNSteps: Int
    
    /// Number of validation images to generate
    public var numValidationImages: Int
    
    /// Seed for validation image generation (nil = random)
    public var validationSeed: UInt64?
    
    // MARK: - Logging
    
    /// Log training metrics every N steps
    public var logEveryNSteps: Int

    /// Evaluate (sync GPU) every N steps - reduces CPU/GPU sync overhead
    /// Set to 1 for exact loss at every step, higher values for faster training
    public var evalEveryNSteps: Int

    /// Enable verbose logging
    public var verbose: Bool
    
    // MARK: - Early Stopping

    /// Enable early stopping when loss plateaus
    public var enableEarlyStopping: Bool

    /// Number of epochs to wait for improvement before stopping
    public var earlyStoppingPatience: Int

    /// Minimum improvement in loss to reset patience counter
    public var earlyStoppingMinDelta: Float

    // MARK: - Overfitting Detection (Early Stopping)

    /// Enable early stopping when overfitting is detected (val loss gap increases)
    public var earlyStoppingOnOverfit: Bool

    /// Maximum allowed gap between validation and training loss before triggering early stop
    public var earlyStoppingMaxValGap: Float

    /// Number of consecutive validation checks where gap increases before stopping
    public var earlyStoppingGapPatience: Int

    // MARK: - Val Loss Stagnation Detection (Early Stopping, Epoch-based)

    /// Enable early stopping when validation loss improvement stagnates (checked at end of each epoch)
    public var earlyStoppingOnValStagnation: Bool

    /// Minimum val loss improvement per epoch required to consider progress (absolute value)
    public var earlyStoppingMinValImprovement: Float

    /// Number of consecutive epochs with insufficient val loss improvement before stopping
    public var earlyStoppingValStagnationPatience: Int

    // MARK: - EMA (Exponential Moving Average)

    /// Use EMA for weight averaging (smoother training, recommended for LoRA)
    public var useEMA: Bool

    /// EMA decay factor (0.99-0.9999, higher = slower averaging)
    public var emaDecay: Float

    // MARK: - Resume Training

    /// Path to checkpoint to resume from (nil = start fresh)
    public var resumeFromCheckpoint: URL?
    
    // MARK: - Initializer
    
    public init(
        // Dataset
        datasetPath: URL,
        validationDatasetPath: URL? = nil,
        captionExtension: String = "txt",
        triggerWord: String? = nil,
        imageSize: Int = 512,
        enableBucketing: Bool = false,
        bucketResolutions: [Int] = [512, 768, 1024],
        shuffleDataset: Bool = true,
        captionDropoutRate: Float = 0.0,
        // LoRA
        rank: Int = 16,
        alpha: Float = 16.0,
        dropout: Float = 0.0,
        targetLayers: LoRATargetLayers = .attention,
        // Training
        learningRate: Float = 1e-4,
        batchSize: Int = 1,
        epochs: Int = 10,
        maxSteps: Int? = nil,
        warmupSteps: Int = 100,
        lrScheduler: LRSchedulerType = .cosine,
        weightDecay: Float = 0.01,
        adamBeta1: Float = 0.9,
        adamBeta2: Float = 0.999,
        adamEpsilon: Float = 1e-8,
        maxGradNorm: Float = 1.0,
        gradientAccumulationSteps: Int = 1,
        // Memory
        quantization: TrainingQuantization = .bf16,
        gradientCheckpointing: Bool = true,
        cacheLatents: Bool = true,
        cacheTextEmbeddings: Bool = true,
        cpuOffloadTextEncoder: Bool = false,
        mixedPrecision: Bool = true,
        // Output
        outputPath: URL,
        saveEveryNSteps: Int = 500,
        keepOnlyLastNCheckpoints: Int = 0,
        validationPrompt: String? = nil,
        validationEveryNSteps: Int = 500,
        numValidationImages: Int = 1,
        validationSeed: UInt64? = nil,
        // Logging
        logEveryNSteps: Int = 10,
        evalEveryNSteps: Int = 10,
        verbose: Bool = false,
        // Early stopping
        enableEarlyStopping: Bool = false,
        earlyStoppingPatience: Int = 5,
        earlyStoppingMinDelta: Float = 0.01,
        // Overfitting detection
        earlyStoppingOnOverfit: Bool = false,
        earlyStoppingMaxValGap: Float = 0.5,
        earlyStoppingGapPatience: Int = 3,
        // Val loss stagnation detection
        earlyStoppingOnValStagnation: Bool = true,
        earlyStoppingMinValImprovement: Float = 0.1,
        earlyStoppingValStagnationPatience: Int = 2,
        // EMA
        useEMA: Bool = true,
        emaDecay: Float = 0.99,
        // Resume
        resumeFromCheckpoint: URL? = nil
    ) {
        self.datasetPath = datasetPath
        self.validationDatasetPath = validationDatasetPath
        self.captionExtension = captionExtension
        self.triggerWord = triggerWord
        self.imageSize = imageSize
        self.enableBucketing = enableBucketing
        self.bucketResolutions = bucketResolutions
        self.shuffleDataset = shuffleDataset
        self.captionDropoutRate = captionDropoutRate
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetLayers = targetLayers
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.epochs = epochs
        self.maxSteps = maxSteps
        self.warmupSteps = warmupSteps
        self.lrScheduler = lrScheduler
        self.weightDecay = weightDecay
        self.adamBeta1 = adamBeta1
        self.adamBeta2 = adamBeta2
        self.adamEpsilon = adamEpsilon
        self.maxGradNorm = maxGradNorm
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.quantization = quantization
        self.gradientCheckpointing = gradientCheckpointing
        self.cacheLatents = cacheLatents
        self.cacheTextEmbeddings = cacheTextEmbeddings
        self.cpuOffloadTextEncoder = cpuOffloadTextEncoder
        self.mixedPrecision = mixedPrecision
        self.outputPath = outputPath
        self.saveEveryNSteps = saveEveryNSteps
        self.keepOnlyLastNCheckpoints = keepOnlyLastNCheckpoints
        self.validationPrompt = validationPrompt
        self.validationEveryNSteps = validationEveryNSteps
        self.numValidationImages = numValidationImages
        self.validationSeed = validationSeed
        self.logEveryNSteps = logEveryNSteps
        self.evalEveryNSteps = evalEveryNSteps
        self.verbose = verbose
        self.enableEarlyStopping = enableEarlyStopping
        self.earlyStoppingPatience = earlyStoppingPatience
        self.earlyStoppingMinDelta = earlyStoppingMinDelta
        self.earlyStoppingOnOverfit = earlyStoppingOnOverfit
        self.earlyStoppingMaxValGap = earlyStoppingMaxValGap
        self.earlyStoppingGapPatience = earlyStoppingGapPatience
        self.earlyStoppingOnValStagnation = earlyStoppingOnValStagnation
        self.earlyStoppingMinValImprovement = earlyStoppingMinValImprovement
        self.earlyStoppingValStagnationPatience = earlyStoppingValStagnationPatience
        self.useEMA = useEMA
        self.emaDecay = emaDecay
        self.resumeFromCheckpoint = resumeFromCheckpoint
    }
    
    // MARK: - Computed Properties
    
    /// Effective scale factor (alpha / rank)
    public var scale: Float {
        alpha / Float(rank)
    }
    
    /// Effective batch size (batch * accumulation steps)
    public var effectiveBatchSize: Int {
        batchSize * gradientAccumulationSteps
    }
    
    // MARK: - Presets
    
    /// Minimal memory preset for 8GB Macs (Klein 4B only)
    public static func minimal8GB(
        datasetPath: URL,
        outputPath: URL,
        triggerWord: String? = nil
    ) -> LoRATrainingConfig {
        LoRATrainingConfig(
            datasetPath: datasetPath,
            triggerWord: triggerWord,
            imageSize: 512,
            rank: 8,
            alpha: 8.0,
            targetLayers: .attention,
            learningRate: 1e-4,
            batchSize: 1,
            epochs: 10,
            gradientAccumulationSteps: 1,
            quantization: .nf4,
            gradientCheckpointing: true,
            cacheLatents: true,
            outputPath: outputPath
        )
    }
    
    /// Balanced preset for 16GB Macs (Klein 4B, 9B)
    public static func balanced16GB(
        datasetPath: URL,
        outputPath: URL,
        triggerWord: String? = nil
    ) -> LoRATrainingConfig {
        LoRATrainingConfig(
            datasetPath: datasetPath,
            triggerWord: triggerWord,
            imageSize: 512,
            rank: 16,
            alpha: 16.0,
            targetLayers: .attentionOutput,
            learningRate: 1e-4,
            batchSize: 1,
            epochs: 10,
            gradientAccumulationSteps: 2,
            quantization: .int8,
            gradientCheckpointing: true,
            cacheLatents: true,
            outputPath: outputPath
        )
    }
    
    /// Quality preset for 32GB+ Macs (all models)
    public static func quality32GB(
        datasetPath: URL,
        outputPath: URL,
        triggerWord: String? = nil
    ) -> LoRATrainingConfig {
        LoRATrainingConfig(
            datasetPath: datasetPath,
            triggerWord: triggerWord,
            imageSize: 1024,
            rank: 32,
            alpha: 32.0,
            targetLayers: .attentionFFN,
            learningRate: 1e-4,
            batchSize: 2,
            epochs: 10,
            gradientAccumulationSteps: 2,
            quantization: .bf16,
            gradientCheckpointing: true,
            cacheLatents: true,
            outputPath: outputPath
        )
    }
    
    // MARK: - Validation
    
    /// Validate the configuration
    public func validate() throws {
        // Check dataset path exists
        guard FileManager.default.fileExists(atPath: datasetPath.path) else {
            throw LoRATrainingConfigError.datasetNotFound(datasetPath)
        }
        
        // Check rank is reasonable
        guard rank >= 1 && rank <= 256 else {
            throw LoRATrainingConfigError.invalidRank(rank)
        }
        
        // Check alpha is positive
        guard alpha > 0 else {
            throw LoRATrainingConfigError.invalidAlpha(alpha)
        }
        
        // Check learning rate
        guard learningRate > 0 && learningRate < 1 else {
            throw LoRATrainingConfigError.invalidLearningRate(learningRate)
        }
        
        // Check batch size
        guard batchSize >= 1 else {
            throw LoRATrainingConfigError.invalidBatchSize(batchSize)
        }
        
        // Check epochs
        guard epochs >= 1 else {
            throw LoRATrainingConfigError.invalidEpochs(epochs)
        }
        
        // Check image size
        guard imageSize >= 256 && imageSize <= 2048 else {
            throw LoRATrainingConfigError.invalidImageSize(imageSize)
        }
        
        // Check output directory parent exists
        let outputDir = outputPath.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: outputDir.path) else {
            throw LoRATrainingConfigError.outputDirectoryNotFound(outputDir)
        }
    }
    
    // MARK: - Persistence
    
    /// Save configuration to JSON file
    public func save(to path: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: path)
    }
    
    /// Load configuration from JSON file
    public static func load(from path: URL) throws -> LoRATrainingConfig {
        let data = try Data(contentsOf: path)
        let decoder = JSONDecoder()
        return try decoder.decode(LoRATrainingConfig.self, from: data)
    }
}

// MARK: - Errors

public enum LoRATrainingConfigError: Error, LocalizedError {
    case datasetNotFound(URL)
    case invalidRank(Int)
    case invalidAlpha(Float)
    case invalidLearningRate(Float)
    case invalidBatchSize(Int)
    case invalidEpochs(Int)
    case invalidImageSize(Int)
    case outputDirectoryNotFound(URL)
    
    public var errorDescription: String? {
        switch self {
        case .datasetNotFound(let url):
            return "Dataset directory not found: \(url.path)"
        case .invalidRank(let rank):
            return "Invalid LoRA rank \(rank). Must be between 1 and 256."
        case .invalidAlpha(let alpha):
            return "Invalid LoRA alpha \(alpha). Must be positive."
        case .invalidLearningRate(let lr):
            return "Invalid learning rate \(lr). Must be between 0 and 1."
        case .invalidBatchSize(let size):
            return "Invalid batch size \(size). Must be at least 1."
        case .invalidEpochs(let epochs):
            return "Invalid epochs \(epochs). Must be at least 1."
        case .invalidImageSize(let size):
            return "Invalid image size \(size). Must be between 256 and 2048."
        case .outputDirectoryNotFound(let url):
            return "Output directory not found: \(url.path)"
        }
    }
}

// MARK: - Memory Estimation

extension LoRATrainingConfig {
    
    /// Estimate memory required for training in GB
    public func estimateMemoryGB(for model: Flux2Model) -> Float {
        var baseMemory: Float = 0
        
        // Base model memory (depends on quantization)
        switch model {
        case .klein4B:
            switch quantization {
            case .bf16: baseMemory = 10
            case .int8: baseMemory = 6
            case .int4, .nf4: baseMemory = 4
            }
        case .klein9B:
            switch quantization {
            case .bf16: baseMemory = 15
            case .int8: baseMemory = 10
            case .int4, .nf4: baseMemory = 6
            }
        case .dev:
            switch quantization {
            case .bf16: baseMemory = 30
            case .int8: baseMemory = 18
            case .int4, .nf4: baseMemory = 13
            }
        }
        
        // LoRA parameters (small addition)
        let loraParamsGB: Float = Float(rank * 2) * 0.001  // Rough estimate
        
        // Optimizer states (2x LoRA params for AdamW)
        let optimizerGB = loraParamsGB * 2
        
        // Activations and gradients (depends on batch size and image size)
        let activationsGB = Float(batchSize) * Float(imageSize * imageSize) / (512 * 512) * 2.0
        
        // Gradient checkpointing saves ~30-40%
        let checkpointSavings: Float = gradientCheckpointing ? 0.35 : 0
        
        // Total
        let total = baseMemory + loraParamsGB + optimizerGB + activationsGB * (1 - checkpointSavings)
        
        // VAE and text encoder if not cached
        let vaeMemory: Float = cacheLatents ? 0 : 2
        let textEncoderMemory: Float = cacheTextEmbeddings ? 0 : 5
        
        return total + vaeMemory + textEncoderMemory
    }
    
    /// Check if training can fit in available memory
    public func canFitInMemory(for model: Flux2Model, availableGB: Int) -> Bool {
        estimateMemoryGB(for: model) <= Float(availableGB)
    }
    
    /// Suggest configuration adjustments to fit in memory
    public func suggestAdjustments(for model: Flux2Model, availableGB: Int) -> [String] {
        var suggestions: [String] = []
        let required = estimateMemoryGB(for: model)
        
        if required > Float(availableGB) {
            _ = required - Float(availableGB)  // Calculate excess for potential future use
            
            if !gradientCheckpointing {
                suggestions.append("Enable gradient checkpointing (saves ~30% memory)")
            }
            
            if quantization == .bf16 {
                suggestions.append("Use int8 or int4 quantization")
            } else if quantization == .int8 {
                suggestions.append("Use int4 or nf4 quantization")
            }
            
            if !cacheLatents {
                suggestions.append("Enable latent caching (pre-encode with VAE)")
            }
            
            if batchSize > 1 {
                suggestions.append("Reduce batch size to 1")
            }
            
            if imageSize > 512 {
                suggestions.append("Reduce image size to 512x512")
            }
            
            if rank > 16 {
                suggestions.append("Reduce LoRA rank to 16 or 8")
            }
            
            if targetLayers == .attentionFFN || targetLayers == .all {
                suggestions.append("Target only attention layers")
            }
        }
        
        return suggestions
    }
}
