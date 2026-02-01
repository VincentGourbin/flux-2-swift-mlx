// TrainLoRACommand.swift - CLI command for LoRA training
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import MLX

// MARK: - Train LoRA Command

struct TrainLoRA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train-lora",
        abstract: "Train a LoRA adapter for Flux.2 models"
    )

    // MARK: - Dataset Arguments

    @Argument(help: "Path to the training dataset directory")
    var dataset: String

    @Option(name: .long, help: "Path to validation dataset directory (for validation loss)")
    var validationDataset: String?

    @Option(name: .shortAndLong, help: "Output path for the trained LoRA (.safetensors)")
    var output: String

    @Option(name: .long, help: "Trigger word to replace [trigger] in captions")
    var triggerWord: String?

    @Option(name: .long, help: "Caption file extension (txt or jsonl)")
    var captionFormat: String = "txt"

    @Option(name: .long, help: "Caption dropout rate for generalization (0.0-1.0, default: 0.0)")
    var captionDropout: Float = 0.0

    // MARK: - Model Arguments

    @Option(name: .long, help: "Model to train on: dev, klein-4b, klein-9b")
    var model: String = "klein-4b"

    @Option(name: .long, help: "Quantization for base model: bf16, int8, int4, nf4")
    var quantization: String = "int8"

    @Flag(name: .long, help: "Use base (non-distilled) model for training (recommended for LoRA)")
    var useBaseModel: Bool = false

    // MARK: - LoRA Arguments

    @Option(name: .shortAndLong, help: "LoRA rank (typically 8-64)")
    var rank: Int = 16

    @Option(name: .long, help: "LoRA alpha for scaling")
    var alpha: Float = 16.0

    @Option(name: .long, help: "Target layers: attention, attention_output, attention_ffn, all")
    var targetLayers: String = "attention"

    @Option(name: .long, help: "Dropout rate for regularization")
    var dropout: Float = 0.0

    // MARK: - Training Arguments

    @Option(name: .long, help: "Learning rate")
    var learningRate: Float = 1e-4

    @Option(name: .shortAndLong, help: "Batch size")
    var batchSize: Int = 1

    @Option(name: .shortAndLong, help: "Number of training epochs")
    var epochs: Int = 10

    @Option(name: .long, help: "Maximum training steps (overrides epochs if set)")
    var maxSteps: Int?

    @Option(name: .long, help: "Number of warmup steps")
    var warmupSteps: Int = 100

    @Option(name: .long, help: "LR scheduler: constant, linear, cosine, cosine_with_restarts")
    var lrScheduler: String = "cosine"

    @Option(name: .long, help: "Weight decay for AdamW")
    var weightDecay: Float = 0.01

    @Option(name: .long, help: "Gradient accumulation steps")
    var gradientAccumulation: Int = 1

    @Option(name: .long, help: "Max gradient norm for clipping")
    var maxGradNorm: Float = 1.0

    // MARK: - Memory Optimization Arguments

    @Flag(name: .long, help: "Enable gradient checkpointing (saves ~30-40% memory)")
    var gradientCheckpointing: Bool = false

    @Flag(name: .long, help: "Pre-cache latents with VAE before training")
    var cacheLatents: Bool = false

    @Flag(name: .long, help: "Cache text embeddings")
    var cacheTextEmbeddings: Bool = false

    @Flag(name: .long, help: "Offload text encoder to CPU after encoding")
    var cpuOffload: Bool = false

    // MARK: - Output Arguments

    @Option(name: .long, help: "Save checkpoint every N steps (0 to disable)")
    var saveEveryNSteps: Int = 500

    @Option(name: .long, help: "Keep only the last N checkpoints (0 = keep all)")
    var keepCheckpoints: Int = 0

    @Option(name: .long, help: "Validation prompt for preview generation")
    var validationPrompt: String?

    @Option(name: .long, help: "Generate validation image every N steps")
    var validateEveryNSteps: Int = 500

    @Option(name: .long, help: "Seed for validation image generation (default: 42)")
    var validationSeed: UInt64 = 42

    // MARK: - Image Arguments

    @Option(name: .long, help: "Target image size for training")
    var imageSize: Int = 512

    @Flag(name: .long, help: "Enable aspect ratio bucketing for multi-resolution training")
    var bucketing: Bool = false

    @Option(name: .long, help: "Resolutions for bucketing (comma-separated, e.g., '512,768,1024')")
    var bucketResolutions: String = "512,768,1024"

    // MARK: - Early Stopping Arguments

    @Flag(name: .long, help: "Enable early stopping when loss plateaus")
    var earlyStop: Bool = false

    @Option(name: .long, help: "Epochs without improvement before stopping (default: 5)")
    var earlyStopPatience: Int = 5

    @Option(name: .long, help: "Minimum loss improvement to reset patience (default: 0.01)")
    var earlyStopMinDelta: Float = 0.01

    @Flag(name: .long, help: "Enable early stopping on overfitting (val-train gap increases)")
    var earlyStopOnOverfit: Bool = false

    @Option(name: .long, help: "Maximum val-train gap before stopping (default: 0.5)")
    var earlyStopMaxGap: Float = 0.5

    @Option(name: .long, help: "Consecutive gap increases before stopping (default: 3)")
    var earlyStopGapPatience: Int = 3

    @Flag(name: .long, help: "Enable early stopping on val loss stagnation (epoch-based, default: true)")
    var earlyStopOnValStagnation: Bool = false

    @Option(name: .long, help: "Min val loss improvement per epoch to consider progress (default: 0.1)")
    var earlyStopMinValImprovement: Float = 0.1

    @Option(name: .long, help: "Consecutive epochs without val improvement before stopping (default: 2)")
    var earlyStopValPatience: Int = 2

    // MARK: - EMA Arguments

    @Flag(name: .long, help: "Use EMA for weight averaging (default: enabled)")
    var ema: Bool = false

    @Flag(name: .long, help: "Disable EMA weight averaging")
    var noEma: Bool = false

    @Option(name: .long, help: "EMA decay factor (0.99-0.9999, higher = slower averaging)")
    var emaDecay: Float = 0.99

    // MARK: - Misc Arguments

    @Option(name: .long, help: "Resume from checkpoint directory")
    var resume: String?

    @Option(name: .long, help: "Log training metrics every N steps")
    var logEveryNSteps: Int = 10

    @Option(name: .long, help: "Evaluate (sync GPU) every N steps - higher = faster but less frequent loss updates")
    var evalEveryNSteps: Int = 10

    @Flag(name: .long, help: "Enable verbose output")
    var verbose: Bool = false

    @Flag(name: .long, help: "Dry run - validate configuration without training")
    var dryRun: Bool = false

    // MARK: - Run

    func run() async throws {
        // Configure logging
        if verbose {
            Flux2Debug.enableDebugMode()
        }

        // Parse model variant
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use: dev, klein-4b, klein-9b")
        }

        // Parse quantization
        guard let quant = TrainingQuantization(rawValue: quantization) else {
            throw ValidationError("Invalid quantization: \(quantization). Use: bf16, int8, int4, nf4")
        }

        // Parse target layers
        guard let targets = LoRATargetLayers(rawValue: targetLayers) else {
            throw ValidationError("Invalid target layers: \(targetLayers). Use: attention, attention_output, attention_ffn, all")
        }

        // Parse LR scheduler
        guard let scheduler = LRSchedulerType(rawValue: lrScheduler) else {
            throw ValidationError("Invalid LR scheduler: \(lrScheduler). Use: constant, linear, cosine, cosine_with_restarts")
        }

        // Validate dataset path
        let datasetURL = URL(fileURLWithPath: dataset)
        guard FileManager.default.fileExists(atPath: datasetURL.path) else {
            throw ValidationError("Dataset not found: \(dataset)")
        }

        // Validate output path
        let outputURL = URL(fileURLWithPath: output)
        let outputDir = outputURL.deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: outputDir.path) else {
            throw ValidationError("Output directory not found: \(outputDir.path)")
        }

        // Create validation dataset URL if provided
        let validationDatasetURL = validationDataset.map { URL(fileURLWithPath: $0) }

        // Create training configuration
        let config = LoRATrainingConfig(
            // Dataset
            datasetPath: datasetURL,
            validationDatasetPath: validationDatasetURL,
            captionExtension: captionFormat,
            triggerWord: triggerWord,
            imageSize: imageSize,
            enableBucketing: bucketing,
            bucketResolutions: parseBucketResolutions(bucketResolutions),
            shuffleDataset: true,
            captionDropoutRate: captionDropout,
            // LoRA
            rank: rank,
            alpha: alpha,
            dropout: dropout,
            targetLayers: targets,
            // Training
            learningRate: learningRate,
            batchSize: batchSize,
            epochs: epochs,
            maxSteps: maxSteps,
            warmupSteps: warmupSteps,
            lrScheduler: scheduler,
            weightDecay: weightDecay,
            adamBeta1: 0.9,
            adamBeta2: 0.999,
            adamEpsilon: 1e-8,
            maxGradNorm: maxGradNorm,
            gradientAccumulationSteps: gradientAccumulation,
            // Memory
            quantization: quant,
            gradientCheckpointing: gradientCheckpointing,
            cacheLatents: cacheLatents,
            cacheTextEmbeddings: cacheTextEmbeddings,
            cpuOffloadTextEncoder: cpuOffload,
            mixedPrecision: true,
            // Output
            outputPath: outputURL,
            saveEveryNSteps: saveEveryNSteps,
            keepOnlyLastNCheckpoints: keepCheckpoints,
            validationPrompt: validationPrompt,
            validationEveryNSteps: validateEveryNSteps,
            numValidationImages: 1,
            validationSeed: validationSeed,
            // Logging
            logEveryNSteps: logEveryNSteps,
            evalEveryNSteps: evalEveryNSteps,
            verbose: verbose,
            // Early stopping
            enableEarlyStopping: earlyStop,
            earlyStoppingPatience: earlyStopPatience,
            earlyStoppingMinDelta: earlyStopMinDelta,
            // Overfitting detection
            earlyStoppingOnOverfit: earlyStopOnOverfit,
            earlyStoppingMaxValGap: earlyStopMaxGap,
            earlyStoppingGapPatience: earlyStopGapPatience,
            // Val loss stagnation detection
            earlyStoppingOnValStagnation: earlyStopOnValStagnation,
            earlyStoppingMinValImprovement: earlyStopMinValImprovement,
            earlyStoppingValStagnationPatience: earlyStopValPatience,
            // EMA - default is enabled unless --no-ema is passed
            useEMA: !noEma,
            emaDecay: emaDecay,
            // Resume
            resumeFromCheckpoint: resume.map { URL(fileURLWithPath: $0) }
        )

        // Validate configuration
        do {
            try config.validate()
        } catch {
            throw ValidationError("Configuration error: \(error.localizedDescription)")
        }

        // Print configuration summary
        printConfigSummary(config: config, model: modelVariant, useBaseModel: useBaseModel)

        // Memory estimation
        let estimatedMemory = config.estimateMemoryGB(for: modelVariant)
        let systemMemory = ModelRegistry.systemRAMGB
        print()
        print("Memory:")
        print("  Estimated requirement: \(String(format: "%.1f", estimatedMemory)) GB")
        print("  System RAM: \(systemMemory) GB")

        if !config.canFitInMemory(for: modelVariant, availableGB: systemMemory - 8) {
            print()
            print("⚠️  Warning: Training may not fit in available memory!")
            print("   Suggestions:")
            for suggestion in config.suggestAdjustments(for: modelVariant, availableGB: systemMemory - 8) {
                print("     - \(suggestion)")
            }
        }

        // Validate dataset
        print()
        print("Validating dataset...")
        let parser = CaptionParser(triggerWord: triggerWord)
        let validation = parser.validateDataset(at: datasetURL, extension: captionFormat)
        print(validation.summary)

        if !validation.isValid {
            throw ValidationError("Dataset validation failed")
        }

        // Dry run - stop here
        if dryRun {
            print()
            print("Dry run complete. Configuration is valid.")
            return
        }

        // Create trainer
        print()
        print("Initializing trainer...")
        let trainer = LoRATrainer(config: config, modelType: modelVariant)

        // Set up event handler
        let eventHandler = ConsoleTrainingEventHandler()
        trainer.setEventHandler(eventHandler)

        // Prepare training
        try await trainer.prepare()

        print()
        print("=" .repeating(60))
        print("Loading models...")
        print("=" .repeating(60))
        print()

        // Load VAE
        print("Loading VAE...")
        let vae = try await loadVAE()
        print("  VAE loaded ✓")

        // Pre-cache latents if enabled
        if config.cacheLatents {
            print()
            print("Pre-caching latents with VAE...")
            try await trainer.preCacheLatents(vae: vae)
            print("  Latent caching complete ✓")
        }

        // Load text encoder
        print()
        print("Loading text encoder...")
        let textEncoder = try await loadTextEncoder(for: modelVariant, quantization: quant)
        print("  Text encoder loaded ✓")

        // Load transformer
        print()
        print("Loading transformer...")
        let transformer = try await loadTransformer(for: modelVariant, quantization: quant, useBase: useBaseModel)
        print("  Transformer loaded ✓")

        print()
        print("=" .repeating(60))
        print("Starting LoRA training...")
        print("=" .repeating(60))
        print()

        // Run training
        // Always pass VAE - even with cacheLatents, we need it for validation image generation
        try await trainer.train(
            transformer: transformer,
            vae: vae,
            textEncoder: { prompt in
                try textEncoder.encode(prompt)
            }
        )

        print()
        print("=" .repeating(60))
        print("Training complete!")
        print("=" .repeating(60))
        print()
        print("Output saved to: \(config.outputPath.path)")
    }

    // MARK: - Model Loading

    private func loadVAE() async throws -> AutoencoderKLFlux2 {
        guard let modelPath = Flux2ModelDownloader.findModelPath(for: .vae(.standard)) else {
            throw ValidationError("VAE not found. Run: flux2 download --vae")
        }

        let vaePath = modelPath.appendingPathComponent("vae")
        let weightsPath = FileManager.default.fileExists(atPath: vaePath.path) ? vaePath : modelPath

        let vae = AutoencoderKLFlux2()
        let weights = try Flux2WeightLoader.loadWeights(from: weightsPath)
        try Flux2WeightLoader.applyVAEWeights(weights, to: vae)
        eval(vae.parameters())

        return vae
    }

    private func loadTextEncoder(
        for model: Flux2Model,
        quantization: TrainingQuantization
    ) async throws -> KleinTextEncoder {
        // Map training quantization to Mistral quantization for text encoder
        let mistralQuant: MistralQuantization
        switch quantization {
        case .bf16:
            mistralQuant = .bf16
        case .int8:
            mistralQuant = .mlx8bit
        case .int4, .nf4:
            mistralQuant = .mlx4bit
        }

        // For Klein models, use KleinTextEncoder
        // For Dev, would need to use T5/CLIP (not implemented here)
        let variant: KleinVariant
        switch model {
        case .klein4B:
            variant = .klein4B
        case .klein9B:
            variant = .klein9B
        case .dev:
            throw ValidationError("Dev model training not yet supported. Use Klein 4B or 9B.")
        }

        let encoder = KleinTextEncoder(variant: variant, quantization: mistralQuant)
        try await encoder.load()

        return encoder
    }

    private func loadTransformer(
        for model: Flux2Model,
        quantization: TrainingQuantization,
        useBase: Bool
    ) async throws -> Flux2Transformer2DModel {
        // Map training quantization to transformer quantization
        // Note: Training currently only supports bf16 and qint8
        let transformerQuant: TransformerQuantization
        switch quantization {
        case .bf16:
            transformerQuant = .bf16
        case .int8, .int4, .nf4:
            transformerQuant = .qint8  // Use qint8 for all quantized modes
        }

        // Determine variant - use base model if requested and available
        let variant: ModelRegistry.TransformerVariant
        if useBase && model == .klein4B && transformerQuant == .bf16 {
            // Use base (non-distilled) Klein 4B for better LoRA training
            variant = .klein4B_base_bf16
            print("  Using base (non-distilled) model for training")
        } else {
            variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: transformerQuant)
            if useBase {
                print("  Note: Base model only available for Klein 4B bf16, using distilled version")
            }
        }

        // Check if model exists, download if needed
        let component = ModelRegistry.ModelComponent.transformer(variant)
        var modelPath = Flux2ModelDownloader.findModelPath(for: component)

        if modelPath == nil {
            print("  Model not found locally, downloading from HuggingFace...")
            let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
            let downloader = Flux2ModelDownloader(hfToken: hfToken)
            modelPath = try await downloader.download(component) { progress, message in
                print("  Download: \(Int(progress * 100))% - \(message)")
            }
        }

        guard let modelPath = modelPath else {
            throw ValidationError("Failed to download transformer for \(model.displayName)")
        }

        let transformer = Flux2Transformer2DModel(
            config: model.transformerConfig,
            memoryOptimization: .aggressive  // Use aggressive memory mode for training
        )

        let weights = try Flux2WeightLoader.loadWeights(from: modelPath)
        try Flux2WeightLoader.applyTransformerWeights(weights, to: transformer)
        eval(transformer.parameters())

        return transformer
    }

    private func printConfigSummary(config: LoRATrainingConfig, model: Flux2Model, useBaseModel: Bool) {
        print()
        print("LoRA Training Configuration")
        print("-" .repeating(40))
        print()
        print("Model: \(model.displayName)\(useBaseModel ? " (base)" : "")")
        print("Quantization: \(config.quantization.displayName)")
        print()
        print("LoRA:")
        print("  Rank: \(config.rank)")
        print("  Alpha: \(config.alpha)")
        print("  Scale: \(String(format: "%.2f", config.scale))")
        print("  Target layers: \(config.targetLayers.displayName)")
        if config.dropout > 0 {
            print("  Dropout: \(config.dropout)")
        }
        print()
        print("Image Processing:")
        if config.enableBucketing {
            print("  Bucketing: enabled (resolutions: \(config.bucketResolutions.map { String($0) }.joined(separator: ", ")))")
        } else {
            print("  Image size: \(config.imageSize)x\(config.imageSize)")
        }
        print()
        print("Training:")
        print("  Learning rate: \(String(format: "%.2e", config.learningRate))")
        if config.captionDropoutRate > 0 {
            print("  Caption dropout: \(String(format: "%.1f", config.captionDropoutRate * 100))%")
        }
        print("  Batch size: \(config.batchSize)")
        print("  Gradient accumulation: \(config.gradientAccumulationSteps)")
        print("  Effective batch size: \(config.effectiveBatchSize)")
        print("  Epochs: \(config.epochs)")
        if let maxSteps = config.maxSteps {
            print("  Max steps: \(maxSteps)")
        }
        print("  Warmup steps: \(config.warmupSteps)")
        print("  LR scheduler: \(config.lrScheduler.displayName)")
        if config.enableEarlyStopping {
            print("  Early stopping: enabled (patience=\(config.earlyStoppingPatience), minDelta=\(config.earlyStoppingMinDelta))")
        }
        if config.earlyStoppingOnOverfit {
            print("  Overfitting detection: enabled (maxGap=\(config.earlyStoppingMaxValGap), patience=\(config.earlyStoppingGapPatience))")
        }
        print()
        print("Weight Averaging:")
        print("  EMA: \(config.useEMA ? "enabled (decay=\(config.emaDecay))" : "disabled")")
        print()
        print("Memory optimizations:")
        print("  Gradient checkpointing: \(config.gradientCheckpointing ? "enabled" : "disabled")")
        print("  Cache latents: \(config.cacheLatents ? "enabled" : "disabled")")
        print("  Cache text embeddings: \(config.cacheTextEmbeddings ? "enabled" : "disabled")")
        print("  CPU offload: \(config.cpuOffloadTextEncoder ? "enabled" : "disabled")")
        print()
        print("Output: \(config.outputPath.path)")
        if config.saveEveryNSteps > 0 {
            print("  Checkpoint every \(config.saveEveryNSteps) steps")
        }
        if let validationPrompt = config.validationPrompt {
            print("  Validation prompt: \"\(validationPrompt.prefix(50))...\"")
        }
    }
}

// MARK: - String Extension

private extension String {
    func repeating(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}

// MARK: - Helper Functions

/// Parse comma-separated resolution string into array of integers
private func parseBucketResolutions(_ input: String) -> [Int] {
    input.split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        .filter { $0 >= 256 && $0 <= 2048 }
}
