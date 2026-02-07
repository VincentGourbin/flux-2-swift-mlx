// TrainLoRACommand.swift - CLI command for LoRA training
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import MLX
import Yams

// MARK: - Train LoRA Command

struct TrainLoRA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train-lora",
        abstract: "Train a LoRA adapter for Flux.2 models"
    )

    // MARK: - Config File Argument

    @Option(name: .shortAndLong, help: "Path to YAML configuration file (CLI args override YAML values)")
    var config: String?

    // MARK: - Dataset Arguments

    @Argument(help: "Path to the training dataset directory (or use --config)")
    var dataset: String?

    @Option(name: .long, help: "Path to validation dataset directory (for validation loss)")
    var validationDataset: String?

    @Option(name: .shortAndLong, help: "Output path for the trained LoRA (.safetensors)")
    var output: String?

    @Option(name: .long, help: "Trigger word to replace [trigger] in captions")
    var triggerWord: String?

    @Option(name: .long, help: "Caption file extension (txt or jsonl)")
    var captionFormat: String = "txt"

    @Option(name: .long, help: "Caption dropout rate for generalization (0.0-1.0, default: 0.0)")
    var captionDropout: Float = 0.0

    // MARK: - Model Arguments

    @Option(name: .long, help: "Model to train on: dev, klein-4b, klein-9b")
    var model: String = "klein-4b"

    @Option(name: .long, help: "Text encoder quantization: bf16, int8, int4, nf4 (transformer always bf16)")
    var quantization: String = "int8"

    // Note: --use-base-model flag removed - LoRA training ALWAYS uses base model (mandatory)

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

    // MARK: - Timestep Sampling Arguments

    @Option(name: .long, help: "Timestep sampling: uniform, logit_normal, flux_shift, content, style")
    var timestepSampling: String = "uniform"

    @Option(name: .long, help: "Logit-normal mean (for logit_normal sampling, default: 0.0)")
    var logitNormalMean: Float = 0.0

    @Option(name: .long, help: "Logit-normal std (for logit_normal sampling, default: 1.0)")
    var logitNormalStd: Float = 1.0

    @Option(name: .long, help: "Flux shift value (for flux_shift sampling, default: 1.0)")
    var fluxShift: Float = 1.0

    @Option(name: .long, help: "Loss weighting strategy: none, bell_shaped, snr")
    var lossWeighting: String = "none"

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

        // Determine if using YAML config or CLI args
        let trainingConfig: LoRATrainingConfig
        let modelVariant: Flux2Model
        // Note: useBase removed - LoRA training ALWAYS uses base model (mandatory)

        if let configPath = config {
            // Load from YAML config file
            print("Loading configuration from: \(configPath)")

            let yamlConfig = try YAMLConfigParser.load(from: configPath)

            // Build CLI overrides (CLI args take precedence over YAML)
            let overrides = CLIOverrides(
                dataset: dataset,
                validationDataset: validationDataset,
                output: output,
                triggerWord: triggerWord,
                model: model != "klein-4b" ? model : nil,  // Only override if explicitly set
                quantization: quantization != "int8" ? quantization : nil,
                rank: rank != 16 ? rank : nil,
                alpha: alpha != 16.0 ? alpha : nil,
                targetLayers: targetLayers != "attention" ? targetLayers : nil,
                learningRate: learningRate != 1e-4 ? learningRate : nil,
                batchSize: batchSize != 1 ? batchSize : nil,
                maxSteps: maxSteps,
                validationPrompt: validationPrompt,
                verbose: verbose ? verbose : nil
            )

            let result = try YAMLConfigParser.toLoRATrainingConfig(yaml: yamlConfig, cliOverrides: overrides)
            trainingConfig = result.config
            modelVariant = result.modelVariant

            print("Configuration loaded successfully")
            print()
        } else {
            // Use CLI arguments (original behavior)
            guard let datasetArg = dataset else {
                throw ValidationError("Dataset path is required. Use --config <file.yaml> or provide dataset as argument.")
            }

            // Parse model variant
            guard let parsedModel = Flux2Model(rawValue: model) else {
                throw ValidationError("Invalid model: \(model). Use: dev, klein-4b, klein-9b")
            }
            modelVariant = parsedModel

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
            let datasetURL = URL(fileURLWithPath: datasetArg)
            guard FileManager.default.fileExists(atPath: datasetURL.path) else {
                throw ValidationError("Dataset not found: \(datasetArg)")
            }

            // Validate output path (required when not using config file)
            guard let outputArg = output else {
                throw ValidationError("Output path is required. Use --output or --config with checkpoints.output")
            }
            let outputURL = URL(fileURLWithPath: outputArg)

            // Create validation dataset URL if provided
            let validationDatasetURL = validationDataset.map { URL(fileURLWithPath: $0) }

            // Note: base model is ALWAYS used for LoRA training (mandatory)

            // Create training configuration
            trainingConfig = LoRATrainingConfig(
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
            // Timestep sampling
            timestepSampling: parseTimestepSampling(timestepSampling),
            logitNormalMean: logitNormalMean,
            logitNormalStd: logitNormalStd,
            fluxShiftValue: fluxShift,
            // Loss weighting
            lossWeighting: parseLossWeighting(lossWeighting),
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
        }

        // Validate configuration
        do {
            try trainingConfig.validate()
        } catch {
            throw ValidationError("Configuration error: \(error.localizedDescription)")
        }

        // Print configuration summary
        printConfigSummary(config: trainingConfig, model: modelVariant)

        // Memory estimation
        let estimatedMemory = trainingConfig.estimateMemoryGB(for: modelVariant)
        let systemMemory = ModelRegistry.systemRAMGB
        print()
        print("Memory:")
        print("  Estimated requirement: \(String(format: "%.1f", estimatedMemory)) GB")
        print("  System RAM: \(systemMemory) GB")

        if !trainingConfig.canFitInMemory(for: modelVariant, availableGB: systemMemory - 8) {
            print()
            print("⚠️  Warning: Training may not fit in available memory!")
            print("   Suggestions:")
            for suggestion in trainingConfig.suggestAdjustments(for: modelVariant, availableGB: systemMemory - 8) {
                print("     - \(suggestion)")
            }
        }

        // Validate dataset
        print()
        print("Validating dataset...")
        let parser = CaptionParser(triggerWord: trainingConfig.triggerWord)
        let validation = parser.validateDataset(at: trainingConfig.datasetPath, extension: trainingConfig.captionExtension)
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

        // Use SimpleLoRATrainer (Ostris-compatible, clean implementation)
        try await runSimpleTrainer(config: trainingConfig, model: modelVariant)
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
    ) async throws -> any TrainingTextEncoder {
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

        switch model {
        case .klein4B:
            // Klein 4B uses Qwen3-4B
            let encoder = KleinTextEncoder(variant: .klein4B, quantization: mistralQuant)
            try await encoder.load()
            return encoder

        case .klein9B:
            // Klein 9B uses Qwen3-8B
            let encoder = KleinTextEncoder(variant: .klein9B, quantization: mistralQuant)
            try await encoder.load()
            return encoder

        case .dev:
            // Dev uses Mistral Small 3.2 (24B)
            // Note: For Dev training, we recommend using 8-bit quantization
            // bf16 requires ~48GB just for the text encoder
            let encoder = DevTextEncoder(quantization: mistralQuant)
            try await encoder.load()
            return encoder
        }
    }

    /// Load transformer for LoRA training
    /// ALWAYS uses base (non-distilled) model - quantization is NOT an option for training
    private func loadTransformer(for model: Flux2Model) async throws -> Flux2Transformer2DModel {
        // LoRA training MUST use base (non-distilled) model
        // Distilled models are trained with guidance distillation and will NOT work for LoRA
        guard let variant = ModelRegistry.TransformerVariant.trainingVariant(for: model) else {
            throw ValidationError("""
                LoRA training is not supported for \(model.displayName).
                No base (non-distilled) model is available for this model type.

                Currently supported for LoRA training:
                - Klein 4B (bf16 base model)
                - Klein 9B (bf16 base model)
                - Dev (32B, already non-distilled)
                """)
        }
        
        print("  Using BASE model (bf16) for LoRA training ✓")

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

    private func printConfigSummary(config: LoRATrainingConfig, model: Flux2Model) {
        print()
        print("LoRA Training Configuration")
        print("-" .repeating(40))
        print()
        print("Model: \(model.displayName)")
        print("  Transformer: BASE bf16 (required for LoRA training)")
        print("  Text encoder: \(config.quantization.displayName)")
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
        print("Timestep Sampling:")
        print("  Strategy: \(config.timestepSampling.displayName)")
        if config.timestepSampling == .logitNormal {
            print("  Mean: \(config.logitNormalMean), Std: \(config.logitNormalStd)")
        } else if config.timestepSampling == .fluxShift {
            print("  Shift: \(config.fluxShiftValue)")
        }
        print()
        print("Loss Weighting:")
        print("  Strategy: \(config.lossWeighting.displayName)")
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

    // MARK: - Training

    /// Run LoRA training (Ostris-compatible implementation)
    /// Note: Base model is ALWAYS used for LoRA training (mandatory)
    private func runSimpleTrainer(
        config: LoRATrainingConfig,
        model: Flux2Model
    ) async throws {

        // Check for resume from checkpoint
        var startStep: Int = 0
        var optimizerStateURL: URL? = nil
        var pauseCheckpointToDelete: URL? = nil  // Pause checkpoints are deleted after resume

        if let resumePath = config.resumeFromCheckpoint {
            // Explicit resume path provided
            let stateURL = resumePath.appendingPathComponent("training_state.json")
            if FileManager.default.fileExists(atPath: stateURL.path) {
                let state = try TrainingState.load(from: stateURL)
                startStep = state.currentStep
                let optURL = resumePath.appendingPathComponent("optimizer_state.safetensors")
                if FileManager.default.fileExists(atPath: optURL.path) {
                    optimizerStateURL = optURL
                }
                // Check if this is a pause checkpoint (should be deleted after resume)
                let pauseMarker = resumePath.appendingPathComponent(".pause_checkpoint")
                if FileManager.default.fileExists(atPath: pauseMarker.path) {
                    pauseCheckpointToDelete = resumePath
                    print("Resuming from pause checkpoint: step \(startStep) (will be cleaned up)")
                } else {
                    print("Resuming from explicit checkpoint: step \(startStep)")
                }
            } else {
                throw ValidationError("Checkpoint not found at: \(resumePath.path)")
            }
        } else {
            // Auto-detect latest checkpoint in output directory
            if let latest = TrainingState.findLatestCheckpoint(in: config.outputPath) {
                let state = try TrainingState.load(from: latest.stateURL)
                startStep = state.currentStep
                let checkpointDir = latest.stateURL.deletingLastPathComponent()
                let optURL = checkpointDir.appendingPathComponent("optimizer_state.safetensors")
                if FileManager.default.fileExists(atPath: optURL.path) {
                    optimizerStateURL = optURL
                }
                // Check if this is a pause checkpoint (should be deleted after resume)
                let pauseMarker = checkpointDir.appendingPathComponent(".pause_checkpoint")
                let isPauseCheckpoint = FileManager.default.fileExists(atPath: pauseMarker.path)
                if isPauseCheckpoint {
                    pauseCheckpointToDelete = checkpointDir
                }
                print()
                print("📂 Found existing checkpoint at step \(startStep)\(isPauseCheckpoint ? " (pause checkpoint)" : "")")
                print("   Auto-resuming from: \(checkpointDir.lastPathComponent)/")
                if isPauseCheckpoint {
                    print("   ⚠️  Pause checkpoint will be deleted after successful resume")
                }
                print()
            }
        }


        // Create simple config from full config
        var simpleConfig = SimpleLoRAConfig(outputDir: config.outputPath)
        simpleConfig.rank = config.rank
        simpleConfig.alpha = config.alpha
        simpleConfig.learningRate = config.learningRate
        simpleConfig.weightDecay = config.weightDecay
        simpleConfig.batchSize = config.batchSize
        simpleConfig.maxSteps = config.maxSteps ?? (config.epochs * 100)  // Estimate if not set
        simpleConfig.saveEveryNSteps = config.saveEveryNSteps
        simpleConfig.logEveryNSteps = config.logEveryNSteps

        // Map timestep sampling
        switch config.timestepSampling {
        case .uniform:
            simpleConfig.timestepSampling = .uniform
        case .content:
            simpleConfig.timestepSampling = .content
        case .style:
            simpleConfig.timestepSampling = .style
        case .balanced:
            simpleConfig.timestepSampling = .balanced
        default:
            simpleConfig.timestepSampling = .balanced  // Default to balanced like Ostris
        }

        // Map loss weighting
        switch config.lossWeighting {
        case .bellShaped:
            simpleConfig.lossWeighting = .bellShaped
        default:
            simpleConfig.lossWeighting = .none
        }

        // Validation settings - convert from LoRATrainingConfig to SimpleLoRAConfig format
        simpleConfig.validationPrompts = config.validationPrompts.map { loraPrompt in
            SimpleLoRAConfig.ValidationPromptConfig(
                prompt: loraPrompt.prompt,
                is512: loraPrompt.is512,
                is1024: loraPrompt.is1024,
                applyTrigger: loraPrompt.applyTrigger,
                seed: loraPrompt.seed
            )
        }
        simpleConfig.validationEveryNSteps = config.validationEveryNSteps
        simpleConfig.validationSeed = config.validationSeed ?? 42
        simpleConfig.validationSteps = config.validationSteps

        // DOP (Differential Output Preservation) settings
        // This is critical for preventing LoRA from overstrengthening
        // Auto-enable DOP when trigger_word is set (unless explicitly disabled)
        simpleConfig.triggerWord = config.triggerWord
        if config.triggerWord != nil && config.diffOutputPreservation {
            simpleConfig.dopEnabled = true
            simpleConfig.dopMultiplier = config.diffOutputPreservationMultiplier
            simpleConfig.dopPreservationClass = config.diffOutputPreservationClass ?? "object"
            simpleConfig.dopEveryNSteps = config.diffOutputPreservationEveryNSteps
        } else {
            simpleConfig.dopEnabled = false
        }
        simpleConfig.gradientAccumulationSteps = config.gradientAccumulationSteps
        simpleConfig.gradientCheckpointing = config.gradientCheckpointing

        // Learning curve visualization
        simpleConfig.generateLearningCurve = config.generateLearningCurve
        simpleConfig.learningCurveSmoothingWindow = config.learningCurveSmoothingWindow

        // Load models
        print("=".repeating(60))
        print("Loading models...")
        print("=".repeating(60))
        print()

        // Load VAE
        print("Loading VAE...")
        let vae = try await loadVAE()
        print("  VAE loaded ✓")

        // Load text encoder
        print()
        print("Loading text encoder...")
        let textEncoder = try await loadTextEncoder(for: model, quantization: config.quantization)
        print("  Text encoder loaded ✓")

        // Load transformer
        print()
        print("Loading transformer...")
        let transformer = try await loadTransformer(for: model)
        print("  Transformer loaded ✓")

        // Create dataset
        print()
        print("Loading dataset...")
        let dataset = try TrainingDataset(config: config)
        print("  Loaded \(dataset.count) samples")

        // Pre-cache latents
        print()
        print("Pre-caching latents...")

        let latentCache = LatentCache(
            config: config,
            cacheDirectory: config.outputPath.appendingPathComponent(".latent_cache"),
            useMemoryCache: true
        )

        // Pre-encode latents
        try await latentCache.preEncodeDataset(dataset, vae: vae) { progress, total in
            print("  Encoding: \(progress)/\(total)")
        }

        // Collect cached latents
        var cachedLatents: [CachedLatentEntry] = []
        for sample in dataset.sampleMetadata {
            let dims = dataset.getTargetDimensions(for: sample.filename)
            if let latent = try latentCache.getLatent(for: sample.filename, width: dims.width, height: dims.height) {
                cachedLatents.append(CachedLatentEntry(
                    filename: sample.filename,
                    latent: latent,
                    width: dims.width,
                    height: dims.height
                ))
            }
        }
        print("  Cached \(cachedLatents.count) latents ✓")

        // Pre-cache text embeddings
        print()
        print("Pre-caching text embeddings...")
        var cachedEmbeddings: [String: CachedEmbeddingEntry] = [:]
        let uniqueCaptions = Set(dataset.allCaptions)

        for caption in uniqueCaptions {
            let embedding = try textEncoder.encodeForTraining(caption)
            cachedEmbeddings[caption] = CachedEmbeddingEntry(
                caption: caption,
                embedding: embedding
            )
        }
        print("  Cached \(cachedEmbeddings.count) embeddings ✓")

        // Update cachedEmbeddings to use filename as key
        var embeddingsByFilename: [String: CachedEmbeddingEntry] = [:]
        for sample in dataset.sampleMetadata {
            if let entry = cachedEmbeddings[sample.caption] {
                embeddingsByFilename[sample.filename] = entry
            }
        }

        // Clean up leftover control files from previous run
        TrainingController.cleanupControlFiles(outputDir: simpleConfig.outputDir)

        // Create training controller for pause/resume/stop support
        let controller = TrainingController(outputDirectory: simpleConfig.outputDir)

        // Create trainer with controller
        let trainer = SimpleLoRATrainer(config: simpleConfig, modelType: model, controller: controller)

        // Delete pause checkpoint now that everything is loaded and ready
        // This prevents disk saturation from accumulating pause checkpoints
        if let checkpointToDelete = pauseCheckpointToDelete {
            try? FileManager.default.removeItem(at: checkpointToDelete)
            print("🗑️  Cleaned up pause checkpoint: \(checkpointToDelete.lastPathComponent)")
            print()
        }

        // Run training
        print()
        try await trainer.train(
            transformer: transformer,
            cachedLatents: cachedLatents,
            cachedEmbeddings: embeddingsByFilename,
            vae: vae,
            textEncoder: { prompt in
                // Reload text encoder if it was unloaded (e.g., by baseline image generation)
                if !textEncoder.isLoaded {
                    try await textEncoder.load()
                }
                return try textEncoder.encodeForTraining(prompt)
            },
            startStep: startStep,
            optimizerState: optimizerStateURL
        )

        print()
        print("=".repeating(60))
        print("Training complete!")
        print("=".repeating(60))
        print()
        print("Output saved to: \(config.outputPath.path)")
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

/// Parse timestep sampling strategy string
private func parseTimestepSampling(_ input: String) -> TimestepSampling {
    switch input.lowercased() {
    case "logit_normal", "logit-normal", "logitnormal":
        return .logitNormal
    case "flux_shift", "flux-shift", "fluxshift":
        return .fluxShift
    case "content":
        return .content
    case "style":
        return .style
    case "balanced":
        return .balanced
    default:
        return .uniform
    }
}

/// Parse loss weighting strategy string
private func parseLossWeighting(_ input: String) -> LossWeighting {
    switch input.lowercased() {
    case "bell_shaped", "bell-shaped", "bellshaped", "weighted":
        return .bellShaped
    default:
        return .none
    }
}
