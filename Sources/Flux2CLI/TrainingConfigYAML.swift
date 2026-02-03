// TrainingConfigYAML.swift - YAML configuration parser for LoRA training
// Copyright 2025 Vincent Gourbin

import Foundation
import Yams
import Flux2Core

// MARK: - YAML Configuration Structure

/// Root structure matching the YAML config file format
struct TrainingConfigYAML: Codable {
    var model: YAMLModelConfig?
    var lora: YAMLLoRAConfig?
    var dataset: DatasetConfig?
    var training: TrainingParams?
    var loss: LossConfig?
    var memory: MemoryConfig?
    var checkpoints: CheckpointsConfig?
    var validation: YAMLValidationConfig?
    var ema: EMAConfig?
    var earlyStop: EarlyStopConfig?

    enum CodingKeys: String, CodingKey {
        case model, lora, dataset, training, loss, memory, checkpoints, validation, ema
        case earlyStop = "early_stop"
    }
}

// MARK: - Model Configuration

struct YAMLModelConfig: Codable {
    var name: String?           // klein-4b, klein-9b, dev
    var quantization: String?   // bf16, int8, int4, nf4
    var useBase: Bool?          // Use base (non-distilled) model

    enum CodingKeys: String, CodingKey {
        case name, quantization
        case useBase = "use_base"
    }
}

// MARK: - LoRA Configuration

struct YAMLLoRAConfig: Codable {
    var rank: Int?
    var alpha: Float?
    var dropout: Float?
    var targetLayers: String?   // attention, attention_output, attention_ffn, all

    enum CodingKeys: String, CodingKey {
        case rank, alpha, dropout
        case targetLayers = "target_layers"
    }
}

// MARK: - Dataset Configuration

struct DatasetConfig: Codable {
    var path: String?
    var validationPath: String?
    var triggerWord: String?
    var captionFormat: String?  // txt, jsonl
    var imageSize: Int?

    enum CodingKeys: String, CodingKey {
        case path
        case validationPath = "validation_path"
        case triggerWord = "trigger_word"
        case captionFormat = "caption_format"
        case imageSize = "image_size"
    }
}

// MARK: - Training Parameters

struct TrainingParams: Codable {
    var batchSize: Int?
    var gradientAccumulation: Int?
    var epochs: Int?
    var maxSteps: Int?
    var warmupSteps: Int?
    var learningRate: Float?
    var weightDecay: Float?
    var captionDropout: Float?
    var maxGradNorm: Float?
    var lrScheduler: String?
    var evalEveryNSteps: Int?
    var logEveryNSteps: Int?

    enum CodingKeys: String, CodingKey {
        case batchSize = "batch_size"
        case gradientAccumulation = "gradient_accumulation"
        case epochs
        case maxSteps = "max_steps"
        case warmupSteps = "warmup_steps"
        case learningRate = "learning_rate"
        case weightDecay = "weight_decay"
        case captionDropout = "caption_dropout"
        case maxGradNorm = "max_grad_norm"
        case lrScheduler = "lr_scheduler"
        case evalEveryNSteps = "eval_every_n_steps"
        case logEveryNSteps = "log_every_n_steps"
    }
}

// MARK: - Loss Configuration

struct LossConfig: Codable {
    var weighting: String?          // uniform, snr_weighted, bell_shaped
    var timestepSampling: String?   // uniform, logit_normal, sigmoid
    var logitNormalMean: Float?
    var logitNormalStd: Float?
    var fluxShift: Float?

    enum CodingKeys: String, CodingKey {
        case weighting
        case timestepSampling = "timestep_sampling"
        case logitNormalMean = "logit_normal_mean"
        case logitNormalStd = "logit_normal_std"
        case fluxShift = "flux_shift"
    }
}

// MARK: - Memory Configuration

struct MemoryConfig: Codable {
    var gradientCheckpointing: Bool?
    var cacheLatents: Bool?
    var cacheTextEmbeddings: Bool?
    var cpuOffload: Bool?
    var bucketing: BucketingConfig?

    enum CodingKeys: String, CodingKey {
        case gradientCheckpointing = "gradient_checkpointing"
        case cacheLatents = "cache_latents"
        case cacheTextEmbeddings = "cache_text_embeddings"
        case cpuOffload = "cpu_offload"
        case bucketing
    }
}

struct BucketingConfig: Codable {
    var enabled: Bool?
    var resolutions: [Int]?
}

// MARK: - Checkpoints Configuration

struct CheckpointsConfig: Codable {
    var output: String?
    var saveEvery: Int?
    var keepLast: Int?

    enum CodingKeys: String, CodingKey {
        case output
        case saveEvery = "save_every"
        case keepLast = "keep_last"
    }
}

// MARK: - Validation Configuration

struct YAMLValidationConfig: Codable {
    var prompt: String?
    var everyNSteps: Int?
    var seed: UInt64?
    var guidance: Float?
    var steps: Int?

    enum CodingKeys: String, CodingKey {
        case prompt
        case everyNSteps = "every_n_steps"
        case seed, guidance, steps
    }
}

// MARK: - EMA Configuration

struct EMAConfig: Codable {
    var enabled: Bool?
    var decay: Float?
}

// MARK: - Early Stopping Configuration

struct EarlyStopConfig: Codable {
    var enabled: Bool?
    var patience: Int?
    var minDelta: Float?
    var onOverfit: Bool?
    var maxGap: Float?
    var gapPatience: Int?
    var onValStagnation: Bool?
    var minValImprovement: Float?
    var valPatience: Int?

    enum CodingKeys: String, CodingKey {
        case enabled, patience
        case minDelta = "min_delta"
        case onOverfit = "on_overfit"
        case maxGap = "max_gap"
        case gapPatience = "gap_patience"
        case onValStagnation = "on_val_stagnation"
        case minValImprovement = "min_val_improvement"
        case valPatience = "val_patience"
    }
}

// MARK: - YAML Parser

enum YAMLConfigError: LocalizedError {
    case fileNotFound(String)
    case parseError(String)
    case validationError(String)

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Configuration file not found: \(path)"
        case .parseError(let message):
            return "Failed to parse YAML configuration: \(message)"
        case .validationError(let message):
            return "Configuration validation error: \(message)"
        }
    }
}

/// Parser for YAML training configuration files
struct YAMLConfigParser {

    /// Load and parse a YAML configuration file
    static func load(from path: String) throws -> TrainingConfigYAML {
        let url = URL(fileURLWithPath: path)

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw YAMLConfigError.fileNotFound(path)
        }

        let yamlString = try String(contentsOf: url, encoding: .utf8)

        do {
            let decoder = YAMLDecoder()
            let config = try decoder.decode(TrainingConfigYAML.self, from: yamlString)
            return config
        } catch {
            throw YAMLConfigError.parseError(error.localizedDescription)
        }
    }

    /// Convert YAML config to LoRATrainingConfig with CLI overrides
    static func toLoRATrainingConfig(
        yaml: TrainingConfigYAML,
        cliOverrides: CLIOverrides
    ) throws -> (config: LoRATrainingConfig, modelVariant: Flux2Model, useBaseModel: Bool) {

        // Model
        let modelName = cliOverrides.model ?? yaml.model?.name ?? "klein-4b"
        guard let modelVariant = Flux2Model(rawValue: modelName) else {
            throw YAMLConfigError.validationError("Invalid model: \(modelName)")
        }

        let quantizationStr = cliOverrides.quantization ?? yaml.model?.quantization ?? "int8"
        guard let quantization = TrainingQuantization(rawValue: quantizationStr) else {
            throw YAMLConfigError.validationError("Invalid quantization: \(quantizationStr)")
        }

        let useBaseModel = cliOverrides.useBaseModel ?? yaml.model?.useBase ?? false

        // Dataset
        guard let datasetPath = cliOverrides.dataset ?? yaml.dataset?.path else {
            throw YAMLConfigError.validationError("Dataset path is required")
        }
        let datasetURL = URL(fileURLWithPath: datasetPath)

        let validationDatasetURL = (cliOverrides.validationDataset ?? yaml.dataset?.validationPath)
            .map { URL(fileURLWithPath: $0) }

        // LoRA
        let rank = cliOverrides.rank ?? yaml.lora?.rank ?? 16
        let alpha = cliOverrides.alpha ?? yaml.lora?.alpha ?? Float(rank)
        let dropout = yaml.lora?.dropout ?? 0.0

        let targetLayersStr = cliOverrides.targetLayers ?? yaml.lora?.targetLayers ?? "attention"
        guard let targetLayers = LoRATargetLayers(rawValue: targetLayersStr) else {
            throw YAMLConfigError.validationError("Invalid target layers: \(targetLayersStr)")
        }

        // Training
        let batchSize = cliOverrides.batchSize ?? yaml.training?.batchSize ?? 1
        let gradientAccumulation = yaml.training?.gradientAccumulation ?? 1
        let epochs = yaml.training?.epochs ?? 10
        let maxSteps = cliOverrides.maxSteps ?? yaml.training?.maxSteps
        let warmupSteps = yaml.training?.warmupSteps ?? 100
        let learningRate = cliOverrides.learningRate ?? yaml.training?.learningRate ?? 1e-4
        let weightDecay = yaml.training?.weightDecay ?? 0.01
        let captionDropout = yaml.training?.captionDropout ?? 0.0
        let maxGradNorm = yaml.training?.maxGradNorm ?? 1.0

        let lrSchedulerStr = yaml.training?.lrScheduler ?? "cosine"
        guard let lrScheduler = LRSchedulerType(rawValue: lrSchedulerStr) else {
            throw YAMLConfigError.validationError("Invalid LR scheduler: \(lrSchedulerStr)")
        }

        // Loss
        let lossWeighting = parseLossWeighting(yaml.loss?.weighting ?? "none")
        let timestepSampling = parseTimestepSampling(yaml.loss?.timestepSampling ?? "uniform")
        let logitNormalMean = yaml.loss?.logitNormalMean ?? 0.0
        let logitNormalStd = yaml.loss?.logitNormalStd ?? 1.0
        let fluxShift = yaml.loss?.fluxShift ?? 1.0

        // Memory
        let gradientCheckpointing = yaml.memory?.gradientCheckpointing ?? false
        let cacheLatents = yaml.memory?.cacheLatents ?? false
        let cacheTextEmbeddings = yaml.memory?.cacheTextEmbeddings ?? false
        let cpuOffload = yaml.memory?.cpuOffload ?? false

        // Bucketing
        let enableBucketing = yaml.memory?.bucketing?.enabled ?? false
        let bucketResolutions = yaml.memory?.bucketing?.resolutions ?? [512, 768, 1024]
        let imageSize = yaml.dataset?.imageSize ?? 512

        // Output
        guard let outputPath = cliOverrides.output ?? yaml.checkpoints?.output else {
            throw YAMLConfigError.validationError("Output path is required")
        }
        let outputURL = URL(fileURLWithPath: outputPath)

        let saveEveryNSteps = yaml.checkpoints?.saveEvery ?? 500
        let keepCheckpoints = yaml.checkpoints?.keepLast ?? 0

        // Validation
        let validationPrompt = cliOverrides.validationPrompt ?? yaml.validation?.prompt
        let validateEveryNSteps = yaml.validation?.everyNSteps ?? 500
        let validationSeed = yaml.validation?.seed ?? 42

        // EMA
        let useEMA = yaml.ema?.enabled ?? true
        let emaDecay = yaml.ema?.decay ?? 0.99

        // Early stopping
        let earlyStop = yaml.earlyStop?.enabled ?? false
        let earlyStopPatience = yaml.earlyStop?.patience ?? 5
        let earlyStopMinDelta = yaml.earlyStop?.minDelta ?? 0.01
        let earlyStopOnOverfit = yaml.earlyStop?.onOverfit ?? false
        let earlyStopMaxGap = yaml.earlyStop?.maxGap ?? 0.5
        let earlyStopGapPatience = yaml.earlyStop?.gapPatience ?? 3
        let earlyStopOnValStagnation = yaml.earlyStop?.onValStagnation ?? false
        let earlyStopMinValImprovement = yaml.earlyStop?.minValImprovement ?? 0.1
        let earlyStopValPatience = yaml.earlyStop?.valPatience ?? 2

        // Misc
        let triggerWord = cliOverrides.triggerWord ?? yaml.dataset?.triggerWord
        let captionFormat = yaml.dataset?.captionFormat ?? "txt"
        let evalEveryNSteps = yaml.training?.evalEveryNSteps ?? 10
        let logEveryNSteps = yaml.training?.logEveryNSteps ?? 10
        let verbose = cliOverrides.verbose ?? false

        // Create config
        let config = LoRATrainingConfig(
            // Dataset
            datasetPath: datasetURL,
            validationDatasetPath: validationDatasetURL,
            captionExtension: captionFormat,
            triggerWord: triggerWord,
            imageSize: imageSize,
            enableBucketing: enableBucketing,
            bucketResolutions: bucketResolutions,
            shuffleDataset: true,
            captionDropoutRate: captionDropout,
            // LoRA
            rank: rank,
            alpha: alpha,
            dropout: dropout,
            targetLayers: targetLayers,
            // Training
            learningRate: learningRate,
            batchSize: batchSize,
            epochs: epochs,
            maxSteps: maxSteps,
            warmupSteps: warmupSteps,
            lrScheduler: lrScheduler,
            weightDecay: weightDecay,
            adamBeta1: 0.9,
            adamBeta2: 0.999,
            adamEpsilon: 1e-8,
            maxGradNorm: maxGradNorm,
            gradientAccumulationSteps: gradientAccumulation,
            // Timestep sampling
            timestepSampling: timestepSampling,
            logitNormalMean: logitNormalMean,
            logitNormalStd: logitNormalStd,
            fluxShiftValue: fluxShift,
            // Loss weighting
            lossWeighting: lossWeighting,
            // Memory
            quantization: quantization,
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
            // EMA
            useEMA: useEMA,
            emaDecay: emaDecay,
            // Resume
            resumeFromCheckpoint: nil
        )

        return (config, modelVariant, useBaseModel)
    }

    // MARK: - Helper Functions

    private static func parseTimestepSampling(_ input: String) -> TimestepSampling {
        switch input.lowercased() {
        case "logit_normal", "logit-normal", "logitnormal":
            return .logitNormal
        case "flux_shift", "flux-shift", "fluxshift":
            return .fluxShift
        case "content":
            return .content
        case "style":
            return .style
        default:
            return .uniform
        }
    }

    private static func parseLossWeighting(_ input: String) -> LossWeighting {
        switch input.lowercased() {
        case "bell_shaped", "bell-shaped", "bellshaped", "weighted":
            return .bellShaped
        case "snr", "snr_weighted":
            return .snr
        default:
            return .none
        }
    }
}

// MARK: - CLI Overrides

/// Struct to hold CLI argument overrides that take precedence over YAML config
struct CLIOverrides {
    var dataset: String?
    var validationDataset: String?
    var output: String?
    var triggerWord: String?
    var model: String?
    var quantization: String?
    var useBaseModel: Bool?
    var rank: Int?
    var alpha: Float?
    var targetLayers: String?
    var learningRate: Float?
    var batchSize: Int?
    var maxSteps: Int?
    var validationPrompt: String?
    var verbose: Bool?
}
