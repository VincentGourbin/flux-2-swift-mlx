// TrainingSession.swift - High-level training session management
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// High-level training session that can be controlled and observed
/// Designed for integration with GUI applications
public final class TrainingSession: @unchecked Sendable {

    // MARK: - Types

    public enum SessionError: LocalizedError {
        case alreadyRunning
        case notRunning
        case noCheckpointFound
        case incompatibleCheckpoint(String)
        case missingOptimizerState
        case invalidState(String)

        public var errorDescription: String? {
            switch self {
            case .alreadyRunning:
                return "Training session is already running"
            case .notRunning:
                return "Training session is not running"
            case .noCheckpointFound:
                return "No checkpoint found to resume from"
            case .incompatibleCheckpoint(let reason):
                return "Checkpoint is incompatible: \(reason)"
            case .missingOptimizerState:
                return "Optimizer state not found in checkpoint"
            case .invalidState(let reason):
                return "Invalid training state: \(reason)"
            }
        }
    }

    /// Resume mode when starting a session
    public enum ResumeMode {
        /// Start fresh, fail if checkpoint exists
        case fresh

        /// Resume from latest checkpoint if available, start fresh otherwise
        case autoResume

        /// Resume from specific checkpoint step
        case fromStep(Int)

        /// Force fresh start, ignore existing checkpoints
        case forceFresh
    }

    // MARK: - Properties

    /// The training controller for pause/resume/stop
    public let controller: TrainingController

    /// Current training state
    public var state: TrainingState? { controller.state }

    /// Current status
    public var status: TrainingStatus { controller.status }

    /// Output directory
    public var outputDirectory: URL { controller.outputDirectory }

    /// The underlying trainer (set when training starts)
    private var trainer: SimpleLoRATrainer?

    /// Task running the training
    private var trainingTask: Task<Void, Error>?

    // MARK: - Initialization

    public init(outputDirectory: URL) {
        self.controller = TrainingController(outputDirectory: outputDirectory)
    }

    // MARK: - Training Control

    /// Start or resume training
    /// - Parameters:
    ///   - config: Training configuration
    ///   - modelType: Model to train
    ///   - resumeMode: How to handle existing checkpoints
    ///   - transformer: Pre-loaded transformer (optional, will load if nil)
    ///   - cachedLatents: Pre-cached latents
    ///   - cachedEmbeddings: Pre-cached embeddings
    ///   - textEncoder: Text encoder function for DOP
    public func start(
        config: SimpleLoRAConfig,
        modelType: Flux2Model,
        resumeMode: ResumeMode = .autoResume,
        transformer: Flux2Transformer2DModel,
        cachedLatents: [CachedLatentEntry],
        cachedEmbeddings: [String: CachedEmbeddingEntry],
        vae: AutoencoderKLFlux2? = nil,
        textEncoder: ((String) async throws -> MLXArray)? = nil
    ) async throws {

        guard status == .idle || status == .completed || status == .failed else {
            throw SessionError.alreadyRunning
        }

        // Check for resume
        let (startStep, optimizerState) = try resolveResumeMode(
            resumeMode: resumeMode,
            config: config,
            modelType: modelType
        )

        // Create trainer with controller
        var trainerConfig = config
        let trainer = SimpleLoRATrainer(
            config: trainerConfig,
            modelType: modelType,
            controller: controller
        )
        self.trainer = trainer

        // Initialize training state
        let configHash = TrainingState.hashConfig(
            modelType: modelType.rawValue,
            rank: config.rank,
            alpha: config.alpha,
            learningRate: config.learningRate,
            datasetPath: config.outputDir.path
        )

        var initialState = TrainingState(
            currentStep: startStep,
            totalSteps: config.maxSteps,
            rngSeed: config.validationSeed,
            configHash: configHash,
            modelType: modelType.rawValue,
            loraRank: config.rank,
            loraAlpha: config.alpha
        )

        controller.updateState(initialState)
        controller.setStatus(.running)

        // Run training directly (not in background task to avoid Sendable issues)
        do {
            try await trainer.train(
                transformer: transformer,
                cachedLatents: cachedLatents,
                cachedEmbeddings: cachedEmbeddings,
                vae: vae,
                textEncoder: textEncoder,
                startStep: startStep,
                optimizerState: optimizerState
            )
            controller.notifyFinished(success: true, message: "Training completed successfully")
        } catch {
            controller.notifyFinished(success: false, message: error.localizedDescription)
            throw error
        }
    }

    /// Wait for training to complete (no-op when training runs synchronously)
    public func wait() async throws {
        // Training now runs synchronously in start(), so nothing to wait for
    }

    /// Pause training
    public func pause() {
        controller.requestPause()
    }

    /// Resume training
    public func resume() {
        controller.resume()
    }

    /// Stop training gracefully (saves checkpoint)
    public func stop() {
        controller.requestStop()
    }

    /// Force stop training immediately
    public func forceStop() {
        controller.forceStop()
        trainingTask?.cancel()
    }

    /// Request immediate checkpoint
    public func checkpoint() {
        controller.requestCheckpoint()
    }

    // MARK: - Observers

    public func addObserver(_ observer: TrainingObserver) {
        controller.addObserver(observer)
    }

    public func removeObserver(_ observer: TrainingObserver) {
        controller.removeObserver(observer)
    }

    // MARK: - Resume Logic

    private func resolveResumeMode(
        resumeMode: ResumeMode,
        config: SimpleLoRAConfig,
        modelType: Flux2Model
    ) throws -> (startStep: Int, optimizerState: URL?) {

        switch resumeMode {
        case .fresh:
            // Check if checkpoint exists
            if let latest = TrainingState.findLatestCheckpoint(in: config.outputDir) {
                throw SessionError.invalidState(
                    "Checkpoint exists at step \(latest.step). Use .autoResume or .forceFresh"
                )
            }
            return (startStep: 0, optimizerState: nil)

        case .forceFresh:
            return (startStep: 0, optimizerState: nil)

        case .autoResume:
            guard let latest = TrainingState.findLatestCheckpoint(in: config.outputDir) else {
                print("No checkpoint found, starting fresh")
                return (startStep: 0, optimizerState: nil)
            }

            // Load and verify state
            let state = try TrainingState.load(from: latest.stateURL)
            try verifyCheckpointCompatibility(state: state, config: config, modelType: modelType)

            let checkpointDir = latest.stateURL.deletingLastPathComponent()
            let optimizerStateURL = checkpointDir.appendingPathComponent("optimizer_state.safetensors")

            print("Resuming from checkpoint at step \(latest.step)")
            return (
                startStep: latest.step,
                optimizerState: FileManager.default.fileExists(atPath: optimizerStateURL.path) ? optimizerStateURL : nil
            )

        case .fromStep(let step):
            let checkpointDir = config.outputDir.appendingPathComponent("checkpoint_\(String(format: "%06d", step))")
            let stateURL = checkpointDir.appendingPathComponent("training_state.json")

            guard FileManager.default.fileExists(atPath: stateURL.path) else {
                throw SessionError.noCheckpointFound
            }

            let state = try TrainingState.load(from: stateURL)
            try verifyCheckpointCompatibility(state: state, config: config, modelType: modelType)

            let optimizerStateURL = checkpointDir.appendingPathComponent("optimizer_state.safetensors")

            print("Resuming from checkpoint at step \(step)")
            return (
                startStep: step,
                optimizerState: FileManager.default.fileExists(atPath: optimizerStateURL.path) ? optimizerStateURL : nil
            )
        }
    }

    private func verifyCheckpointCompatibility(
        state: TrainingState,
        config: SimpleLoRAConfig,
        modelType: Flux2Model
    ) throws {
        // Verify model type
        if state.modelType != modelType.rawValue {
            throw SessionError.incompatibleCheckpoint(
                "Model type mismatch: checkpoint=\(state.modelType), config=\(modelType.rawValue)"
            )
        }

        // Verify LoRA rank
        if state.loraRank != config.rank {
            throw SessionError.incompatibleCheckpoint(
                "LoRA rank mismatch: checkpoint=\(state.loraRank), config=\(config.rank)"
            )
        }

        // Verify LoRA alpha
        if state.loraAlpha != config.alpha {
            throw SessionError.incompatibleCheckpoint(
                "LoRA alpha mismatch: checkpoint=\(state.loraAlpha), config=\(config.alpha)"
            )
        }
    }
}

// Note: SimpleLoRATrainer has a built-in init that accepts controller
