// TrainingState.swift - Persistable training state for checkpoint/resume
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Complete training state that can be saved and restored
public struct TrainingState: Codable, Sendable {

    // MARK: - Progress

    /// Current training step (1-based)
    public var currentStep: Int

    /// Total steps planned
    public var totalSteps: Int

    /// Current epoch (if applicable)
    public var currentEpoch: Int

    /// Total epochs planned
    public var totalEpochs: Int

    // MARK: - Loss History

    /// Recent loss values (last N steps for averaging)
    public var recentLosses: [Float]

    /// Best loss achieved
    public var bestLoss: Float

    /// Step at which best loss was achieved
    public var bestLossStep: Int

    /// Average loss over recent window
    public var averageLoss: Float {
        guard !recentLosses.isEmpty else { return 0 }
        return recentLosses.reduce(0, +) / Float(recentLosses.count)
    }

    // MARK: - Timing

    /// Training start timestamp
    public var startedAt: Date

    /// Last checkpoint timestamp
    public var lastCheckpointAt: Date?

    /// Total training time in seconds (excluding pauses)
    public var totalTrainingTime: TimeInterval

    /// Estimated time remaining based on current pace
    public var estimatedTimeRemaining: TimeInterval {
        guard currentStep > 0 else { return 0 }
        let avgTimePerStep = totalTrainingTime / Double(currentStep)
        let remainingSteps = totalSteps - currentStep
        return avgTimePerStep * Double(remainingSteps)
    }

    // MARK: - RNG State

    /// Random seed used for training
    public var rngSeed: UInt64

    // MARK: - Config Verification

    /// Hash of the training config (to verify resume compatibility)
    public var configHash: String

    /// Model type used
    public var modelType: String

    /// LoRA rank
    public var loraRank: Int

    /// LoRA alpha
    public var loraAlpha: Float

    // MARK: - Checkpoint Info

    /// Path to optimizer state (relative to checkpoint dir)
    public var optimizerStatePath: String?

    /// List of saved checkpoints
    public var checkpointSteps: [Int]

    // MARK: - Initialization

    public init(
        currentStep: Int = 0,
        totalSteps: Int,
        currentEpoch: Int = 0,
        totalEpochs: Int = 1,
        rngSeed: UInt64 = 42,
        configHash: String,
        modelType: String,
        loraRank: Int,
        loraAlpha: Float
    ) {
        self.currentStep = currentStep
        self.totalSteps = totalSteps
        self.currentEpoch = currentEpoch
        self.totalEpochs = totalEpochs
        self.recentLosses = []
        self.bestLoss = Float.infinity
        self.bestLossStep = 0
        self.startedAt = Date()
        self.lastCheckpointAt = nil
        self.totalTrainingTime = 0
        self.rngSeed = rngSeed
        self.configHash = configHash
        self.modelType = modelType
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.optimizerStatePath = nil
        self.checkpointSteps = []
    }

    // MARK: - Updates

    /// Record a loss value
    public mutating func recordLoss(_ loss: Float, maxHistory: Int = 100) {
        recentLosses.append(loss)
        if recentLosses.count > maxHistory {
            recentLosses.removeFirst()
        }

        if loss < bestLoss {
            bestLoss = loss
            bestLossStep = currentStep
        }
    }

    /// Record a checkpoint
    public mutating func recordCheckpoint(step: Int) {
        checkpointSteps.append(step)
        lastCheckpointAt = Date()
    }

    // MARK: - Persistence

    /// Save state to JSON file
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    /// Load state from JSON file
    public static func load(from url: URL) throws -> TrainingState {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(TrainingState.self, from: data)
    }

    /// Find the latest checkpoint in a directory
    public static func findLatestCheckpoint(in outputDir: URL) -> (step: Int, stateURL: URL)? {
        let fm = FileManager.default

        // Look for checkpoint_XXXXXX directories
        guard let contents = try? fm.contentsOfDirectory(at: outputDir, includingPropertiesForKeys: nil) else {
            return nil
        }

        let checkpointDirs = contents
            .filter { $0.lastPathComponent.hasPrefix("checkpoint_") }
            .compactMap { dir -> (step: Int, url: URL)? in
                let name = dir.lastPathComponent
                guard let stepStr = name.split(separator: "_").last,
                      let step = Int(stepStr) else { return nil }
                let stateURL = dir.appendingPathComponent("training_state.json")
                guard fm.fileExists(atPath: stateURL.path) else { return nil }
                return (step: step, url: stateURL)
            }
            .sorted { $0.step > $1.step }

        return checkpointDirs.first.map { (step: $0.step, stateURL: $0.url) }
    }
}

// MARK: - Config Hash Helper

extension TrainingState {
    /// Generate a hash from training config for verification
    public static func hashConfig(
        modelType: String,
        rank: Int,
        alpha: Float,
        learningRate: Float,
        datasetPath: String
    ) -> String {
        let configString = "\(modelType)|\(rank)|\(alpha)|\(learningRate)|\(datasetPath)"
        var hash: UInt64 = 5381
        for char in configString.utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt64(char)
        }
        return String(format: "%016llx", hash)
    }
}
