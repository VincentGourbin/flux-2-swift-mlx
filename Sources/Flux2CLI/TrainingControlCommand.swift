// TrainingControlCommand.swift - CLI commands for controlling training sessions
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core

// MARK: - Training Control Command Group

struct TrainingControlCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "training",
        abstract: "Control running training sessions",
        subcommands: [
            PauseCommand.self,
            ResumeCommand.self,
            StopCommand.self,
            StatusCommand.self
        ]
    )
}

// MARK: - Pause Command

struct PauseCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pause",
        abstract: "Pause a running training session"
    )

    @Argument(help: "Output directory of the training session")
    var outputDir: String

    func run() throws {
        let outputURL = URL(fileURLWithPath: outputDir)

        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw ValidationError("Output directory not found: \(outputDir)")
        }

        try TrainingController.pauseTraining(outputDir: outputURL)
    }
}

// MARK: - Resume Command

struct ResumeCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "resume",
        abstract: "Resume a paused training session"
    )

    @Argument(help: "Output directory of the training session")
    var outputDir: String

    func run() throws {
        let outputURL = URL(fileURLWithPath: outputDir)

        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw ValidationError("Output directory not found: \(outputDir)")
        }

        try TrainingController.resumeTraining(outputDir: outputURL)
    }
}

// MARK: - Stop Command

struct StopCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract: "Stop a running training session gracefully (saves checkpoint)"
    )

    @Argument(help: "Output directory of the training session")
    var outputDir: String

    func run() throws {
        let outputURL = URL(fileURLWithPath: outputDir)

        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw ValidationError("Output directory not found: \(outputDir)")
        }

        try TrainingController.stopTraining(outputDir: outputURL)
    }
}

// MARK: - Status Command

struct StatusCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Show status of a training session"
    )

    @Argument(help: "Output directory of the training session")
    var outputDir: String

    func run() throws {
        let outputURL = URL(fileURLWithPath: outputDir)

        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw ValidationError("Output directory not found: \(outputDir)")
        }

        // Check pause status
        let isPaused = TrainingController.isPaused(outputDir: outputURL)

        // Find latest checkpoint
        let latestCheckpoint = TrainingState.findLatestCheckpoint(in: outputURL)

        print("Training Session Status")
        print("-".repeating(40))
        print("Output directory: \(outputDir)")
        print()

        if isPaused {
            print("Status: PAUSED")
            print("  A .pause file exists in the output directory.")
            print("  Run 'flux2 training resume \(outputDir)' to continue.")
        } else {
            print("Status: Running or Idle")
        }
        print()

        if let checkpoint = latestCheckpoint {
            print("Latest checkpoint: step \(checkpoint.step)")

            // Load and display training state
            do {
                let state = try TrainingState.load(from: checkpoint.stateURL)
                print()
                print("Training Progress:")
                print("  Current step: \(state.currentStep) / \(state.totalSteps)")
                let progress = Float(state.currentStep) / Float(state.totalSteps) * 100
                print("  Progress: \(String(format: "%.1f", progress))%")
                print()
                print("Loss:")
                print("  Recent average: \(String(format: "%.4f", state.averageLoss))")
                print("  Best loss: \(String(format: "%.4f", state.bestLoss)) (step \(state.bestLossStep))")
                print()
                print("Timing:")
                let hours = Int(state.totalTrainingTime) / 3600
                let minutes = (Int(state.totalTrainingTime) % 3600) / 60
                print("  Total training time: \(hours)h \(minutes)m")
                let etaMinutes = Int(state.estimatedTimeRemaining / 60)
                let etaHours = etaMinutes / 60
                if etaHours > 0 {
                    print("  Estimated remaining: \(etaHours)h \(etaMinutes % 60)m")
                } else if etaMinutes > 0 {
                    print("  Estimated remaining: \(etaMinutes)m")
                }
                print()
                print("Configuration:")
                print("  Model: \(state.modelType)")
                print("  LoRA rank: \(state.loraRank)")
                print("  LoRA alpha: \(state.loraAlpha)")
                print()
                print("Checkpoints saved: \(state.checkpointSteps.count)")
                if !state.checkpointSteps.isEmpty {
                    let checkpointList = state.checkpointSteps.suffix(5).map { String($0) }.joined(separator: ", ")
                    print("  Recent: \(checkpointList)")
                }
            } catch {
                print("  (Could not load training state: \(error.localizedDescription))")
            }
        } else {
            print("No checkpoints found.")
            print("Training may not have started or saved any checkpoints yet.")
        }
    }
}

// MARK: - String Extension

private extension String {
    func repeating(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}
