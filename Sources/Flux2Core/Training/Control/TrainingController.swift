// TrainingController.swift - Training control protocol and implementation
// Copyright 2025 Vincent Gourbin

import Foundation

/// Training status enum
public enum TrainingStatus: String, Codable, Sendable {
    case idle = "idle"
    case running = "running"
    case paused = "paused"
    case checkpointing = "checkpointing"
    case completed = "completed"
    case failed = "failed"
    case cancelled = "cancelled"
}

/// Protocol for controlling training sessions
/// Designed to be used from both CLI and GUI applications
public protocol TrainingControllerProtocol: AnyObject, Sendable {

    /// Current training status
    var status: TrainingStatus { get }

    /// Current training state (progress, losses, etc.)
    var state: TrainingState? { get }

    /// Output directory for this training session
    var outputDirectory: URL { get }

    // MARK: - Control Commands

    /// Request pause after current step completes
    /// Returns immediately, training pauses asynchronously
    func requestPause()

    /// Resume paused training
    func resume()

    /// Request stop after current step (saves checkpoint)
    func requestStop()

    /// Force immediate stop (no checkpoint)
    func forceStop()

    /// Request immediate checkpoint
    func requestCheckpoint()

    // MARK: - Observation

    /// Add observer for training events
    func addObserver(_ observer: TrainingObserver)

    /// Remove observer
    func removeObserver(_ observer: TrainingObserver)
}

/// Observer protocol for training events
/// Implement this to receive training updates in your application
public protocol TrainingObserver: AnyObject {

    /// Called when training status changes
    func trainingStatusChanged(_ status: TrainingStatus)

    /// Called after each training step
    func trainingStepCompleted(step: Int, totalSteps: Int, loss: Float)

    /// Called when a checkpoint is saved
    func trainingCheckpointSaved(step: Int, path: URL)

    /// Called when validation images are generated
    func trainingValidationCompleted(step: Int, images: [URL])

    /// Called when training completes or fails
    func trainingFinished(success: Bool, message: String)

    /// Called when training is paused
    func trainingPaused(atStep: Int)

    /// Called when training is resumed
    func trainingResumed(atStep: Int)
}

/// Default implementations for optional observer methods
public extension TrainingObserver {
    func trainingStatusChanged(_ status: TrainingStatus) {}
    func trainingStepCompleted(step: Int, totalSteps: Int, loss: Float) {}
    func trainingCheckpointSaved(step: Int, path: URL) {}
    func trainingValidationCompleted(step: Int, images: [URL]) {}
    func trainingFinished(success: Bool, message: String) {}
    func trainingPaused(atStep: Int) {}
    func trainingResumed(atStep: Int) {}
}

/// Concrete implementation of TrainingController
/// Uses file-based signaling for pause/resume (works across processes)
public final class TrainingController: TrainingControllerProtocol, @unchecked Sendable {

    // MARK: - Properties

    public private(set) var status: TrainingStatus = .idle
    public private(set) var state: TrainingState?
    public let outputDirectory: URL

    private let lock = NSLock()
    private var observers: [WeakObserver] = []

    // Control flags (checked by training loop)
    private var _pauseRequested: Bool = false
    private var _stopRequested: Bool = false
    private var _forceStopRequested: Bool = false
    private var _checkpointRequested: Bool = false

    // File-based control (for cross-process communication)
    private var pauseFileURL: URL { outputDirectory.appendingPathComponent(".pause") }
    private var stopFileURL: URL { outputDirectory.appendingPathComponent(".stop") }
    private var checkpointFileURL: URL { outputDirectory.appendingPathComponent(".checkpoint") }

    // MARK: - Initialization

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    // MARK: - Control Commands

    public func requestPause() {
        lock.lock()
        _pauseRequested = true
        // Also create file for cross-process signaling
        FileManager.default.createFile(atPath: pauseFileURL.path, contents: nil)
        lock.unlock()

        // Notify AFTER releasing lock to avoid deadlock
        notifyObservers { $0.trainingStatusChanged(.paused) }
    }

    public func resume() {
        lock.lock()
        _pauseRequested = false
        // Remove pause file
        try? FileManager.default.removeItem(at: pauseFileURL)

        var shouldNotify = false
        var step = 0
        if status == .paused {
            status = .running
            shouldNotify = true
            step = state?.currentStep ?? 0
        }
        lock.unlock()

        // Notify AFTER releasing lock to avoid deadlock
        if shouldNotify {
            notifyObservers { $0.trainingResumed(atStep: step) }
        }
    }

    public func requestStop() {
        lock.lock()
        defer { lock.unlock() }

        _stopRequested = true

        // Create stop file for cross-process signaling
        FileManager.default.createFile(atPath: stopFileURL.path, contents: nil)
    }

    public func forceStop() {
        lock.lock()
        defer { lock.unlock() }

        _forceStopRequested = true
        _stopRequested = true
    }

    public func requestCheckpoint() {
        lock.lock()
        defer { lock.unlock() }

        _checkpointRequested = true
    }

    // MARK: - Status Checking (called by training loop)

    /// Check if pause is requested (call after each step)
    public func shouldPause() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        // Check both flag and file (for cross-process control)
        return _pauseRequested || FileManager.default.fileExists(atPath: pauseFileURL.path)
    }

    /// Check if stop is requested
    public func shouldStop() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        return _stopRequested || FileManager.default.fileExists(atPath: stopFileURL.path)
    }

    /// Check if force stop is requested
    public func shouldForceStop() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        return _forceStopRequested
    }

    /// Check if checkpoint is requested (and clear the flag)
    public func shouldCheckpoint() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        // Check programmatic flag
        if _checkpointRequested {
            _checkpointRequested = false
            return true
        }

        // Check file-based signal (for cross-process control)
        if FileManager.default.fileExists(atPath: checkpointFileURL.path) {
            // Remove the file to acknowledge the request
            try? FileManager.default.removeItem(at: checkpointFileURL)
            print("📸 Checkpoint requested via .checkpoint file")
            return true
        }

        return false
    }

    /// Wait while paused, checking periodically
    /// Returns true if should continue, false if should stop
    public func waitWhilePaused() -> Bool {
        guard shouldPause() else { return true }

        lock.lock()
        status = .paused
        let step = state?.currentStep ?? 0
        lock.unlock()

        notifyObservers { $0.trainingPaused(atStep: step) }
        print("⏸️  Training paused at step \(step)")
        print("   To resume: delete '\(pauseFileURL.path)' or call resume()")

        // Poll until unpaused or stopped
        while shouldPause() && !shouldStop() {
            Thread.sleep(forTimeInterval: 0.5)
        }

        if shouldStop() {
            return false
        }

        lock.lock()
        status = .running
        lock.unlock()

        notifyObservers { $0.trainingResumed(atStep: step) }
        print("▶️  Training resumed")

        return true
    }

    // MARK: - State Updates (called by training loop)

    /// Update training state
    public func updateState(_ state: TrainingState) {
        lock.lock()
        self.state = state
        lock.unlock()
    }

    /// Set training status
    public func setStatus(_ newStatus: TrainingStatus) {
        lock.lock()
        status = newStatus
        lock.unlock()

        notifyObservers { $0.trainingStatusChanged(newStatus) }
    }

    /// Notify step completion
    public func notifyStepCompleted(step: Int, totalSteps: Int, loss: Float) {
        notifyObservers { $0.trainingStepCompleted(step: step, totalSteps: totalSteps, loss: loss) }
    }

    /// Notify checkpoint saved
    public func notifyCheckpointSaved(step: Int, path: URL) {
        notifyObservers { $0.trainingCheckpointSaved(step: step, path: path) }
    }

    /// Notify validation completed
    public func notifyValidationCompleted(step: Int, images: [URL]) {
        notifyObservers { $0.trainingValidationCompleted(step: step, images: images) }
    }

    /// Notify training finished
    public func notifyFinished(success: Bool, message: String) {
        setStatus(success ? .completed : .failed)
        notifyObservers { $0.trainingFinished(success: success, message: message) }

        // Clean up control files
        try? FileManager.default.removeItem(at: pauseFileURL)
        try? FileManager.default.removeItem(at: stopFileURL)
    }

    // MARK: - Observers

    public func addObserver(_ observer: TrainingObserver) {
        lock.lock()
        defer { lock.unlock() }

        // Clean up dead references
        observers.removeAll { $0.observer == nil }
        observers.append(WeakObserver(observer))
    }

    public func removeObserver(_ observer: TrainingObserver) {
        lock.lock()
        defer { lock.unlock() }

        observers.removeAll { $0.observer === observer || $0.observer == nil }
    }

    private func notifyObservers(_ block: (TrainingObserver) -> Void) {
        lock.lock()
        let currentObservers = observers.compactMap { $0.observer }
        lock.unlock()

        for observer in currentObservers {
            block(observer)
        }
    }

    // MARK: - Weak Reference Wrapper

    private class WeakObserver {
        weak var observer: TrainingObserver?
        init(_ observer: TrainingObserver) {
            self.observer = observer
        }
    }
}

// MARK: - CLI Helper

extension TrainingController {

    /// Create a pause file for the given output directory (for CLI use)
    public static func pauseTraining(outputDir: URL) throws {
        let pauseFile = outputDir.appendingPathComponent(".pause")
        FileManager.default.createFile(atPath: pauseFile.path, contents: nil)
        print("Created pause signal: \(pauseFile.path)")
        print("Training will pause after the current step completes.")
    }

    /// Remove pause file to resume training (for CLI use)
    public static func resumeTraining(outputDir: URL) throws {
        let pauseFile = outputDir.appendingPathComponent(".pause")
        if FileManager.default.fileExists(atPath: pauseFile.path) {
            try FileManager.default.removeItem(at: pauseFile)
            print("Removed pause signal. Training will resume.")
        } else {
            print("Training is not paused (no .pause file found).")
        }
    }

    /// Check if training is paused
    public static func isPaused(outputDir: URL) -> Bool {
        let pauseFile = outputDir.appendingPathComponent(".pause")
        return FileManager.default.fileExists(atPath: pauseFile.path)
    }

    /// Request graceful stop
    public static func stopTraining(outputDir: URL) throws {
        let stopFile = outputDir.appendingPathComponent(".stop")
        FileManager.default.createFile(atPath: stopFile.path, contents: nil)
        print("Created stop signal: \(stopFile.path)")
        print("Training will stop after the current step and save a checkpoint.")
    }

    /// Clean up leftover control files from previous runs
    /// Call this when explicitly starting/resuming training
    public static func cleanupControlFiles(outputDir: URL) {
        let pauseFile = outputDir.appendingPathComponent(".pause")
        let stopFile = outputDir.appendingPathComponent(".stop")

        var cleaned: [String] = []

        if FileManager.default.fileExists(atPath: pauseFile.path) {
            try? FileManager.default.removeItem(atPath: pauseFile.path)
            cleaned.append(".pause")
        }

        if FileManager.default.fileExists(atPath: stopFile.path) {
            try? FileManager.default.removeItem(atPath: stopFile.path)
            cleaned.append(".stop")
        }

        if !cleaned.isEmpty {
            print("🧹 Cleaned up control files from previous run: \(cleaned.joined(separator: ", "))")
        }
    }
}
