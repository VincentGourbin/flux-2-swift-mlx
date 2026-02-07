// TrainingControlTests.swift - Tests for training control functionality
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core

final class TrainingControlTests: XCTestCase {

    var tempDir: URL!

    override func setUpWithError() throws {
        // Create a temporary directory for each test
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("TrainingControlTests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        // Clean up temporary directory
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - TrainingState Tests

    func testTrainingStateInitialization() {
        let state = TrainingState(
            currentStep: 0,
            totalSteps: 1000,
            rngSeed: 42,
            configHash: "test_hash",
            modelType: "klein-4b",
            loraRank: 32,
            loraAlpha: 32.0
        )

        XCTAssertEqual(state.currentStep, 0)
        XCTAssertEqual(state.totalSteps, 1000)
        XCTAssertEqual(state.rngSeed, 42)
        XCTAssertEqual(state.modelType, "klein-4b")
        XCTAssertEqual(state.loraRank, 32)
        XCTAssertEqual(state.loraAlpha, 32.0)
        XCTAssertTrue(state.recentLosses.isEmpty)
        XCTAssertEqual(state.bestLoss, Float.infinity)
    }

    func testTrainingStateRecordLoss() {
        var state = TrainingState(
            currentStep: 1,
            totalSteps: 100,
            rngSeed: 42,
            configHash: "test",
            modelType: "klein-4b",
            loraRank: 32,
            loraAlpha: 32.0
        )

        state.recordLoss(1.5)
        state.recordLoss(1.3)
        state.recordLoss(1.1)

        XCTAssertEqual(state.recentLosses.count, 3)
        XCTAssertEqual(state.bestLoss, 1.1)
        XCTAssertEqual(state.averageLoss, (1.5 + 1.3 + 1.1) / 3.0, accuracy: 0.001)
    }

    func testTrainingStateSaveAndLoad() throws {
        var state = TrainingState(
            currentStep: 50,
            totalSteps: 500,
            rngSeed: 123,
            configHash: "abc123",
            modelType: "klein-9b",
            loraRank: 16,
            loraAlpha: 16.0
        )
        state.recordLoss(1.2)
        state.recordLoss(1.0)
        state.recordCheckpoint(step: 50)

        // Save
        let saveURL = tempDir.appendingPathComponent("training_state.json")
        try state.save(to: saveURL)

        // Verify file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: saveURL.path))

        // Load
        let loadedState = try TrainingState.load(from: saveURL)

        XCTAssertEqual(loadedState.currentStep, 50)
        XCTAssertEqual(loadedState.totalSteps, 500)
        XCTAssertEqual(loadedState.rngSeed, 123)
        XCTAssertEqual(loadedState.modelType, "klein-9b")
        XCTAssertEqual(loadedState.loraRank, 16)
        XCTAssertEqual(loadedState.recentLosses.count, 2)
        XCTAssertEqual(loadedState.checkpointSteps, [50])
    }

    func testFindLatestCheckpoint() throws {
        // Create checkpoint directories
        let checkpoint100 = tempDir.appendingPathComponent("checkpoint_000100")
        let checkpoint200 = tempDir.appendingPathComponent("checkpoint_000200")
        let checkpoint150 = tempDir.appendingPathComponent("checkpoint_000150")

        try FileManager.default.createDirectory(at: checkpoint100, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: checkpoint200, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: checkpoint150, withIntermediateDirectories: true)

        // Create training_state.json in each
        for (dir, step) in [(checkpoint100, 100), (checkpoint200, 200), (checkpoint150, 150)] {
            var state = TrainingState(
                currentStep: step,
                totalSteps: 500,
                rngSeed: 42,
                configHash: "test",
                modelType: "klein-4b",
                loraRank: 32,
                loraAlpha: 32.0
            )
            // Record a loss to avoid Float.infinity (which can't be serialized to JSON)
            state.recordLoss(1.0)
            try state.save(to: dir.appendingPathComponent("training_state.json"))
        }

        // Find latest
        let latest = TrainingState.findLatestCheckpoint(in: tempDir)

        XCTAssertNotNil(latest)
        XCTAssertEqual(latest?.step, 200)
    }

    func testFindLatestCheckpointNoCheckpoints() {
        let latest = TrainingState.findLatestCheckpoint(in: tempDir)
        XCTAssertNil(latest)
    }

    func testConfigHash() {
        let hash1 = TrainingState.hashConfig(
            modelType: "klein-4b",
            rank: 32,
            alpha: 32.0,
            learningRate: 1e-4,
            datasetPath: "/path/to/dataset"
        )

        let hash2 = TrainingState.hashConfig(
            modelType: "klein-4b",
            rank: 32,
            alpha: 32.0,
            learningRate: 1e-4,
            datasetPath: "/path/to/dataset"
        )

        let hash3 = TrainingState.hashConfig(
            modelType: "klein-9b",  // Different model
            rank: 32,
            alpha: 32.0,
            learningRate: 1e-4,
            datasetPath: "/path/to/dataset"
        )

        XCTAssertEqual(hash1, hash2)  // Same config = same hash
        XCTAssertNotEqual(hash1, hash3)  // Different config = different hash
    }

    // MARK: - TrainingController Tests

    func testTrainingControllerInitialization() {
        let controller = TrainingController(outputDirectory: tempDir)

        XCTAssertEqual(controller.status, .idle)
        XCTAssertNil(controller.state)
        XCTAssertEqual(controller.outputDirectory, tempDir)
    }

    func testTrainingControllerPauseResume() {
        let controller = TrainingController(outputDirectory: tempDir)

        // Request pause
        controller.requestPause()

        // Check pause file exists
        let pauseFile = tempDir.appendingPathComponent(".pause")
        XCTAssertTrue(FileManager.default.fileExists(atPath: pauseFile.path))
        XCTAssertTrue(controller.shouldPause())

        // Resume
        controller.resume()

        // Check pause file removed
        XCTAssertFalse(FileManager.default.fileExists(atPath: pauseFile.path))
        XCTAssertFalse(controller.shouldPause())
    }

    func testTrainingControllerStop() {
        let controller = TrainingController(outputDirectory: tempDir)

        // Request stop
        controller.requestStop()

        // Check stop file exists
        let stopFile = tempDir.appendingPathComponent(".stop")
        XCTAssertTrue(FileManager.default.fileExists(atPath: stopFile.path))
        XCTAssertTrue(controller.shouldStop())
    }

    func testTrainingControllerForceStop() {
        let controller = TrainingController(outputDirectory: tempDir)

        XCTAssertFalse(controller.shouldForceStop())

        controller.forceStop()

        XCTAssertTrue(controller.shouldForceStop())
        XCTAssertTrue(controller.shouldStop())  // Force stop also sets stop flag
    }

    func testTrainingControllerCheckpointRequest() {
        let controller = TrainingController(outputDirectory: tempDir)

        XCTAssertFalse(controller.shouldCheckpoint())

        controller.requestCheckpoint()

        XCTAssertTrue(controller.shouldCheckpoint())
        // Second call should return false (flag is cleared)
        XCTAssertFalse(controller.shouldCheckpoint())
    }

    func testTrainingControllerStateUpdate() {
        let controller = TrainingController(outputDirectory: tempDir)

        let state = TrainingState(
            currentStep: 100,
            totalSteps: 500,
            rngSeed: 42,
            configHash: "test",
            modelType: "klein-4b",
            loraRank: 32,
            loraAlpha: 32.0
        )

        controller.updateState(state)

        XCTAssertNotNil(controller.state)
        XCTAssertEqual(controller.state?.currentStep, 100)
    }

    func testTrainingControllerStatusChange() {
        let controller = TrainingController(outputDirectory: tempDir)

        XCTAssertEqual(controller.status, .idle)

        controller.setStatus(.running)
        XCTAssertEqual(controller.status, .running)

        controller.setStatus(.paused)
        XCTAssertEqual(controller.status, .paused)

        controller.setStatus(.completed)
        XCTAssertEqual(controller.status, .completed)
    }

    func testTrainingControllerStaticPauseResume() throws {
        // Test static CLI helper methods
        try TrainingController.pauseTraining(outputDir: tempDir)

        XCTAssertTrue(TrainingController.isPaused(outputDir: tempDir))

        try TrainingController.resumeTraining(outputDir: tempDir)

        XCTAssertFalse(TrainingController.isPaused(outputDir: tempDir))
    }

    func testTrainingControllerStaticStop() throws {
        try TrainingController.stopTraining(outputDir: tempDir)

        let stopFile = tempDir.appendingPathComponent(".stop")
        XCTAssertTrue(FileManager.default.fileExists(atPath: stopFile.path))
    }

    // MARK: - Observer Tests

    func testTrainingObserver() {
        let controller = TrainingController(outputDirectory: tempDir)
        let observer = MockTrainingObserver()

        controller.addObserver(observer)

        // Trigger events
        controller.notifyStepCompleted(step: 10, totalSteps: 100, loss: 1.5)
        controller.setStatus(.paused)

        XCTAssertEqual(observer.lastStep, 10)
        XCTAssertEqual(observer.lastLoss, 1.5)
        XCTAssertEqual(observer.lastStatus, .paused)

        // Remove observer
        controller.removeObserver(observer)
        controller.notifyStepCompleted(step: 20, totalSteps: 100, loss: 1.0)

        // Should not be updated
        XCTAssertEqual(observer.lastStep, 10)
    }

    // MARK: - Pause Checkpoint Marker Tests

    func testPauseCheckpointMarker() throws {
        // Create a checkpoint directory with pause marker
        let checkpointDir = tempDir.appendingPathComponent("checkpoint_000100")
        try FileManager.default.createDirectory(at: checkpointDir, withIntermediateDirectories: true)

        let pauseMarker = checkpointDir.appendingPathComponent(".pause_checkpoint")
        FileManager.default.createFile(atPath: pauseMarker.path, contents: nil)

        // Verify marker exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: pauseMarker.path))

        // Simulate cleanup after resume
        try FileManager.default.removeItem(at: checkpointDir)

        // Verify directory is removed
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkpointDir.path))
    }
}

// MARK: - Mock Observer

private class MockTrainingObserver: TrainingObserver {
    var lastStep: Int = 0
    var lastLoss: Float = 0
    var lastStatus: TrainingStatus = .idle
    var checkpointPaths: [URL] = []

    func trainingStatusChanged(_ status: TrainingStatus) {
        lastStatus = status
    }

    func trainingStepCompleted(step: Int, totalSteps: Int, loss: Float) {
        lastStep = step
        lastLoss = loss
    }

    func trainingCheckpointSaved(step: Int, path: URL) {
        checkpointPaths.append(path)
    }
}
