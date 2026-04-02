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

    // MARK: - VLM Score Tests

    func testVLMPromptScoreCodable() throws {
        let score = VLMPromptScore(
            promptIndex: 0,
            sceneScore: 62,
            styleScore: 70,
            sceneReason: "Good subject match",
            styleReason: "Color palette matches",
            baselineSceneScore: 15,
            baselineStyleScore: 20,
            isTriggered: false
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(score)
        let decoded = try JSONDecoder().decode(VLMPromptScore.self, from: data)

        XCTAssertEqual(decoded.promptIndex, 0)
        XCTAssertEqual(decoded.sceneScore, 62)
        XCTAssertEqual(decoded.styleScore, 70)
        XCTAssertEqual(decoded.sceneReason, "Good subject match")
        XCTAssertEqual(decoded.styleReason, "Color palette matches")
        XCTAssertEqual(decoded.baselineSceneScore, 15)
        XCTAssertEqual(decoded.baselineStyleScore, 20)
        XCTAssertFalse(decoded.isTriggered)
    }

    func testVLMPromptScoreIsTriggeredDefault() throws {
        // isTriggered defaults to true for backward compatibility
        let score = VLMPromptScore(
            promptIndex: 0, sceneScore: 50, styleScore: 60,
            sceneReason: "", styleReason: ""
        )
        XCTAssertTrue(score.isTriggered)

        // Round-trip preserves default
        let data = try JSONEncoder().encode(score)
        let decoded = try JSONDecoder().decode(VLMPromptScore.self, from: data)
        XCTAssertTrue(decoded.isTriggered)
    }

    func testVLMPromptScoreBackwardCompatNoisTriggered() throws {
        // Old JSON without isTriggered field should decode with default true
        let oldJson = """
        {
            "promptIndex": 0,
            "sceneScore": 55,
            "styleScore": 65,
            "sceneReason": "OK",
            "styleReason": "OK"
        }
        """
        let data = oldJson.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(VLMPromptScore.self, from: data)
        XCTAssertEqual(decoded.sceneScore, 55)
        XCTAssertTrue(decoded.isTriggered)  // default when missing
    }

    func testVLMScoreRecordCodable() throws {
        let record = VLMScoreRecord(
            step: 50,
            promptScores: [
                VLMPromptScore(promptIndex: 0, sceneScore: 60, styleScore: 70,
                               sceneReason: "test", styleReason: "test",
                               baselineSceneScore: 10, baselineStyleScore: 15),
                VLMPromptScore(promptIndex: 1, sceneScore: 55, styleScore: 65,
                               sceneReason: "test2", styleReason: "test2")
            ],
            compositeScore: 62.5,
            baselineComposite: 12.5,
            improvement: 50.0
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(record)
        let decoded = try JSONDecoder().decode(VLMScoreRecord.self, from: data)

        XCTAssertEqual(decoded.step, 50)
        XCTAssertEqual(decoded.promptScores.count, 2)
        XCTAssertEqual(decoded.compositeScore, 62.5, accuracy: 0.01)
        XCTAssertEqual(decoded.baselineComposite, 12.5)
        XCTAssertEqual(decoded.improvement, 50.0)
        // Second prompt has nil baseline scores
        XCTAssertNil(decoded.promptScores[1].baselineSceneScore)
    }

    func testRecordVLMScoreUpdatesBest() {
        var state = TrainingState(
            currentStep: 0,
            totalSteps: 100,
            rngSeed: 42,
            configHash: "test",
            modelType: "klein-4b",
            loraRank: 32,
            loraAlpha: 32.0
        )

        XCTAssertEqual(state.bestVLMScore, 0)
        XCTAssertEqual(state.bestVLMStep, 0)
        XCTAssertTrue(state.vlmScoreHistory.isEmpty)

        // Record first score
        let record1 = VLMScoreRecord(step: 25, promptScores: [],
                                      compositeScore: 30.0)
        state.recordVLMScore(record1)

        XCTAssertEqual(state.bestVLMScore, 30.0, accuracy: 0.01)
        XCTAssertEqual(state.bestVLMStep, 25)
        XCTAssertEqual(state.vlmScoreHistory.count, 1)

        // Record better score
        let record2 = VLMScoreRecord(step: 50, promptScores: [],
                                      compositeScore: 61.0)
        state.recordVLMScore(record2)

        XCTAssertEqual(state.bestVLMScore, 61.0, accuracy: 0.01)
        XCTAssertEqual(state.bestVLMStep, 50)

        // Record worse score — best should NOT change
        let record3 = VLMScoreRecord(step: 75, promptScores: [],
                                      compositeScore: 42.0)
        state.recordVLMScore(record3)

        XCTAssertEqual(state.bestVLMScore, 61.0, accuracy: 0.01)
        XCTAssertEqual(state.bestVLMStep, 50)
        XCTAssertEqual(state.vlmScoreHistory.count, 3)
    }

    func testVLMScoreHistoryPersistence() throws {
        var state = TrainingState(
            currentStep: 50,
            totalSteps: 100,
            rngSeed: 42,
            configHash: "test",
            modelType: "klein-4b",
            loraRank: 32,
            loraAlpha: 32.0
        )
        state.recordLoss(0.5)  // Avoid Float.infinity in JSON

        let record = VLMScoreRecord(
            step: 25,
            promptScores: [
                VLMPromptScore(promptIndex: 0, sceneScore: 58, styleScore: 66,
                               sceneReason: "Good", styleReason: "Great")
            ],
            compositeScore: 62.0,
            baselineComposite: 10.0,
            improvement: 52.0
        )
        state.recordVLMScore(record)

        // Save and reload
        let saveURL = tempDir.appendingPathComponent("vlm_state.json")
        try state.save(to: saveURL)
        let loaded = try TrainingState.load(from: saveURL)

        XCTAssertEqual(loaded.vlmScoreHistory.count, 1)
        XCTAssertEqual(loaded.vlmScoreHistory[0].step, 25)
        XCTAssertEqual(loaded.vlmScoreHistory[0].compositeScore, 62.0, accuracy: 0.01)
        XCTAssertEqual(loaded.vlmScoreHistory[0].promptScores[0].sceneScore, 58)
        XCTAssertEqual(loaded.bestVLMScore, 62.0, accuracy: 0.01)
        XCTAssertEqual(loaded.bestVLMStep, 25)
    }

    func testVLMScoreBackwardCompatibility() throws {
        // Simulate loading a training_state.json from BEFORE VLM scoring was added
        // (no vlmScoreHistory, bestVLMScore, bestVLMStep fields)
        let oldJson = """
        {
            "currentStep": 100,
            "totalSteps": 500,
            "currentEpoch": 0,
            "totalEpochs": 1,
            "recentLosses": [0.5, 0.4],
            "bestLoss": 0.4,
            "bestLossStep": 100,
            "startedAt": "2025-01-01T00:00:00Z",
            "totalTrainingTime": 3600,
            "rngSeed": 42,
            "configHash": "abc",
            "modelType": "klein-4b",
            "loraRank": 32,
            "loraAlpha": 32.0,
            "checkpointSteps": [50, 100]
        }
        """
        let saveURL = tempDir.appendingPathComponent("old_state.json")
        try oldJson.data(using: .utf8)!.write(to: saveURL)

        let loaded = try TrainingState.load(from: saveURL)

        // VLM fields should have defaults
        XCTAssertTrue(loaded.vlmScoreHistory.isEmpty)
        XCTAssertEqual(loaded.bestVLMScore, 0)
        XCTAssertEqual(loaded.bestVLMStep, 0)
        // Original fields still work
        XCTAssertEqual(loaded.currentStep, 100)
        XCTAssertEqual(loaded.bestLoss, 0.4, accuracy: 0.01)
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
