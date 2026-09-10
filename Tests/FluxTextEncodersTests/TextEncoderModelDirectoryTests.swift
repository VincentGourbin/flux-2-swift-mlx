/**
 * TextEncoderModelDirectoryTests.swift
 * Tests for custom model directory configuration in TextEncoderModelDownloader
 */

import XCTest
@testable import FluxTextEncoders

final class TextEncoderModelDirectoryTests: XCTestCase {

    // MARK: - Setup / Teardown

    override func tearDown() {
        // Always reset to default after each test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
        super.tearDown()
    }

    // MARK: - Default Directory

    func testDefaultModelsDirectoryIsMistralModels() {
        let expected = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral")
            .appendingPathComponent("models")
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, expected)
    }

    // MARK: - Custom Directory

    func testCustomModelsDirectoryOverridesDefault() {
        let custom = URL(fileURLWithPath: "/tmp/test-text-models")
        TextEncoderModelDownloader.customModelsDirectory = custom
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, custom)
    }

    func testCustomModelsDirectoryNilFallsBackToDefault() {
        let custom = URL(fileURLWithPath: "/tmp/test-text-models")
        TextEncoderModelDownloader.customModelsDirectory = custom
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, custom)

        TextEncoderModelDownloader.customModelsDirectory = nil
        let expected = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral")
            .appendingPathComponent("models")
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, expected)
    }

    // MARK: - reconfigureHubApi

    func testReconfigureHubApiDoesNotCrash() {
        // Verify reconfigureHubApi can be called without errors
        TextEncoderModelDownloader.customModelsDirectory = URL(fileURLWithPath: "/tmp/test-models")
        TextEncoderModelDownloader.reconfigureHubApi()

        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    // MARK: - hubCachePath with custom directory

    func testHubCachePathUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        // Create a fake model directory for a known repo
        let model = ModelInfo(
            id: "test",
            repoId: "test-org/test-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("test-model")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        let cachePath = TextEncoderModelDownloader.hubCachePath(for: model)
        XCTAssertNotNil(cachePath)
        XCTAssertEqual(cachePath!.standardizedFileURL.path, modelDir.standardizedFileURL.path)
    }

    func testHubCachePathReturnsNilWhenModelNotInCustomDir() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        let model = ModelInfo(
            id: "test",
            repoId: "test-org/missing-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        let cachePath = TextEncoderModelDownloader.hubCachePath(for: model)
        XCTAssertNil(cachePath)
    }

    // MARK: - findModelPath with custom directory

    func testFindModelPathUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        let model = ModelInfo(
            id: "test",
            repoId: "test-org/test-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        // Create fake model in hub download location (custom dir)
        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("test-model")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        let found = TextEncoderModelDownloader.findModelPath(for: model)
        XCTAssertNotNil(found)
        XCTAssertTrue(found!.path.hasPrefix(tempDir.path))
    }

    // MARK: - findQwen3ModelPath with custom directory

    func testFindQwen3ModelPathUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        let model = Qwen3ModelInfo(
            id: "test-qwen",
            repoId: "test-org/qwen3-test",
            name: "Qwen3 Test",
            description: "Test Qwen3 model",
            variant: .qwen3_4B_8bit,
            parameters: "4B"
        )

        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("qwen3-test")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        let found = TextEncoderModelDownloader.findQwen3ModelPath(for: model)
        XCTAssertNotNil(found)
        XCTAssertTrue(found!.path.hasPrefix(tempDir.path))
    }

    func testFindQwen3ModelPathTreatsDanglingWeightSymlinkAsMissing() throws {
        // A text encoder relocated to an external disk that is unplugged: the
        // weight is a dangling symlink and must not count as present-by-name.
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-dangling-\(UUID().uuidString)")
            .appendingPathComponent("models")
        let model = Qwen3ModelInfo(
            id: "test-qwen-dangling",
            repoId: "test-org/qwen3-dangling",
            name: "Qwen3 Test",
            description: "Test Qwen3 model",
            variant: .qwen3_4B_8bit,
            parameters: "4B"
        )
        let modelDir = tempDir.appendingPathComponent("test-org").appendingPathComponent("qwen3-dangling")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: modelDir.appendingPathComponent("model.safetensors"),
            withDestinationURL: tempDir.appendingPathComponent("unplugged/model.safetensors"))
        defer { try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent()) }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        XCTAssertFalse(TextEncoderModelDownloader.verifyShardedModel(at: modelDir).complete)
        XCTAssertNil(TextEncoderModelDownloader.findQwen3ModelPath(for: model))
    }

    // MARK: - Multiple switches

    func testSwitchingCustomDirectories() {
        let dir1 = URL(fileURLWithPath: "/tmp/models-a")
        let dir2 = URL(fileURLWithPath: "/tmp/models-b")

        TextEncoderModelDownloader.customModelsDirectory = dir1
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, dir1)

        TextEncoderModelDownloader.customModelsDirectory = dir2
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, dir2)

        TextEncoderModelDownloader.customModelsDirectory = nil
        let defaultDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral")
            .appendingPathComponent("models")
        XCTAssertEqual(TextEncoderModelDownloader.modelsDirectory, defaultDir)
    }

    // MARK: - isModelDownloaded with custom directory

    func testIsModelDownloadedUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        let model = ModelInfo(
            id: "test",
            repoId: "test-org/test-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        // Empty custom dir — model should not be found
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir
        XCTAssertFalse(TextEncoderModelDownloader.isModelDownloaded(model))

        // Add model files — now it should be found
        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("test-model")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))

        XCTAssertTrue(TextEncoderModelDownloader.isModelDownloaded(model))
    }
}
