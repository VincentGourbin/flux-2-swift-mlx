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

    func testFindQwen3ModelPathTreatsDanglingWeightSymlinkAsMissing() async throws {
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

        // And downloading must refuse (before any network call) rather than
        // let the Hub client replace the link with a local copy.
        do {
            _ = try await TextEncoderModelDownloader().downloadQwen3(model)
            XCTFail("Expected downloadQwen3 to refuse a relocated model")
        } catch let error as TextEncoderModelDownloaderError {
            // Dangling link: the disk is gone, distinct from a live but
            // incomplete relocation (.weightsRelocated).
            guard case .weightsUnreachable(_, let files) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(files, ["model.safetensors"])
        }
        XCTAssertNotNil(try? FileManager.default.destinationOfSymbolicLink(
            atPath: modelDir.appendingPathComponent("model.safetensors").path))
    }

    func testDownloadQwen3RefusesIncompleteRelocatedSeriesWithMissingShardsNamed() async throws {
        // Disk mounted (live links), but shard 2 of 2 was never relocated:
        // the error must name the actually-missing shard, not the ones present.
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-relocated-\(UUID().uuidString)")
            .appendingPathComponent("models")
        let model = Qwen3ModelInfo(
            id: "test-qwen-relocated",
            repoId: "test-org/qwen3-relocated",
            name: "Qwen3 Test",
            description: "Test Qwen3 model",
            variant: .qwen3_4B_8bit,
            parameters: "4B"
        )
        let modelDir = tempDir.appendingPathComponent("test-org").appendingPathComponent("qwen3-relocated")
        let externalDir = tempDir.appendingPathComponent("external")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: externalDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let shard1 = "model-00001-of-00002.safetensors"
        try Data(repeating: 0x42, count: 8).write(to: externalDir.appendingPathComponent(shard1))
        try FileManager.default.createSymbolicLink(
            at: modelDir.appendingPathComponent(shard1), withDestinationURL: externalDir.appendingPathComponent(shard1))
        defer { try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent()) }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        do {
            _ = try await TextEncoderModelDownloader().downloadQwen3(model)
            XCTFail("Expected downloadQwen3 to refuse an incomplete relocated series")
        } catch let error as TextEncoderModelDownloaderError {
            guard case .weightsRelocated(_, let missing) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(missing, ["model-00002-of-00002.safetensors"])
        }
        // The live link must survive the refused download.
        XCTAssertNotNil(try? FileManager.default.destinationOfSymbolicLink(
            atPath: modelDir.appendingPathComponent(shard1).path))
    }

    func testVerifyShardedModelGroupsSeriesByStemAndTotal() throws {
        // The text-encoder verifier shares Flux2Core's series logic now: a
        // non-"model" stem is parsed, and a leftover shard of another series
        // neither completes nor breaks the real one.
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("te-series-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        let write = { (name: String) in
            try Data(repeating: 0x42, count: 8).write(to: dir.appendingPathComponent(name))
        }

        try write("qwen3-00001-of-00002.safetensors")
        let partial = TextEncoderModelDownloader.verifyShardedModel(at: dir)
        XCTAssertFalse(partial.complete)
        XCTAssertEqual(partial.missing, ["qwen3-00002-of-00002.safetensors"])

        try write("qwen3-00002-of-00002.safetensors")
        try write("model-00001-of-00007.safetensors")  // leftover of another series
        XCTAssertTrue(TextEncoderModelDownloader.verifyShardedModel(at: dir).complete)
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
