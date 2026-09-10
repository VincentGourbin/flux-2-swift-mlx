// ModelPathOverrideTests.swift - Tests for ModelRegistry per-component path overrides
// Copyright 2025 Vincent Gourbin
//
// Lets a caller redirect a single model component to an arbitrary URL (e.g.
// one relocated to an external disk) without moving the rest of the catalog
// under customModelsDirectory. See
// Fluxforge Studio/docs/FRAMEWORK_ASKS_STORAGE.md ask #2.

import XCTest
@testable import Flux2Core

final class ModelPathOverrideTests: XCTestCase {

    private let fm = FileManager.default

    override func tearDown() {
        ModelRegistry.clearPathOverrides()
        ModelRegistry.customModelsDirectory = nil
        super.tearDown()
    }

    private func makeTempDir(_ label: String) throws -> URL {
        let dir = fm.temporaryDirectory.appendingPathComponent("flux2-\(label)-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeCompleteModel(at dir: URL) throws {
        try "{}".write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data(repeating: 0x42, count: 64).write(to: dir.appendingPathComponent("model.safetensors"))
    }

    // MARK: - Registry API

    func testLocalPathReturnsOverrideWhenSet() throws {
        let override = URL(fileURLWithPath: "/Volumes/External/flux2-vae")
        try ModelRegistry.setPathOverride(override, for: .vae(.standard))

        XCTAssertEqual(ModelRegistry.localPath(for: .vae(.standard)), override)
        XCTAssertEqual(ModelRegistry.pathOverride(for: .vae(.standard)), override)
    }

    func testDefaultLocalPathIgnoresOverride() throws {
        try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/flux2-vae"), for: .vae(.standard))

        XCTAssertFalse(ModelRegistry.defaultLocalPath(for: .vae(.standard)).path.hasPrefix("/Volumes/External"))
    }

    func testOverrideOnlyAffectsItsOwnComponent() throws {
        try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/flux2-vae"), for: .vae(.standard))

        XCTAssertFalse(ModelRegistry.localPath(for: .transformer(.klein4B_bf16)).path.hasPrefix("/Volumes/External"))
        XCTAssertNil(ModelRegistry.pathOverride(for: .transformer(.klein4B_bf16)))
    }

    func testSettingNilClearsOverride() throws {
        try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/flux2-vae"), for: .vae(.standard))
        try ModelRegistry.setPathOverride(nil, for: .vae(.standard))

        XCTAssertNil(ModelRegistry.pathOverride(for: .vae(.standard)))
        XCTAssertEqual(ModelRegistry.localPath(for: .vae(.standard)), ModelRegistry.defaultLocalPath(for: .vae(.standard)))
    }

    func testTextEncoderOverrideIsRejectedNotSilentlyInert() {
        XCTAssertThrowsError(
            try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/mistral"), for: .textEncoder(.mlx8bit))
        )
        XCTAssertNil(ModelRegistry.pathOverride(for: .textEncoder(.mlx8bit)))
    }

    // MARK: - findModelPath / isDownloaded

    func testFindModelPathUsesOverrideWhenComplete() throws {
        let overrideDir = try makeTempDir("override")
        defer { try? fm.removeItem(at: overrideDir) }
        try writeCompleteModel(at: overrideDir)

        try ModelRegistry.setPathOverride(overrideDir, for: .vae(.standard))

        let found = Flux2ModelDownloader.findModelPath(for: .vae(.standard))
        XCTAssertEqual(found?.standardizedFileURL.path, overrideDir.standardizedFileURL.path)
        XCTAssertTrue(ModelRegistry.isDownloaded(.vae(.standard)))
    }

    func testFindModelPathDoesNotFallBackWhenOverrideIsIncomplete() throws {
        // Override points at an empty directory while a fully valid copy sits
        // at the default location. The override is authoritative: no fallback.
        let overrideDir = try makeTempDir("override-empty")
        let customDir = try makeTempDir("default")
        defer {
            try? fm.removeItem(at: overrideDir)
            try? fm.removeItem(at: customDir)
        }
        ModelRegistry.customModelsDirectory = customDir
        let defaultModelDir = ModelRegistry.defaultLocalPath(for: .vae(.standard))
        try fm.createDirectory(at: defaultModelDir, withIntermediateDirectories: true)
        try writeCompleteModel(at: defaultModelDir)

        try ModelRegistry.setPathOverride(overrideDir, for: .vae(.standard))

        XCTAssertNil(Flux2ModelDownloader.findModelPath(for: .vae(.standard)))
        XCTAssertFalse(ModelRegistry.isDownloaded(.vae(.standard)))
    }

    func testRegistryAndDownloaderAgreeOnExistingButEmptyDirectory() throws {
        // Previously ModelRegistry.isDownloaded said "yes" for any existing
        // directory while Flux2ModelDownloader required completeness.
        let customDir = try makeTempDir("agree")
        defer { try? fm.removeItem(at: customDir) }
        ModelRegistry.customModelsDirectory = customDir
        try fm.createDirectory(at: ModelRegistry.defaultLocalPath(for: .vae(.standard)), withIntermediateDirectories: true)

        XCTAssertFalse(ModelRegistry.isDownloaded(.vae(.standard)))
        XCTAssertEqual(ModelRegistry.isDownloaded(.vae(.standard)), Flux2ModelDownloader.isDownloaded(.vae(.standard)))
    }

    // MARK: - delete / download guards

    func testDeleteRefusesComponentWithActiveOverride() throws {
        let overrideDir = try makeTempDir("override-delete")
        defer { try? fm.removeItem(at: overrideDir) }
        try writeCompleteModel(at: overrideDir)

        try ModelRegistry.setPathOverride(overrideDir, for: .vae(.standard))

        XCTAssertThrowsError(try Flux2ModelDownloader.delete(.vae(.standard))) { error in
            guard case Flux2DownloadError.deletionRefusedForOverride = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        XCTAssertTrue(fm.fileExists(atPath: overrideDir.appendingPathComponent("model.safetensors").path))
    }

    func testDownloadRefusesOverrideOnUnmountedVolumeBeforeAnyNetworkCall() async throws {
        // A /Volumes/<disk> that doesn't exist stands in for an unplugged
        // external disk. The check runs before fetchFileList, so this test
        // needs no network and must leave nothing behind.
        let override = URL(fileURLWithPath: "/Volumes/flux2-test-\(UUID().uuidString)/models/flux2-vae")
        try ModelRegistry.setPathOverride(override, for: .vae(.standard))

        do {
            _ = try await Flux2ModelDownloader().download(.vae(.standard))
            XCTFail("Expected download() to refuse an unmounted override volume")
        } catch let error as Flux2DownloadError {
            guard case .overrideVolumeUnavailable = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        XCTAssertFalse(fm.fileExists(atPath: override.deletingLastPathComponent().deletingLastPathComponent().path))
    }

    func testUnmountedVolumeDetectionOnlyTriggersForMissingMountPoint() throws {
        XCTAssertTrue(Flux2ModelDownloader.isOnUnmountedVolume(
            URL(fileURLWithPath: "/Volumes/flux2-nope-\(UUID().uuidString)/a/b/c")))

        // A missing subfolder under an existing directory is not "unmounted".
        let existing = try makeTempDir("mounted")
        defer { try? fm.removeItem(at: existing) }
        XCTAssertFalse(Flux2ModelDownloader.isOnUnmountedVolume(existing.appendingPathComponent("not/yet/created")))
        XCTAssertFalse(Flux2ModelDownloader.isOnUnmountedVolume(existing))
    }
}
