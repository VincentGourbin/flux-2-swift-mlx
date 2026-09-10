// ModelPathOverrideTests.swift - Tests for ModelRegistry per-component path overrides
// Copyright 2025 Vincent Gourbin
//
// Lets a caller redirect a single model component to an arbitrary directory
// (e.g. one relocated to an external disk) without moving the rest of the
// catalog under customModelsDirectory. See
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

    func testResolvedPathReturnsOverrideWhileLocalPathStaysPure() throws {
        let override = URL(fileURLWithPath: "/Volumes/External/flux2-vae", isDirectory: true)
        try ModelRegistry.setPathOverride(override, for: .vae(.standard))

        XCTAssertEqual(ModelRegistry.resolvedPath(for: .vae(.standard)), override)
        XCTAssertEqual(ModelRegistry.pathOverride(for: .vae(.standard)), override)
        XCTAssertEqual(ModelRegistry.pathOverride(forComponent: .vae(.standard)), override)
        // localPath is the catalog layout consumers derive relative paths from.
        XCTAssertFalse(ModelRegistry.localPath(for: .vae(.standard)).path.hasPrefix("/Volumes/External"))
    }

    func testOverrideOnlyAffectsItsOwnComponent() throws {
        try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/flux2-vae"), for: .vae(.standard))

        XCTAssertNil(ModelRegistry.pathOverride(for: .transformer(.klein4B_bf16)))
        XCTAssertEqual(
            ModelRegistry.resolvedPath(for: .transformer(.klein4B_bf16)),
            ModelRegistry.localPath(for: .transformer(.klein4B_bf16)))
    }

    func testSettingNilClearsOverride() throws {
        try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/flux2-vae"), for: .vae(.standard))
        try ModelRegistry.setPathOverride(nil, for: .vae(.standard))

        XCTAssertNil(ModelRegistry.pathOverride(for: .vae(.standard)))
    }

    func testNonFileURLIsRejected() {
        XCTAssertThrowsError(try ModelRegistry.setPathOverride(URL(string: "https://example.com/vae")!, for: .vae(.standard)))
        XCTAssertThrowsError(try ModelRegistry.setPathOverride(URL(string: "/Volumes/External/vae")!, for: .vae(.standard)))
        XCTAssertNil(ModelRegistry.pathOverride(for: .vae(.standard)))
    }

    func testStoredOverrideIsStandardizedDirectoryURL() throws {
        try ModelRegistry.setPathOverride(URL(fileURLWithPath: "/Volumes/External/./x/../flux2-vae"), for: .vae(.standard))

        let stored = try XCTUnwrap(ModelRegistry.pathOverride(for: .vae(.standard)))
        XCTAssertEqual(stored.path, "/Volumes/External/flux2-vae")
        XCTAssertTrue(stored.hasDirectoryPath)
    }

    func testTextEncoderOverrideIsUnrepresentable() {
        // `.textEncoder` is not an OverridableComponent; the ModelComponent
        // read side simply never has an override for it.
        XCTAssertNil(ModelRegistry.OverridableComponent(.textEncoder(.mlx8bit)))
        XCTAssertNil(ModelRegistry.pathOverride(forComponent: .textEncoder(.mlx8bit)))
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
        XCTAssertNil(Flux2ModelDownloader.unavailableReason(for: .vae(.standard)))
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
        let defaultModelDir = ModelRegistry.localPath(for: .vae(.standard))
        try fm.createDirectory(at: defaultModelDir, withIntermediateDirectories: true)
        try writeCompleteModel(at: defaultModelDir)

        try ModelRegistry.setPathOverride(overrideDir, for: .vae(.standard))

        XCTAssertNil(Flux2ModelDownloader.findModelPath(for: .vae(.standard)))
        XCTAssertFalse(ModelRegistry.isDownloaded(.vae(.standard)))
        XCTAssertNotNil(Flux2ModelDownloader.unavailableReason(for: .vae(.standard)))
    }

    func testRegistryAndDownloaderAgreeOnExistingButEmptyDirectory() throws {
        // Previously ModelRegistry.isDownloaded said "yes" for any existing
        // directory while Flux2ModelDownloader required completeness.
        let customDir = try makeTempDir("agree")
        defer { try? fm.removeItem(at: customDir) }
        ModelRegistry.customModelsDirectory = customDir
        try fm.createDirectory(at: ModelRegistry.localPath(for: .vae(.standard)), withIntermediateDirectories: true)

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
        XCTAssertNotNil(Flux2ModelDownloader.unavailableReason(for: .vae(.standard)))
    }

    func testDownloadRefusesDanglingWeightSymlinksInDefaultLayout() async throws {
        // The consumer's shipped relocation: default directory, weights are
        // absolute symlinks to the external disk, disk unplugged. This must not
        // become a re-download (which would overwrite the links and orphan the
        // external copy), and must fail before any network call.
        let customDir = try makeTempDir("dangling")
        defer { try? fm.removeItem(at: customDir) }
        ModelRegistry.customModelsDirectory = customDir
        let modelDir = ModelRegistry.localPath(for: .vae(.standard))
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let link = modelDir.appendingPathComponent("model.safetensors")
        try fm.createSymbolicLink(at: link, withDestinationURL: customDir.appendingPathComponent("unplugged/model.safetensors"))

        XCTAssertEqual(Flux2ModelDownloader.unreachableWeights(at: modelDir), ["model.safetensors"])
        XCTAssertNil(Flux2ModelDownloader.findModelPath(for: .vae(.standard)))
        XCTAssertNotNil(Flux2ModelDownloader.unavailableReason(for: .vae(.standard)))

        do {
            _ = try await Flux2ModelDownloader().download(.vae(.standard))
            XCTFail("Expected download() to refuse dangling weight symlinks")
        } catch let error as Flux2DownloadError {
            guard case .weightsUnreachable(_, _, let files) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(files, ["model.safetensors"])
        }
        // The symlink is untouched.
        XCTAssertNotNil(try? fm.destinationOfSymbolicLink(atPath: link.path))
    }

    func testDownloadRefusesReadOnlyDestinationBeforeAnyNetworkCall() async throws {
        let overrideDir = try makeTempDir("readonly")
        defer {
            try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: overrideDir.path)
            try? fm.removeItem(at: overrideDir)
        }
        try fm.setAttributes([.posixPermissions: 0o555], ofItemAtPath: overrideDir.path)
        try XCTSkipIf(fm.isWritableFile(atPath: overrideDir.path), "Running as a user that ignores permission bits")
        try ModelRegistry.setPathOverride(overrideDir, for: .vae(.standard))

        do {
            _ = try await Flux2ModelDownloader().download(.vae(.standard))
            XCTFail("Expected download() to refuse a read-only destination")
        } catch let error as Flux2DownloadError {
            guard case .destinationNotWritable = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testUnmountedVolumeDetection() throws {
        XCTAssertTrue(Flux2ModelDownloader.isOnUnmountedVolume(
            URL(fileURLWithPath: "/Volumes/flux2-nope-\(UUID().uuidString)/a/b/c")))
        XCTAssertTrue(Flux2ModelDownloader.isOnUnmountedVolume(
            URL(fileURLWithPath: "/volumes/flux2-nope-\(UUID().uuidString)")))

        // A missing subfolder under an existing directory is not "unmounted".
        let existing = try makeTempDir("mounted")
        defer { try? fm.removeItem(at: existing) }
        XCTAssertFalse(Flux2ModelDownloader.isOnUnmountedVolume(existing.appendingPathComponent("not/yet/created")))
        XCTAssertFalse(Flux2ModelDownloader.isOnUnmountedVolume(existing))
    }
}
