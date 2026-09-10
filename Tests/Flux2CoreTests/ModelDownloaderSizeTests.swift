// ModelDownloaderSizeTests.swift - Regression tests for Flux2ModelDownloader.downloadedSize()
// Copyright 2025 Vincent Gourbin
//
// Flux2ModelDownloader.directorySize(at:) must follow a file symlink to its
// target's real size (a component relocated to an external disk), not report
// the symlink's own near-zero size, and a broken symlink (unmounted external
// disk) must contribute 0 rather than leak the symlink's own size. See
// Fluxforge Studio/docs/FRAMEWORK_ASKS_STORAGE.md ask #1.

import XCTest
@testable import Flux2Core

final class ModelDownloaderSizeTests: XCTestCase {

    override func tearDown() {
        ModelRegistry.customModelsDirectory = nil
        super.tearDown()
    }

    /// Runs `body` with ModelRegistry.customModelsDirectory sandboxed into a
    /// fresh temp directory, restoring global state afterwards.
    private func withSandbox<T>(_ body: (URL) throws -> T) rethrows -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-modelsize-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let previous = ModelRegistry.customModelsDirectory
        ModelRegistry.customModelsDirectory = dir
        defer {
            ModelRegistry.customModelsDirectory = previous
            try? FileManager.default.removeItem(at: dir)
        }
        return try body(dir)
    }

    func testDownloadedSizeFollowsSymlinkedWeightToItsRealSize() throws {
        try withSandbox { customDir in
            let fm = FileManager.default
            let modelDir = ModelRegistry.localPath(for: .vae(.standard))
            let externalDir = customDir.appendingPathComponent("external")
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
            try fm.createDirectory(at: externalDir, withIntermediateDirectories: true)

            // A regular, non-relocated file.
            let configData = Data(repeating: 0x41, count: 500)
            try configData.write(to: modelDir.appendingPathComponent("config.json"))

            // A "relocated" weight: real bytes live on the external target, the
            // model directory only holds an absolute file symlink to it.
            let targetData = Data(repeating: 0x42, count: 50_000)
            let targetURL = externalDir.appendingPathComponent("model.safetensors")
            try targetData.write(to: targetURL)
            let symlinkURL = modelDir.appendingPathComponent("model.safetensors")
            try fm.createSymbolicLink(at: symlinkURL, withDestinationURL: targetURL)

            let size = Flux2ModelDownloader.downloadedSize()

            XCTAssertEqual(size, Int64(configData.count + targetData.count))
        }
    }

    func testDownloadedSizeBrokenSymlinkContributesZero() throws {
        try withSandbox { customDir in
            let fm = FileManager.default
            let modelDir = ModelRegistry.localPath(for: .vae(.standard))
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)

            let configData = Data(repeating: 0x41, count: 1_234)
            try configData.write(to: modelDir.appendingPathComponent("config.json"))

            // Simulates an unmounted external disk: the symlink target doesn't exist.
            let missingTarget = customDir.appendingPathComponent("not-mounted/model.safetensors")
            let symlinkURL = modelDir.appendingPathComponent("model.safetensors")
            try fm.createSymbolicLink(at: symlinkURL, withDestinationURL: missingTarget)

            let size = Flux2ModelDownloader.downloadedSize()

            XCTAssertEqual(size, Int64(configData.count))
        }
    }
}
