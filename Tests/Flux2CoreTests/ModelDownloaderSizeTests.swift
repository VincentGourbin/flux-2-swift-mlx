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
import FluxTextEncoders

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
        let previousHub = Flux2ModelDownloader.legacyHubCacheDirectory
        ModelRegistry.customModelsDirectory = dir
        Flux2ModelDownloader.legacyHubCacheDirectory = dir.appendingPathComponent("no-hub-cache")
        defer {
            ModelRegistry.customModelsDirectory = previous
            Flux2ModelDownloader.legacyHubCacheDirectory = previousHub
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

            // The walk itself: the dangling link adds 0, not its own lstat size.
            XCTAssertEqual(Flux2ModelDownloader.directorySize(at: modelDir), Int64(configData.count))

            // And at the API level the model is no longer "downloaded" at all
            // (verifyModel follows symlinks), so it contributes nothing.
            XCTAssertEqual(Flux2ModelDownloader.downloadedSize(), 0)
        }
    }

    // MARK: - verifyModel follows symlinks

    func testVerifyModelCountsSymlinkedWeightWhoseTargetExists() throws {
        try withSandbox { customDir in
            let fm = FileManager.default
            let modelDir = ModelRegistry.localPath(for: .vae(.standard))
            let externalDir = customDir.appendingPathComponent("external")
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
            try fm.createDirectory(at: externalDir, withIntermediateDirectories: true)
            try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

            let target = externalDir.appendingPathComponent("model.safetensors")
            try Data(repeating: 0x42, count: 64).write(to: target)
            try fm.createSymbolicLink(at: modelDir.appendingPathComponent("model.safetensors"), withDestinationURL: target)

            XCTAssertTrue(Flux2ModelDownloader.verifyModel(at: modelDir).complete)
            XCTAssertNotNil(Flux2ModelDownloader.findModelPath(for: .vae(.standard)))
        }
    }

    func testVerifyModelTreatsBrokenSymlinkedWeightAsMissing() throws {
        try withSandbox { customDir in
            let fm = FileManager.default
            let modelDir = ModelRegistry.localPath(for: .vae(.standard))
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
            try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

            // Unplugged external disk: the weight file's symlink dangles.
            let missingTarget = customDir.appendingPathComponent("not-mounted/model.safetensors")
            try fm.createSymbolicLink(at: modelDir.appendingPathComponent("model.safetensors"), withDestinationURL: missingTarget)

            XCTAssertFalse(Flux2ModelDownloader.verifyModel(at: modelDir).complete)
            XCTAssertNil(Flux2ModelDownloader.findModelPath(for: .vae(.standard)))
        }
    }

    func testVerifyModelDiffusersShardsRequireTheWholeReachableSeries() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("flux2-shards-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try "{}".write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        // Shards 3-4 local, 1-2 dangling (partially unplugged relocation).
        for i in 3...4 {
            try Data(repeating: 0x42, count: 8).write(
                to: dir.appendingPathComponent("diffusion_pytorch_model-0000\(i)-of-00004.safetensors"))
        }
        for i in 1...2 {
            try fm.createSymbolicLink(
                at: dir.appendingPathComponent("diffusion_pytorch_model-0000\(i)-of-00004.safetensors"),
                withDestinationURL: dir.appendingPathComponent("unplugged/\(i).safetensors"))
        }

        let result = Flux2ModelDownloader.verifyModel(at: dir)
        XCTAssertFalse(result.complete)
        XCTAssertEqual(result.missing, [
            "diffusion_pytorch_model-00001-of-00004.safetensors",
            "diffusion_pytorch_model-00002-of-00004.safetensors",
        ])

        // Complete the series and it verifies.
        for i in 1...2 {
            let link = dir.appendingPathComponent("diffusion_pytorch_model-0000\(i)-of-00004.safetensors")
            try fm.removeItem(at: link)
            try Data(repeating: 0x42, count: 8).write(to: link)
        }
        XCTAssertTrue(Flux2ModelDownloader.verifyModel(at: dir).complete)
    }

    func testVerifyModelSeriesAreGroupedByStemAndTotal() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("flux2-series-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        let write = { (name: String) in
            try Data(repeating: 0x42, count: 8).write(to: dir.appendingPathComponent(name))
        }

        // A complete 2-shard series plus a leftover shard of a 7-shard series:
        // complete, whatever order the listing yields.
        try write("model-00001-of-00002.safetensors")
        try write("model-00002-of-00002.safetensors")
        try write("diffusion_pytorch_model-00001-of-00007.safetensors")
        XCTAssertTrue(Flux2ModelDownloader.verifyModel(at: dir).complete)

        // Two half series of different stems must not union into "complete".
        try fm.removeItem(at: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try fm.removeItem(at: dir.appendingPathComponent("diffusion_pytorch_model-00001-of-00007.safetensors"))
        try write("diffusion_pytorch_model-00001-of-00002.safetensors")
        let result = Flux2ModelDownloader.verifyModel(at: dir)
        XCTAssertFalse(result.complete)
        XCTAssertEqual(result.missing.count, 1)
    }

    func testFilesToLoadExcludesLeftoverShardOfADifferentSeries() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("flux2-filestoload-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        // A complete 2-shard series (the real, current model)...
        try Data(repeating: 0x41, count: 8).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try Data(repeating: 0x42, count: 8).write(to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        // ...plus a leftover shard of an unrelated 5-shard series (a stale
        // download from a different HF revision).
        try Data(repeating: 0x43, count: 8).write(to: dir.appendingPathComponent("model-00001-of-00005.safetensors"))

        // Verification is still complete (a real series is whole)...
        XCTAssertTrue(SafetensorsDirectory.verifySeries(at: dir).complete)
        // ...but loading must not pull the stale leftover's tensors in too.
        let files = SafetensorsDirectory.filesToLoad(at: dir)
        XCTAssertEqual(files, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }

    func testVerifyModelIgnoresAppleDoubleSidecars() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("flux2-sidecar-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try "{}".write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        // Only the exFAT sidecar is left after the real weight went away.
        try Data(repeating: 0, count: 4).write(to: dir.appendingPathComponent("._flux-2-klein-4b.safetensors"))

        XCTAssertFalse(Flux2ModelDownloader.verifyModel(at: dir).complete)
    }

    // MARK: - directorySize(at:) edge cases (multi-hop chains, symlinked directories)

    func testDirectorySizeFollowsMultiHopSymlinkChain() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("flux2-chain-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let real = root.appendingPathComponent("real.safetensors")
        try Data(repeating: 0x41, count: 10_000).write(to: real)

        let link1 = root.appendingPathComponent("link1.safetensors")
        try fm.createSymbolicLink(at: link1, withDestinationURL: real)
        let link2 = root.appendingPathComponent("link2.safetensors")
        try fm.createSymbolicLink(at: link2, withDestinationURL: link1)

        // Only link2 sits in the model dir; it must resolve through link1 to
        // real.safetensors's actual size, not link1's own tiny symlink size.
        let modelDir = root.appendingPathComponent("model")
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let chainedLink = modelDir.appendingPathComponent("model.safetensors")
        try fm.createSymbolicLink(at: chainedLink, withDestinationURL: link2)

        XCTAssertEqual(Flux2ModelDownloader.directorySize(at: modelDir), 10_000)
    }

    func testDirectorySizeFollowsSymlinkedSubdirectory() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("flux2-dirlink-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let realDir = root.appendingPathComponent("real-weights")
        try fm.createDirectory(at: realDir, withIntermediateDirectories: true)
        try Data(repeating: 0x41, count: 4_000).write(to: realDir.appendingPathComponent("shard1.safetensors"))
        try Data(repeating: 0x42, count: 6_000).write(to: realDir.appendingPathComponent("shard2.safetensors"))

        let modelDir = root.appendingPathComponent("model")
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let dirLink = modelDir.appendingPathComponent("weights")
        try fm.createSymbolicLink(at: dirLink, withDestinationURL: realDir)

        XCTAssertEqual(Flux2ModelDownloader.directorySize(at: modelDir), 10_000)
    }

    func testDirectorySizeIgnoresSymlinkCycleThroughAncestorDirectory() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("flux2-cycle-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let modelDir = root.appendingPathComponent("model")
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data(repeating: 0x41, count: 10_000).write(to: modelDir.appendingPathComponent("weights.safetensors"))

        // A stray symlink pointing back at the model directory itself. Without
        // cycle detection this would cause weights.safetensors to be re-summed
        // on every hop instead of the cycle being recognized.
        let selfLink = modelDir.appendingPathComponent("self-link")
        try fm.createSymbolicLink(at: selfLink, withDestinationURL: modelDir)

        XCTAssertEqual(Flux2ModelDownloader.directorySize(at: modelDir), 10_000)
    }

    func testDirectorySizeEmptySymlinkTargetContributesZero() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("flux2-emptytarget-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        try Data(repeating: 0x41, count: 500).write(to: root.appendingPathComponent("config.json"))

        // A malformed symlink with an empty stored target (readlink() succeeds
        // and returns ""). Foundation's createSymbolicLink API always stores a
        // non-empty destination, so this needs the raw POSIX call.
        let brokenLink = root.appendingPathComponent("model.safetensors")
        let result = brokenLink.path.withCString { symlink("", $0) }
        try XCTSkipIf(result != 0, "This filesystem doesn't allow an empty symlink target")

        XCTAssertEqual(Flux2ModelDownloader.directorySize(at: root), 500)
    }
}
