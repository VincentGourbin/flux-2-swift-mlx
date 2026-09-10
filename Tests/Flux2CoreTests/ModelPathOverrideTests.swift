// ModelPathOverrideTests.swift - Tests for ModelRegistry.pathOverrides
// Copyright 2025 Vincent Gourbin
//
// Lets a caller redirect a single model component to an arbitrary URL
// (e.g. one relocated to an external disk) without moving the rest of the
// catalog under customModelsDirectory. See
// Fluxforge Studio/docs/FRAMEWORK_ASKS_STORAGE.md ask #2.

import XCTest
@testable import Flux2Core

final class ModelPathOverrideTests: XCTestCase {

    override func tearDown() {
        ModelRegistry.pathOverrides = [:]
        ModelRegistry.customModelsDirectory = nil
        super.tearDown()
    }

    func testLocalPathReturnsOverrideWhenSet() {
        let override = URL(fileURLWithPath: "/Volumes/External/flux2-vae")
        ModelRegistry.pathOverrides[.vae(.standard)] = override

        XCTAssertEqual(ModelRegistry.localPath(for: .vae(.standard)), override)
    }

    func testLocalPathIgnoresOverrideForOtherComponents() {
        ModelRegistry.pathOverrides[.vae(.standard)] = URL(fileURLWithPath: "/Volumes/External/flux2-vae")

        let defaultPath = ModelRegistry.localPath(for: .transformer(.klein4B_bf16))
        XCTAssertFalse(defaultPath.path.hasPrefix("/Volumes/External"))
    }

    func testFindModelPathUsesOverrideWhenComplete() throws {
        let fm = FileManager.default
        let overrideDir = fm.temporaryDirectory
            .appendingPathComponent("flux2-override-\(UUID().uuidString)")
        try fm.createDirectory(at: overrideDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: overrideDir) }

        try "{}".write(to: overrideDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: overrideDir.appendingPathComponent("model.safetensors"))

        ModelRegistry.pathOverrides[.vae(.standard)] = overrideDir

        let found = Flux2ModelDownloader.findModelPath(for: .vae(.standard))
        XCTAssertEqual(found?.standardizedFileURL.path, overrideDir.standardizedFileURL.path)
    }

    func testFindModelPathDoesNotFallBackWhenOverrideIsIncomplete() throws {
        // Point the override at an empty directory, but leave a fully valid
        // model at the default computed location. An override must be
        // authoritative: it should NOT silently fall back to the default
        // location just because that one happens to be valid.
        let fm = FileManager.default
        let overrideDir = fm.temporaryDirectory
            .appendingPathComponent("flux2-override-empty-\(UUID().uuidString)")
        try fm.createDirectory(at: overrideDir, withIntermediateDirectories: true)

        let customDir = fm.temporaryDirectory
            .appendingPathComponent("flux2-default-\(UUID().uuidString)")
        let defaultModelDir = customDir
            .appendingPathComponent("black-forest-labs")
            .appendingPathComponent("FLUX.2-klein-4B-vae")
        try fm.createDirectory(at: defaultModelDir, withIntermediateDirectories: true)
        try "{}".write(to: defaultModelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: defaultModelDir.appendingPathComponent("model.safetensors"))

        defer {
            try? fm.removeItem(at: overrideDir)
            try? fm.removeItem(at: customDir)
        }

        ModelRegistry.customModelsDirectory = customDir
        ModelRegistry.pathOverrides[.vae(.standard)] = overrideDir

        XCTAssertNil(Flux2ModelDownloader.findModelPath(for: .vae(.standard)))
    }
}
