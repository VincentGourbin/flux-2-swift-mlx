// RuntimeBeaconTests.swift — Unit tests for the opt-in activity beacon
// Copyright 2026 Vincent Gourbin

import XCTest
@testable import Flux2Core

/// RuntimeBeacon.isEnabled and directoryOverride are global state, so every
/// test runs inside `withSandbox`, which restores both afterwards. XCTest
/// runs the methods of one class serially, which is all the isolation needed.
final class RuntimeBeaconTests: XCTestCase {

    /// Runs `body` with the beacon enabled and sandboxed into a fresh temp
    /// directory, restoring global state afterwards.
    private func withSandbox<T>(
        enabled: Bool = true,
        _ body: (URL) throws -> T
    ) throws -> T {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("beacon-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // An exported FLUX2_RUNTIME_BEACON=1 would force-enable begin() and
        // break the enabled:false assertions — neutralize it for the test.
        unsetenv("FLUX2_RUNTIME_BEACON")
        RuntimeBeacon.directoryOverride = dir
        RuntimeBeacon.isEnabled = enabled
        defer {
            RuntimeBeacon.isEnabled = false
            RuntimeBeacon.directoryOverride = nil
            try? FileManager.default.removeItem(at: dir)
        }
        return try body(dir)
    }

    private func manifestFiles(in dir: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "json" } ?? []
    }

    func testDisabledByDefaultBeginReturnsNilAndWritesNothing() throws {
        try withSandbox(enabled: false) { dir in
            XCTAssertNil(RuntimeBeacon.begin(task: "generate"))
            XCTAssertTrue(manifestFiles(in: dir).isEmpty)
        }
    }

    func testSessionLifecycleManifestCreatedUpdatedThenDeletedOnEnd() throws {
        try withSandbox { dir in
            let session = try XCTUnwrap(RuntimeBeacon.begin(task: "generate", model: "klein-9b"))

            let files = manifestFiles(in: dir)
            XCTAssertEqual(files.count, 1)
            let url = try XCTUnwrap(files.first)

            var json = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            XCTAssertEqual(json["runtime"] as? String, "flux-2-swift-mlx")
            XCTAssertEqual(json["displayName"] as? String, "FLUX.2")
            XCTAssertEqual(json["task"] as? String, "generate")
            XCTAssertEqual(json["model"] as? String, "klein-9b")
            XCTAssertEqual(json["pid"] as? Int32, ProcessInfo.processInfo.processIdentifier)
            XCTAssertEqual(json["version"] as? Int, RuntimeBeacon.schemaVersion)
            XCTAssertNil(json["phase"])

            session.update(phase: "denoising", step: 3, totalSteps: 4)
            json = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            XCTAssertEqual(json["phase"] as? String, "denoising")
            XCTAssertEqual(json["step"] as? Int, 3)
            XCTAssertEqual(json["totalSteps"] as? Int, 4)

            session.end()
            XCTAssertTrue(manifestFiles(in: dir).isEmpty)

            // end() is idempotent and update() after end() writes nothing back.
            session.end()
            session.update(phase: "vae-decode")
            XCTAssertTrue(manifestFiles(in: dir).isEmpty)
        }
    }

    func testDeinitRemovesTheManifestEvenWithoutAnExplicitEnd() throws {
        try withSandbox { dir in
            do {
                let session = try XCTUnwrap(RuntimeBeacon.begin(task: "train"))
                XCTAssertEqual(manifestFiles(in: dir).count, 1)
                _ = session  // silence unused warning; deallocated at scope exit
            }
            XCTAssertTrue(manifestFiles(in: dir).isEmpty)
        }
    }

    func testBeginRemovesStaleManifestsLeftByDeadProcessesKeepsLiveOnes() throws {
        try withSandbox { dir in
            // Fake leftover from a crashed process. PID 99999997 is outside
            // macOS's PID range, so kill(pid, 0) fails with ESRCH.
            let stale = dir.appendingPathComponent("99999997-deadbeef.json")
            try Data("{}".utf8).write(to: stale)
            // Manifest of a live foreign process (launchd, pid 1) must survive.
            let live = dir.appendingPathComponent("1-cafebabe.json")
            try Data("{}".utf8).write(to: live)

            let session = try XCTUnwrap(RuntimeBeacon.begin(task: "generate"))
            defer { session.end() }

            let names = manifestFiles(in: dir).map(\.lastPathComponent)
            XCTAssertFalse(names.contains("99999997-deadbeef.json"))
            XCTAssertTrue(names.contains("1-cafebabe.json"))
            XCTAssertEqual(names.count, 2)  // live foreign + our session
        }
    }
}
