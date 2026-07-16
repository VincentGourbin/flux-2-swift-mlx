// RuntimeBeacon.swift - Opt-in presence beacon for external activity monitors
// Copyright 2026 Vincent Gourbin

import Foundation
import os

/// Opt-in beacon advertising heavy GPU/memory activity (generation, training,
/// model loading) to external monitors such as SiliconScope.
///
/// When enabled, each heavy operation writes a small JSON manifest to
/// `~/Library/Application Support/ai-runtime-beacons/<pid>-<id>.json` for its
/// duration and deletes it when the operation ends (including on error, via
/// `defer` at every call site). Nothing is ever written unless the host
/// explicitly opts in.
///
/// ## Enabling
/// ```swift
/// RuntimeBeacon.isEnabled = true   // from code
/// ```
/// or set the environment variable `FLUX2_RUNTIME_BEACON=1` (used by the CLI's
/// `--beacon` flag as well).
///
/// ## Manifest schema (version 1)
/// The format is deliberately runtime-agnostic — same convention as
/// ltx-video-swift-mlx's RuntimeBeacon — so monitors only need one reader:
/// ```json
/// {
///   "version": 1,
///   "pid": 1234,
///   "runtime": "flux-2-swift-mlx",
///   "displayName": "FLUX.2",
///   "task": "generate",
///   "model": "klein-9b",
///   "phase": "denoising",
///   "step": 3,
///   "totalSteps": 4,
///   "startedAt": "2026-07-16T09:00:00Z",
///   "updatedAt": "2026-07-16T09:00:42Z"
/// }
/// ```
///
/// ## Crash safety
/// A `kill -9` can leave a manifest behind; both `begin(task:model:)` and
/// well-behaved consumers treat a manifest whose `pid` is dead as garbage and
/// delete it, so stale files never outlive the next beacon-enabled run.
///
/// - Note: Sandboxed apps write inside their container, where external
///   monitors cannot see the manifest. The beacon is a no-op in practice
///   there; it targets CLI tools and non-sandboxed apps.
public enum RuntimeBeacon {
    /// Manifest schema version.
    public static let schemaVersion = 1

    /// Identity written to every manifest produced by this package.
    static let runtimeID = "flux-2-swift-mlx"
    static let runtimeDisplayName = "FLUX.2"

    private static let enabledState = OSAllocatedUnfairLock(initialState: false)

    /// Global opt-in toggle. Defaults to `false`: no file is ever written
    /// unless the host sets this to `true` or exports `FLUX2_RUNTIME_BEACON=1`.
    public static var isEnabled: Bool {
        get { enabledState.withLock { $0 } }
        set { enabledState.withLock { $0 = newValue } }
    }

    private static var environmentEnabled: Bool {
        ProcessInfo.processInfo.environment["FLUX2_RUNTIME_BEACON"] == "1"
    }

    /// Test hook: redirects all manifest writes away from the real shared directory.
    private static let directoryOverrideState = OSAllocatedUnfairLock<URL?>(initialState: nil)
    static var directoryOverride: URL? {
        get { directoryOverrideState.withLock { $0 } }
        set { directoryOverrideState.withLock { $0 = newValue } }
    }

    /// Directory holding the manifests of all live beacon sessions
    /// (shared, runtime-agnostic location).
    public static var directory: URL {
        directoryOverride
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("ai-runtime-beacons", isDirectory: true)
    }

    /// Start a beacon session for one heavy operation. Returns `nil` when the
    /// beacon is disabled (the default), so call sites stay one-liners:
    /// ```swift
    /// let beacon = RuntimeBeacon.begin(task: "generate", model: model.rawValue)
    /// defer { beacon?.end() }
    /// ```
    /// Never throws and never blocks generation: every filesystem failure is
    /// swallowed and simply disables the session.
    public static func begin(task: String, model: String? = nil) -> Session? {
        guard isEnabled || environmentEnabled else { return nil }
        let dir = directory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        removeStaleManifests(in: dir)
        return Session(directory: dir, task: task, model: model)
    }

    /// Delete manifests whose owning process is dead (leftovers from crashes /
    /// force-kills). `kill(pid, 0)` probes liveness without sending a signal.
    private static func removeStaleManifests(in dir: URL) {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return }
        for url in entries where url.pathExtension == "json" {
            guard let pidToken = url.lastPathComponent.split(separator: "-").first,
                  let pid = pid_t(pidToken) else { continue }
            if pid != ProcessInfo.processInfo.processIdentifier,
               kill(pid, 0) == -1 && errno == ESRCH {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    // MARK: - Session

    /// One live manifest: created by ``RuntimeBeacon/begin(task:model:)``,
    /// refreshed by ``update(phase:step:totalSteps:)``, deleted by ``end()``.
    /// `deinit` also ends the session as a safety net, but call sites must
    /// still `defer { beacon?.end() }` — actors can keep references alive
    /// longer than the operation.
    public final class Session: Sendable {
        private struct Manifest: Encodable {
            let version: Int
            let pid: Int32
            let runtime: String
            let displayName: String
            let task: String
            let model: String?
            var phase: String?
            var step: Int?
            var totalSteps: Int?
            let startedAt: Date
            var updatedAt: Date
        }

        private struct State {
            var manifest: Manifest
            var ended: Bool
        }

        private let fileURL: URL
        private let state: OSAllocatedUnfairLock<State>

        private static let encoder: JSONEncoder = {
            let e = JSONEncoder()
            e.dateEncodingStrategy = .iso8601
            e.outputFormatting = [.sortedKeys]
            return e
        }()

        fileprivate init(directory: URL, task: String, model: String?) {
            let pid = ProcessInfo.processInfo.processIdentifier
            let id = UUID().uuidString.prefix(8)
            self.fileURL = directory.appendingPathComponent("\(pid)-\(id).json")
            let now = Date()
            self.state = OSAllocatedUnfairLock(initialState: State(
                manifest: Manifest(
                    version: RuntimeBeacon.schemaVersion,
                    pid: pid,
                    runtime: RuntimeBeacon.runtimeID,
                    displayName: RuntimeBeacon.runtimeDisplayName,
                    task: task,
                    model: model,
                    phase: nil,
                    step: nil,
                    totalSteps: nil,
                    startedAt: now,
                    updatedAt: now
                ),
                ended: false
            ))
            write()
        }

        deinit {
            end()
        }

        /// Refresh the manifest with the operation's current phase/step.
        /// Safe to call from any thread; no-op after ``end()``.
        public func update(phase: String, step: Int? = nil, totalSteps: Int? = nil) {
            let stillLive = state.withLock { s -> Bool in
                guard !s.ended else { return false }
                s.manifest.phase = phase
                s.manifest.step = step
                s.manifest.totalSteps = totalSteps
                s.manifest.updatedAt = Date()
                return true
            }
            if stillLive { write() }
        }

        /// Delete the manifest. Idempotent; also runs from `deinit`.
        public func end() {
            let shouldRemove = state.withLock { s -> Bool in
                guard !s.ended else { return false }
                s.ended = true
                return true
            }
            if shouldRemove {
                try? FileManager.default.removeItem(at: fileURL)
            }
        }

        /// Atomic write so a monitor never reads a half-written manifest.
        private func write() {
            let manifest = state.withLock { $0.manifest }
            guard let data = try? Self.encoder.encode(manifest) else { return }
            try? data.write(to: fileURL, options: .atomic)
        }
    }
}
