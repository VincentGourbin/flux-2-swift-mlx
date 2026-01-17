// Flux2Debug.swift - Debug and logging utilities
// Copyright 2025 Vincent Gourbin

import Foundation

/// Debug utilities for Flux.2
public enum Flux2Debug {
    /// Enable/disable debug logging
    nonisolated(unsafe) public static var enabled: Bool = true

    /// Log level
    public enum Level: Int, Comparable, Sendable {
        case verbose = 0  // Detailed debug info
        case info = 1     // General progress
        case warning = 2  // Warnings
        case error = 3    // Errors only

        public static func < (lhs: Level, rhs: Level) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    /// Minimum log level to display (default: warning - only show warnings/errors)
    nonisolated(unsafe) public static var minLevel: Level = .warning

    /// Enable debug mode (shows all logs including verbose)
    public static func enableDebugMode() {
        enabled = true
        minLevel = .verbose
    }

    /// Set to normal mode (only show warnings and errors)
    public static func setNormalMode() {
        enabled = true
        minLevel = .warning
    }

    /// Log a debug message
    public static func log(_ message: String, level: Level = .info) {
        guard enabled, level >= minLevel else { return }

        let prefix: String
        switch level {
        case .verbose:
            prefix = "[Flux2:V]"
        case .info:
            prefix = "[Flux2]"
        case .warning:
            prefix = "[Flux2:W]"
        case .error:
            prefix = "[Flux2:E]"
        }

        print("\(prefix) \(message)")
        fflush(stdout)
    }

    /// Convenience methods
    public static func verbose(_ message: String) {
        log(message, level: .verbose)
    }

    public static func info(_ message: String) {
        log(message, level: .info)
    }

    public static func warning(_ message: String) {
        log(message, level: .warning)
    }

    public static func error(_ message: String) {
        log(message, level: .error)
    }

    /// Log with timing
    public static func timed<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        log("\(label): \(String(format: "%.3f", elapsed))s")
        return result
    }

    /// Async version of timed
    public static func timedAsync<T>(_ label: String, _ block: () async throws -> T) async rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        log("\(label): \(String(format: "%.3f", elapsed))s")
        return result
    }
}
