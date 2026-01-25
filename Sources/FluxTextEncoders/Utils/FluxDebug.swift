/**
 * FluxDebug.swift
 * Debug utilities for FluxTextEncoders
 */

import Foundation

public enum FluxDebug {
    nonisolated(unsafe) public static var isEnabled = false

    public static func log(_ message: String, file: String = #file, line: Int = #line) {
        guard isEnabled else { return }
        let filename = URL(fileURLWithPath: file).lastPathComponent
        print("[\(filename):\(line)] \(message)")
    }

    public static func error(_ message: String, file: String = #file, line: Int = #line) {
        let filename = URL(fileURLWithPath: file).lastPathComponent
        print("[ERROR][\(filename):\(line)] \(message)")
    }
}
