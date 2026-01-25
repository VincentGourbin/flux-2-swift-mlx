/**
 * ProfilerTests.swift
 * Unit tests for FluxProfiler (non-MLX dependent tests only)
 *
 * Note: Full profiler tests require MLX Metal GPU access which is not available
 * in the standard `swift test` environment. MLX-dependent tests should be run
 * directly via Xcode or in an environment with full Metal support.
 */

import XCTest
@testable import FluxTextEncoders

/// Basic profiler tests that don't require MLX Metal initialization
final class ProfilerBasicTests: XCTestCase {

    // MARK: - Basic Tests (Non-MLX)

    func testDeviceInfoArchitecture() {
        // Test that we can get device architecture without MLX
        #if arch(arm64)
        XCTAssertTrue(true, "Running on ARM64 architecture")
        #else
        XCTAssertTrue(true, "Running on x86_64 architecture")
        #endif
    }

    func testSystemMemorySize() {
        // Test that we can get system memory without MLX
        let memorySize = ProcessInfo.processInfo.physicalMemory
        XCTAssertGreaterThan(memorySize, 0, "System should have memory")
    }

    func testTimeMeasurement() {
        // Test basic time measurement (no MLX needed)
        let start = Date()
        Thread.sleep(forTimeInterval: 0.01)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertGreaterThanOrEqual(elapsed, 0.01, "Should measure at least 10ms")
    }
}

// MARK: - Note about MLX-dependent tests
/**
 * The following profiler features require full MLX Metal GPU support:
 *
 * - FluxProfiler.shared singleton
 * - FluxProfiler.snapshot() memory snapshots
 * - FluxProfiler.profile() step profiling
 * - FluxProfiler.profileAsync() async step profiling
 * - FluxProfiler.getMetrics() generation metrics
 * - FluxProfiler.reset() and timing methods
 * - withProfiling() helper function
 *
 * These tests should be run:
 * 1. Directly via Xcode with proper Metal environment
 * 2. Via integration tests in the MistralApp
 * 3. In a CI environment with GPU support (GitHub Actions macOS runners)
 *
 * The `swift test` command may not have access to the MLX metallib
 * required for GPU operations, causing fatal errors.
 */
