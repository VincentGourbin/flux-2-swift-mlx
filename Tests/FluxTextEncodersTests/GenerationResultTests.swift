/**
 * GenerationResultTests.swift
 * Unit tests for GenerationResult
 */

import XCTest
@testable import FluxTextEncoders

final class GenerationResultTests: XCTestCase {

    // MARK: - Initialization Tests

    func testGenerationResultInit() {
        let result = GenerationResult(
            text: "Hello world",
            tokens: [100, 200, 300],
            promptTokens: 10,
            generatedTokens: 3,
            totalTime: 1.5,
            tokensPerSecond: 2.0
        )

        XCTAssertEqual(result.text, "Hello world")
        XCTAssertEqual(result.tokens, [100, 200, 300])
        XCTAssertEqual(result.promptTokens, 10)
        XCTAssertEqual(result.generatedTokens, 3)
        XCTAssertEqual(result.totalTime, 1.5)
        XCTAssertEqual(result.tokensPerSecond, 2.0)
    }

    func testGenerationResultEmptyText() {
        let result = GenerationResult(
            text: "",
            tokens: [],
            promptTokens: 5,
            generatedTokens: 0,
            totalTime: 0.1,
            tokensPerSecond: 0.0
        )

        XCTAssertTrue(result.text.isEmpty)
        XCTAssertTrue(result.tokens.isEmpty)
        XCTAssertEqual(result.generatedTokens, 0)
    }

    // MARK: - Summary Tests

    func testSummaryFormat() {
        let result = GenerationResult(
            text: "Test output",
            tokens: [1, 2, 3, 4, 5],
            promptTokens: 100,
            generatedTokens: 5,
            totalTime: 2.0,
            tokensPerSecond: 2.5
        )

        let summary = result.summary()

        XCTAssertTrue(summary.contains("Prompt: 100 tokens"), "Should show prompt tokens")
        XCTAssertTrue(summary.contains("Generated: 5 tokens"), "Should show generated tokens")
        XCTAssertTrue(summary.contains("2.5 tokens/s"), "Should show tokens per second")
        XCTAssertTrue(summary.contains("2.00s"), "Should show time")
    }

    func testSummaryWithZeroTokens() {
        let result = GenerationResult(
            text: "",
            tokens: [],
            promptTokens: 0,
            generatedTokens: 0,
            totalTime: 0.0,
            tokensPerSecond: 0.0
        )

        let summary = result.summary()

        XCTAssertTrue(summary.contains("Prompt: 0 tokens"))
        XCTAssertTrue(summary.contains("Generated: 0 tokens"))
    }

    func testSummaryWithLargeNumbers() {
        let result = GenerationResult(
            text: String(repeating: "a", count: 1000),
            tokens: Array(0..<1000),
            promptTokens: 5000,
            generatedTokens: 1000,
            totalTime: 50.0,
            tokensPerSecond: 20.0
        )

        let summary = result.summary()

        XCTAssertTrue(summary.contains("5000 tokens"))
        XCTAssertTrue(summary.contains("1000 tokens"))
        XCTAssertTrue(summary.contains("20.0 tokens/s"))
    }

    // MARK: - Token Count Consistency Tests

    func testTokenCountMatchesArray() {
        let tokens = [10, 20, 30, 40, 50]
        let result = GenerationResult(
            text: "test",
            tokens: tokens,
            promptTokens: 10,
            generatedTokens: tokens.count,
            totalTime: 1.0,
            tokensPerSecond: 5.0
        )

        XCTAssertEqual(result.tokens.count, result.generatedTokens,
                      "Token count should match tokens array length")
    }

    // MARK: - Performance Metrics Tests

    func testHighPerformanceMetrics() {
        let result = GenerationResult(
            text: "Fast generation",
            tokens: Array(0..<100),
            promptTokens: 50,
            generatedTokens: 100,
            totalTime: 1.0,
            tokensPerSecond: 100.0
        )

        XCTAssertEqual(result.tokensPerSecond, 100.0)

        let summary = result.summary()
        XCTAssertTrue(summary.contains("100.0 tokens/s"))
    }

    func testLowPerformanceMetrics() {
        let result = GenerationResult(
            text: "Slow generation",
            tokens: [1],
            promptTokens: 1000,
            generatedTokens: 1,
            totalTime: 10.0,
            tokensPerSecond: 0.1
        )

        let summary = result.summary()
        XCTAssertTrue(summary.contains("0.1 tokens/s"))
    }

    // MARK: - Sendable Conformance

    func testGenerationResultIsSendable() {
        let result: Sendable = GenerationResult(
            text: "test",
            tokens: [1],
            promptTokens: 1,
            generatedTokens: 1,
            totalTime: 1.0,
            tokensPerSecond: 1.0
        )
        XCTAssertNotNil(result)
    }

    // MARK: - Edge Cases

    func testVeryLongText() {
        let longText = String(repeating: "Hello world. ", count: 10000)
        let result = GenerationResult(
            text: longText,
            tokens: Array(0..<10000),
            promptTokens: 100,
            generatedTokens: 10000,
            totalTime: 100.0,
            tokensPerSecond: 100.0
        )

        XCTAssertEqual(result.text.count, longText.count)
        XCTAssertEqual(result.tokens.count, 10000)
    }

    func testUnicodeText() {
        let unicodeText = "Hello 世界 🌍 مرحبا"
        let result = GenerationResult(
            text: unicodeText,
            tokens: [1, 2, 3, 4, 5],
            promptTokens: 5,
            generatedTokens: 5,
            totalTime: 0.5,
            tokensPerSecond: 10.0
        )

        XCTAssertEqual(result.text, unicodeText)
    }

    func testSpecialCharactersInText() {
        let specialText = "Line1\nLine2\tTab\"Quote'Single"
        let result = GenerationResult(
            text: specialText,
            tokens: [1, 2, 3],
            promptTokens: 3,
            generatedTokens: 3,
            totalTime: 0.1,
            tokensPerSecond: 30.0
        )

        XCTAssertEqual(result.text, specialText)
    }

    // MARK: - Time Formatting Tests

    func testTimeFormattingInSummary() {
        // Test various time values
        let shortResult = GenerationResult(
            text: "quick",
            tokens: [1],
            promptTokens: 1,
            generatedTokens: 1,
            totalTime: 0.01,
            tokensPerSecond: 100.0
        )

        let summary = shortResult.summary()
        XCTAssertTrue(summary.contains("0.01s"), "Should format short time correctly")
    }

    func testDecimalPrecisionInSummary() {
        let result = GenerationResult(
            text: "test",
            tokens: [1, 2, 3],
            promptTokens: 10,
            generatedTokens: 3,
            totalTime: 1.23456,
            tokensPerSecond: 2.43902
        )

        let summary = result.summary()
        // Should be formatted with limited decimal places
        XCTAssertTrue(summary.contains("1.23s") || summary.contains("1.2s"),
                     "Time should be formatted with limited decimals")
    }
}
