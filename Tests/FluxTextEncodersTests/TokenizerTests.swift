/**
 * TokenizerTests.swift
 * Unit tests for TekkenTokenizer
 */

import XCTest
@testable import FluxTextEncoders

final class TokenizerTests: XCTestCase {

    var tokenizer: TekkenTokenizer!

    override func setUp() {
        super.setUp()
        // Use default tokenizer (no model path)
        tokenizer = TekkenTokenizer()
    }

    override func tearDown() {
        tokenizer = nil
        super.tearDown()
    }

    // MARK: - Basic Encoding Tests

    func testEncodeEmptyString() {
        let tokens = tokenizer.encode("")
        XCTAssertTrue(tokens.isEmpty, "Empty string should produce empty token array")
    }

    func testEncodeSimpleText() {
        let tokens = tokenizer.encode("Hello")
        XCTAssertFalse(tokens.isEmpty, "Non-empty string should produce tokens")
    }

    func testEncodeWithSpecialTokens() {
        let tokensWithoutSpecial = tokenizer.encode("Hello", addSpecialTokens: false)
        let tokensWithSpecial = tokenizer.encode("Hello", addSpecialTokens: true)

        // With special tokens should have BOS at start and EOS at end
        XCTAssertGreaterThan(tokensWithSpecial.count, tokensWithoutSpecial.count,
                            "Adding special tokens should increase token count")

        // First token should be BOS
        if !tokensWithSpecial.isEmpty {
            XCTAssertEqual(tokensWithSpecial.first, tokenizer.bosToken,
                          "First token with special tokens should be BOS")
        }

        // Last token should be EOS
        if !tokensWithSpecial.isEmpty {
            XCTAssertEqual(tokensWithSpecial.last, tokenizer.eosToken,
                          "Last token with special tokens should be EOS")
        }
    }

    // MARK: - Special Token Properties

    func testSpecialTokenProperties() {
        // BOS token should be 1 by default
        XCTAssertEqual(tokenizer.bosToken, 1, "BOS token should be 1")

        // EOS token should be 2 by default
        XCTAssertEqual(tokenizer.eosToken, 2, "EOS token should be 2")

        // PAD token should be 11 by default
        XCTAssertEqual(tokenizer.padToken, 11, "PAD token should be 11")
    }

    func testVocabSize() {
        let vocabSize = tokenizer.vocabSize
        XCTAssertGreaterThan(vocabSize, 0, "Vocab size should be positive")
    }

    // MARK: - Chat Template Tests

    func testApplyChatTemplateSimple() {
        let messages: [[String: String]] = [
            ["role": "user", "content": "Hello"]
        ]

        let prompt = tokenizer.applyChatTemplate(messages: messages)

        // Should contain the user message
        XCTAssertTrue(prompt.contains("Hello"), "Chat template should contain user message")

        // Should contain instruction markers
        XCTAssertTrue(prompt.contains("[INST]") || prompt.contains("user"),
                     "Chat template should contain instruction markers")
    }

    func testApplyChatTemplateWithSystem() {
        let messages: [[String: String]] = [
            ["role": "system", "content": "You are helpful"],
            ["role": "user", "content": "Hi"]
        ]

        let prompt = tokenizer.applyChatTemplate(messages: messages)

        // Should contain both system and user content
        XCTAssertTrue(prompt.contains("You are helpful") || prompt.contains("helpful"),
                     "Chat template should contain system message")
        XCTAssertTrue(prompt.contains("Hi"), "Chat template should contain user message")
    }

    func testApplyChatTemplateMultiTurn() {
        let messages: [[String: String]] = [
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Hi there!"],
            ["role": "user", "content": "How are you?"]
        ]

        let prompt = tokenizer.applyChatTemplate(messages: messages)

        // Should contain all messages
        XCTAssertTrue(prompt.contains("Hello"), "Should contain first user message")
        XCTAssertTrue(prompt.contains("Hi there!"), "Should contain assistant message")
        XCTAssertTrue(prompt.contains("How are you?"), "Should contain second user message")
    }

    func testEncodeChatMessagesProducesTokens() {
        let messages: [[String: String]] = [
            ["role": "user", "content": "Hello world"]
        ]

        let tokens = tokenizer.encodeChatMessages(messages: messages)

        XCTAssertFalse(tokens.isEmpty, "Chat messages should produce tokens")

        // First token should be BOS
        XCTAssertEqual(tokens.first, tokenizer.bosToken,
                      "First token should be BOS")
    }

    // MARK: - Decoding Tests

    func testDecodeEmptyArray() {
        let text = tokenizer.decode([])
        XCTAssertTrue(text.isEmpty, "Empty token array should decode to empty string")
    }

    func testDecodeSkipsSpecialTokensByDefault() {
        // Decode with special token IDs
        let tokens = [tokenizer.bosToken, tokenizer.eosToken]
        let text = tokenizer.decode(tokens, skipSpecialTokens: true)

        // Should not contain special token strings
        XCTAssertFalse(text.contains("<s>"), "Should skip BOS token")
        XCTAssertFalse(text.contains("</s>"), "Should skip EOS token")
    }

    func testBatchDecode() {
        let tokenLists = [
            [tokenizer.bosToken],
            [tokenizer.eosToken]
        ]

        let decoded = tokenizer.batchDecode(tokenLists)

        XCTAssertEqual(decoded.count, 2, "Batch decode should return same number of strings")
    }

    // MARK: - Edge Cases

    func testEncodeUnicodeText() {
        let tokens = tokenizer.encode("Hello 世界 🌍")
        // Should not crash and should produce some tokens
        XCTAssertFalse(tokens.isEmpty, "Unicode text should produce tokens")
    }

    func testEncodeLongText() {
        let longText = String(repeating: "Hello world. ", count: 100)
        let tokens = tokenizer.encode(longText)

        XCTAssertFalse(tokens.isEmpty, "Long text should produce tokens")
        XCTAssertGreaterThan(tokens.count, 10, "Long text should produce many tokens")
    }

    func testEncodeSpecialCharacters() {
        let specialText = "Hello\nWorld\tTab\"Quote'Single"
        let tokens = tokenizer.encode(specialText)

        XCTAssertFalse(tokens.isEmpty, "Text with special characters should produce tokens")
    }

    // MARK: - MLXArray Support
    // Note: Tests that require Metal/MLX GPU access are not included here.
    // They would need to be run in an environment with full MLX support.
    // The tokenizer's MLXArray functionality can be tested via integration tests
    // or directly in Xcode with proper Metal environment.
}

