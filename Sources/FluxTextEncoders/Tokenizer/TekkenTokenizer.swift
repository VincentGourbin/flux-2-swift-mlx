/**
 * TekkenTokenizer.swift
 *
 * BPE Tokenizer for Mistral models - based on tiktoken (like Python mistral-common)
 * Adapted from mlx-voxtral-swift project
 *
 * Supports both:
 * - Legacy tekken.json format (Mistral-common)
 * - HuggingFace tokenizer.json format (via swift-transformers)
 */

import Foundation
import MLX
import Tokenizers

// MARK: - Tekken Tokenizer

/**
 * TekkenTokenizer - BPE implementation equivalent to mistral_common.tokens.tokenizers.tekken.Tekkenizer
 * Falls back to HuggingFace tokenizer format when tekken.json is not available
 */
public class TekkenTokenizer {

    // Vocabulary (mergeable_ranks in tiktoken)
    private var mergeableRanks: [Data: Int] = [:]
    private var reverseVocabulary: [Int: String] = [:]
    private var rankToBytes: [Int: Data] = [:]  // Fast reverse lookup for decoding
    private var modelPath: String?

    // HuggingFace tokenizer fallback
    private var hfTokenizer: Tokenizer?
    private var useHFTokenizer: Bool = false

    // Regex pattern for text splitting (pat_str in tiktoken)
    private var regexPattern: String = ""
    private var compiledRegex: NSRegularExpression?

    // Configuration
    private var numSpecialTokens: Int = 0

    // Special tokens mapping (string -> rank)
    private var specialTokensMap: [String: Int] = [:]
    private var reverseSpecialTokensMap: [Int: String] = [:]

    // Special tokens (loaded from config files)
    private var bosTokenId = 1
    private var eosTokenId = 2
    private let unkTokenId = 0
    private var padTokenId = 11
    private var instTokenId = 3
    private var endInstTokenId = 4
    private var systemPromptTokenId = 17
    private var endSystemPromptTokenId = 18

    // Public access to vocabulary
    public var vocab: [String: Int] {
        var result: [String: Int] = [:]
        for (bytes, rank) in mergeableRanks {
            if let string = String(data: bytes, encoding: .utf8) {
                result[string] = rank
            }
        }
        return result
    }

    // MARK: - JSON Structures

    private struct TekkenVocab: Codable {
        let config: TekkenConfig
        let vocab: [TekkenToken]
        let special_tokens: [TekkenSpecialToken]?
    }

    private struct TekkenConfig: Codable {
        let pattern: String
        let num_vocab_tokens: Int
        let default_vocab_size: Int
        let default_num_special_tokens: Int
        let version: String
    }

    private struct TekkenToken: Codable {
        let rank: Int
        let token_bytes: String  // base64 encoded
        let token_str: String?
    }

    private struct TekkenSpecialToken: Codable {
        let rank: Int
        let token_str: String
        let is_control: Bool
    }

    private struct GenerationConfigJSON: Codable {
        let bos_token_id: Int?
        let eos_token_id: Int?
        let pad_token_id: Int?
    }

    // MARK: - Initialization

    public init(modelPath: String? = nil) {
        self.modelPath = modelPath
        loadTokenizerData()
    }

    private func loadTokenizerData() {
        if let modelPath = modelPath {
            loadTekkenTokenizerFromFile(modelPath: modelPath)
        } else {
            loadDefaultTokenizer()
        }
    }

    public func loadTekkenTokenizerFromFile(modelPath: String) {
        mergeableRanks.removeAll()
        reverseVocabulary.removeAll()
        rankToBytes.removeAll()
        specialTokensMap.removeAll()
        reverseSpecialTokensMap.removeAll()
        numSpecialTokens = 0

        let tekkenPath = "\(modelPath)/tekken.json"

        guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: tekkenPath)) else {
            FluxDebug.log("Cannot load \(tekkenPath), trying HuggingFace tokenizer format...")
            loadHuggingFaceTokenizer(modelPath: modelPath)
            return
        }

        do {
            let tekkenVocab = try JSONDecoder().decode(TekkenVocab.self, from: jsonData)

            // Load regex pattern
            regexPattern = tekkenVocab.config.pattern
            compiledRegex = try? NSRegularExpression(pattern: regexPattern, options: [])

            // Truncate vocabulary (like Python)
            numSpecialTokens = tekkenVocab.config.default_num_special_tokens
            let defaultVocabSize = tekkenVocab.config.default_vocab_size
            let maxVocab = defaultVocabSize - numSpecialTokens

            let truncatedVocab = Array(tekkenVocab.vocab.prefix(maxVocab))

            for token in truncatedVocab {
                if let tokenData = Data(base64Encoded: token.token_bytes) {
                    mergeableRanks[tokenData] = token.rank
                    rankToBytes[token.rank] = tokenData  // Fast reverse lookup

                    if let tokenString = token.token_str {
                        reverseVocabulary[token.rank + numSpecialTokens] = tokenString
                    } else if let decodedString = String(data: tokenData, encoding: .utf8) {
                        reverseVocabulary[token.rank + numSpecialTokens] = decodedString
                    }
                }
            }

            // Load special tokens from tekken.json
            if let specialTokens = tekkenVocab.special_tokens {
                for specialToken in specialTokens {
                    specialTokensMap[specialToken.token_str] = specialToken.rank
                    reverseSpecialTokensMap[specialToken.rank] = specialToken.token_str

                    // Update known special token IDs
                    switch specialToken.token_str {
                    case "<s>": bosTokenId = specialToken.rank
                    case "</s>": eosTokenId = specialToken.rank
                    case "<pad>": padTokenId = specialToken.rank
                    case "[INST]": instTokenId = specialToken.rank
                    case "[/INST]": endInstTokenId = specialToken.rank
                    case "[SYSTEM_PROMPT]": systemPromptTokenId = specialToken.rank
                    case "[/SYSTEM_PROMPT]": endSystemPromptTokenId = specialToken.rank
                    default: break
                    }
                }
                FluxDebug.log("Loaded \(specialTokens.count) special tokens")
            }

            // Load additional config from generation_config.json
            loadSpecialTokens(modelPath: modelPath)

            FluxDebug.log("Loaded tokenizer with \(mergeableRanks.count) tokens, numSpecialTokens=\(numSpecialTokens)")

        } catch {
            FluxDebug.error("Error parsing Tekken JSON: \(error)")
            loadDefaultTokenizer()
        }
    }

    private func loadSpecialTokens(modelPath: String) {
        let generationConfigPath = "\(modelPath)/generation_config.json"
        if let generationData = try? Data(contentsOf: URL(fileURLWithPath: generationConfigPath)) {
            if let generationConfig = try? JSONDecoder().decode(GenerationConfigJSON.self, from: generationData) {
                if let bos = generationConfig.bos_token_id { bosTokenId = bos }
                if let eos = generationConfig.eos_token_id { eosTokenId = eos }
                if let pad = generationConfig.pad_token_id { padTokenId = pad }
            }
        }
    }

    private func loadDefaultTokenizer() {
        regexPattern = "[\\w]+|[^\\w\\s]"
        compiledRegex = try? NSRegularExpression(pattern: regexPattern, options: [])

        let demoTokens = ["hello", "world", "test", "user", "assistant"]
        for (index, token) in demoTokens.enumerated() {
            if let tokenData = token.data(using: .utf8) {
                mergeableRanks[tokenData] = index
                reverseVocabulary[index + numSpecialTokens] = token
            }
        }

        numSpecialTokens = 1000
    }

    /// Thread-safe result container for async loading
    private final class TokenizerResult: @unchecked Sendable {
        var tokenizer: Tokenizer?
        var error: Error?
    }

    /// Load HuggingFace tokenizer format (tokenizer.json + tokenizer_config.json)
    private func loadHuggingFaceTokenizer(modelPath: String) {
        let folderURL = URL(fileURLWithPath: modelPath)

        // Use a synchronous Task to load the tokenizer
        let semaphore = DispatchSemaphore(value: 0)
        let result = TokenizerResult()

        Task {
            do {
                result.tokenizer = try await AutoTokenizer.from(modelFolder: folderURL)
            } catch {
                result.error = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let tokenizer = result.tokenizer {
            self.hfTokenizer = tokenizer
            self.useHFTokenizer = true

            // Extract special tokens from the HF tokenizer
            if let bosId = tokenizer.bosTokenId {
                bosTokenId = bosId
            }
            if let eosId = tokenizer.eosTokenId {
                eosTokenId = eosId
            }

            FluxDebug.log("Loaded HuggingFace tokenizer successfully")
        } else {
            if let error = result.error {
                FluxDebug.error("Failed to load HuggingFace tokenizer: \(error)")
            }
            FluxDebug.log("Falling back to default tokenizer")
            loadDefaultTokenizer()
        }
    }

    // MARK: - Encoding

    /**
     * Encode text using BPE (equivalent to tiktoken.Encoding.encode + Tekkenizer offset)
     */
    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        guard !text.isEmpty else { return [] }

        // Use HuggingFace tokenizer if available
        if useHFTokenizer, let tokenizer = hfTokenizer {
            var tokens = tokenizer.encode(text: text)
            if addSpecialTokens {
                // The HF tokenizer might already add special tokens, check if we need to add them
                if tokens.first != bosTokenId {
                    tokens.insert(bosTokenId, at: 0)
                }
                if tokens.last != eosTokenId {
                    tokens.append(eosTokenId)
                }
            }
            return tokens
        }

        // Legacy tekken tokenizer
        let chunks = splitByRegexPattern(text)

        var rawTokens: [Int] = []
        for chunk in chunks {
            let chunkTokens = encodeBPEChunk(chunk)
            rawTokens.append(contentsOf: chunkTokens)
        }

        // Apply Tekkenizer offset
        var finalTokens = rawTokens.map { $0 + numSpecialTokens }

        if addSpecialTokens {
            finalTokens.insert(bosTokenId, at: 0)
            finalTokens.append(eosTokenId)
        }

        return finalTokens
    }

    private func splitByRegexPattern(_ text: String) -> [String] {
        guard let regex = compiledRegex else {
            return text.components(separatedBy: CharacterSet.whitespacesAndNewlines).filter { !$0.isEmpty }
        }

        let range = NSRange(location: 0, length: text.utf16.count)
        let matches = regex.matches(in: text, options: [], range: range)

        return matches.compactMap { match in
            guard let swiftRange = Range(match.range, in: text) else { return nil }
            return String(text[swiftRange])
        }
    }

    private func encodeBPEChunk(_ chunk: String) -> [Int] {
        guard !chunk.isEmpty else { return [] }
        guard let chunkData = chunk.data(using: .utf8) else { return [] }

        // Direct lookup if chunk exists in vocabulary
        if let directRank = mergeableRanks[chunkData] {
            return [directRank]
        }

        let bytes = Array(chunkData)

        if bytes.count == 1 {
            let byteData = Data([bytes[0]])
            if let rank = mergeableRanks[byteData] {
                return [rank]
            } else {
                return [unkTokenId]
            }
        }

        // BPE merge algorithm
        var word: [Data] = bytes.map { Data([$0]) }

        while word.count >= 2 {
            var pairs: [(Data, Data, Int)] = []

            for i in 0..<(word.count - 1) {
                let pair = word[i] + word[i + 1]
                if mergeableRanks[pair] != nil {
                    pairs.append((word[i], word[i + 1], i))
                }
            }

            if pairs.isEmpty { break }

            let bestPair = pairs.min { pair1, pair2 in
                let rank1 = mergeableRanks[pair1.0 + pair1.1] ?? Int.max
                let rank2 = mergeableRanks[pair2.0 + pair2.1] ?? Int.max
                return rank1 < rank2
            }!

            let newData = bestPair.0 + bestPair.1
            let position = bestPair.2

            var newWord: [Data] = []
            var i = 0
            while i < word.count {
                if i == position {
                    newWord.append(newData)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }

            word = newWord
        }

        let tokens = word.compactMap { data -> Int? in
            if let rank = mergeableRanks[data] {
                return rank
            }
            return nil
        }

        return tokens.isEmpty ? [unkTokenId] : tokens
    }

    // MARK: - Decoding

    /**
     * Decode tokens back to text
     */
    public func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        // Use HuggingFace tokenizer if available
        if useHFTokenizer, let tokenizer = hfTokenizer {
            // swift-transformers 0.1.14+ doesn't have skipSpecialTokens parameter
            return tokenizer.decode(tokens: tokens)
        }

        // Tekken tokenizer - accumulate bytes for proper UTF-8 decoding
        var byteBuffer = Data()
        var result = Data()

        for tokenId in tokens {
            // Check if it's a special token (0-999 range)
            if tokenId < numSpecialTokens {
                // Flush byte buffer before special token
                if !byteBuffer.isEmpty {
                    result.append(byteBuffer)
                    byteBuffer.removeAll()
                }
                if !skipSpecialTokens {
                    if let specialToken = reverseSpecialTokensMap[tokenId],
                       let data = specialToken.data(using: .utf8) {
                        result.append(data)
                    }
                }
                continue
            }

            // Regular token - get raw rank
            let rawTokenId = tokenId - numSpecialTokens

            // Fast O(1) lookup using rankToBytes
            if let bytes = rankToBytes[rawTokenId] {
                byteBuffer.append(bytes)
            }
        }

        // Append remaining bytes
        if !byteBuffer.isEmpty {
            result.append(byteBuffer)
        }

        // Decode accumulated bytes as UTF-8, replacing invalid sequences
        return String(data: result, encoding: .utf8)
            ?? String(decoding: result, as: UTF8.self)
    }

    public func batchDecode(_ tokenIdsList: [[Int]], skipSpecialTokens: Bool = true) -> [String] {
        return tokenIdsList.map { decode($0, skipSpecialTokens: skipSpecialTokens) }
    }

    // MARK: - Properties

    public var vocabSize: Int {
        // For HF tokenizer, use the config vocabSize or default
        if useHFTokenizer {
            return 131072  // Mistral default vocab size
        }
        return mergeableRanks.count + numSpecialTokens
    }
    public var bosToken: Int { bosTokenId }
    public var eosToken: Int { eosTokenId }
    public var padToken: Int { padTokenId }

    // MARK: - Chat Template

    /**
     * Apply Mistral Small 3.2 chat template to messages
     * Format: [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] for system
     *         [INST]...[/INST] for user
     *         content</s> for assistant
     */
    public func applyChatTemplate(
        messages: [[String: String]],
        addGenerationPrompt: Bool = true
    ) -> String {
        // Try using HuggingFace tokenizer's chat template if available
        if useHFTokenizer, let tokenizer = hfTokenizer {
            do {
                let tokens = try tokenizer.applyChatTemplate(messages: messages)
                // Decode and return
                return tokenizer.decode(tokens: tokens)
            } catch {
                FluxDebug.log("HF chat template failed: \(error), using manual format")
            }
        }

        // Manual Mistral Small 3.2 format
        var result = "<s>"

        for message in messages {
            guard let role = message["role"], let content = message["content"] else { continue }

            if role == "system" {
                result += "[SYSTEM_PROMPT]\(content)[/SYSTEM_PROMPT]"
            } else if role == "user" {
                result += "[INST]\(content)[/INST]"
            } else if role == "assistant" {
                result += "\(content)</s>"
            }
        }

        return result
    }

    /**
     * Encode messages using chat template - returns token IDs directly
     * Uses special token IDs directly instead of BPE-encoding special token strings
     */
    public func encodeChatMessages(
        messages: [[String: String]],
        addGenerationPrompt: Bool = true
    ) -> [Int] {
        // Try using HuggingFace tokenizer's chat template if available
        if useHFTokenizer, let tokenizer = hfTokenizer {
            do {
                return try tokenizer.applyChatTemplate(messages: messages)
            } catch {
                FluxDebug.log("HF chat template failed: \(error), using manual format")
            }
        }

        // Build token array directly with proper special token IDs
        var tokens: [Int] = [bosTokenId]  // <s>

        for message in messages {
            guard let role = message["role"], let content = message["content"] else { continue }

            if role == "system" {
                tokens.append(systemPromptTokenId)  // [SYSTEM_PROMPT]
                tokens.append(contentsOf: encode(content, addSpecialTokens: false))
                tokens.append(endSystemPromptTokenId)  // [/SYSTEM_PROMPT]
            } else if role == "user" {
                tokens.append(instTokenId)  // [INST]
                tokens.append(contentsOf: encode(content, addSpecialTokens: false))
                tokens.append(endInstTokenId)  // [/INST]
            } else if role == "assistant" {
                tokens.append(contentsOf: encode(content, addSpecialTokens: false))
                tokens.append(eosTokenId)  // </s>
            }
        }

        return tokens
    }

    // MARK: - Factory

    public static func fromPretrained(_ modelPath: String) throws -> TekkenTokenizer {
        return TekkenTokenizer(modelPath: modelPath)
    }

    // MARK: - MLXArray Support

    public func callAsFunction(
        text: String,
        returnTensors: String = "mlx",
        padding: Bool = true
    ) throws -> [String: MLXArray] {
        let tokenIds = encode(text)
        var result: [String: MLXArray] = [
            "input_ids": MLXArray(tokenIds).reshaped([1, tokenIds.count])
        ]

        if padding {
            result["attention_mask"] = MLXArray.ones([1, tokenIds.count], type: Int32.self)
        }

        return result
    }
}
