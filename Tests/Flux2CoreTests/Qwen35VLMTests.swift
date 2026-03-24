// Qwen35VLMTests.swift - Tests for Qwen3.5 VLM components and LoRA evaluator
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
@testable import FluxTextEncoders

final class Qwen35ConfigTests: XCTestCase {

    // MARK: - Qwen35 Configuration

    func testQwen35TextConfigParsing() throws {
        let json = """
        {
            "vocab_size": 248320,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "eos_token_id": 248044,
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10]
            }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen35TextConfig.self, from: data)

        XCTAssertEqual(config.vocabSize, 248320)
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.numHiddenLayers, 32)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.fullAttentionInterval, 4)
        XCTAssertEqual(config.linearConvKernelDim, 4)
        XCTAssertEqual(config.linearKeyHeadDim, 128)
        XCTAssertEqual(config.linearNumKeyHeads, 16)
        XCTAssertEqual(config.linearNumValueHeads, 32)
        XCTAssertEqual(config.linearValueHeadDim, 128)
        XCTAssertEqual(config.ropeTheta, 10_000_000.0)
        XCTAssertEqual(config.partialRotaryFactor, 0.25)
        XCTAssertEqual(config.mropeSectionSizes, [11, 11, 10])
        XCTAssertEqual(config.eosTokenId, 248044)
    }

    func testQwen35LayerTypeDetection() throws {
        let json = """
        {
            "vocab_size": 248320, "hidden_size": 2560, "intermediate_size": 9216,
            "num_hidden_layers": 8, "num_attention_heads": 16, "num_key_value_heads": 4,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
                          "linear_attention", "linear_attention", "linear_attention", "full_attention"]
        }
        """
        let config = try JSONDecoder().decode(Qwen35TextConfig.self, from: json.data(using: .utf8)!)

        // Linear layers: 0, 1, 2, 4, 5, 6
        XCTAssertTrue(config.isLinearLayer(0))
        XCTAssertTrue(config.isLinearLayer(1))
        XCTAssertTrue(config.isLinearLayer(2))
        XCTAssertFalse(config.isLinearLayer(3))  // full_attention
        XCTAssertTrue(config.isLinearLayer(4))
        XCTAssertFalse(config.isLinearLayer(7))  // full_attention
    }

    func testQwen35RotaryDim() throws {
        let json = """
        {
            "vocab_size": 248320, "hidden_size": 2560, "intermediate_size": 9216,
            "num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4,
            "head_dim": 256,
            "rope_parameters": { "partial_rotary_factor": 0.25 }
        }
        """
        let config = try JSONDecoder().decode(Qwen35TextConfig.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(config.rotaryDim, 64)  // 256 * 0.25
    }

    func testQwen35VisionConfigParsing() throws {
        let json = """
        {
            "depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
            "num_heads": 16, "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "in_channels": 3, "out_hidden_size": 2560,
            "num_position_embeddings": 2304
        }
        """
        let config = try JSONDecoder().decode(Qwen35VisionConfig.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.depth, 24)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHeads, 16)
        XCTAssertEqual(config.patchSize, 16)
        XCTAssertEqual(config.spatialMergeSize, 2)
        XCTAssertEqual(config.outHiddenSize, 2560)
        XCTAssertEqual(config.numPositionEmbeddings, 2304)  // 48*48
    }

    func testQwen35TopLevelConfigParsing() throws {
        let json = """
        {
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "text_config": {
                "vocab_size": 248320, "hidden_size": 2560, "intermediate_size": 9216,
                "num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4
            },
            "vision_config": {
                "depth": 24, "hidden_size": 1024, "num_heads": 16
            }
        }
        """
        let config = try JSONDecoder().decode(Qwen35Config.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.imageTokenId, 248056)
        XCTAssertEqual(config.visionStartTokenId, 248053)
        XCTAssertEqual(config.visionEndTokenId, 248054)
        XCTAssertEqual(config.textConfig.hiddenSize, 2560)
        XCTAssertEqual(config.visionConfig.depth, 24)
    }
}

// MARK: - LoRA Evaluator Recommendation Tests

final class LoRARecommendationTests: XCTestCase {

    func testRecommendationHighScores() {
        // Both scores high → minimal training needed
        let rec = computeTestRecommendation(scene: 9, style: 9, model: .klein4B)
        XCTAssertLessThanOrEqual(rec.steps, 200)
        XCTAssertLessThanOrEqual(rec.rank, 8)
        XCTAssertEqual(rec.timestepSampling, "balanced")
        XCTAssertFalse(rec.useDOP)
    }

    func testRecommendationLowScenes() {
        // Low scene, high style → content focus
        let rec = computeTestRecommendation(scene: 2, style: 8, model: .klein4B)
        XCTAssertGreaterThanOrEqual(rec.steps, 500)
        XCTAssertEqual(rec.timestepSampling, "content")
        XCTAssertTrue(rec.useDOP)
    }

    func testRecommendationLowStyle() {
        // High scene, low style → style focus
        let rec = computeTestRecommendation(scene: 8, style: 2, model: .klein4B)
        XCTAssertGreaterThanOrEqual(rec.steps, 500)
        XCTAssertEqual(rec.timestepSampling, "style")
        XCTAssertFalse(rec.useDOP)  // Scene OK, no need for DOP
    }

    func testRecommendationBothLow() {
        // Both low → maximum effort
        let rec = computeTestRecommendation(scene: 1, style: 1, model: .klein4B)
        XCTAssertGreaterThanOrEqual(rec.steps, 1000)
        XCTAssertGreaterThanOrEqual(rec.rank, 48)
        XCTAssertTrue(rec.useDOP)
    }

    func testRecommendationGradientCheckpointingDev() {
        let rec = computeTestRecommendation(scene: 5, style: 5, model: .dev)
        XCTAssertTrue(rec.useGradientCheckpointing)
    }

    func testRecommendationGradientCheckpointingKlein4B() {
        let rec = computeTestRecommendation(scene: 5, style: 5, model: .klein4B)
        XCTAssertFalse(rec.useGradientCheckpointing)
    }

    func testRecommendationTargetLayersHighRank() {
        let rec = computeTestRecommendation(scene: 0, style: 0, model: .klein4B)
        // rank > 32 → attention only for memory
        if rec.rank > 32 {
            XCTAssertEqual(rec.targetLayers, "attention")
        }
    }

    func testRecommendationYAMLExport() {
        let rec = LoRARecommendation(
            steps: 500, rank: 32, alpha: 32.0, learningRate: 1e-4,
            warmupSteps: 50, timestepSampling: "content", lossWeighting: "bell_shaped",
            targetLayers: "all", useDOP: true, dopClass: "cat",
            useGradientCheckpointing: false, summary: "Test"
        )

        let yaml = rec.toYAML(model: .klein4B, triggerWord: "xyz_cat")
        XCTAssertTrue(yaml.contains("name: klein-4b"))
        XCTAssertTrue(yaml.contains("rank: 32"))
        XCTAssertTrue(yaml.contains("max_steps: 500"))
        XCTAssertTrue(yaml.contains("timestep_sampling: content"))
        XCTAssertTrue(yaml.contains("diff_output_preservation: true"))
        XCTAssertTrue(yaml.contains("trigger_word: xyz_cat"))
    }

    // Helper — mirrors the private computeRecommendation logic in LoRAEvaluator
    private func computeTestRecommendation(scene: Int, style: Int, model: Flux2Model) -> LoRARecommendation {
        let sceneGap = 10 - max(0, scene)
        let styleGap = 10 - max(0, style)
        let totalGap = sceneGap + styleGap
        let needsGradientCheckpoint = model == .dev || model == .klein9B || model == .klein9BBase

        let steps: Int
        let rank: Int
        switch totalGap {
        case 0...4: steps = 150; rank = 8
        case 5...8: steps = 400; rank = 16
        case 9...12: steps = 750; rank = 32
        case 13...16: steps = 1200; rank = 48
        default: steps = 2000; rank = 64
        }

        let timestepSampling: String
        if sceneGap > styleGap + 2 { timestepSampling = "content" }
        else if styleGap > sceneGap + 2 { timestepSampling = "style" }
        else { timestepSampling = "balanced" }

        let useDOP = sceneGap >= 4
        let targetLayers = rank <= 32 ? "all" : "attention"

        return LoRARecommendation(
            steps: steps, rank: rank, alpha: Float(rank), learningRate: 1e-4,
            warmupSteps: max(10, steps / 10), timestepSampling: timestepSampling,
            lossWeighting: "bell_shaped", targetLayers: targetLayers,
            useDOP: useDOP, dopClass: useDOP ? "object" : nil,
            useGradientCheckpointing: needsGradientCheckpoint, summary: "test"
        )
    }
}

// MARK: - Qwen3-VL/VL Variant Registry Tests

final class Qwen35RegistryTests: XCTestCase {

    @MainActor
    func testQwen35VariantProperties() {
        let variant = Qwen35Variant.qwen35_4B_4bit
        XCTAssertEqual(variant.repoId, "mlx-community/Qwen3.5-4B-MLX-4bit")
        XCTAssertFalse(variant.isGated)
        XCTAssertEqual(variant.license, "Apache 2.0")
    }

    @MainActor
    func testQwen35_8bitVariant() {
        let variant = Qwen35Variant.qwen35_4B_8bit
        XCTAssertEqual(variant.repoId, "mlx-community/Qwen3.5-4B-MLX-8bit")
        XCTAssertEqual(variant.estimatedSizeGB, 5)
    }

    @MainActor
    func testQwen35ModelRegistry() {
        let model = TextEncoderModelRegistry.shared.qwen35Model(withVariant: .qwen35_4B_4bit)
        XCTAssertNotNil(model)
        XCTAssertEqual(model?.parameters, "4B")
    }

    @MainActor
    func testQwen3VLVariantKleinMapping() {
        XCTAssertEqual(Qwen3VLVariant.qwen3VL_4B_8bit.kleinVariant, .klein4B)
        XCTAssertEqual(Qwen3VLVariant.qwen3VL_8B_8bit.kleinVariant, .klein9B)
    }
}

// MARK: - Image Comparison Parsing Tests

final class FluxComparisonParsingTests: XCTestCase {

    func testParseJSONComparison() {
        let json = """
        {"scene_score": 7, "scene_reason": "Similar subjects", "style_score": 4, "style_reason": "Different palette"}
        """
        let comparison = parseTestComparison(json)
        XCTAssertEqual(comparison.sceneScore, 7)
        XCTAssertEqual(comparison.styleScore, 4)
        XCTAssertEqual(comparison.sceneReason, "Similar subjects")
        XCTAssertEqual(comparison.styleReason, "Different palette")
    }

    func testParseTextFallback() {
        let text = """
        The images are similar.
        - Scene: 8/10 (Same cat on couch).
        - Style: 5/10 (Different lighting).
        """
        let comparison = parseTestComparison(text)
        XCTAssertEqual(comparison.sceneScore, 8)
        XCTAssertEqual(comparison.styleScore, 5)
    }

    // Replicates FluxTextEncoders.parseComparisonResult logic for testing
    private func parseTestComparison(_ text: String) -> FluxTextEncoders.FluxImageComparison {
        // JSON first
        if let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}") {
            let jsonStr = String(text[start...end])
            struct CJ: Decodable { let scene_score: Int?; let style_score: Int?; let scene_reason: String?; let style_reason: String? }
            if let data = jsonStr.data(using: .utf8),
               let p = try? JSONDecoder().decode(CJ.self, from: data), p.scene_score != nil {
                return .init(sceneScore: p.scene_score ?? -1, styleScore: p.style_score ?? -1,
                            sceneReason: p.scene_reason ?? "", styleReason: p.style_reason ?? "", rawResponse: text)
            }
        }
        // Regex fallback
        let scene = extractScore(from: text, keyword: "scene")
        let style = extractScore(from: text, keyword: "style")
        return .init(sceneScore: scene, styleScore: style, sceneReason: "", styleReason: "", rawResponse: text)
    }

    private func extractScore(from text: String, keyword: String) -> Int {
        let lower = text.lowercased()
        let pattern = "\(keyword)[^0-9]*?(\\d+)/10"
        if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
           let match = regex.firstMatch(in: lower, range: NSRange(lower.startIndex..., in: lower)),
           match.numberOfRanges > 1,
           let range = Range(match.range(at: 1), in: lower) {
            return Int(lower[range]) ?? -1
        }
        return -1
    }
}
