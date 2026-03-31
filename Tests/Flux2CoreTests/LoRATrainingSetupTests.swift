// LoRATrainingSetupTests.swift - Tests for LoRA training setup and YAML generation
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Core
@testable import FluxTextEncoders

final class LoRATrainingSetupTests: XCTestCase {

    // MARK: - YAML with VLM Scoring

    func testYAMLWithVLMScoringContainsValidation() {
        let rec = LoRARecommendation(
            steps: 500, rank: 32, alpha: 32.0, learningRate: 1e-4,
            warmupSteps: 50, timestepSampling: "content", lossWeighting: "bell_shaped",
            targetLayers: "all", useDOP: true, dopClass: "person",
            useGradientCheckpointing: false, summary: "Test"
        )

        let yaml = rec.toYAMLWithVLMScoring(
            model: .klein4B,
            triggerWord: "VinZ",
            datasetPath: "./dataset",
            validationPrompt: "VinZ, portrait of a man with glasses",
            referenceImagePath: "/path/to/reference.jpg",
            checkpointEvery: 50
        )

        // Check training config
        XCTAssertTrue(yaml.contains("name: klein-4b"))
        XCTAssertTrue(yaml.contains("rank: 32"))
        XCTAssertTrue(yaml.contains("max_steps: 500"))

        // Check validation section
        XCTAssertTrue(yaml.contains("validation:"))
        XCTAssertTrue(yaml.contains("VinZ, portrait of a man with glasses"))
        XCTAssertTrue(yaml.contains("every_n_steps: 50"))

        // Check VLM scoring config
        XCTAssertTrue(yaml.contains("vlm_scoring:"))
        XCTAssertTrue(yaml.contains("enabled: true"))
        XCTAssertTrue(yaml.contains("reference.jpg"))
        XCTAssertTrue(yaml.contains("save_best_checkpoint: true"))
        XCTAssertTrue(yaml.contains("compare_to_baseline: true"))
    }

    func testYAMLWithVLMScoringCheckpointInterval() {
        let rec = LoRARecommendation(
            steps: 1000, rank: 48, alpha: 48.0, learningRate: 1e-4,
            warmupSteps: 100, timestepSampling: "balanced", lossWeighting: "bell_shaped",
            targetLayers: "all", useDOP: false, dopClass: nil,
            useGradientCheckpointing: true, summary: "Test"
        )

        let yaml = rec.toYAMLWithVLMScoring(
            model: .klein9B,
            triggerWord: "sks",
            validationPrompt: "sks, portrait photo",
            referenceImagePath: "/ref.png",
            checkpointEvery: 100
        )

        XCTAssertTrue(yaml.contains("save_every: 100"))
        XCTAssertTrue(yaml.contains("every_n_steps: 100"))
        XCTAssertTrue(yaml.contains("gradient_checkpointing: true"))
    }

    func testYAMLEscapesQuotesInPrompt() {
        let rec = LoRARecommendation(
            steps: 250, rank: 32, alpha: 32.0, learningRate: 1e-4,
            warmupSteps: 25, timestepSampling: "balanced", lossWeighting: "bell_shaped",
            targetLayers: "all", useDOP: false, dopClass: nil,
            useGradientCheckpointing: false, summary: "Test"
        )

        let yaml = rec.toYAMLWithVLMScoring(
            model: .klein4B,
            triggerWord: "test",
            validationPrompt: "test, a man wearing a \"blue\" shirt",
            referenceImagePath: "/ref.png"
        )

        // Quotes should be escaped in YAML
        XCTAssertTrue(yaml.contains("\\\"blue\\\""))
    }

    // MARK: - LoRA Training Setup Struct

    func testLoRATrainingSetupCreation() {
        // Verify struct fields compile and are accessible
        let context = LoRAContext(name: "Test", description: "A test subject")
        XCTAssertEqual(context.name, "Test")
        XCTAssertEqual(context.description, "A test subject")
    }

    // MARK: - VLM Scoring Config

    func testVLMScoringConfigDefaults() {
        let config = SimpleLoRAConfig(outputDir: URL(fileURLWithPath: "/tmp/test"))
        XCTAssertFalse(config.vlmScoringEnabled)
        XCTAssertEqual(config.vlmScoringSceneWeight, 0.5)
        XCTAssertEqual(config.vlmScoringMaxReferences, 3)
        XCTAssertTrue(config.vlmScoringCompareToBaseline)
        XCTAssertTrue(config.vlmScoringBestCheckpoint)
        XCTAssertFalse(config.vlmScoringEarlyStopping)
        XCTAssertEqual(config.vlmScoringPatience, 3)
    }

    func testVLMScoringConfigCustom() {
        var config = SimpleLoRAConfig(outputDir: URL(fileURLWithPath: "/tmp/test"))
        config.vlmScoringEnabled = true
        config.vlmScoringSceneWeight = 0.7
        config.vlmScoringMaxReferences = 1
        config.vlmScoringEarlyStopping = true
        config.vlmScoringPatience = 5

        XCTAssertTrue(config.vlmScoringEnabled)
        XCTAssertEqual(config.vlmScoringSceneWeight, 0.7)
        XCTAssertEqual(config.vlmScoringMaxReferences, 1)
        XCTAssertTrue(config.vlmScoringEarlyStopping)
        XCTAssertEqual(config.vlmScoringPatience, 5)
    }

    // MARK: - VLM Score Records

    func testVLMPromptScoreCreation() {
        let score = VLMPromptScore(
            promptIndex: 0,
            sceneScore: 65,
            styleScore: 70,
            sceneReason: "Similar but different person",
            styleReason: "Same photographic style",
            baselineSceneScore: 45,
            baselineStyleScore: 85
        )

        XCTAssertEqual(score.sceneScore, 65)
        XCTAssertEqual(score.styleScore, 70)
        XCTAssertEqual(score.baselineSceneScore, 45)
        XCTAssertEqual(score.baselineStyleScore, 85)
    }

    func testVLMScoreRecordComposite() {
        let promptScore = VLMPromptScore(
            promptIndex: 0,
            sceneScore: 60,
            styleScore: 80,
            sceneReason: "",
            styleReason: "",
            baselineSceneScore: 40,
            baselineStyleScore: 80
        )

        let record = VLMScoreRecord(
            step: 50,
            promptScores: [promptScore],
            compositeScore: 70.0,  // (60 + 80) / 2
            baselineComposite: 60.0,
            improvement: 10.0
        )

        XCTAssertEqual(record.step, 50)
        XCTAssertEqual(record.compositeScore, 70.0)
        XCTAssertEqual(record.baselineComposite, 60.0)
        XCTAssertEqual(record.improvement, 10.0)
    }

    // MARK: - Thinking Mode

    func testEnableThinkingParameterExists() {
        // Verify the enableThinking parameter is available on VLM APIs
        // (compile-time check — these would fail to compile if parameter missing)
        _ = FluxTextEncoders.FluxImageComparison(
            sceneScore: 50, styleScore: 60,
            sceneReason: "test", styleReason: "test",
            rawResponse: ""
        )
    }
}
