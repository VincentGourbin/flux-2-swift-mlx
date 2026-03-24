// EvaluateLoRA.swift - Automated LoRA training evaluation CLI command
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import ImageIO
import UniformTypeIdentifiers

struct EvaluateLoRA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "evaluate-lora",
        abstract: "Evaluate how far a reference image is from base model output and recommend LoRA training parameters"
    )

    @Option(name: .shortAndLong, help: "Reference image path")
    var image: String

    @Option(name: .long, help: "Model variant: klein-4b (default), klein-9b, dev")
    var model: String = "klein-4b"

    @Option(name: .long, help: "Random seed for baseline generation")
    var seed: UInt64 = 42

    @Option(name: .shortAndLong, help: "Baseline image width")
    var width: Int = 512

    @Option(name: .shortAndLong, help: "Baseline image height")
    var height: Int = 512

    @Option(name: .long, help: "Output directory for evaluation results")
    var outputDir: String = "./evaluation"

    @Option(name: .long, help: "Trigger word for the LoRA (used in generated YAML)")
    var triggerWord: String = "xyz_trigger"

    @Option(name: .long, help: "Dataset path for training config")
    var datasetPath: String = "./dataset"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4")
    var transformerQuant: String = "qint8"

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    func run() async throws {
        let startTime = Date()

        // Parse model
        guard let modelVariant = Flux2Model(rawValue: model) else {
            throw ValidationError("Invalid model: \(model). Use klein-4b, klein-9b, or dev")
        }

        guard let transQuant = TransformerQuantization(rawValue: transformerQuant) else {
            throw ValidationError("Invalid transformer quantization: \(transformerQuant)")
        }

        // Load reference image
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: image) as CFURL, nil),
              let refImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ValidationError("Failed to load image: \(image)")
        }

        // Create output directory
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        print("=== LoRA Evaluation Pipeline ===")
        print("Model: \(modelVariant.displayName)")
        print("Reference: \(image) (\(refImage.width)x\(refImage.height))")
        print("Baseline: \(width)x\(height), seed=\(seed)")
        print("Output: \(outputDir)")
        print()

        // Run evaluation
        let quantConfig = Flux2QuantizationConfig(textEncoder: .mlx8bit, transformer: transQuant)

        let evaluator = LoRAEvaluator()
        let result = try await evaluator.evaluate(
            referenceImage: refImage,
            model: modelVariant,
            quantization: quantConfig,
            seed: seed,
            width: width,
            height: height,
            hfToken: token
        ) { message in
            print(message)
            fflush(stdout)
        }

        // Save results
        print("\n--- Saving results ---")

        // Copy reference image into output dir
        let refPath = "\(outputDir)/reference.png"
        try saveImage(refImage, to: refPath)
        print("  Reference:   \(refPath)")

        // Baseline image
        let baselinePath = "\(outputDir)/baseline.png"
        try saveImage(result.baselineImage, to: baselinePath)
        print("  Baseline:    \(baselinePath)")

        // Description (the prompt used to generate baseline)
        let descPath = "\(outputDir)/prompt.txt"
        try result.description.write(toFile: descPath, atomically: true, encoding: .utf8)
        print("  Prompt:      \(descPath)")

        // YAML config
        let yamlPath = "\(outputDir)/recommended_config.yaml"
        let yaml = result.recommendation.toYAML(
            model: modelVariant, triggerWord: triggerWord, datasetPath: datasetPath
        )
        try yaml.write(toFile: yamlPath, atomically: true, encoding: .utf8)
        print("  Config:      \(yamlPath)")

        // Report
        let report = buildReport(result: result, model: modelVariant)
        let reportPath = "\(outputDir)/report.txt"
        try report.write(toFile: reportPath, atomically: true, encoding: .utf8)
        print("  Report:      \(reportPath)")

        // Print recommendation
        let rec = result.recommendation
        print()
        print("+" + String(repeating: "=", count: 46) + "+")
        print("|     LoRA Training Recommendation            |")
        print("+" + String(repeating: "=", count: 46) + "+")
        print("| Scene: \(pad(result.sceneScore))/10  Style: \(pad(result.styleScore))/10                  |")
        print("|                                              |")
        print("| Steps: \(pad4(rec.steps))   Rank: \(pad(rec.rank))   LR: \(rec.learningRate)      |")
        print("| Timestep: \(rec.timestepSampling.padding(toLength: 10, withPad: " ", startingAt: 0))  Layers: \(rec.targetLayers.padding(toLength: 10, withPad: " ", startingAt: 0))   |")
        print("| DOP: \(rec.useDOP ? "yes" : "no ")   Grad Ckpt: \(rec.useGradientCheckpointing ? "yes" : "no ")              |")
        print("|                                              |")
        print("| Config: \(yamlPath.suffix(30).padding(toLength: 36, withPad: " ", startingAt: 0)) |")
        print("+" + String(repeating: "=", count: 46) + "+")

        let elapsed = Date().timeIntervalSince(startTime)
        print("\nTotal time: \(String(format: "%.1f", elapsed))s")
    }

    private func buildReport(result: LoRAEvaluation, model: Flux2Model) -> String {
        let rec = result.recommendation
        return """
        === LoRA Evaluation Report ===
        Date: \(ISO8601DateFormatter().string(from: Date()))
        Model: \(model.displayName)

        Reference Image Description:
        \(result.description)

        Comparison Scores:
          Scene: \(result.sceneScore)/10 — \(result.sceneReason)
          Style: \(result.styleScore)/10 — \(result.styleReason)

        Recommended Training Parameters:
          Steps: \(rec.steps)
          Rank: \(rec.rank)
          Alpha: \(rec.alpha)
          Learning Rate: \(rec.learningRate)
          Warmup Steps: \(rec.warmupSteps)
          Timestep Sampling: \(rec.timestepSampling)
          Loss Weighting: \(rec.lossWeighting)
          Target Layers: \(rec.targetLayers)
          DOP: \(rec.useDOP) \(rec.dopClass.map { "(class: \($0))" } ?? "")
          Gradient Checkpointing: \(rec.useGradientCheckpointing)

        Summary: \(rec.summary)
        """
    }

    private func pad(_ n: Int) -> String {
        String(format: "%2d", n)
    }

    private func pad4(_ n: Int) -> String {
        String(format: "%4d", n)
    }
}
