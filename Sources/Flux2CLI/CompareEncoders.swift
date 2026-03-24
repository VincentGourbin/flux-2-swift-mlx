// CompareEncoders.swift - Compare Qwen3 vs Qwen3-VL text encoder embeddings and image generation
// Copyright 2025 Vincent Gourbin

import Foundation
import ArgumentParser
import Flux2Core
import FluxTextEncoders
import MLX

struct CompareEncoders: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compare-encoders",
        abstract: "Compare Qwen3 vs Qwen3-VL text encoder embeddings and generated images"
    )

    @Argument(help: "Text prompt for comparison")
    var prompt: String

    @Option(name: .long, help: "Model variant: klein-4b, klein-9b")
    var model: String = "klein-4b"

    @Option(name: .long, help: "Random seed for reproducible generation")
    var seed: UInt64 = 42

    @Option(name: .shortAndLong, help: "Image width")
    var width: Int = 512

    @Option(name: .shortAndLong, help: "Image height")
    var height: Int = 512

    @Option(name: .shortAndLong, help: "Denoising steps (default: 4 for Klein)")
    var steps: Int = 4

    @Option(name: .shortAndLong, help: "Guidance scale")
    var guidance: Float = 1.0

    @Option(name: .long, help: "Output directory for comparison results")
    var outputDir: String = "./comparison"

    @Option(name: .long, help: "Transformer quantization: bf16, qint8, int4")
    var transformerQuant: String = "qint8"

    @Option(name: .long, help: "Local path to Qwen3-VL model (if not set, auto-downloads)")
    var vlModelPath: String?

    @Option(name: .long, help: "Qwen3-VL variant: vl-4b-8bit, vl-4b-4bit, vl-8b-8bit, vl-8b-4bit (auto-detected if not set)")
    var vlVariant: String?

    @Option(name: .long, help: "HuggingFace token for gated models")
    var hfToken: String?

    @Flag(name: .long, help: "Skip image generation (embeddings comparison only)")
    var embeddingsOnly: Bool = false

    func run() async throws {
        let startTime = Date()

        // Parse model variant
        guard model == "klein-4b" || model == "klein-9b" else {
            throw ValidationError("Invalid model: \(model). Use klein-4b or klein-9b")
        }
        let kleinVariant: KleinVariant = model == "klein-4b" ? .klein4B : .klein9B

        // Parse VL variant if specified
        let parsedVLVariant: Qwen3VLVariant?
        if let vlVar = vlVariant {
            guard let v = Qwen3VLVariant(rawValue: "qwen3\(vlVar)") else {
                throw ValidationError("Invalid VL variant: \(vlVar). Use vl-4b-8bit, vl-4b-4bit, vl-8b-8bit, vl-8b-4bit")
            }
            parsedVLVariant = v
        } else {
            parsedVLVariant = nil
        }

        // Create output directory
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

        let vlLabel = parsedVLVariant?.displayName ?? (kleinVariant == .klein4B ? "Qwen3-VL-4B" : "Qwen3-VL-8B")
        let standardLabel = kleinVariant == .klein4B ? "Qwen3-4B" : "Qwen3-8B"
        print("=== Text Encoder Comparison: \(standardLabel) vs \(vlLabel) ===")
        print("Prompt: \"\(prompt)\"")
        print("Seed: \(seed)")
        print("Size: \(width)x\(height)")
        print("Steps: \(steps)")
        if let path = vlModelPath {
            print("VL model path: \(path)")
        }
        if let v = parsedVLVariant {
            print("VL variant: \(v.displayName)")
        }
        print()

        let token = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]

        // ── Step 1: Standard Qwen3 embeddings ──
        print("--- Step 1: Loading \(standardLabel) (standard) ---")
        let standardStart = Date()

        try await FluxTextEncoders.shared.loadKleinModel(variant: kleinVariant)
        let standardEmbeddings = try FluxTextEncoders.shared.extractKleinEmbeddings(prompt: prompt)
        eval(standardEmbeddings)

        let standardStats = computeStats(standardEmbeddings, label: standardLabel)
        let standardElapsed = Date().timeIntervalSince(standardStart)
        print("  Encoding time: \(String(format: "%.1f", standardElapsed))s")

        // Unload standard
        await MainActor.run { FluxTextEncoders.shared.unloadKleinModel() }
        Memory.clearCache()

        // ── Step 2: Qwen3-VL embeddings ──
        print("\n--- Step 2: Loading \(vlLabel) (vision-language) ---")
        let vlStart = Date()

        if let path = vlModelPath {
            // Load from local path
            try await FluxTextEncoders.shared.loadKleinVLModel(variant: kleinVariant, from: path)
        } else {
            // Auto-download
            try await FluxTextEncoders.shared.loadKleinVLModel(
                variant: kleinVariant,
                qwen3VLVariant: parsedVLVariant,
                hfToken: token
            ) { progress, message in
                print("\r  [\(Int(progress * 100))%] \(message)", terminator: "")
                fflush(stdout)
            }
            print()
        }
        let vlEmbeddings = try FluxTextEncoders.shared.extractKleinVLEmbeddings(prompt: prompt)
        eval(vlEmbeddings)

        let vlStats = computeStats(vlEmbeddings, label: vlLabel)
        let vlElapsed = Date().timeIntervalSince(vlStart)
        print("  Encoding time: \(String(format: "%.1f", vlElapsed))s")

        // Unload VL
        await MainActor.run { FluxTextEncoders.shared.unloadKleinModel() }
        Memory.clearCache()

        // ── Step 3: Compare embeddings ──
        print("\n--- Step 3: Comparing embeddings ---")
        let comparison = compareEmbeddings(standard: standardEmbeddings, vl: vlEmbeddings)

        // Print report
        print()
        printReport(standardStats: standardStats, vlStats: vlStats, comparison: comparison)

        // ── Step 4: Generate images (optional) ──
        if !embeddingsOnly {
            guard let transformerQuantization = TransformerQuantization(rawValue: transformerQuant) else {
                throw ValidationError("Invalid transformer quantization: \(transformerQuant)")
            }

            let quantConfig = Flux2QuantizationConfig(
                textEncoder: .mlx8bit,
                transformer: transformerQuantization
            )

            let modelVariant: Flux2Model = kleinVariant == .klein4B ? .klein4B : .klein9B

            // Generate with standard embeddings
            print("\n--- Step 4a: Generating image with \(standardLabel) embeddings ---")
            let pipeline = Flux2Pipeline(model: modelVariant, quantization: quantConfig, hfToken: token)

            let imageA = try await pipeline.generateTextToImage(
                prompt: prompt,
                height: height,
                width: width,
                steps: steps,
                guidance: guidance,
                seed: seed,
                precomputedEmbeddings: standardEmbeddings
            ) { current, total in
                print("\r  Step \(current)/\(total)", terminator: "")
                fflush(stdout)
            }
            print()

            let standardPath = "\(outputDir)/standard_qwen3.png"
            try saveImage(imageA, to: standardPath)
            print("  Saved: \(standardPath)")

            // Generate with VL embeddings (same pipeline, same seed)
            print("\n--- Step 4b: Generating image with \(vlLabel) embeddings ---")

            let imageB = try await pipeline.generateTextToImage(
                prompt: prompt,
                height: height,
                width: width,
                steps: steps,
                guidance: guidance,
                seed: seed,
                precomputedEmbeddings: vlEmbeddings
            ) { current, total in
                print("\r  Step \(current)/\(total)", terminator: "")
                fflush(stdout)
            }
            print()

            let vlPath = "\(outputDir)/vl_qwen3vl.png"
            try saveImage(imageB, to: vlPath)
            print("  Saved: \(vlPath)")
        }

        // ── Save report ──
        let reportPath = "\(outputDir)/report.txt"
        let report = buildReport(
            prompt: prompt,
            standardStats: standardStats,
            vlStats: vlStats,
            comparison: comparison
        )
        try report.write(toFile: reportPath, atomically: true, encoding: .utf8)
        print("\nReport saved: \(reportPath)")

        let totalElapsed = Date().timeIntervalSince(startTime)
        print("Total time: \(String(format: "%.1f", totalElapsed))s")
    }

    // MARK: - Statistics

    struct EmbeddingStats {
        let label: String
        let shape: [Int]
        let mean: Float
        let std: Float
        let min: Float
        let max: Float
        let l2NormPerToken: Float
    }

    func computeStats(_ embeddings: MLXArray, label: String) -> EmbeddingStats {
        let flat = embeddings.reshaped([-1])
        let mean = MLX.mean(flat).item(Float.self)
        let std = MLX.sqrt(MLX.mean(MLX.pow(flat - mean, 2))).item(Float.self)
        let minVal = MLX.min(flat).item(Float.self)
        let maxVal = MLX.max(flat).item(Float.self)

        // L2 norm per token: average of norm([1, seq, hidden]) along hidden dim
        let squeezed = embeddings.squeezed(axis: 0)  // [seq, hidden]
        let norms = MLX.sqrt(MLX.sum(squeezed * squeezed, axis: -1))  // [seq]
        let avgNorm = MLX.mean(norms).item(Float.self)

        let stats = EmbeddingStats(
            label: label,
            shape: embeddings.shape,
            mean: mean,
            std: std,
            min: minVal,
            max: maxVal,
            l2NormPerToken: avgNorm
        )

        print("  \(label):")
        print("    Shape: \(embeddings.shape)")
        print("    Mean: \(String(format: "%.6f", mean))  Std: \(String(format: "%.6f", std))")
        print("    Min: \(String(format: "%.6f", minVal))  Max: \(String(format: "%.6f", maxVal))")
        print("    L2 norm/token: \(String(format: "%.4f", avgNorm))")

        return stats
    }

    // MARK: - Comparison

    struct EmbeddingComparison {
        let cosineSimilarity: Float
        let mae: Float
        let perLayerCosine: [Float]  // For layers [9, 18, 27] → splits at [0:2560, 2560:5120, 5120:7680]
    }

    func compareEmbeddings(standard: MLXArray, vl: MLXArray) -> EmbeddingComparison {
        let s = standard.squeezed(axis: 0)  // [512, 7680]
        let v = vl.squeezed(axis: 0)

        // Global cosine similarity (flatten everything)
        let sFlat = s.reshaped([-1])
        let vFlat = v.reshaped([-1])
        let dotProduct = MLX.sum(sFlat * vFlat).item(Float.self)
        let normS = MLX.sqrt(MLX.sum(sFlat * sFlat)).item(Float.self)
        let normV = MLX.sqrt(MLX.sum(vFlat * vFlat)).item(Float.self)
        let cosine = dotProduct / (normS * normV + 1e-8)

        // MAE
        let mae = MLX.mean(MLX.abs(sFlat - vFlat)).item(Float.self)

        // Per-layer cosine: split outputDim into 3 equal parts (one per extracted layer)
        let hiddenSize = s.dim(1) / 3  // 7680/3=2560 for Klein4B, 12288/3=4096 for Klein9B
        var perLayer: [Float] = []
        for i in 0..<3 {
            let start = i * hiddenSize
            let end = start + hiddenSize
            let sLayer = s[0..., start..<end].reshaped([-1])
            let vLayer = v[0..., start..<end].reshaped([-1])
            let dot = MLX.sum(sLayer * vLayer).item(Float.self)
            let nS = MLX.sqrt(MLX.sum(sLayer * sLayer)).item(Float.self)
            let nV = MLX.sqrt(MLX.sum(vLayer * vLayer)).item(Float.self)
            perLayer.append(dot / (nS * nV + 1e-8))
        }

        let comp = EmbeddingComparison(
            cosineSimilarity: cosine,
            mae: mae,
            perLayerCosine: perLayer
        )

        print("  Cosine similarity (global): \(String(format: "%.6f", cosine))")
        print("  MAE: \(String(format: "%.6f", mae))")
        print("  Per-layer cosine [9, 18, 27]: \(perLayer.map { String(format: "%.4f", $0) }.joined(separator: ", "))")

        return comp
    }

    // MARK: - Report

    func printReport(standardStats: EmbeddingStats, vlStats: EmbeddingStats, comparison: EmbeddingComparison) {
        print("╔══════════════════════════════════════════════════╗")
        print("║       Encoder Comparison Report                 ║")
        print("╠══════════════════════════════════════════════════╣")
        print("║ Prompt: \"\(prompt.prefix(40))\"")
        print("╠══════════════════════════════════════════════════╣")
        print("║ Qwen3-4B (standard):                            ║")
        print("║   Mean=\(f6(standardStats.mean)) Std=\(f6(standardStats.std)) L2=\(f4(standardStats.l2NormPerToken))")
        print("║ Qwen3-VL-4B (vision-language):                  ║")
        print("║   Mean=\(f6(vlStats.mean)) Std=\(f6(vlStats.std)) L2=\(f4(vlStats.l2NormPerToken))")
        print("╠══════════════════════════════════════════════════╣")
        print("║ Cosine similarity: \(f6(comparison.cosineSimilarity))")
        print("║ MAE: \(f6(comparison.mae))")
        print("║ Per-layer cosine [9,18,27]: \(comparison.perLayerCosine.map { f4($0) }.joined(separator: ", "))")
        print("╚══════════════════════════════════════════════════╝")
    }

    func buildReport(prompt: String, standardStats: EmbeddingStats, vlStats: EmbeddingStats, comparison: EmbeddingComparison) -> String {
        """
        === Text Encoder Comparison Report ===
        Date: \(ISO8601DateFormatter().string(from: Date()))
        Prompt: "\(prompt)"
        Seed: \(seed)
        Size: \(width)x\(height)
        Steps: \(steps)

        Qwen3-4B (standard):
          Shape: \(standardStats.shape)
          Mean: \(standardStats.mean)  Std: \(standardStats.std)
          Min: \(standardStats.min)  Max: \(standardStats.max)
          L2 norm/token: \(standardStats.l2NormPerToken)

        Qwen3-VL-4B (vision-language):
          Shape: \(vlStats.shape)
          Mean: \(vlStats.mean)  Std: \(vlStats.std)
          Min: \(vlStats.min)  Max: \(vlStats.max)
          L2 norm/token: \(vlStats.l2NormPerToken)

        Comparison:
          Cosine similarity (global): \(comparison.cosineSimilarity)
          MAE: \(comparison.mae)
          Per-layer cosine [layer 9, 18, 27]: \(comparison.perLayerCosine.map { String($0) }.joined(separator: ", "))

        Images:
          Standard: standard_qwen3.png
          VL:       vl_qwen3vl.png
        """
    }

    private func f6(_ v: Float) -> String { String(format: "%.6f", v) }
    private func f4(_ v: Float) -> String { String(format: "%.4f", v) }
}
