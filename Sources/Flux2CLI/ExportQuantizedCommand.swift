// ExportQuantizedCommand.swift — `flux2 export-quantized`
// Copyright 2026 Vincent Gourbin
//
// Loads a transformer with on-the-fly quantization (paying the standard
// bf16 read + quantize pass ONCE), then writes the resulting MLX-native
// pre-quantized checkpoint next to the source weights. Subsequent loads
// with the same model/quantization pick it up automatically and skip both
// the bf16 read and the quantize pass — the win is largest on machines
// whose page cache cannot hold the bf16 file (16-32 GB Macs), where every
// load is otherwise a cold load.

import Foundation
import ArgumentParser
import Flux2Core

struct ExportQuantized: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-quantized",
        abstract: "Export a pre-quantized MLX transformer checkpoint (skips bf16 read + quantize pass on later loads)"
    )

    @Option(name: .long, help: "Custom models directory (for sandboxed apps or custom storage).")
    var modelsDir: String?

    @Option(name: .long, help: "FLUX.2 model: klein-9b | klein-9b-base | klein-9b-kv | dev | klein-4b | klein-4b-base.")
    var fluxModel: String = "klein-9b"

    @Option(name: .long, help: "Transformer quantization to export (bf16 is rejected — the original checkpoint already is bf16): \(TransformerQuantization.cliValueList)")
    var transformerQuant: String = "qint8"

    func run() async throws {
        configureModelsDirectory(modelsDir)

        let modelChoice: Flux2Model
        switch fluxModel.lowercased() {
        case "klein-9b", "klein9b":               modelChoice = .klein9B
        case "klein-9b-base", "klein9b-base":     modelChoice = .klein9BBase
        case "klein-9b-kv", "klein9b-kv":         modelChoice = .klein9BKV
        case "klein-4b", "klein4b":               modelChoice = .klein4B
        case "klein-4b-base", "klein4b-base":     modelChoice = .klein4BBase
        case "dev":                               modelChoice = .dev
        default:
            throw ValidationError("Unsupported --flux-model '\(fluxModel)'.")
        }

        let quant = try TransformerQuantization.parseCLI(transformerQuant)
        guard quant != .bf16 else {
            throw ValidationError("--transformer-quant bf16 makes no sense for a pre-quantized export.")
        }

        // Text encoder config is irrelevant here (never loaded), any value works.
        let pipeline = Flux2Pipeline(
            model: modelChoice,
            quantization: Flux2QuantizationConfig(textEncoder: .mlx4bit, transformer: quant),
            vaeVariant: .smallDecoder
        )

        print("Loading \(modelChoice.displayName) with on-the-fly \(quant.rawValue) quantization (one-time cost)...")
        let start = Date()
        let url = try await pipeline.exportPrequantizedTransformer()
        print("✓ Export done in \(String(format: "%.1fs", Date().timeIntervalSince(start)))")
        print("✓ Checkpoint: \(url.path)")
        print("Subsequent loads of \(modelChoice.displayName) with --transformer-quant \(quant.rawValue) will use it automatically.")
        print("To reclaim the disk space, delete the model's 'mlx-prequantized/' subdirectory.")
    }
}
