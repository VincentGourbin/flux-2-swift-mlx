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

    @Option(name: .long, help: "\(Flux2Model.cliValueList).")
    var fluxModel: String = "klein-9b"

    @Option(name: .long, help: "Transformer quantization to export (bf16 is rejected — the original checkpoint already is bf16): \(TransformerQuantization.cliValueList)")
    var transformerQuant: String = "qint8"

    @Flag(name: .long, help: "Regenerate from the source weights even when a checkpoint already exists (without this flag, an existing checkpoint is kept as-is and the command is a no-op).")
    var force: Bool = false

    func run() async throws {
        configureModelsDirectory(modelsDir)

        let modelChoice = try Flux2Model.parseCLI(fluxModel)
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

        // Distinguish the no-op case up front so the user isn't told a
        // fresh export happened when a valid checkpoint was simply kept.
        if !force,
           let sourcePath = Flux2ModelDownloader.findModelPath(for: .transformer(
               ModelRegistry.TransformerVariant.variant(for: modelChoice, quantization: quant))),
           Flux2PrequantizedCheckpoint.isValid(sourceModelPath: sourcePath, quantization: quant)
        {
            let url = Flux2PrequantizedCheckpoint.weightsURL(
                sourceModelPath: sourcePath, quantization: quant)
            print("✓ A valid pre-quantized checkpoint already exists — nothing to do (pass --force to regenerate from the source weights).")
            print("✓ Checkpoint: \(url.path)")
            return
        }

        print("Loading \(modelChoice.displayName) with on-the-fly \(quant.rawValue) quantization (one-time cost)...")
        let start = Date()
        let url = try await pipeline.exportPrequantizedTransformer(force: force)
        print("✓ Export done in \(String(format: "%.1fs", Date().timeIntervalSince(start)))")
        print("✓ Checkpoint: \(url.path)")
        print("Subsequent loads of \(modelChoice.displayName) with --transformer-quant \(quant.rawValue) will use it automatically.")
        print("To reclaim the disk space, delete the model's 'mlx-prequantized/' subdirectory.")
    }
}
