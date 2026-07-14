// PrequantizedCheckpoint.swift — save/load MLX-native pre-quantized transformer weights
// Copyright 2026 Vincent Gourbin
//
// On-the-fly quantization pays, at EVERY load, the full bf16 read (17 GB for
// Klein 9B) plus a GPU quantize pass. On machines whose page cache cannot
// hold the bf16 file (16-32 GB Macs) every load is a cold load, so this cost
// is structural. A pre-quantized checkpoint stores the QuantizedLinear
// parameters (packed uint32 `weight` + `scales` + optional `biases`) plus the
// non-quantized parameters, in the framework's OWN module naming — reloading
// is then a plain `loadArrays` + `update(parameters:)`: no key mapping, no
// bf16 materialization, no quantize pass, and ~half the bytes read.
//
// Layout: `<sourceModelDir>/mlx-prequantized/<quant>/transformer.safetensors`.
// Living inside the source model directory keeps the artifact self-contained
// (deleting the model removes its derivatives; host apps listing top-level
// model directories don't see a phantom model) and sidesteps any registry
// change. Writing is EXPLICIT ONLY (CLI `export-quantized` or
// `Flux2Pipeline.exportPrequantizedTransformer()`) — the framework never
// surprises the host with a multi-GB disk write; shipping/deleting exports is
// host policy.

import Foundation
import MLX
import MLXNN

public enum Flux2PrequantizedCheckpoint {

    /// Format identifier stored in (and required from) the safetensors metadata.
    public static let format = "flux2-mlx-prequantized-v1"

    static let subdirectory = "mlx-prequantized"
    static let filename = "transformer.safetensors"

    /// Location of the pre-quantized export for `quantization`, derived from
    /// the source model directory (the one `findModelPath` resolves).
    public static func weightsURL(
        sourceModelPath: URL,
        quantization: TransformerQuantization
    ) -> URL {
        sourceModelPath
            .appendingPathComponent(subdirectory)
            .appendingPathComponent(quantization.rawValue)
            .appendingPathComponent(filename)
    }

    /// Whether an export exists for this source/quantization pair.
    public static func exists(
        sourceModelPath: URL,
        quantization: TransformerQuantization
    ) -> Bool {
        FileManager.default.fileExists(
            atPath: weightsURL(sourceModelPath: sourceModelPath, quantization: quantization).path)
    }

    // MARK: - Save

    /// Save an already-quantized transformer as a pre-quantized checkpoint.
    ///
    /// The full flattened parameter set is written (quantized triplets and
    /// plain f16 parameters alike) so the file is self-sufficient. Modes
    /// without biases (mxfp4/mxfp8/nvfp4) simply have no `.biases` keys.
    ///
    /// - Returns: URL of the written safetensors file.
    @discardableResult
    public static func save(
        model: Flux2Transformer2DModel,
        sourceModelPath: URL,
        quantization: TransformerQuantization
    ) throws -> URL {
        guard quantization != .bf16 else {
            throw Flux2Error.invalidConfiguration(
                "Pre-quantized export requires a quantized transformer (got bf16). Use the original checkpoint instead.")
        }

        var params: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            params[key] = value
        }
        // Materialize before writing — save() would do it array by array,
        // one batched eval lets the IO/compute pipeline overlap instead.
        eval(params.values.map { $0 })

        let url = weightsURL(sourceModelPath: sourceModelPath, quantization: quantization)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        let metadata: [String: String] = [
            "format": format,
            "quantization": quantization.rawValue,
            "bits": String(quantization.bits),
            "group_size": String(quantization.groupSize),
            "mode": quantization.mode.rawValue,
            "source": sourceModelPath.lastPathComponent,
            "created_by": "flux-2-swift-mlx",
        ]
        try MLX.save(arrays: params, metadata: metadata, url: url)

        let sizeGB = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int)
            .flatMap { Double($0 ?? 0) / 1_000_000_000 } ?? 0
        Flux2Debug.log(
            "Pre-quantized checkpoint saved: \(url.path) (\(params.count) tensors, \(String(format: "%.1f", sizeGB)) GB)")
        return url
    }

    // MARK: - Load

    /// Try to load a pre-quantized checkpoint into a freshly-initialized
    /// transformer. Returns `false` (after logging why) when the checkpoint
    /// is absent or fails validation — the caller then falls back to the
    /// standard bf16 + on-the-fly path. Never throws for recoverable
    /// situations: a bad export must not break generation.
    ///
    /// On success the model holds the loaded QuantizedLinear parameters.
    /// Note: `quantize(model:)` is applied here only to give the model the
    /// QuantizedLinear STRUCTURE; the lazy quantization of the random init
    /// weights is never evaluated — `update(parameters:)` replaces those
    /// arrays before any eval, so the structural pass costs nothing.
    public static func load(
        into model: Flux2Transformer2DModel,
        sourceModelPath: URL,
        quantization: TransformerQuantization
    ) -> Bool {
        let url = weightsURL(sourceModelPath: sourceModelPath, quantization: quantization)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }

        let loaded: [String: MLXArray]
        let metadata: [String: String]
        do {
            (loaded, metadata) = try loadArraysAndMetadata(url: url)
        } catch {
            Flux2Debug.warning("Pre-quantized checkpoint unreadable (\(error)) — falling back to standard load: \(url.path)")
            return false
        }

        // Strict metadata validation — a stale or foreign file must not be
        // silently half-applied.
        let expected: [(String, String)] = [
            ("format", format),
            ("bits", String(quantization.bits)),
            ("group_size", String(quantization.groupSize)),
            ("mode", quantization.mode.rawValue),
        ]
        for (key, value) in expected where metadata[key] != value {
            Flux2Debug.warning(
                "Pre-quantized checkpoint metadata mismatch (\(key): \(metadata[key] ?? "nil") ≠ \(value)) — falling back to standard load: \(url.path)")
            return false
        }

        // Give the model the QuantizedLinear structure (lazy, see doc above).
        quantize(
            model: model,
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            mode: quantization.mode)

        // Full-coverage key check in BOTH directions: missing keys would
        // leave random weights in place; extra keys mean a layout drift.
        var modelKeys = Set<String>()
        for (key, _) in model.parameters().flattened() {
            modelKeys.insert(key)
        }
        let loadedKeys = Set(loaded.keys)
        guard loadedKeys == modelKeys else {
            let missing = modelKeys.subtracting(loadedKeys)
            let extra = loadedKeys.subtracting(modelKeys)
            Flux2Debug.warning(
                "Pre-quantized checkpoint key set mismatch (missing \(missing.count), extra \(extra.count); e.g. \(missing.prefix(3)) / \(extra.prefix(3))) — falling back to standard load: \(url.path)")
            return false
        }

        _ = model.update(parameters: ModuleParameters.unflattened(loaded))
        eval(model.parameters())
        Flux2Debug.log(
            "Loaded pre-quantized transformer (\(quantization.rawValue), \(loaded.count) tensors) — skipped bf16 read and quantize pass")
        return true
    }
}
