// PrequantizedCheckpoint.swift — save/load MLX-native pre-quantized weights
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
// Layout: `<sourceModelDir>/mlx-prequantized/<quant>/<component>.safetensors`.
// Living inside the source model directory keeps the artifact self-contained
// (deleting the model removes its derivatives; host apps listing top-level
// model directories don't see a phantom model) and sidesteps any registry
// change. Writing is EXPLICIT ONLY (CLI `export-quantized` or
// `Flux2Pipeline.exportPrequantizedTransformer()`) — the framework never
// surprises the host with a multi-GB disk write; shipping/deleting exports is
// host policy.
//
// Safety model (review #116 hardening):
// - the save is ATOMIC (temp file + rename) so a crash or disk-full can never
//   leave a truncated file at the auto-discovered path — critical because a
//   valid-header/truncated-data safetensors would otherwise load as SILENT
//   uninitialized weights (MLX's deferred pread errors are not surfaced);
// - `load(into:)` validates everything (metadata, source fingerprint, key
//   sets, shapes/dtypes) BEFORE mutating the caller's model — on any failure
//   it returns false having touched nothing, so the standard-path fallback
//   runs on a pristine model;
// - LoRA-baked exports are tagged in metadata and loudly warned about at
//   load (they intentionally restyle every generation of that model/quant).

import Foundation
import MLX
import MLXNN

public enum Flux2PrequantizedCheckpoint {

    /// Format identifier stored in (and required from) the safetensors metadata.
    public static let format = "flux2-mlx-prequantized-v1"

    static let subdirectory = "mlx-prequantized"

    /// Location of the pre-quantized export for `quantization`, derived from
    /// the source model directory (the one `findModelPath` resolves).
    /// `component` names the model being stored ("transformer" for the
    /// pipeline's use; the API itself only needs `Module`, so other heavy
    /// components can reuse it).
    public static func weightsURL(
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String = "transformer"
    ) -> URL {
        sourceModelPath
            .appendingPathComponent(subdirectory)
            .appendingPathComponent(quantization.rawValue)
            .appendingPathComponent("\(component).safetensors")
    }

    /// Whether an export exists for this source/quantization pair.
    public static func exists(
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String = "transformer"
    ) -> Bool {
        FileManager.default.fileExists(
            atPath: weightsURL(
                sourceModelPath: sourceModelPath, quantization: quantization,
                component: component
            ).path)
    }

    /// Remove the export for this source/quantization pair (idempotent).
    /// Used by `export(force:)` to guarantee regeneration from the source.
    public static func remove(
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String = "transformer"
    ) {
        let url = weightsURL(
            sourceModelPath: sourceModelPath, quantization: quantization, component: component)
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Source fingerprint

    /// Cheap identity of the source weights: name/size/mtime of every
    /// `.safetensors` directly in the source directory. Detects a re-download
    /// (new HF revision, repaired shards) or a checkpoint directory copied
    /// under a different same-architecture model, without hashing gigabytes.
    static func sourceFingerprint(of sourceModelPath: URL) -> String {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: sourceModelPath.path) else {
            return "unknown"
        }
        let parts: [String] = entries.filter { $0.hasSuffix(".safetensors") }.sorted().map { name in
            let attrs = try? fm.attributesOfItem(
                atPath: sourceModelPath.appendingPathComponent(name).path)
            let size = (attrs?[.size] as? NSNumber)?.int64Value ?? -1
            let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? -1
            return "\(name):\(size):\(Int(mtime))"
        }
        return parts.isEmpty ? "unknown" : parts.joined(separator: ",")
    }

    /// Payload integrity check: a safetensors file whose header parses but
    /// whose tensor data is truncated (killed copy, corrupted volume) would
    /// otherwise pass every header-derived check (metadata, keys, shapes)
    /// and then load SILENTLY UNINITIALIZED tensors — MLX's deferred pread
    /// errors are swallowed by the scheduler, so nothing surfaces. The
    /// header's `data_offsets` give the exact expected payload size; compare
    /// it against the real file size before trusting anything else.
    static func payloadIsComplete(url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let lenData = try? handle.read(upToCount: 8), lenData.count == 8 else { return false }
        let headerLen = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }  // little-endian per spec; Apple platforms are LE
        guard headerLen > 0, headerLen < 512 * 1024 * 1024,
              let headerData = try? handle.read(upToCount: Int(headerLen)),
              headerData.count == Int(headerLen),
              let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { return false }
        var maxEnd: Int64 = 0
        for (key, value) in header where key != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let offsets = entry["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let end = (offsets[1] as? NSNumber)?.int64Value
            else { return false }
            maxEnd = max(maxEnd, end)
        }
        let expectedSize = Int64(8) + Int64(headerLen) + maxEnd
        let actualSize = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            .map(Int64.init) ?? -1
        if actualSize != expectedSize {
            Flux2Debug.warning(
                "Pre-quantized checkpoint payload is incomplete (\(actualSize) bytes on disk, header declares \(expectedSize)) — the file is truncated or corrupt: \(url.path)")
            return false
        }
        return true
    }

    /// Lightweight validity probe: payload-integrity + header + metadata
    /// checks only (lazy read, no tensor materialization). Used by exports
    /// to distinguish "a valid checkpoint already exists" (no-op without
    /// force) from "an invalid or stale file squats the path" (regenerate) —
    /// otherwise the load-side "re-run export-quantized" advice would loop
    /// into a no-op.
    public static func isValid(
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String = "transformer"
    ) -> Bool {
        let url = weightsURL(
            sourceModelPath: sourceModelPath, quantization: quantization, component: component)
        guard FileManager.default.fileExists(atPath: url.path),
              payloadIsComplete(url: url),
              let (_, metadata) = try? loadArraysAndMetadata(url: url)
        else { return false }
        return validateMetadata(
            metadata, url: url, sourceModelPath: sourceModelPath,
            quantization: quantization, component: component)
    }

    /// Shared metadata validation (see `load` step 1 and `isValid`).
    private static func validateMetadata(
        _ metadata: [String: String],
        url: URL,
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String
    ) -> Bool {
        let expected: [(String, String)] = [
            ("format", format),
            ("quantization", quantization.rawValue),
            ("bits", String(quantization.bits)),
            ("group_size", String(quantization.groupSize)),
            ("mode", quantization.mode.rawValue),
            ("component", component),
            ("source", sourceModelPath.lastPathComponent),
        ]
        for (key, value) in expected where metadata[key] != value {
            Flux2Debug.warning(
                "Pre-quantized checkpoint metadata mismatch (\(key): \(metadata[key] ?? "nil") ≠ \(value)) — falling back to standard load: \(url.path)")
            return false
        }
        let fingerprint = sourceFingerprint(of: sourceModelPath)
        if metadata["source_fingerprint"] != fingerprint {
            Flux2Debug.warning(
                "Pre-quantized checkpoint is stale (source weights changed since export — re-download or update?) — falling back to standard load. Re-run `flux2 export-quantized` with force to refresh: \(url.path)")
            return false
        }
        return true
    }

    // MARK: - Save

    /// Save an already-quantized model as a pre-quantized checkpoint.
    ///
    /// The full flattened parameter set is written (quantized triplets and
    /// plain f16 parameters alike) so the file is self-sufficient. Modes
    /// without biases (mxfp4/mxfp8/nvfp4) simply have no `.biases` keys.
    ///
    /// The write is atomic: data goes to a temporary file in the destination
    /// directory, then replaces the final path in one rename. On any error
    /// the temporary file is removed and the previous checkpoint (if any) is
    /// left untouched.
    ///
    /// - Parameter loRABaked: set when the model contains merged LoRA
    ///   weights; recorded in metadata so `load` can warn (the export then
    ///   intentionally restyles every generation of that model/quant).
    /// - Returns: URL of the written safetensors file.
    @discardableResult
    public static func save(
        model: Module,
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String = "transformer",
        loRABaked: Bool = false
    ) throws -> URL {
        guard quantization != .bf16 else {
            throw Flux2Error.invalidConfiguration(
                "Pre-quantized export requires a quantized model (got bf16). Use the original checkpoint instead.")
        }

        let params = Dictionary(uniqueKeysWithValues: Array(model.parameters().flattened()))
        eval(Array(params.values))

        let url = weightsURL(
            sourceModelPath: sourceModelPath, quantization: quantization, component: component)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        var metadata: [String: String] = [
            "format": format,
            "quantization": quantization.rawValue,
            "bits": String(quantization.bits),
            "group_size": String(quantization.groupSize),
            "mode": quantization.mode.rawValue,
            "component": component,
            "source": sourceModelPath.lastPathComponent,
            "source_fingerprint": sourceFingerprint(of: sourceModelPath),
            "created_by": "flux-2-swift-mlx",
        ]
        if loRABaked {
            metadata["lora_baked"] = "true"
        }

        // Atomic write: never leave a truncated file at the discovered path.
        let tmpURL = url.deletingLastPathComponent()
            .appendingPathComponent(".tmp-\(url.lastPathComponent)")
        try? FileManager.default.removeItem(at: tmpURL)
        do {
            try MLX.save(arrays: params, metadata: metadata, url: tmpURL)
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmpURL)
        } catch {
            try? FileManager.default.removeItem(at: tmpURL)
            throw error
        }

        let sizeBytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        Flux2Debug.log(
            "Pre-quantized checkpoint saved: \(url.path) (\(params.count) tensors, \(String(format: "%.1f", Double(sizeBytes) / 1_000_000_000)) GB\(loRABaked ? ", LoRA-baked" : ""))")
        return url
    }

    // MARK: - Load

    /// Try to load a pre-quantized checkpoint into a freshly-initialized,
    /// NOT-yet-quantized model. Returns `false` (after logging why) when the
    /// checkpoint is absent or fails validation — in that case the model has
    /// NOT been mutated in any way, so the caller can safely fall back to
    /// the standard load path on the same instance. Never throws for
    /// recoverable situations: a bad export must not break generation.
    ///
    /// Validation order (everything before any mutation):
    /// 0. payload integrity (file size vs header `data_offsets` — the only
    ///    gate a valid-header/truncated-data file cannot pass);
    /// 1. metadata: format, quantization identity, bits/group/mode, source
    ///    name + fingerprint, LoRA-bake warning;
    /// 2. key sets in both directions against the post-quantization
    ///    parameter manifest — computed on a THROWAWAY structure clone, not
    ///    on the caller's model;
    /// 3. shapes and dtypes of every tensor against that manifest (the
    ///    update below runs unverified, so this is the only shape gate).
    /// Only then is the caller's model quantized (structure only, lazy —
    /// `update(parameters:)` replaces the arrays before anything evaluates)
    /// and updated.
    public static func load(
        into model: Module,
        makeStructureClone: () -> Module,
        sourceModelPath: URL,
        quantization: TransformerQuantization,
        component: String = "transformer"
    ) -> Bool {
        let url = weightsURL(
            sourceModelPath: sourceModelPath, quantization: quantization, component: component)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }

        // 0. Payload integrity — MUST precede everything: all later checks
        // are header-derived and would pass on a truncated-data file.
        guard payloadIsComplete(url: url) else { return false }

        let loaded: [String: MLXArray]
        let metadata: [String: String]
        do {
            (loaded, metadata) = try loadArraysAndMetadata(url: url)
        } catch {
            Flux2Debug.warning(
                "Pre-quantized checkpoint unreadable (\(error)) — falling back to standard load: \(url.path)")
            return false
        }

        // 1. Strict metadata validation — a stale or foreign file must not
        // be silently half-applied.
        guard validateMetadata(
            metadata, url: url, sourceModelPath: sourceModelPath,
            quantization: quantization, component: component)
        else {
            return false
        }
        if metadata["lora_baked"] == "true" {
            Flux2Debug.warning(
                "Pre-quantized checkpoint has LoRA weights BAKED IN — every generation of this model/quant will carry that LoRA's style. Delete \(url.deletingLastPathComponent().path) if this is not intended.")
        }

        // 2+3. Build the expected post-quantization manifest on a throwaway
        // structure clone so the caller's model stays pristine until every
        // check has passed. The clone's random weights are lazy and never
        // evaluated — only key names, shapes and dtypes are consulted.
        // Cast the clone to float16 first: the standard load path converts
        // every parameter to f16 before quantizing (WeightLoader), while a
        // fresh init is f32 — without the cast the manifest's scales/biases
        // dtypes would spuriously mismatch every legitimate checkpoint.
        let reference = makeStructureClone()
        let castParams = Dictionary(
            uniqueKeysWithValues: reference.parameters().flattened().map {
                ($0.0, $0.1.asType(.float16))
            })
        _ = reference.update(parameters: ModuleParameters.unflattened(castParams))
        quantize(
            model: reference,
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            mode: quantization.mode)
        let manifest = Dictionary(uniqueKeysWithValues: Array(reference.parameters().flattened()))

        let manifestKeys = Set(manifest.keys)
        let loadedKeys = Set(loaded.keys)
        guard loadedKeys == manifestKeys else {
            let missing = manifestKeys.subtracting(loadedKeys)
            let extra = loadedKeys.subtracting(manifestKeys)
            Flux2Debug.warning(
                "Pre-quantized checkpoint key set mismatch (missing \(missing.count), extra \(extra.count); e.g. \(missing.prefix(3)) / \(extra.prefix(3))) — falling back to standard load: \(url.path)")
            return false
        }
        for (key, expectedArray) in manifest {
            guard let actual = loaded[key] else { continue }  // covered by key check
            if actual.shape != expectedArray.shape || actual.dtype != expectedArray.dtype {
                Flux2Debug.warning(
                    "Pre-quantized checkpoint tensor mismatch at \(key) (shape \(actual.shape) dtype \(actual.dtype) ≠ expected \(expectedArray.shape) \(expectedArray.dtype)) — falling back to standard load: \(url.path)")
                return false
            }
        }

        // All checks passed — NOW mutate the caller's model: structural
        // quantize (lazy, never evaluated) then replace all parameters.
        quantize(
            model: model,
            groupSize: quantization.groupSize,
            bits: quantization.bits,
            mode: quantization.mode)
        _ = model.update(parameters: ModuleParameters.unflattened(loaded))
        eval(model.parameters())
        Flux2Debug.log(
            "Loaded pre-quantized \(component) (\(quantization.rawValue), \(loaded.count) tensors) — skipped source read and quantize pass")
        return true
    }
}
