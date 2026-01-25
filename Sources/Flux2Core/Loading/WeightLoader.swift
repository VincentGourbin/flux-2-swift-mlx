// WeightLoader.swift - Load model weights from safetensors
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Utilities for loading Flux.2 model weights from safetensors files
public class Flux2WeightLoader {

    /// Load all weights from a model directory
    /// - Parameter modelPath: Path to directory containing safetensors files
    /// - Returns: Dictionary of weight name to MLXArray
    public static func loadWeights(from modelPath: String) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(atPath: modelPath)
        let safetensorFiles = contents.filter { $0.hasSuffix(".safetensors") }.sorted()

        if safetensorFiles.isEmpty {
            throw Flux2WeightLoaderError.noWeightsFound(modelPath)
        }

        Flux2Debug.log("Found \(safetensorFiles.count) safetensor files in \(modelPath)")

        var allWeights: [String: MLXArray] = [:]

        for filename in safetensorFiles {
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))

            for (key, value) in weights {
                allWeights[key] = value
            }

            Flux2Debug.log("Loaded \(weights.count) tensors from \(filename)")
        }

        return allWeights
    }

    /// Load weights from URL
    public static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        try loadWeights(from: url.path)
    }

    // MARK: - Transformer Weight Mapping

    /// Detect if weights are in BFL (Black Forest Labs) native format
    private static func isBFLFormat(_ weights: [String: MLXArray]) -> Bool {
        // BFL format uses "double_blocks" and "single_blocks" directly
        // Diffusers format uses "transformer_blocks" and "single_transformer_blocks"
        return weights.keys.contains { $0.hasPrefix("double_blocks.") || $0.hasPrefix("single_blocks.") }
    }

    /// Convert HuggingFace transformer weight keys to Swift module paths
    /// Also handles dequantization of quanto qint8 weights
    /// Supports both Diffusers format and BFL native format
    public static func mapTransformerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        // Detect format
        if isBFLFormat(weights) {
            Flux2Debug.log("Detected BFL native format weights")
            return mapBFLTransformerWeights(weights)
        }

        Flux2Debug.log("Detected Diffusers format weights")
        return mapDiffusersTransformerWeights(weights)
    }

    /// Map weights in BFL (Black Forest Labs) native format
    /// BFL format uses fused QKV and different naming conventions
    private static func mapBFLTransformerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        // Debug: count double_blocks keys
        let doubleBlockKeys = weights.keys.filter { $0.hasPrefix("double_blocks.") }
        Flux2Debug.log("BFL format: found \(doubleBlockKeys.count) double_blocks keys")
        for key in doubleBlockKeys.prefix(5) {
            Flux2Debug.verbose("  double_block key: \(key)")
        }

        for (key, value) in weights {
            // BFL format uses fused QKV weights that need to be split
            // double_blocks.{i}.img_attn.qkv.weight -> split into toQ, toK, toV
            // double_blocks.{i}.txt_attn.qkv.weight -> split into addQProj, addKProj, addVProj

            if key.contains(".img_attn.qkv.weight") {
                Flux2Debug.verbose("Splitting img_attn QKV: \(key) shape=\(value.shape)")
                // Extract block index
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                // Split fused QKV into Q, K, V (shape: [3*dim, dim] -> 3x [dim, dim])
                let dim = value.shape[0] / 3
                let q = value[0..<dim, 0...]
                let k = value[dim..<(2*dim), 0...]
                let v = value[(2*dim)..., 0...]
                mapped["transformerBlocks.\(blockIdx).attn.toQ.weight"] = q
                mapped["transformerBlocks.\(blockIdx).attn.toK.weight"] = k
                mapped["transformerBlocks.\(blockIdx).attn.toV.weight"] = v
            } else if key.contains(".txt_attn.qkv.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                let dim = value.shape[0] / 3
                let q = value[0..<dim, 0...]
                let k = value[dim..<(2*dim), 0...]
                let v = value[(2*dim)..., 0...]
                mapped["transformerBlocks.\(blockIdx).attn.addQProj.weight"] = q
                mapped["transformerBlocks.\(blockIdx).attn.addKProj.weight"] = k
                mapped["transformerBlocks.\(blockIdx).attn.addVProj.weight"] = v
            } else if key.contains(".img_attn.proj.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).attn.toOut.weight"] = value
            } else if key.contains(".txt_attn.proj.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).attn.toAddOut.weight"] = value
            } else if key.contains(".img_attn.norm.query_norm.scale") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).attn.normQ.weight"] = value
            } else if key.contains(".img_attn.norm.key_norm.scale") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).attn.normK.weight"] = value
            } else if key.contains(".txt_attn.norm.query_norm.scale") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).attn.normAddedQ.weight"] = value
            } else if key.contains(".txt_attn.norm.key_norm.scale") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).attn.normAddedK.weight"] = value
            } else if key.contains(".img_mlp.0.weight") {
                // BFL: img_mlp.0 is the gated linear (produces 2x for SwiGLU)
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).ff.activation.proj.weight"] = value
            } else if key.contains(".img_mlp.2.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).ff.linearOut.weight"] = value
            } else if key.contains(".txt_mlp.0.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).ffContext.activation.proj.weight"] = value
            } else if key.contains(".txt_mlp.2.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "double_blocks.")
                mapped["transformerBlocks.\(blockIdx).ffContext.linearOut.weight"] = value
            } else if key.hasPrefix("single_blocks.") && key.contains(".linear1.weight") {
                // Single blocks have fused QKV+MLP
                let blockIdx = extractBlockIndex(from: key, prefix: "single_blocks.")
                mapped["singleTransformerBlocks.\(blockIdx).attn.toQkvMlp.weight"] = value
            } else if key.hasPrefix("single_blocks.") && key.contains(".linear2.weight") {
                let blockIdx = extractBlockIndex(from: key, prefix: "single_blocks.")
                mapped["singleTransformerBlocks.\(blockIdx).attn.toOut.weight"] = value
            } else if key.hasPrefix("single_blocks.") && key.contains(".norm.query_norm.scale") {
                let blockIdx = extractBlockIndex(from: key, prefix: "single_blocks.")
                mapped["singleTransformerBlocks.\(blockIdx).attn.normQ.weight"] = value
            } else if key.hasPrefix("single_blocks.") && key.contains(".norm.key_norm.scale") {
                let blockIdx = extractBlockIndex(from: key, prefix: "single_blocks.")
                mapped["singleTransformerBlocks.\(blockIdx).attn.normK.weight"] = value
            } else if key == "img_in.weight" {
                Flux2Debug.log("BFL: mapping img_in -> xEmbedder: \(value.shape)")
                mapped["xEmbedder.weight"] = value
            } else if key == "txt_in.weight" {
                Flux2Debug.log("BFL: mapping txt_in -> contextEmbedder: \(value.shape)")
                mapped["contextEmbedder.weight"] = value
            } else if key == "time_in.in_layer.weight" {
                Flux2Debug.log("BFL: mapping time_in.in_layer -> timestepEmbedder.linear1: \(value.shape)")
                mapped["timeGuidanceEmbed.timestepEmbedder.linear1.weight"] = value
            } else if key == "time_in.out_layer.weight" {
                Flux2Debug.log("BFL: mapping time_in.out_layer -> timestepEmbedder.linear2: \(value.shape)")
                mapped["timeGuidanceEmbed.timestepEmbedder.linear2.weight"] = value
            } else if key == "double_stream_modulation_img.lin.weight" {
                mapped["doubleStreamModulationImg.linear.weight"] = value
            } else if key == "double_stream_modulation_txt.lin.weight" {
                mapped["doubleStreamModulationTxt.linear.weight"] = value
            } else if key == "single_stream_modulation.lin.weight" {
                mapped["singleStreamModulation.linear.weight"] = value
            } else if key == "final_layer.adaLN_modulation.1.weight" {
                // IMPORTANT: BFL format vs Diffusers format weight layout difference
                //
                // The normOut layer (AdaLayerNormContinuous) projects conditioning to scale+shift:
                //   params = linear(silu(conditioning))  // [B, dim * 2]
                //   scale = params[0:dim]
                //   shift = params[dim:]
                //
                // BFL format stores the linear weight as [shift_weights | scale_weights]
                // Diffusers format stores it as [scale_weights | shift_weights]
                //
                // Without this swap, bf16 BFL models produce inverted scale/shift values,
                // causing ~10x higher output magnitude and posterized/noisy images.
                //
                // Fix: Swap the two halves of the weight matrix to match diffusers layout.
                let dim = value.shape[0] / 2  // 3072 for Klein 4B, 6144 for Dev
                let shiftRows = value[0..<dim, 0...]  // First half (shift in BFL)
                let scaleRows = value[dim..., 0...]   // Second half (scale in BFL)
                let swapped = concatenated([scaleRows, shiftRows], axis: 0)
                Flux2Debug.log("BFL: Swapped normOut.linear.weight halves (dim=\(dim)): \(value.shape) -> \(swapped.shape)")
                mapped["normOut.linear.weight"] = swapped
            } else if key == "final_layer.linear.weight" {
                mapped["projOut.weight"] = value
            } else {
                Flux2Debug.verbose("BFL format: unmapped key \(key)")
            }
        }

        Flux2Debug.log("Mapped \(mapped.count) BFL format weights")

        // Debug: print mapped keys by category
        let transformerBlockKeys = mapped.keys.filter { $0.hasPrefix("transformerBlocks.") }
        let singleBlockKeys = mapped.keys.filter { $0.hasPrefix("singleTransformerBlocks.") }
        Flux2Debug.log("  - transformerBlocks keys: \(transformerBlockKeys.count)")
        Flux2Debug.log("  - singleTransformerBlocks keys: \(singleBlockKeys.count)")

        Flux2Debug.verbose("transformerBlocks keys:")
        for key in transformerBlockKeys.sorted().prefix(10) {
            Flux2Debug.verbose("  - \(key): \(mapped[key]!.shape)")
        }

        // DEBUG: Print raw BFL weight statistics BEFORE conversion
        Flux2Debug.log("=== Raw BFL Weight Statistics (BEFORE conversion) ===")
        for key in ["xEmbedder.weight", "contextEmbedder.weight",
                    "transformerBlocks.0.attn.toQ.weight",
                    "transformerBlocks.0.ff.activation.proj.weight",
                    "singleTransformerBlocks.0.attn.toQkvMlp.weight"] {
            if let w = mapped[key] {
                eval(w)
                // Convert to float32 for accurate stats
                let wf32 = w.asType(.float32)
                let absVal = MLX.abs(wf32)
                let meanVal = mean(absVal).item(Float.self)
                let maxVal = MLX.max(absVal).item(Float.self)
                let minVal = MLX.min(absVal).item(Float.self)
                let hasNaN = any(isNaN(wf32)).item(Bool.self)
                let numInf = sum(MLX.abs(wf32) .> 1e30).item(Int.self)  // Count very large values as proxy for inf
                Flux2Debug.log("  \(key): dtype=\(w.dtype), shape=\(w.shape), mean=\(meanVal), max=\(maxVal), min=\(minVal), hasNaN=\(hasNaN), numInf=\(numInf)")
            }
        }

        // Convert bfloat16 → float32 → float16 (via float32 to avoid precision loss)
        // Direct bf16 → f16 can cause inf/nan due to exponent range difference
        var converted: [String: MLXArray] = [:]
        var convertedCount = 0
        for (key, value) in mapped {
            if value.dtype == .bfloat16 {
                // bf16 → f32 → f16 preserves values within f16's range
                converted[key] = value.asType(.float32).asType(.float16)
                convertedCount += 1
            } else {
                converted[key] = value
            }
        }
        Flux2Debug.log("Converted \(convertedCount) bfloat16 weights to float16 via float32")

        // DEBUG: Print weight statistics for key layers (AFTER processing)
        Flux2Debug.log("=== BFL Weight Statistics (AFTER processing) ===")
        for key in ["xEmbedder.weight", "contextEmbedder.weight",
                    "transformerBlocks.0.attn.toQ.weight",
                    "transformerBlocks.0.ff.activation.proj.weight",
                    "singleTransformerBlocks.0.attn.toQkvMlp.weight"] {
            if let w = converted[key] {
                eval(w)
                let wf32 = w.asType(.float32)
                let absVal = MLX.abs(wf32)
                let meanVal = mean(absVal).item(Float.self)
                let maxVal = MLX.max(absVal).item(Float.self)
                let minVal = MLX.min(absVal).item(Float.self)
                let hasNaN = any(isNaN(wf32)).item(Bool.self)
                let numInf = sum(MLX.abs(wf32) .> 1e30).item(Int.self)  // Count very large values as proxy for inf
                Flux2Debug.log("  \(key): dtype=\(w.dtype), shape=\(w.shape), mean=\(meanVal), max=\(maxVal), min=\(minVal), hasNaN=\(hasNaN), numInf=\(numInf)")
            } else {
                Flux2Debug.log("  \(key): NOT FOUND")
            }
        }

        return converted
    }

    /// Extract block index from key like "double_blocks.3.img_attn.qkv.weight"
    private static func extractBlockIndex(from key: String, prefix: String) -> Int {
        guard key.hasPrefix(prefix) else { return 0 }
        let afterPrefix = String(key.dropFirst(prefix.count))
        guard let dotIndex = afterPrefix.firstIndex(of: ".") else { return 0 }
        let indexStr = String(afterPrefix[..<dotIndex])
        return Int(indexStr) ?? 0
    }

    /// Map weights in Diffusers format (HuggingFace/quanto quantized)
    private static func mapDiffusersTransformerWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        var quantizedData: [String: MLXArray] = [:]  // base_key -> ._data
        var quantizedScales: [String: MLXArray] = [:]  // base_key -> ._scale

        // First pass: collect quantized weights and their scales
        for (key, value) in weights {
            var processedKey = key

            // Remove common prefixes
            if processedKey.hasPrefix("transformer.") {
                processedKey = String(processedKey.dropFirst(12))
            }

            // Check for quanto quantization suffixes
            if processedKey.hasSuffix("._data") {
                let baseKey = String(processedKey.dropLast(6))  // Remove "._data"
                let mappedKey = mapTransformerKeySimple(baseKey)
                quantizedData[mappedKey] = value
            } else if processedKey.hasSuffix("._scale") {
                let baseKey = String(processedKey.dropLast(7))  // Remove "._scale"
                let mappedKey = mapTransformerKeySimple(baseKey)
                quantizedScales[mappedKey] = value
            } else if processedKey.hasSuffix(".input_scale") || processedKey.hasSuffix(".output_scale") {
                // Skip activation scales for quanto
                continue
            } else {
                // Non-quantized weight, map directly
                let mappedKey = mapTransformerKeySimple(processedKey)
                mapped[mappedKey] = value
            }
        }

        // Second pass: dequantize qint8 weights to float16 for memory efficiency
        // Process in smaller batches to reduce peak memory
        var dequantCount = 0
        for (key, data) in quantizedData {
            if let scale = quantizedScales[key] {
                // Dequantize: weight = data.float() * scale, then convert to float16
                let dequantized = (data.asType(.float32) * scale).asType(.float16)
                mapped[key] = dequantized
                dequantCount += 1
            } else {
                // No scale found, use data as-is
                mapped[key] = data.asType(.float16)
                Flux2Debug.warning("No scale found for quantized weight: \(key)")
            }
            // Periodically clear unused memory
            if dequantCount % 50 == 0 {
                eval(mapped.values.map { $0 })
            }
        }
        Flux2Debug.log("Dequantized \(dequantCount) qint8 weights to float16")

        // DEBUG: Print weight statistics for key layers
        Flux2Debug.log("=== Diffusers Weight Statistics ===")
        for key in ["xEmbedder.weight", "contextEmbedder.weight",
                    "transformerBlocks.0.attn.toQ.weight",
                    "transformerBlocks.0.ff.activation.proj.weight",
                    "singleTransformerBlocks.0.attn.toQkvMlp.weight"].prefix(5) {
            if let w = mapped[key] {
                eval(w)
                let absVal = MLX.abs(w)
                let meanVal = mean(absVal).item(Float.self)
                let maxVal = MLX.max(absVal).item(Float.self)
                let minVal = MLX.min(absVal).item(Float.self)
                Flux2Debug.log("  \(key): shape=\(w.shape), mean=\(meanVal), max=\(maxVal), min=\(minVal)")
            } else {
                Flux2Debug.log("  \(key): NOT FOUND")
            }
        }

        return mapped
    }

    /// Map key without handling quantization suffixes (called after suffix processing)
    private static func mapTransformerKeySimple(_ key: String) -> String {
        var mapped = key

        // Map single stream blocks (checkpoint may use various formats)
        mapped = mapped.replacingOccurrences(of: "single_transformer_blocks.", with: "singleTransformerBlocks.")
        mapped = mapped.replacingOccurrences(of: "single_transformerBlocks.", with: "singleTransformerBlocks.")

        // Map double stream blocks (both formats)
        mapped = mapped.replacingOccurrences(of: "transformer_blocks.", with: "transformerBlocks.")

        // Map attention components - both underscore and camelCase from checkpoint
        mapped = mapped.replacingOccurrences(of: "attn.to_q.", with: "attn.toQ.")
        mapped = mapped.replacingOccurrences(of: "attn.to_k.", with: "attn.toK.")
        mapped = mapped.replacingOccurrences(of: "attn.to_v.", with: "attn.toV.")
        mapped = mapped.replacingOccurrences(of: "attn.toQ.", with: "attn.toQ.")
        mapped = mapped.replacingOccurrences(of: "attn.toK.", with: "attn.toK.")
        mapped = mapped.replacingOccurrences(of: "attn.toV.", with: "attn.toV.")
        mapped = mapped.replacingOccurrences(of: "attn.to_out.0.", with: "attn.toOut.")
        mapped = mapped.replacingOccurrences(of: "attn.toOut.0.", with: "attn.toOut.")
        mapped = mapped.replacingOccurrences(of: "attn.add_q_proj.", with: "attn.addQProj.")
        mapped = mapped.replacingOccurrences(of: "attn.add_k_proj.", with: "attn.addKProj.")
        mapped = mapped.replacingOccurrences(of: "attn.add_v_proj.", with: "attn.addVProj.")
        mapped = mapped.replacingOccurrences(of: "attn.addQProj.", with: "attn.addQProj.")
        mapped = mapped.replacingOccurrences(of: "attn.addKProj.", with: "attn.addKProj.")
        mapped = mapped.replacingOccurrences(of: "attn.addVProj.", with: "attn.addVProj.")
        mapped = mapped.replacingOccurrences(of: "attn.to_add_out.", with: "attn.toAddOut.")
        mapped = mapped.replacingOccurrences(of: "attn.toAddOut.", with: "attn.toAddOut.")

        // Map single block attention (fused QKV+MLP)
        // Model uses toQkvMlp, checkpoint uses to_qkv_mlp_proj
        mapped = mapped.replacingOccurrences(of: "attn.to_qkv_mlp_proj.", with: "attn.toQkvMlp.")

        // Map to_out without .0. (single stream blocks)
        mapped = mapped.replacingOccurrences(of: "attn.to_out.", with: "attn.toOut.")

        // Map norms - both formats
        mapped = mapped.replacingOccurrences(of: "attn.norm_q.", with: "attn.normQ.")
        mapped = mapped.replacingOccurrences(of: "attn.norm_k.", with: "attn.normK.")
        mapped = mapped.replacingOccurrences(of: "attn.normQ.", with: "attn.normQ.")
        mapped = mapped.replacingOccurrences(of: "attn.normK.", with: "attn.normK.")
        mapped = mapped.replacingOccurrences(of: "attn.norm_added_q.", with: "attn.normAddedQ.")
        mapped = mapped.replacingOccurrences(of: "attn.norm_added_k.", with: "attn.normAddedK.")
        mapped = mapped.replacingOccurrences(of: "attn.normAddedQ.", with: "attn.normAddedQ.")
        mapped = mapped.replacingOccurrences(of: "attn.normAddedK.", with: "attn.normAddedK.")
        mapped = mapped.replacingOccurrences(of: "norm1_context.", with: "norm1Context.")
        mapped = mapped.replacingOccurrences(of: "norm2_context.", with: "norm2Context.")

        // Map feedforward
        // Model uses SwiGLU with activation.proj for input, checkpoint uses linear_in
        mapped = mapped.replacingOccurrences(of: "ff_context.linear_in.", with: "ffContext.activation.proj.")
        mapped = mapped.replacingOccurrences(of: "ff_context.linear_out.", with: "ffContext.linearOut.")
        mapped = mapped.replacingOccurrences(of: ".ff.linear_in.", with: ".ff.activation.proj.")
        mapped = mapped.replacingOccurrences(of: ".ff.linear_out.", with: ".ff.linearOut.")
        // Generic fallbacks
        mapped = mapped.replacingOccurrences(of: "ff_context.", with: "ffContext.")
        mapped = mapped.replacingOccurrences(of: "linear_in.", with: "activation.proj.")
        mapped = mapped.replacingOccurrences(of: "linear_out.", with: "linearOut.")

        // Map embeddings
        mapped = mapped.replacingOccurrences(of: "x_embedder.", with: "xEmbedder.")
        mapped = mapped.replacingOccurrences(of: "context_embedder.", with: "contextEmbedder.")
        mapped = mapped.replacingOccurrences(of: "contextEmbedder.", with: "contextEmbedder.")
        mapped = mapped.replacingOccurrences(of: "time_text_embed.", with: "timeGuidanceEmbed.")
        mapped = mapped.replacingOccurrences(of: "time_guidance_embed.", with: "timeGuidanceEmbed.")

        // Map timestep embedder
        mapped = mapped.replacingOccurrences(of: "timestep_embedder.", with: "timestepEmbedder.")
        mapped = mapped.replacingOccurrences(of: "guidance_embedder.", with: "guidanceEmbedder.")
        mapped = mapped.replacingOccurrences(of: "linear_1.", with: "linear1.")
        mapped = mapped.replacingOccurrences(of: "linear_2.", with: "linear2.")

        // Map modulation layers
        mapped = mapped.replacingOccurrences(of: "double_stream_modulation_img.", with: "doubleStreamModulationImg.")
        mapped = mapped.replacingOccurrences(of: "double_stream_modulation_txt.", with: "doubleStreamModulationTxt.")
        mapped = mapped.replacingOccurrences(of: "single_stream_modulation.", with: "singleStreamModulation.")

        // Map output
        mapped = mapped.replacingOccurrences(of: "norm_out.", with: "normOut.")
        mapped = mapped.replacingOccurrences(of: "proj_out.", with: "projOut.")
        mapped = mapped.replacingOccurrences(of: "normOut.", with: "normOut.")
        mapped = mapped.replacingOccurrences(of: "projOut.", with: "projOut.")

        return mapped
    }

    // MARK: - VAE Weight Mapping

    /// Convert HuggingFace VAE weight keys to Swift module paths
    /// Also transposes Conv2d weights from PyTorch OIHW format to MLX OHWI format
    public static func mapVAEWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        for (key, var value) in weights {
            var newKey = key

            // Transpose Conv2d weights from PyTorch OIHW to MLX OHWI format
            // Conv2d weights have 4 dimensions and key ends with ".weight"
            // PyTorch: [out_channels, in_channels, kernel_h, kernel_w]
            // MLX: [out_channels, kernel_h, kernel_w, in_channels]
            if key.hasSuffix(".weight") && value.ndim == 4 {
                value = value.transposed(0, 2, 3, 1)  // OIHW -> OHWI
            }

            // Map encoder/decoder
            newKey = newKey.replacingOccurrences(of: "encoder.conv_in.", with: "encoder.convIn.")
            newKey = newKey.replacingOccurrences(of: "encoder.conv_out.", with: "encoder.convOut.")
            newKey = newKey.replacingOccurrences(of: "encoder.conv_norm_out.", with: "encoder.convNormOut.")
            newKey = newKey.replacingOccurrences(of: "decoder.conv_in.", with: "decoder.convIn.")
            newKey = newKey.replacingOccurrences(of: "decoder.conv_out.", with: "decoder.convOut.")
            newKey = newKey.replacingOccurrences(of: "decoder.conv_norm_out.", with: "decoder.convNormOut.")

            // Map down/up blocks
            newKey = newKey.replacingOccurrences(of: "down_blocks.", with: "downBlocks.")
            newKey = newKey.replacingOccurrences(of: "up_blocks.", with: "upBlocks.")

            // Map mid_block - special handling for tuple structure
            // mid_block.resnets.0 -> midBlock.0 (resnet1)
            // mid_block.attentions.0 -> midBlock.1 (attention)
            // mid_block.resnets.1 -> midBlock.2 (resnet2)
            // Apply to both encoder and decoder
            newKey = newKey.replacingOccurrences(of: "encoder.mid_block.resnets.0.", with: "encoder.midBlock.0.")
            newKey = newKey.replacingOccurrences(of: "encoder.mid_block.attentions.0.", with: "encoder.midBlock.1.")
            newKey = newKey.replacingOccurrences(of: "encoder.mid_block.resnets.1.", with: "encoder.midBlock.2.")
            newKey = newKey.replacingOccurrences(of: "decoder.mid_block.resnets.0.", with: "decoder.midBlock.0.")
            newKey = newKey.replacingOccurrences(of: "decoder.mid_block.attentions.0.", with: "decoder.midBlock.1.")
            newKey = newKey.replacingOccurrences(of: "decoder.mid_block.resnets.1.", with: "decoder.midBlock.2.")

            // Map ResNet blocks in down/up blocks
            // The model uses tuples: (blocks: [ResnetBlock2D], upsample/downsample: Upsample2D?)
            // This flattens as: upBlocks.{idx}.0.{resnet_idx} for blocks, upBlocks.{idx}.1 for upsample
            // Checkpoint uses: up_blocks.{idx}.resnets.{resnet_idx}
            newKey = newKey.replacingOccurrences(of: "resnets.", with: "0.")
            newKey = newKey.replacingOccurrences(of: "conv_shortcut.", with: "convShortcut.")

            // Map upsamplers and downsamplers
            // upsamplers.0.conv -> 1.conv (position 1 in tuple)
            // downsamplers.0.conv -> 1.conv (position 1 in tuple)
            newKey = newKey.replacingOccurrences(of: "upsamplers.0.conv.", with: "1.conv.")
            newKey = newKey.replacingOccurrences(of: "downsamplers.0.conv.", with: "1.conv.")

            // Map attention
            newKey = newKey.replacingOccurrences(of: "attentions.", with: "attentions.")
            newKey = newKey.replacingOccurrences(of: "group_norm.", with: "groupNorm.")
            newKey = newKey.replacingOccurrences(of: "to_q.", with: "toQ.")
            newKey = newKey.replacingOccurrences(of: "to_k.", with: "toK.")
            newKey = newKey.replacingOccurrences(of: "to_v.", with: "toV.")
            newKey = newKey.replacingOccurrences(of: "to_out.0.", with: "toOut.")  // VAE attention has indexed to_out
            newKey = newKey.replacingOccurrences(of: "to_out.", with: "toOut.")

            // Map quant conv - order matters! More specific patterns first
            newKey = newKey.replacingOccurrences(of: "post_quant_conv.", with: "postQuantConv.")
            newKey = newKey.replacingOccurrences(of: "quant_conv.", with: "quantConv.")

            // Map batch norm (bn -> latentBatchNorm)
            newKey = newKey.replacingOccurrences(of: "bn.", with: "latentBatchNorm.")
            newKey = newKey.replacingOccurrences(of: "running_mean", with: "runningMean")
            newKey = newKey.replacingOccurrences(of: "running_var", with: "runningVar")
            newKey = newKey.replacingOccurrences(of: "num_batches_tracked", with: "numBatchesTracked")

            mapped[newKey] = value
        }

        return mapped
    }

    // MARK: - Weight Application

    /// Apply weights to a transformer model
    public static func applyTransformerWeights(
        _ weights: [String: MLXArray],
        to model: Flux2Transformer2DModel
    ) throws {
        let mapped = mapTransformerWeights(weights)

        // Use MLX's built-in weight loading - flatten to get full paths
        let flattenedArray = model.parameters().flattened()
        var flatParameters: [String: MLXArray] = [:]
        for (key, value) in flattenedArray {
            flatParameters[key] = value
        }

        var updates: [String: MLXArray] = [:]
        var notFound = 0

        // Debug: print flattened model parameter keys by category
        let modelTransformerBlocks = flatParameters.keys.filter { $0.hasPrefix("transformerBlocks.") }
        let modelSingleBlocks = flatParameters.keys.filter { $0.hasPrefix("singleTransformerBlocks.") }
        let modelTimeEmbed = flatParameters.keys.filter { $0.contains("timeGuidanceEmbed") || $0.contains("xEmbedder") }
        Flux2Debug.verbose("Model parameter keys:")
        Flux2Debug.verbose("  - transformerBlocks params: \(modelTransformerBlocks.count)")
        Flux2Debug.verbose("  - singleTransformerBlocks params: \(modelSingleBlocks.count)")
        Flux2Debug.verbose("  - time/embedder params: \(modelTimeEmbed.count)")
        Flux2Debug.verbose("  - total params: \(flatParameters.count)")
        for key in modelTimeEmbed.sorted() {
            Flux2Debug.verbose("    \(key)")
        }

        for (key, value) in mapped {
            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 10 {
                    Flux2Debug.log("Warning: No parameter found for key: \(key)")
                }
            }
        }

        if notFound > 10 {
            Flux2Debug.log("... and \(notFound - 10) more missing parameters")
        }

        // Update model with new weights using the flattened format
        _ = model.update(parameters: ModuleParameters.unflattened(updates))

        // Debug: print some weight shapes
        for (key, value) in updates.sorted(by: { $0.key < $1.key }).prefix(10) {
            Flux2Debug.log("  Weight \(key): shape=\(value.shape), dtype=\(value.dtype)")
        }

        Flux2Debug.log("Applied \(updates.count) weights to transformer (\(notFound) not found)")
    }

    /// Apply weights to a VAE model
    public static func applyVAEWeights(
        _ weights: [String: MLXArray],
        to model: AutoencoderKLFlux2
    ) throws {
        let mapped = mapVAEWeights(weights)

        // Check for BatchNorm running stats
        if let runningMean = mapped["latentBatchNorm.runningMean"],
           let runningVar = mapped["latentBatchNorm.runningVar"] {
            model.loadBatchNormStats(runningMean: runningMean, runningVar: runningVar)
        }

        // Flatten model parameters for key matching
        let flattenedArray = model.parameters().flattened()
        var flatParameters: [String: MLXArray] = [:]
        for (key, value) in flattenedArray {
            flatParameters[key] = value
        }

        // Debug: print some VAE parameter keys
        Flux2Debug.verbose("VAE model parameter keys (first 20 of \(flatParameters.count) flattened):")
        for key in flatParameters.keys.sorted().prefix(20) {
            Flux2Debug.verbose("  - \(key)")
        }

        var updates: [String: MLXArray] = [:]
        var notFound = 0

        for (key, value) in mapped {
            if key.contains("running") { continue }  // Skip running stats, handled above

            if flatParameters.keys.contains(key) {
                updates[key] = value
            } else {
                notFound += 1
                if notFound <= 10 {
                    Flux2Debug.log("VAE Warning: No parameter found for key: \(key)")
                }
            }
        }

        if notFound > 10 {
            Flux2Debug.log("... and \(notFound - 10) more missing VAE parameters")
        }

        _ = model.update(parameters: ModuleParameters.unflattened(updates))

        Flux2Debug.log("Applied \(updates.count) weights to VAE (\(notFound) not found)")
    }

    // MARK: - Quantized Weight Loading

    /// Load quantized transformer weights
    public static func loadQuantizedTransformer(
        from modelPath: String,
        quantization: TransformerQuantization
    ) throws -> [String: MLXArray] {
        let weights = try loadWeights(from: modelPath)

        // For qint8, weights may already be quantized in safetensors
        // If not, we need to quantize them here
        if quantization != .bf16 {
            Flux2Debug.log("Loading \(quantization.rawValue) quantized weights")
        }

        return weights
    }

    // MARK: - Verification

    /// Verify that all required weights are present
    public static func verifyTransformerWeights(_ weights: [String: MLXArray]) -> (complete: Bool, missing: [String]) {
        let requiredPrefixes = [
            "xEmbedder",
            "contextEmbedder",
            "timeGuidanceEmbed",
            "transformerBlocks.0",
            "singleTransformerBlocks.0",
            "normOut",
            "projOut"
        ]

        var missing: [String] = []

        for prefix in requiredPrefixes {
            let hasPrefix = weights.keys.contains { $0.hasPrefix(prefix) }
            if !hasPrefix {
                missing.append(prefix)
            }
        }

        return (missing.isEmpty, missing)
    }

    // MARK: - LoRA Weight Merging

    /// Merge LoRA weights into a transformer model
    ///
    /// This computes `newWeight = originalWeight + scale * (loraB @ loraA)` for each LoRA pair
    /// and updates the model weights in place.
    ///
    /// - Parameters:
    ///   - loraManager: The LoRA manager containing loaded LoRA weights
    ///   - model: The transformer model to merge weights into
    public static func mergeLoRAWeights(
        from loraManager: LoRAManager,
        into model: Flux2Transformer2DModel
    ) {
        // Get flattened model parameters
        let flattenedArray = model.parameters().flattened()
        var flatParameters: [String: MLXArray] = [:]
        for (key, value) in flattenedArray {
            flatParameters[key] = value
        }

        var updates: [String: MLXArray] = [:]
        var mergedCount = 0
        var notFoundCount = 0

        Flux2Debug.log("[LoRA] Merging weights into transformer...")

        var dtypeLogged = false

        // Iterate through all LoRA pairs
        for layerPath in loraManager.loadedLayerPaths {
            let pairs = loraManager.getLoRAPairs(for: layerPath)
            guard !pairs.isEmpty else { continue }

            // The layer path needs ".weight" suffix to match model parameters
            let weightKey = layerPath + ".weight"

            guard let originalWeight = flatParameters[weightKey] else {
                notFoundCount += 1
                if notFoundCount <= 10 {
                    Flux2Debug.log("[LoRA] Warning: No weight found for layer: \(weightKey)")
                }
                continue
            }

            // Start with the original weight
            var mergedWeight = originalWeight

            // Apply all LoRA pairs for this layer
            for (scale, loraA, loraB) in pairs {
                // Log dtypes once for debugging
                if !dtypeLogged {
                    Flux2Debug.log("[LoRA] dtypes - original: \(originalWeight.dtype), loraA: \(loraA.dtype), loraB: \(loraB.dtype)")
                    dtypeLogged = true
                }

                // Compute LoRA delta: scale * (loraB @ loraA)
                // loraA: [rank, in_features], loraB: [out_features, rank]
                // Result: [out_features, in_features]
                //
                // IMPORTANT: Convert LoRA weights to same dtype as original weight
                // to avoid dtype mismatches that can cause color issues
                let loraAConverted = loraA.asType(originalWeight.dtype)
                let loraBConverted = loraB.asType(originalWeight.dtype)
                let loraDelta = scale * matmul(loraBConverted, loraAConverted)

                // Add to weight
                mergedWeight = mergedWeight + loraDelta
            }

            updates[weightKey] = mergedWeight
            mergedCount += 1
        }

        if notFoundCount > 10 {
            Flux2Debug.log("[LoRA] ... and \(notFoundCount - 10) more layers not found")
        }

        // Apply merged weights to model
        if !updates.isEmpty {
            _ = model.update(parameters: ModuleParameters.unflattened(updates))
            eval(model.parameters())
            Flux2Debug.log("[LoRA] Merged \(mergedCount) layers (\(notFoundCount) not found)")
        }
    }

    /// Get summary of loaded weights
    public static func summarizeWeights(_ weights: [String: MLXArray]) {
        var totalParams: Int64 = 0
        var byPrefix: [String: Int64] = [:]

        for (key, array) in weights {
            let params = Int64(array.shape.reduce(1, *))
            totalParams += params

            // Group by first component
            let prefix = String(key.split(separator: ".").first ?? Substring(key))
            byPrefix[prefix, default: 0] += params
        }

        Flux2Debug.log("Weight Summary:")
        for (prefix, params) in byPrefix.sorted(by: { $0.value > $1.value }) {
            let gb = Float(params * 2) / 1_000_000_000  // bf16 = 2 bytes
            Flux2Debug.log("  \(prefix): \(params) params (~\(String(format: "%.2f", gb))GB)")
        }
        Flux2Debug.log("Total: \(totalParams) parameters")
    }
}

// MARK: - Errors

public enum Flux2WeightLoaderError: LocalizedError {
    case noWeightsFound(String)
    case weightMismatch(String)
    case fileNotFound(String)
    case incompatibleShape(expected: [Int], got: [Int], key: String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound(let path):
            return "No safetensors files found in: \(path)"
        case .weightMismatch(let message):
            return "Weight mismatch: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .incompatibleShape(let expected, let got, let key):
            return "Incompatible shape for \(key): expected \(expected), got \(got)"
        }
    }
}
