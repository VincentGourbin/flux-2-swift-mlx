/**
 * TextEncoderWeightLoader.swift
 * Utilities for loading text encoder model weights from safetensors files
 */

import Foundation
import MLX
import MLXNN

/// Load weights from safetensors files in a model directory
public class TextEncoderWeightLoader {

    /// Load all weights from a model directory
    public static func loadWeights(from modelPath: String) throws -> [String: MLXArray] {
        // Same predicate as the verifiers: `._*` AppleDouble sidecars and
        // dangling relocation symlinks are not loadable weights.
        let safetensorFiles = SafetensorsDirectory.reachableWeights(
            at: URL(fileURLWithPath: modelPath, isDirectory: true))

        if safetensorFiles.isEmpty {
            throw TextEncoderWeightLoaderError.noWeightsFound
        }

        FluxDebug.log("Found \(safetensorFiles.count) safetensor files")

        var allWeights: [String: MLXArray] = [:]

        for filename in safetensorFiles {
            let filePath = "\(modelPath)/\(filename)"
            let weights = try loadArrays(url: URL(fileURLWithPath: filePath))
            for (key, value) in weights {
                allWeights[key] = value
            }
            FluxDebug.log("Loaded \(weights.count) tensors from \(filename)")
        }

        return allWeights
    }

    /// Convert HuggingFace weight keys to Swift module paths
    public static func convertWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var converted: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Remove "model." prefix
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }

            converted[newKey] = value
        }

        return converted
    }

    /// Get summary of loaded weights
    public static func summarizeWeights(_ weights: [String: MLXArray]) {
        var totalParams: Int64 = 0

        for (key, array) in weights {
            let params = array.shape.reduce(1, *)
            totalParams += Int64(params)
            FluxDebug.log("  \(key): \(array.shape) (\(params) params)")
        }

        FluxDebug.log("Total parameters: \(totalParams)")
    }
}

/// Errors for weight loading
public enum TextEncoderWeightLoaderError: LocalizedError {
    case noWeightsFound
    case weightMismatch(String)
    case fileNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound:
            return "No safetensors files found in model directory"
        case .weightMismatch(let message):
            return "Weight mismatch: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        }
    }
}
