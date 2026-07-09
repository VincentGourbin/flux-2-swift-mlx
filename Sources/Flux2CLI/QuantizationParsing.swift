// QuantizationParsing.swift - Shared CLI parsing for quantization options
// Copyright 2025 Vincent Gourbin

import ArgumentParser
import Flux2Core

extension TransformerQuantization {
    /// All accepted CLI values, derived from the enum so help/error text can't drift
    static var cliValueList: String {
        allCases.map(\.rawValue).joined(separator: ", ")
    }

    /// Parse a CLI value, accepting legacy aliases ("8bit" → qint8, "4bit" → int4)
    static func parseCLI(_ value: String) throws -> TransformerQuantization {
        switch value {
        case "8bit": return .qint8
        case "4bit": return .int4
        default:
            guard let parsed = TransformerQuantization(rawValue: value) else {
                throw ValidationError("Invalid transformer quantization: \(value). Use: \(cliValueList)")
            }
            return parsed
        }
    }

    /// Help text for --transformer-quant options
    static var cliHelp: String {
        "Transformer quantization: \(cliValueList)"
    }
}
