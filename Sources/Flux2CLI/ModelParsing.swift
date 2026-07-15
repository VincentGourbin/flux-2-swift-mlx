// ModelParsing.swift - Shared CLI parsing for --flux-model options
// Copyright 2026 Vincent Gourbin
//
// Same pattern as QuantizationParsing.swift (PR #108): one parser derived
// from the enum so the accepted values and help text cannot drift between
// commands (inpaint / outpaint / export-quantized all take --flux-model).

import ArgumentParser
import Flux2Core

extension Flux2Model {
    /// All accepted CLI values, derived from the enum raw values.
    static var cliValueList: String {
        allCases.map(\.rawValue).joined(separator: " | ")
    }

    /// Parse a CLI value. Accepts the raw values plus their no-hyphen
    /// aliases ("klein9b" → "klein-9b", "klein9b-kv" → "klein-9b-kv", …).
    static func parseCLI(_ value: String) throws -> Flux2Model {
        let lowered = value.lowercased()
        if let parsed = Flux2Model(rawValue: lowered) {
            return parsed
        }
        // No-hyphen aliases: normalize "klein9b*" to "klein-9b*" etc.
        let normalized = lowered
            .replacingOccurrences(of: "klein9b", with: "klein-9b")
            .replacingOccurrences(of: "klein4b", with: "klein-4b")
        if let parsed = Flux2Model(rawValue: normalized) {
            return parsed
        }
        throw ValidationError("Unsupported --flux-model '\(value)'. Use: \(cliValueList)")
    }

    /// Help text for --flux-model options.
    static var cliHelp: String {
        "FLUX.2 model: \(cliValueList)"
    }
}
