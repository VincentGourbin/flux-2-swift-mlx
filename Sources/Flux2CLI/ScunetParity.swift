// ScunetParity.swift — verify the Swift SCUNet (MLX) matches the Python lab reference.
// Loads the exported weights + a parity tensor file (NHWC float `input`/`output`) and
// reports PSNR of the Swift forward pass against the Python MLX output.

import Foundation
import ArgumentParser
import Flux2Core
import MLX

struct ScunetParity: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scunet-parity",
        abstract: "Verify Swift SCUNet (MLX) numerics vs the Python lab on a parity tensor file"
    )

    @Option(name: .long, help: "Weights .safetensors (MLX OHWI layout, keys match the module)")
    var weights: String

    @Option(name: .long, help: "Parity .safetensors with NHWC float `input` and `output`")
    var parity: String

    func run() throws {
        let model = SCUNetModel()
        let (matched, missing, unused) = try model.loadWeights(fromSafetensors: URL(fileURLWithPath: weights))
        print("Loaded \(matched) params  (missing \(missing.count), unused \(unused.count))")
        if !missing.isEmpty { print("  missing e.g.: \(Array(missing.prefix(8)))") }
        if !unused.isEmpty { print("  unused  e.g.: \(Array(unused.prefix(8)))") }

        let arrays = try loadArrays(url: URL(fileURLWithPath: parity))
        guard let input = arrays["input"], let expected = arrays["output"] else {
            print("Parity file must contain `input` and `output`")
            throw ExitCode.failure
        }

        let pred = clip(model(input.asType(.float32)), min: 0, max: 1)
        let exp = clip(expected.asType(.float32), min: 0, max: 1)
        eval(pred)
        let mse = mean(square(pred - exp)).item(Float.self)
        let maxabs = MLX.max(abs(pred - exp)).item(Float.self)
        let psnr = 10.0 * log10(1.0 / Double(Swift.max(mse, 1e-12)))
        print(String(format: "Swift-vs-PythonMLX  PSNR=%.2f dB  MSE=%.3e  maxabs=%.4f", psnr, mse, maxabs))
        if psnr < 50 {
            print("WARNING: PSNR below 50 dB — investigate before wiring the app path.")
        } else {
            print("OK: Swift SCUNet matches the Python MLX reference.")
        }
    }
}
