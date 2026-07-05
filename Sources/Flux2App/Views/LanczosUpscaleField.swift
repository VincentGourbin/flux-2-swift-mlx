/**
 * LanczosUpscaleField.swift
 * Compact "Lanczos Upscale" number box for the Generated Image row.
 */

import SwiftUI

/// "Lanczos Upscale" label + number box (1–8×, default 1×). The value is the
/// save-time scale factor: 1× saves at native size, > 1 upscales on save.
/// Backed by the shared `imageSaveUpscaleBy` setting.
struct LanczosUpscaleField: View {
    @Binding var factor: Double

    private static let range: ClosedRange<Double> = 1...8

    var body: some View {
        HStack(spacing: 4) {
            Text("Lanczos Upscale")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            TextField("", value: $factor, format: .number.precision(.fractionLength(0...2)))
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .frame(width: 46)
            Text("×")
                .foregroundStyle(.secondary)
            Stepper("", value: $factor, in: Self.range, step: 1)
                .labelsHidden()
        }
        .controlSize(.small)
        .help("Save-time upscale factor (1× = native size, up to 8×)")
        .onChange(of: factor) { _, newValue in
            // Keep the typed value in range; the guard stops the assignment from
            // re-triggering onChange forever.
            let clamped = min(max(newValue, Self.range.lowerBound), Self.range.upperBound)
            if clamped != newValue {
                factor = clamped
            }
        }
    }
}
