/**
 * ResetOnDoubleClickSlider.swift
 * Slider that restores a default value on double-click.
 */

import SwiftUI

struct ResetOnDoubleClickSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let defaultValue: Double

    var body: some View {
        Slider(value: $value, in: range, step: step)
            .simultaneousGesture(
                TapGesture(count: 2).onEnded {
                    value = defaultValue
                }
            )
    }
}

struct ResetOnDoubleClickIntSlider: View {
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int
    let defaultValue: Int

    var body: some View {
        ResetOnDoubleClickSlider(
            value: Binding(
                get: { Double(value) },
                set: { value = Int($0) }
            ),
            range: Double(range.lowerBound)...Double(range.upperBound),
            step: Double(step),
            defaultValue: Double(defaultValue)
        )
    }
}
