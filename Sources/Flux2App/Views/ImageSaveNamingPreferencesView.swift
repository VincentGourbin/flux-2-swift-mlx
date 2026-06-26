/**
 * ImageSaveNamingPreferencesView.swift
 * Path + filename controls shared by the Output palette and Defaults dialog.
 */

import SwiftUI

struct ImageSaveNamingPreferencesView: View {
    @Binding var outputMode: String
    @Binding var preset: String
    @Binding var inputBase: String
    @Binding var filenamePrefix: String
    @Binding var freeText: String
    @Binding var useTimestamp: Bool
    @Binding var timestampFormat: String
    @Binding var useAutoIncrement: Bool
    @Binding var autoIncrementDigits: Int
    @Binding var autoIncrementStart: Int
    @Binding var autoIncrementStep: Int

    var previewPrompt: String = "sample prompt"
    var compact: Bool = false

    private var namingValues: ImageSaveNamingValues {
        ImageSaveNamingValues(
            outputMode: outputMode,
            preset: preset,
            inputBase: inputBase,
            filenamePrefix: filenamePrefix,
            freeText: freeText,
            useTimestamp: useTimestamp,
            timestampFormat: timestampFormat,
            useAutoIncrement: useAutoIncrement,
            autoIncrementDigits: autoIncrementDigits,
            autoIncrementStart: autoIncrementStart,
            autoIncrementStep: autoIncrementStep
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 12) {
            labeledRow("Path") {
                Picker("Path", selection: $outputMode) {
                    ForEach(ImageSaveOutputMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
            }

            labeledRow("Preset") {
                Picker("Preset", selection: $preset) {
                    ForEach(ImageSavePreset.allCases) { preset in
                        Text(preset.rawValue).tag(preset.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(outputMode != ImageSaveOutputMode.preset.rawValue)
            }

            Text("File Names")
                .font(compact ? .caption.bold() : .headline)
                .frame(maxWidth: .infinity)
                .padding(.top, compact ? 2 : 4)

            labeledRow("Base") {
                Picker("Base", selection: $inputBase) {
                    ForEach(ImageSaveInputBase.allCases) { inputBase in
                        Text(inputBase.rawValue).tag(inputBase.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
            }

            labeledRow("Static Prefix") {
                TextField("image", text: $filenamePrefix)
                    .textFieldStyle(.roundedBorder)
                    .disabled(inputBase != ImageSaveInputBase.staticPrefix.rawValue)
            }

            labeledRow("Free Text Segment") {
                TextField("", text: $freeText)
                    .textFieldStyle(.roundedBorder)
            }

            Toggle("Add timestamp", isOn: $useTimestamp)
                .font(compact ? .caption : .body)

            labeledRow("Timestamp") {
                Picker("Timestamp", selection: $timestampFormat) {
                    ForEach(ImageSaveTimestampFormat.allCases) { format in
                        Text(format.rawValue).tag(format.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(!useTimestamp)
            }

            Toggle("Add auto-increment", isOn: $useAutoIncrement)
                .font(compact ? .caption : .body)

            Stepper("Digits: \(autoIncrementDigits)", value: $autoIncrementDigits, in: 1...12)
                .font(compact ? .caption : .body)
                .disabled(!useAutoIncrement)
            Stepper("Start: \(autoIncrementStart)", value: $autoIncrementStart, in: 0...999_999)
                .font(compact ? .caption : .body)
                .disabled(!useAutoIncrement)
            Stepper("Step: \(autoIncrementStep)", value: $autoIncrementStep, in: 1...999)
                .font(compact ? .caption : .body)
                .disabled(!useAutoIncrement)

            Text("Preview: \(ImageSaveService.previewFilename(metadata: ImageSaveMetadata(prompt: previewPrompt), naming: namingValues))")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private func labeledRow<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        if compact {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                content()
            }
        } else {
            HStack(alignment: .center, spacing: 12) {
                Text(title)
                    .frame(width: 120, alignment: .trailing)
                content()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

/// Output palette bindings — working save settings used at save time.
struct ImageSaveWorkingNamingPreferencesView: View {
    @AppStorage(ImageSavePreferenceKeys.outputMode) private var outputMode = ImageSaveNamingValues.factory.outputMode
    @AppStorage(ImageSavePreferenceKeys.preset) private var preset = ImageSaveNamingValues.factory.preset
    @AppStorage(ImageSavePreferenceKeys.inputBase) private var inputBase = ImageSaveNamingValues.factory.inputBase
    @AppStorage(ImageSavePreferenceKeys.filenamePrefix) private var filenamePrefix = ImageSaveNamingValues.factory.filenamePrefix
    @AppStorage(ImageSavePreferenceKeys.freeText) private var freeText = ImageSaveNamingValues.factory.freeText
    @AppStorage(ImageSavePreferenceKeys.useTimestamp) private var useTimestamp = ImageSaveNamingValues.factory.useTimestamp
    @AppStorage(ImageSavePreferenceKeys.timestampFormat) private var timestampFormat = ImageSaveNamingValues.factory.timestampFormat
    @AppStorage(ImageSavePreferenceKeys.useAutoIncrement) private var useAutoIncrement = ImageSaveNamingValues.factory.useAutoIncrement
    @AppStorage(ImageSavePreferenceKeys.autoIncrementDigits) private var autoIncrementDigits = ImageSaveNamingValues.factory.autoIncrementDigits
    @AppStorage(ImageSavePreferenceKeys.autoIncrementStart) private var autoIncrementStart = ImageSaveNamingValues.factory.autoIncrementStart
    @AppStorage(ImageSavePreferenceKeys.autoIncrementStep) private var autoIncrementStep = ImageSaveNamingValues.factory.autoIncrementStep

    var previewPrompt: String = "sample prompt"

    var body: some View {
        ImageSaveNamingPreferencesView(
            outputMode: $outputMode,
            preset: $preset,
            inputBase: $inputBase,
            filenamePrefix: $filenamePrefix,
            freeText: $freeText,
            useTimestamp: $useTimestamp,
            timestampFormat: $timestampFormat,
            useAutoIncrement: $useAutoIncrement,
            autoIncrementDigits: $autoIncrementDigits,
            autoIncrementStart: $autoIncrementStart,
            autoIncrementStep: $autoIncrementStep,
            previewPrompt: previewPrompt,
            compact: true
        )
    }
}

/// Defaults dialog bindings — seeds the Output palette on New Project.
struct ImageSaveDefaultsNamingPreferencesView: View {
    @AppStorage(ImageSavePreferenceKeys.defaultOutputMode) private var outputMode = ImageSaveNamingValues.factory.outputMode
    @AppStorage(ImageSavePreferenceKeys.defaultPreset) private var preset = ImageSaveNamingValues.factory.preset
    @AppStorage(ImageSavePreferenceKeys.defaultInputBase) private var inputBase = ImageSaveNamingValues.factory.inputBase
    @AppStorage(ImageSavePreferenceKeys.defaultFilenamePrefix) private var filenamePrefix = ImageSaveNamingValues.factory.filenamePrefix
    @AppStorage(ImageSavePreferenceKeys.defaultFreeText) private var freeText = ImageSaveNamingValues.factory.freeText
    @AppStorage(ImageSavePreferenceKeys.defaultUseTimestamp) private var useTimestamp = ImageSaveNamingValues.factory.useTimestamp
    @AppStorage(ImageSavePreferenceKeys.defaultTimestampFormat) private var timestampFormat = ImageSaveNamingValues.factory.timestampFormat
    @AppStorage(ImageSavePreferenceKeys.defaultUseAutoIncrement) private var useAutoIncrement = ImageSaveNamingValues.factory.useAutoIncrement
    @AppStorage(ImageSavePreferenceKeys.defaultAutoIncrementDigits) private var autoIncrementDigits = ImageSaveNamingValues.factory.autoIncrementDigits
    @AppStorage(ImageSavePreferenceKeys.defaultAutoIncrementStart) private var autoIncrementStart = ImageSaveNamingValues.factory.autoIncrementStart
    @AppStorage(ImageSavePreferenceKeys.defaultAutoIncrementStep) private var autoIncrementStep = ImageSaveNamingValues.factory.autoIncrementStep

    var body: some View {
        ImageSaveDefaultsViewContent(
            outputMode: $outputMode,
            preset: $preset,
            inputBase: $inputBase,
            filenamePrefix: $filenamePrefix,
            freeText: $freeText,
            useTimestamp: $useTimestamp,
            timestampFormat: $timestampFormat,
            useAutoIncrement: $useAutoIncrement,
            autoIncrementDigits: $autoIncrementDigits,
            autoIncrementStart: $autoIncrementStart,
            autoIncrementStep: $autoIncrementStep
        )
    }
}

struct ImageSaveDefaultsViewContent: View {
    @Binding var outputMode: String
    @Binding var preset: String
    @Binding var inputBase: String
    @Binding var filenamePrefix: String
    @Binding var freeText: String
    @Binding var useTimestamp: Bool
    @Binding var timestampFormat: String
    @Binding var useAutoIncrement: Bool
    @Binding var autoIncrementDigits: Int
    @Binding var autoIncrementStart: Int
    @Binding var autoIncrementStep: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Default path and filename settings for new generation projects.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            ImageSaveNamingPreferencesView(
                outputMode: $outputMode,
                preset: $preset,
                inputBase: $inputBase,
                filenamePrefix: $filenamePrefix,
                freeText: $freeText,
                useTimestamp: $useTimestamp,
                timestampFormat: $timestampFormat,
                useAutoIncrement: $useAutoIncrement,
                autoIncrementDigits: $autoIncrementDigits,
                autoIncrementStart: $autoIncrementStart,
                autoIncrementStep: $autoIncrementStep,
                compact: false
            )
        }
        .padding(20)
        .frame(minWidth: 480)
        .onAppear {
            ImageSavePreferenceKeys.bootstrapStoredDefaultsIfNeeded()
        }
    }
}

struct ImageSaveDefaultsView: View {
    var body: some View {
        ImageSaveDefaultsNamingPreferencesView()
    }
}
