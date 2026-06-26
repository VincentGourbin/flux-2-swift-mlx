/**
 * ImageSavePreferences.swift
 * UserDefaults keys and helpers for image save path / filename settings.
 */

import Foundation

/// Path + filename segment settings shared by the Output palette and Defaults dialog.
struct ImageSaveNamingValues: Equatable {
    var outputMode: String
    var preset: String
    var inputBase: String
    var filenamePrefix: String
    var freeText: String
    var useTimestamp: Bool
    var timestampFormat: String
    var useAutoIncrement: Bool
    var autoIncrementDigits: Int
    var autoIncrementStart: Int
    var autoIncrementStep: Int

    static let factory = ImageSaveNamingValues(
        outputMode: ImageSaveOutputMode.default.rawValue,
        preset: ImageSavePreset.peeps.rawValue,
        inputBase: ImageSaveInputBase.staticPrefix.rawValue,
        filenamePrefix: "image",
        freeText: "",
        useTimestamp: false,
        timestampFormat: ImageSaveTimestampFormat.compactDate.rawValue,
        useAutoIncrement: false,
        autoIncrementDigits: 5,
        autoIncrementStart: 1,
        autoIncrementStep: 1
    )
}

enum ImageSavePreferenceKeys {
    static let outputMode = "imageSaveOutputMode"
    static let preset = "imageSavePreset"
    static let inputBase = "imageSaveInputBase"
    static let filenamePrefix = "imageSaveFilenamePrefix"
    static let freeText = "imageSaveFreeText"
    static let useTimestamp = "imageSaveUseTimestamp"
    static let timestampFormat = "imageSaveTimestampFormat"
    static let useAutoIncrement = "imageSaveUseAutoIncrement"
    static let autoIncrementDigits = "imageSaveAutoIncrementDigits"
    static let autoIncrementStart = "imageSaveAutoIncrementStart"
    static let autoIncrementStep = "imageSaveAutoIncrementStep"

    static let defaultOutputMode = "imageSaveDefaultOutputMode"
    static let defaultPreset = "imageSaveDefaultPreset"
    static let defaultInputBase = "imageSaveDefaultInputBase"
    static let defaultFilenamePrefix = "imageSaveDefaultFilenamePrefix"
    static let defaultFreeText = "imageSaveDefaultFreeText"
    static let defaultUseTimestamp = "imageSaveDefaultUseTimestamp"
    static let defaultTimestampFormat = "imageSaveDefaultTimestampFormat"
    static let defaultUseAutoIncrement = "imageSaveDefaultUseAutoIncrement"
    static let defaultAutoIncrementDigits = "imageSaveDefaultAutoIncrementDigits"
    static let defaultAutoIncrementStart = "imageSaveDefaultAutoIncrementStart"
    static let defaultAutoIncrementStep = "imageSaveDefaultAutoIncrementStep"

    private static let defaultsSeededKey = "imageSaveDefaultKeysSeeded"

    static func readWorking(from defaults: UserDefaults = .standard) -> ImageSaveNamingValues {
        read(from: defaults, outputMode: outputMode, preset: preset, inputBase: inputBase,
             filenamePrefix: filenamePrefix, freeText: freeText, useTimestamp: useTimestamp,
             timestampFormat: timestampFormat, useAutoIncrement: useAutoIncrement,
             autoIncrementDigits: autoIncrementDigits, autoIncrementStart: autoIncrementStart,
             autoIncrementStep: autoIncrementStep)
    }

    static func readStoredDefaults(from defaults: UserDefaults = .standard) -> ImageSaveNamingValues {
        read(from: defaults, outputMode: defaultOutputMode, preset: defaultPreset, inputBase: defaultInputBase,
             filenamePrefix: defaultFilenamePrefix, freeText: defaultFreeText, useTimestamp: defaultUseTimestamp,
             timestampFormat: defaultTimestampFormat, useAutoIncrement: defaultUseAutoIncrement,
             autoIncrementDigits: defaultAutoIncrementDigits, autoIncrementStart: defaultAutoIncrementStart,
             autoIncrementStep: defaultAutoIncrementStep)
    }

    static func writeWorking(_ values: ImageSaveNamingValues, to defaults: UserDefaults = .standard) {
        write(values, to: defaults, outputMode: outputMode, preset: preset, inputBase: inputBase,
              filenamePrefix: filenamePrefix, freeText: freeText, useTimestamp: useTimestamp,
              timestampFormat: timestampFormat, useAutoIncrement: useAutoIncrement,
              autoIncrementDigits: autoIncrementDigits, autoIncrementStart: autoIncrementStart,
              autoIncrementStep: autoIncrementStep)
    }

    static func writeStoredDefaults(_ values: ImageSaveNamingValues, to defaults: UserDefaults = .standard) {
        write(values, to: defaults, outputMode: defaultOutputMode, preset: defaultPreset, inputBase: defaultInputBase,
              filenamePrefix: defaultFilenamePrefix, freeText: defaultFreeText, useTimestamp: defaultUseTimestamp,
              timestampFormat: defaultTimestampFormat, useAutoIncrement: defaultUseAutoIncrement,
              autoIncrementDigits: defaultAutoIncrementDigits, autoIncrementStart: defaultAutoIncrementStart,
              autoIncrementStep: defaultAutoIncrementStep)
    }

    /// Seed stored defaults once from factory values.
    static func bootstrapStoredDefaultsIfNeeded() {
        let defaults = UserDefaults.standard
        guard !defaults.bool(forKey: defaultsSeededKey) else { return }
        writeStoredDefaults(.factory, to: defaults)
        defaults.set(true, forKey: defaultsSeededKey)
    }

    /// Copy stored defaults into the working Output palette keys (e.g. New Project).
    static func applyStoredDefaultsToWorking() {
        let defaults = UserDefaults.standard
        bootstrapStoredDefaultsIfNeeded()
        writeWorking(readStoredDefaults(from: defaults), to: defaults)
    }

    private static func read(
        from defaults: UserDefaults,
        outputMode: String,
        preset: String,
        inputBase: String,
        filenamePrefix: String,
        freeText: String,
        useTimestamp: String,
        timestampFormat: String,
        useAutoIncrement: String,
        autoIncrementDigits: String,
        autoIncrementStart: String,
        autoIncrementStep: String
    ) -> ImageSaveNamingValues {
        ImageSaveNamingValues(
            outputMode: defaults.string(forKey: outputMode) ?? ImageSaveNamingValues.factory.outputMode,
            preset: defaults.string(forKey: preset) ?? ImageSaveNamingValues.factory.preset,
            inputBase: defaults.string(forKey: inputBase) ?? ImageSaveNamingValues.factory.inputBase,
            filenamePrefix: defaults.string(forKey: filenamePrefix) ?? ImageSaveNamingValues.factory.filenamePrefix,
            freeText: defaults.string(forKey: freeText) ?? ImageSaveNamingValues.factory.freeText,
            useTimestamp: defaults.object(forKey: useTimestamp) as? Bool ?? ImageSaveNamingValues.factory.useTimestamp,
            timestampFormat: defaults.string(forKey: timestampFormat) ?? ImageSaveNamingValues.factory.timestampFormat,
            useAutoIncrement: defaults.object(forKey: useAutoIncrement) as? Bool ?? ImageSaveNamingValues.factory.useAutoIncrement,
            autoIncrementDigits: defaults.object(forKey: autoIncrementDigits) as? Int ?? ImageSaveNamingValues.factory.autoIncrementDigits,
            autoIncrementStart: defaults.object(forKey: autoIncrementStart) as? Int ?? ImageSaveNamingValues.factory.autoIncrementStart,
            autoIncrementStep: defaults.object(forKey: autoIncrementStep) as? Int ?? ImageSaveNamingValues.factory.autoIncrementStep
        )
    }

    private static func write(
        _ values: ImageSaveNamingValues,
        to defaults: UserDefaults,
        outputMode: String,
        preset: String,
        inputBase: String,
        filenamePrefix: String,
        freeText: String,
        useTimestamp: String,
        timestampFormat: String,
        useAutoIncrement: String,
        autoIncrementDigits: String,
        autoIncrementStart: String,
        autoIncrementStep: String
    ) {
        defaults.set(values.outputMode, forKey: outputMode)
        defaults.set(values.preset, forKey: preset)
        defaults.set(values.inputBase, forKey: inputBase)
        defaults.set(values.filenamePrefix, forKey: filenamePrefix)
        defaults.set(values.freeText, forKey: freeText)
        defaults.set(values.useTimestamp, forKey: useTimestamp)
        defaults.set(values.timestampFormat, forKey: timestampFormat)
        defaults.set(values.useAutoIncrement, forKey: useAutoIncrement)
        defaults.set(values.autoIncrementDigits, forKey: autoIncrementDigits)
        defaults.set(values.autoIncrementStart, forKey: autoIncrementStart)
        defaults.set(values.autoIncrementStep, forKey: autoIncrementStep)
    }
}
