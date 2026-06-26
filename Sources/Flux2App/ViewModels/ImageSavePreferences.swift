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
        preset: ImageSaveOutputRootPresetStore.factoryPresetName,
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
    static let outputRootPresetsJSON = "imageSaveOutputRootPresetsJSON"

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
        ImageSaveOutputRootPresetStore.bootstrapIfNeeded()
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

/// Named output root preset shared by Settings and the Output palette.
struct ImageSaveOutputRootPreset: Codable, Equatable, Identifiable {
    var name: String
    var path: String

    var id: String { name }
}

/// User-defined output root presets shared by Settings and the Output palette preset picker.
enum ImageSaveOutputRootPresetStore {
    private static let legacyPresetNames: Set<String> = Set(ImageSavePreset.allCases.map(\.rawValue))

    static var factoryPresetName: String {
        nameFromPath(ImageSaveService.defaultOutputRoot)
    }

    static func decode(_ json: String) -> [ImageSaveOutputRootPreset] {
        guard let data = json.data(using: .utf8) else { return [] }

        if let presets = try? JSONDecoder().decode([ImageSaveOutputRootPreset].self, from: data) {
            return sanitize(presets)
        }

        if let paths = try? JSONDecoder().decode([String].self, from: data) {
            return sanitize(paths.map { path in
                ImageSaveOutputRootPreset(name: nameFromPath(path), path: normalize(path))
            })
        }

        return []
    }

    static func encode(_ presets: [ImageSaveOutputRootPreset]) -> String {
        let sanitized = sanitize(presets)
        guard let data = try? JSONEncoder().encode(sanitized),
              let json = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return json
    }

    static func readPresets(from defaults: UserDefaults = .standard) -> [ImageSaveOutputRootPreset] {
        decode(defaults.string(forKey: ImageSavePreferenceKeys.outputRootPresetsJSON) ?? "")
    }

    static func writePresets(_ presets: [ImageSaveOutputRootPreset], to defaults: UserDefaults = .standard) {
        defaults.set(encode(presets), forKey: ImageSavePreferenceKeys.outputRootPresetsJSON)
    }

    static func path(forPresetName name: String, from defaults: UserDefaults = .standard) -> String? {
        readPresets(from: defaults).first(where: { $0.name == name })?.path
    }

    static func normalize(_ path: String) -> String {
        (path.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).standardizingPath
    }

    static func bootstrapIfNeeded() {
        let defaults = UserDefaults.standard
        var presets = readPresets(from: defaults)
        let outputRoot = nonEmpty(
            defaults.string(forKey: "imageSaveOutputRoot"),
            fallback: ImageSaveService.defaultOutputRoot
        )
        let selected = defaults.string(forKey: ImageSavePreferenceKeys.preset) ?? ""

        if presets.isEmpty {
            var seed: [ImageSaveOutputRootPreset] = [
                ImageSaveOutputRootPreset(name: nameFromPath(outputRoot), path: normalize(outputRoot)),
                ImageSaveOutputRootPreset(name: factoryPresetName, path: normalize(ImageSaveService.defaultOutputRoot)),
            ]
            if legacyPresetNames.contains(selected),
               let legacy = ImageSavePreset(rawValue: selected) {
                let legacyPath = URL(fileURLWithPath: outputRoot, isDirectory: true)
                    .appendingPathComponent(legacy.relativeDirectory, isDirectory: true)
                    .path
                seed.insert(ImageSaveOutputRootPreset(name: legacy.rawValue, path: normalize(legacyPath)), at: 0)
            } else if !selected.isEmpty, !legacyPresetNames.contains(selected) {
                seed.insert(
                    ImageSaveOutputRootPreset(name: nameFromPath(selected), path: normalize(selected)),
                    at: 0
                )
            }
            presets = uniquePreservingOrder(seed)
            writePresets(presets, to: defaults)
        }

        if legacyPresetNames.contains(selected),
           let legacy = ImageSavePreset(rawValue: selected) {
            let legacyPath = URL(fileURLWithPath: outputRoot, isDirectory: true)
                .appendingPathComponent(legacy.relativeDirectory, isDirectory: true)
                .path
            let legacyName = legacy.rawValue
            if !presets.contains(where: { $0.name == legacyName }) {
                presets.append(ImageSaveOutputRootPreset(name: legacyName, path: normalize(legacyPath)))
                writePresets(presets, to: defaults)
            }
            defaults.set(legacyName, forKey: ImageSavePreferenceKeys.preset)
        } else if !presets.contains(where: { $0.name == selected }) {
            let fallback = presets.first?.name ?? factoryPresetName
            defaults.set(fallback, forKey: ImageSavePreferenceKeys.preset)
        }
    }

    @discardableResult
    static func addPreset(name: String, path: String, to defaults: UserDefaults = .standard) -> [ImageSaveOutputRootPreset] {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedPath = normalize(path)
        guard !trimmedName.isEmpty, !normalizedPath.isEmpty else { return readPresets(from: defaults) }

        var presets = readPresets(from: defaults)
        guard !presets.contains(where: { $0.name == trimmedName }) else { return presets }

        presets.append(ImageSaveOutputRootPreset(name: trimmedName, path: normalizedPath))
        writePresets(presets, to: defaults)
        defaults.set(trimmedName, forKey: ImageSavePreferenceKeys.preset)
        return presets
    }

    @discardableResult
    static func removePreset(named name: String, from defaults: UserDefaults = .standard) -> [ImageSaveOutputRootPreset] {
        var presets = readPresets(from: defaults)
        presets.removeAll { $0.name == name }
        writePresets(presets, to: defaults)

        let selected = defaults.string(forKey: ImageSavePreferenceKeys.preset) ?? ""
        if selected == name {
            defaults.set(presets.first?.name ?? factoryPresetName, forKey: ImageSavePreferenceKeys.preset)
        }
        return presets
    }

    private static func nameFromPath(_ path: String) -> String {
        let standardized = normalize(path)
        let last = (standardized as NSString).lastPathComponent
        return last.isEmpty ? standardized : last
    }

    private static func sanitize(_ presets: [ImageSaveOutputRootPreset]) -> [ImageSaveOutputRootPreset] {
        uniquePreservingOrder(
            presets.map {
                ImageSaveOutputRootPreset(
                    name: $0.name.trimmingCharacters(in: .whitespacesAndNewlines),
                    path: normalize($0.path)
                )
            }
            .filter { !$0.name.isEmpty && !$0.path.isEmpty }
        )
    }

    private static func uniquePreservingOrder(_ presets: [ImageSaveOutputRootPreset]) -> [ImageSaveOutputRootPreset] {
        var seen = Set<String>()
        var result: [ImageSaveOutputRootPreset] = []
        for preset in presets {
            guard seen.insert(preset.name).inserted else { continue }
            result.append(preset)
        }
        return result
    }

    private static func nonEmpty(_ value: String?, fallback: String) -> String {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return fallback
        }
        return value
    }
}
