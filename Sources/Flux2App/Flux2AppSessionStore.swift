/**
 * Flux2AppSessionStore.swift
 * Persists GUI parameters across app launches (UserDefaults-backed).
 */

import CoreGraphics
import Foundation
import Flux2Core

enum ImageInputSaveSource: String, CaseIterable, Identifiable {
    case raw = "Raw"
    case formatted = "Formatted"
    case prepared = "Prepared"

    var id: String { rawValue }

    var menuLabel: String { rawValue.lowercased() }
}

enum PreviewComparisonSide: String, CaseIterable, Identifiable {
    case formatted = "A"
    case processed = "B"

    var id: String { rawValue }
}

struct Flux2GenerationGUIState: Codable {
    var selectedFamily: String?
    var selectedModel: String
    var textQuantization: String
    var transformerQuantization: String
    var prompt: String
    var upsamplePrompt: Bool
    var clearPromptAfterGeneration: Bool
    var width: Int
    var height: Int
    var steps: Int
    var guidance: Float
    var seed: String
    var sizingFavor: String?
    var sizingMethod: String?
    var preparationScale: Double?
    var preparationOverlayOpacity: Double?
    var megapixelBudget: Double?
    var contextAreaX: Double?
    var contextAreaY: Double?
    var contextAreaWidth: Double?
    var contextAreaHeight: Double?
    var processAreaX: Double?
    var processAreaY: Double?
    var processAreaWidth: Double?
    var processAreaHeight: Double?
    var hasProcessArea: Bool
    var editMode: String?
    var inpaintMaskTool: String?
    var outpaintPadding: OutpaintPadding?
    var inpaintIntent: String?
    var enrichInpaintPromptWithVLM: Bool?
    var vlmContextManual: Bool = false
    var interpretImagePaths: [String]
    var showCheckpoints: Bool
    var checkpointInterval: Int
    var previewZoomScale: Double
    var previewComparisonSide: String
    var inputSaveSource: String
}

struct Flux2AppShellState: Codable {
    var selectedTab: Int
}

enum Flux2AppSessionStore {
    private static let shellKey = "flux2Session.shell"
    private static let i2iKey = "flux2Session.i2i"
    private static let t2iKey = "flux2Session.t2i"

    static func loadShell() -> Flux2AppShellState? {
        load(Flux2AppShellState.self, forKey: shellKey)
    }

    static func saveShell(selectedTab: Int) {
        save(Flux2AppShellState(selectedTab: selectedTab), forKey: shellKey)
    }

    static func loadImageToImage() -> Flux2GenerationGUIState? {
        load(Flux2GenerationGUIState.self, forKey: i2iKey)
    }

    static func saveImageToImage(_ state: Flux2GenerationGUIState) {
        save(state, forKey: i2iKey)
    }

    static func loadTextToImage() -> Flux2GenerationGUIState? {
        load(Flux2GenerationGUIState.self, forKey: t2iKey)
    }

    static func saveTextToImage(_ state: Flux2GenerationGUIState) {
        save(state, forKey: t2iKey)
    }

    private static func load<T: Decodable>(_ type: T.Type, forKey key: String) -> T? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private static func save<T: Encodable>(_ value: T, forKey key: String) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }
}

extension Notification.Name {
    static let flux2PersistSession = Notification.Name("flux2PersistSession")
}
