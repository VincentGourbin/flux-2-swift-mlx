// EditHistoryStore.swift
//
// In-memory linear document history for Image to Image.
//
// Agent invariants:
// - `entries` + `currentIndex` are persisted in the project manifest; JXL masters/thumbs live in the bundle.
// - On load, thumbs are cached; masters load lazily on jump/save via `masterImage(for:bundleRoot:)`.
// - On save, `historyAssets(bundleRoot:)` must resolve every manifest entry (memory or disk) before the bundle is replaced.
// - Append truncates the redo tail when `currentIndex` is not at the end; prune drops oldest at `maxEntryCount`.

import CoreGraphics
import Flux2Core
import Foundation

#if canImport(AppKit)
import AppKit
#endif

@MainActor
final class EditHistoryStore: ObservableObject {
    static let maxEntryCount = 30

    @Published private(set) var entries: [EditHistoryEntry] = []
    @Published private(set) var currentIndex: Int?
    /// Steps removed by the most recent `append` call (0 when nothing was pruned).
    private(set) var lastPrunedStepCount = 0

    private var masterImages: [UUID: CGImage] = [:]
    private var thumbImages: [UUID: CGImage] = [:]
    private var nextSequence = 1

    var canStepBack: Bool {
        guard let currentIndex else { return false }
        return currentIndex > 0
    }

    var canStepForward: Bool {
        guard let currentIndex else { return false }
        return currentIndex + 1 < entries.count
    }

    func reset() {
        entries.removeAll()
        currentIndex = nil
        masterImages.removeAll()
        thumbImages.removeAll()
        nextSequence = 1
        lastPrunedStepCount = 0
    }

    func load(from project: FluxGenerationProject, bundleRoot: URL?) {
        reset()
        entries = project.history ?? []
        if let index = project.currentHistoryIndex, entries.indices.contains(index) {
            currentIndex = index
        } else {
            currentIndex = nil
        }
        nextSequence = (entries.compactMap { sequenceNumber(from: $0.master) }.max() ?? 0) + 1

        guard let bundleRoot else { return }
        for entry in entries {
            let thumbURL = bundleRoot.appendingPathComponent(entry.thumb, isDirectory: false)
            if FileManager.default.fileExists(atPath: thumbURL.path),
               let thumb = try? ProjectBundleImageWriter.loadCGImage(from: thumbURL) {
                thumbImages[entry.id] = thumb
            }
        }
    }

    func append(
        master: CGImage,
        label: String,
        kind: EditHistoryKind,
        prompt: String,
        settings: EditHistorySettings,
        spatial: EditHistorySpatial
    ) throws -> EditHistoryEntry {
        lastPrunedStepCount = 0

        if let currentIndex, currentIndex + 1 < entries.count {
            let removed = entries[(currentIndex + 1)...]
            for entry in removed {
                masterImages.removeValue(forKey: entry.id)
                thumbImages.removeValue(forKey: entry.id)
            }
            entries.removeSubrange((currentIndex + 1)...)
        }

        let sequence = nextSequence
        nextSequence += 1
        let thumb = try ProjectBundleImageWriter.makeThumbnail(from: master)
        let entry = EditHistoryEntry(
            label: label,
            master: EditHistoryPaths.masterPath(sequence: sequence),
            thumb: EditHistoryPaths.thumbPath(sequence: sequence),
            kind: kind,
            prompt: prompt,
            settings: settings,
            spatial: spatial
        )
        entries.append(entry)
        currentIndex = entries.count - 1
        masterImages[entry.id] = master
        thumbImages[entry.id] = thumb
        lastPrunedStepCount = trimToMaxDepth()
        return entry
    }

    func masterImage(for entry: EditHistoryEntry, bundleRoot: URL?) throws -> CGImage {
        if let cached = masterImages[entry.id] {
            return cached
        }
        guard let bundleRoot else {
            throw Flux2Error.invalidConfiguration("History master image is not available.")
        }
        let url = bundleRoot.appendingPathComponent(entry.master, isDirectory: false)
        let image = try ProjectBundleImageWriter.loadCGImage(from: url)
        masterImages[entry.id] = image
        return image
    }

    func thumbImage(for entry: EditHistoryEntry) -> CGImage? {
        thumbImages[entry.id]
    }

    func historyAssets(bundleRoot: URL?) throws -> [FluxGenerationProjectBundle.HistoryAsset] {
        try entries.map { entry in
            let master = try resolvedMaster(for: entry, bundleRoot: bundleRoot)
            let thumb = try resolvedThumb(for: entry, bundleRoot: bundleRoot)
            return FluxGenerationProjectBundle.HistoryAsset(entry: entry, master: master, thumb: thumb)
        }
    }

    func select(index: Int) {
        guard entries.indices.contains(index) else { return }
        currentIndex = index
    }

    func manifestFields() -> (entries: [EditHistoryEntry], currentIndex: Int?) {
        (entries, currentIndex)
    }

    private func resolvedMaster(for entry: EditHistoryEntry, bundleRoot: URL?) throws -> CGImage {
        if let cached = masterImages[entry.id] {
            return cached
        }
        return try loadImage(relativePath: entry.master, bundleRoot: bundleRoot, label: "master")
    }

    private func resolvedThumb(for entry: EditHistoryEntry, bundleRoot: URL?) throws -> CGImage {
        if let cached = thumbImages[entry.id] {
            return cached
        }
        return try loadImage(relativePath: entry.thumb, bundleRoot: bundleRoot, label: "thumbnail")
    }

    private func loadImage(relativePath: String, bundleRoot: URL?, label: String) throws -> CGImage {
        guard let bundleRoot else {
            throw Flux2Error.invalidConfiguration(
                "History \(label) for “\(relativePath)” is not in memory and the project has not been saved to a bundle yet."
            )
        }
        let url = bundleRoot.appendingPathComponent(relativePath, isDirectory: false)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Flux2Error.invalidConfiguration(
                "History \(label) file is missing from the project bundle: \(relativePath)"
            )
        }
        return try ProjectBundleImageWriter.loadCGImage(from: url)
    }

    private func sequenceNumber(from path: String) -> Int? {
        let stem = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        return Int(stem)
    }

    @discardableResult
    private func trimToMaxDepth() -> Int {
        var pruned = 0
        while entries.count > Self.maxEntryCount {
            let removed = entries.removeFirst()
            masterImages.removeValue(forKey: removed.id)
            thumbImages.removeValue(forKey: removed.id)
            if let index = currentIndex {
                currentIndex = max(0, index - 1)
            }
            pruned += 1
        }
        return pruned
    }
}

#if canImport(AppKit)
extension EditHistoryStore {
    func thumbNSImage(for entry: EditHistoryEntry) -> NSImage? {
        guard let cgImage = thumbImages[entry.id] else { return nil }
        return NSImage(
            cgImage: cgImage,
            size: NSSize(width: cgImage.width, height: cgImage.height)
        )
    }
}
#endif
