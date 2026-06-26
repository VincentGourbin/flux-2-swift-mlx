import CoreGraphics
import Flux2Core
import Foundation

#if canImport(AppKit)
import AppKit
#endif

@MainActor
final class EditHistoryStore: ObservableObject {
    @Published private(set) var entries: [EditHistoryEntry] = []
    @Published private(set) var currentIndex: Int?

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
    }

    func load(from project: FluxGenerationProject, bundleRoot: URL?) {
        reset()
        entries = project.history ?? []
        currentIndex = project.currentHistoryIndex
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

    func historyAssets() -> [FluxGenerationProjectBundle.HistoryAsset] {
        entries.compactMap { entry in
            guard let master = masterImages[entry.id],
                  let thumb = thumbImages[entry.id] else {
                return nil
            }
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

    private func sequenceNumber(from path: String) -> Int? {
        let stem = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        return Int(stem)
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
