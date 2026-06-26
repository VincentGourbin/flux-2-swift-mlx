import CoreGraphics
import Flux2Core
import Foundation

// Public edit-history API for ImageGenerationViewModel.
// Restore/capture internals live in sibling extension files in this folder.

extension ImageGenerationViewModel {
    var historyEntries: [EditHistoryEntry] {
        editHistoryStore.entries
    }

    var currentHistoryIndex: Int? {
        editHistoryStore.currentIndex
    }

    var canStepHistoryBack: Bool {
        requiresReferenceImages && editHistoryStore.canStepBack
    }

    var canStepHistoryForward: Bool {
        requiresReferenceImages && editHistoryStore.canStepForward
    }

    func isCurrentHistoryEntry(at index: Int) -> Bool {
        editHistoryStore.currentIndex == index
    }

    func stepHistoryBack() {
        guard let index = editHistoryStore.currentIndex, index > 0 else { return }
        jumpToHistory(at: index - 1)
    }

    func stepHistoryForward() {
        guard let index = editHistoryStore.currentIndex, index + 1 < editHistoryStore.entries.count else { return }
        jumpToHistory(at: index + 1)
    }

    func jumpToHistory(at index: Int) {
        guard requiresReferenceImages else { return }
        guard editHistoryStore.entries.indices.contains(index) else { return }
        let entry = editHistoryStore.entries[index]
        do {
            let master = try editHistoryStore.masterImage(for: entry, bundleRoot: currentBundleRootURL)
            try restoreFromHistoryEntry(entry, master: master)
            editHistoryStore.select(index: index)
        } catch {
            errorMessage = "Failed to restore history: \(error.localizedDescription)"
        }
    }

    func noteFillContextMaskScaleEditBegan() {
        guard requiresReferenceImages, hasLocalFillSelection else { return }
        fillContextMaskScaleUndoBaseline = fillContextMaskScale
    }

    func commitFillContextMaskScaleEditIfChanged() {
        guard let baseline = fillContextMaskScaleUndoBaseline else { return }
        fillContextMaskScaleUndoBaseline = nil
        guard abs(fillContextMaskScale - baseline) > 0.0001 else { return }

        let current = fillContextMaskScale
        fillContextMaskScale = baseline
        recordSelectionUndoPoint()
        fillContextMaskScale = current
    }

    func appendEditHistoryAfterGenerate(image: CGImage) {
        guard requiresReferenceImages else { return }
        appendEditHistory(image: image, kind: .generate, label: historyLabel(for: .generate))
    }

    func appendEditHistoryAfterAdopt(image: CGImage) {
        guard requiresReferenceImages else { return }
        appendEditHistory(image: image, kind: .adopt, label: historyLabel(for: .adopt))
    }

    func maybeRecordImportHistory(cgImage: CGImage) {
        guard requiresReferenceImages, editHistoryStore.entries.isEmpty else { return }
        appendEditHistory(image: cgImage, kind: .import, label: historyLabel(for: .import))
    }

    func loadEditHistory(from project: FluxGenerationProject, bundleRoot: URL?) {
        editHistoryStore.load(from: project, bundleRoot: bundleRoot)
    }

    func applyLoadedHistoryPointer(bundleRoot: URL?) {
        guard requiresReferenceImages,
              let index = editHistoryStore.currentIndex,
              editHistoryStore.entries.indices.contains(index) else {
            return
        }

        let entry = editHistoryStore.entries[index]
        do {
            let root = bundleRoot ?? currentBundleRootURL
            let master = try editHistoryStore.masterImage(for: entry, bundleRoot: root)
            try restoreFromHistoryEntry(entry, master: master)
            editHistoryStore.select(index: index)
        } catch {
            statusMessage = "Project loaded; use History to restore step \(index + 1)."
        }
    }

    func clearEditHistory() {
        editHistoryStore.reset()
        if let cgImage = primaryReferenceImage {
            maybeRecordImportHistory(cgImage: cgImage)
        }
    }

    func replacePrimaryReference(with cgImage: CGImage, preservingSpatialWorkflow: Bool = false) {
        bootstrapImageSlotsIfNeeded()
        if let primaryID = primaryImageSlot?.id {
            loadImageIntoSlot(primaryID, cgImage: cgImage, preservingSpatialWorkflow: preservingSpatialWorkflow)
            return
        }
        if let firstID = imageSlots.first?.id {
            loadImageIntoSlot(firstID, cgImage: cgImage, preservingSpatialWorkflow: preservingSpatialWorkflow)
        }
    }

    func historyAssetsForSave() throws -> [FluxGenerationProjectBundle.HistoryAsset] {
        try editHistoryStore.historyAssets(bundleRoot: currentBundleRootURL)
    }

    func editHistoryManifestFields() -> (entries: [EditHistoryEntry]?, currentIndex: Int?) {
        let fields = editHistoryStore.manifestFields()
        guard !fields.entries.isEmpty else {
            return (nil, nil)
        }
        return (fields.entries, fields.currentIndex)
    }

    var currentBundleRootURL: URL? {
        guard let url = currentProjectURL else { return nil }
        if FluxGenerationProjectBundle.isBundleURL(url) {
            if url.pathExtension == FluxGenerationProjectBundle.packageExtension {
                return url
            }
            if url.lastPathComponent == FluxGenerationProjectBundle.manifestName {
                return url.deletingLastPathComponent()
            }
        }
        return nil
    }

    func editHistorySmokeSummary() -> String {
        let index = editHistoryStore.currentIndex.map(String.init) ?? "none"
        return "history_steps=\(editHistoryStore.entries.count)\nhistory_index=\(index)"
    }
}
