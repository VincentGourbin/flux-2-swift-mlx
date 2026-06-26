import CoreGraphics
import Flux2Core
import Flux2Chains
import Foundation

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

    func beginFillContextMaskScaleEdit() {
        guard requiresReferenceImages, hasLocalFillSelection else { return }
        recordSelectionUndoPoint()
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

    func clearEditHistory() {
        editHistoryStore.reset()
    }

    func historyAssetsForSave() -> [FluxGenerationProjectBundle.HistoryAsset] {
        editHistoryStore.historyAssets()
    }

    func editHistoryManifestFields() -> (entries: [EditHistoryEntry]?, currentIndex: Int?) {
        let fields = editHistoryStore.manifestFields()
        guard !fields.entries.isEmpty else {
            return (nil, nil)
        }
        return (fields.entries, fields.currentIndex)
    }

    func replacePrimaryReference(with cgImage: CGImage) {
        bootstrapImageSlotsIfNeeded()
        if let primaryID = primaryImageSlot?.id {
            loadImageIntoSlot(primaryID, cgImage: cgImage)
            return
        }
        if let firstID = imageSlots.first?.id {
            loadImageIntoSlot(firstID, cgImage: cgImage)
        }
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

    private func appendEditHistory(image: CGImage, kind: EditHistoryKind, label: String) {
        do {
            _ = try editHistoryStore.append(
                master: image,
                label: label,
                kind: kind,
                prompt: prompt,
                settings: captureCurrentHistorySettings(),
                spatial: captureCurrentHistorySpatial()
            )
        } catch {
            errorMessage = "Failed to record edit history: \(error.localizedDescription)"
        }
    }

    private func captureCurrentHistorySpatial() -> EditHistorySpatial {
        EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(contextArea),
            processArea: processArea.map(FluxGenerationProject.NormalizedRect.init),
            inpaintMaskLayers: inpaintMaskLayers.isEmpty ? nil : inpaintMaskLayers,
            fillContextMaskScale: hasLocalFillSelection ? fillContextMaskScale : nil,
            outpaintPadding: outpaintPadding.hasExpansion ? outpaintPadding : nil,
            inpaintIntent: inpaintIntent.rawValue,
            enrichInpaintPromptWithVLM: enrichInpaintPromptWithVLM
        )
    }

    private func captureCurrentHistorySettings() -> EditHistorySettings {
        EditHistorySettings(
            selectedModel: selectedModel.rawValue,
            steps: steps,
            guidance: guidance,
            megapixelBudget: megapixelBudget,
            upsamplePrompt: upsamplePrompt,
            generateRoute: generateRoute.rawValue
        )
    }

    private func restoreFromHistoryEntry(_ entry: EditHistoryEntry, master: CGImage) throws {
        clearSelectionUndoHistory()
        applyHistorySettings(entry.settings)
        applySpatialFromHistory(entry.spatial)
        prompt = entry.prompt
        replacePrimaryReference(with: master)
        generatedImage = master
        previewComparisonSide = .processed
        cacheFormattedComparisonImage(for: master)
        applySizingControls()
        Task { await refreshVisionSubjectMaskCache() }
    }

    private func applyHistorySettings(_ settings: EditHistorySettings) {
        skipNextModelDefaultApplication = true
        selectedModel = Flux2Model(rawValue: settings.selectedModel) ?? selectedModel
        steps = settings.steps
        guidance = settings.guidance
        if let budget = settings.megapixelBudget {
            megapixelBudget = min(max(budget, Self.minMegapixelBudget), Self.maxMegapixelBudget)
        }
        upsamplePrompt = settings.upsamplePrompt
    }

    private func applySpatialFromHistory(_ spatial: EditHistorySpatial) {
        contextArea = Self.clampUnitRect(spatial.contextArea.cgRect)
        processArea = spatial.processArea?.cgRect
        inpaintMaskLayers = spatial.inpaintMaskLayers ?? []
        fillContextMaskScale = spatial.fillContextMaskScale ?? 0
        outpaintPadding = spatial.outpaintPadding ?? .zero
        outpaintCanvasIsDefined = outpaintPadding.hasExpansion
        if let intentRaw = spatial.inpaintIntent,
           let intent = Flux2InpaintIntent(rawValue: intentRaw) {
            inpaintIntent = intent
        }
        if let enrich = spatial.enrichInpaintPromptWithVLM {
            enrichInpaintPromptWithVLM = enrich
        }
        draftPolygonPoints.removeAll()
        draftLassoPoints.removeAll()
        visionSubjectMasks.removeAll()
        visionSubjectStatusMessage = nil
        isDrawingSelection = false
    }

    private func historyLabel(for kind: EditHistoryKind) -> String {
        switch kind {
        case .import:
            return "Import"
        case .generate, .adopt:
            switch generateRoute {
            case .fullImage: return "Prompt edit"
            case .localFill: return "Generative fill"
            case .outpaint: return "Outpaint"
            }
        }
    }
}
