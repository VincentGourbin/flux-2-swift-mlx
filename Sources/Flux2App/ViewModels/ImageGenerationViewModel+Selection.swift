/**
 * ImageGenerationViewModel+Selection.swift
 * Generative-fill selection: mask-tool enablement, layer commits (rectangle /
 * polygon / lasso / vision subject), and session undo/redo. The stored backing
 * (`selectionUndoStore`, `isApplyingSelectionUndo`, `polygonSelectionCommitMode`)
 * lives on the main class; this extension owns the behavior.
 */

import CoreGraphics
import Flux2Chains
import Flux2Core
import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

extension ImageGenerationViewModel {
    func isMaskToolEnabled(_ tool: InpaintMaskTool) -> Bool {
        guard canUseCanvasTools else { return false }
        switch tool {
        case .liveArea:
            return barnDoorToolsApply
        case .rectangle, .polygon, .visionSubject, .cropCanvas:
            return true
        case .pointer:
            return false
        }
    }

    var hasActiveSelection: Bool {
        !inpaintMaskLayers.isEmpty
            || !draftPolygonPoints.isEmpty
            || !draftLassoPoints.isEmpty
            || processArea != nil
    }

    var hasFillMask: Bool {
        !inpaintMaskLayers.isEmpty
    }

    // MARK: - Undo / redo

    var canUndoSelection: Bool {
        workflow == .imageToImage && selectionUndoStore.canUndo
    }

    var canRedoSelection: Bool {
        workflow == .imageToImage && selectionUndoStore.canRedo
    }

    func undoSelection() {
        guard workflow == .imageToImage else { return }
        guard let snapshot = selectionUndoStore.popUndo(current: captureSelectionSnapshot()) else { return }
        restoreSelectionSnapshot(snapshot)
    }

    func redoSelection() {
        guard workflow == .imageToImage else { return }
        guard let snapshot = selectionUndoStore.popRedo(current: captureSelectionSnapshot()) else { return }
        restoreSelectionSnapshot(snapshot)
    }

    func clearSelectionUndoHistory() {
        selectionUndoStore.reset()
        objectWillChange.send()
    }

    private func captureSelectionSnapshot() -> SelectionUndoSnapshot {
        SelectionUndoSnapshot(
            inpaintMaskLayers: inpaintMaskLayers,
            processArea: processArea,
            draftPolygonPoints: draftPolygonPoints,
            draftLassoPoints: draftLassoPoints,
            fillContextMaskScale: fillContextMaskScale,
            inpaintIntent: inpaintIntent,
            enrichInpaintPromptWithVLM: enrichInpaintPromptWithVLM
        )
    }

    func recordSelectionUndoPoint() {
        guard workflow == .imageToImage, !isApplyingSelectionUndo else { return }
        selectionUndoStore.pushUndoPoint(captureSelectionSnapshot())
        objectWillChange.send()
    }

    private func restoreSelectionSnapshot(_ snapshot: SelectionUndoSnapshot) {
        isApplyingSelectionUndo = true
        defer { isApplyingSelectionUndo = false }

        inpaintMaskLayers = snapshot.inpaintMaskLayers
        processArea = snapshot.processArea
        draftPolygonPoints = snapshot.draftPolygonPoints
        draftLassoPoints = snapshot.draftLassoPoints
        fillContextMaskScale = snapshot.fillContextMaskScale
        inpaintIntent = snapshot.inpaintIntent
        enrichInpaintPromptWithVLM = snapshot.enrichInpaintPromptWithVLM
        visionSubjectMasks.removeAll()
        visionSubjectStatusMessage = nil
        isDrawingSelection = false
        objectWillChange.send()

        Task { await refreshVisionSubjectMaskCache() }
    }

    // MARK: - Tools

    func selectMaskTool(_ tool: InpaintMaskTool) {
        guard isMaskToolEnabled(tool) else { return }
        if tool == .cropCanvas {
            deselectSelections()
            resetOutpaintCanvas()
            cancelBarnDoorsIfActive()
        }
        inpaintMaskTool = tool
    }

    /// Drop back to pointer when the active toolbar tool is no longer usable.
    func clearActiveToolIfDisabled() {
        guard inpaintMaskTool != .pointer else { return }
        if !isMaskToolEnabled(inpaintMaskTool) {
            inpaintMaskTool = .pointer
        }
    }

    func deselectSelections(recordUndo: Bool = true) {
        if recordUndo, hasActiveSelection {
            recordSelectionUndoPoint()
        }
        inpaintMaskLayers.removeAll()
        draftPolygonPoints.removeAll()
        draftLassoPoints.removeAll()
        #if canImport(AppKit)
        polygonSelectionCommitMode = nil
        #endif
        visionSubjectMasks.removeAll()
        visionSubjectStatusMessage = nil
        processArea = nil
        isDrawingSelection = false
        fillContextMaskScale = 0
        inpaintMaskTool = .pointer
    }

    private func applyGenerativeFillDefaultsIfNeeded() {
        guard hasLocalFillSelection else { return }
        enrichInpaintPromptWithVLM = true
        inpaintIntent = .fill
        fillContextMaskScale = 0
    }

    #if canImport(AppKit)
    static func selectionModeFromModifierFlags(_ flags: NSEvent.ModifierFlags) -> SelectionMode {
        if flags.contains(.option) { return .subtract }
        if flags.contains(.shift) { return .add }
        return .replace
    }

    private static var currentSelectionCommitMode: SelectionMode {
        selectionModeFromModifierFlags(NSEvent.modifierFlags)
    }
    #else
    private static var currentSelectionCommitMode: SelectionMode { .replace }
    #endif

    // MARK: - Commits

    private func prepareSelectionCommit(mode: SelectionMode) {
        if mode == .replace {
            inpaintMaskLayers.removeAll()
            visionSubjectMasks.removeAll()
            visionSubjectStatusMessage = nil
        }
    }

    private func appendCommittedLayer(_ primitive: InpaintMaskPrimitive, mode: SelectionMode) {
        let hadCommittedSelection = !inpaintMaskLayers.isEmpty
        recordSelectionUndoPoint()
        prepareSelectionCommit(mode: mode)
        inpaintMaskLayers.append(
            InpaintMaskLayer(
                combineMode: mode.combineMode,
                primitive: primitive
            )
        )
        processArea = nil
        if !hadCommittedSelection {
            applyGenerativeFillDefaultsIfNeeded()
        }
    }

    func commitFillRectangle(_ rect: CGRect, mode: SelectionMode? = nil) {
        guard inpaintMaskTool != .visionSubject else { return }
        let resolvedMode = mode ?? Self.currentSelectionCommitMode
        let clamped = Self.clampUnitRect(rect)
        appendCommittedLayer(.rectangle(.init(clamped)), mode: resolvedMode)
    }

    func addDraftPolygonPoint(_ point: CGPoint) {
        #if canImport(AppKit)
        if draftPolygonPoints.isEmpty {
            polygonSelectionCommitMode = Self.currentSelectionCommitMode
        }
        #endif
        draftPolygonPoints.append(Self.clampUnitPoint(point))
    }

    func closeDraftPolygon(mode: SelectionMode? = nil) {
        guard draftPolygonPoints.count >= 3 else { return }
        let points = draftPolygonPoints.map { FluxGenerationProject.NormalizedPoint($0) }
        #if canImport(AppKit)
        let resolvedMode = mode ?? polygonSelectionCommitMode ?? Self.currentSelectionCommitMode
        polygonSelectionCommitMode = nil
        #else
        let resolvedMode = mode ?? Self.currentSelectionCommitMode
        #endif
        if inpaintMaskTool == .visionSubject {
            commitVisionSubject(selection: .polygon(points), mode: resolvedMode)
            return
        }
        appendCommittedLayer(.polygon(points), mode: resolvedMode)
        draftPolygonPoints.removeAll()
    }

    func commitLassoSelection(mode: SelectionMode? = nil) {
        guard inpaintMaskTool == .visionSubject, draftLassoPoints.count >= 3 else { return }
        let resolvedMode = mode ?? Self.currentSelectionCommitMode
        let points = draftLassoPoints.map { FluxGenerationProject.NormalizedPoint($0) }
        draftLassoPoints.removeAll()
        isDrawingSelection = false
        commitVisionSubject(selection: .polygon(points), mode: resolvedMode)
    }

    func appendLassoPoint(_ point: CGPoint) {
        draftLassoPoints.append(Self.clampUnitPoint(point))
    }

    // MARK: - Vision subject

    func commitVisionSubject(
        selection: VisionSubjectSelection,
        mode: SelectionMode? = nil
    ) {
        guard let image = primaryReferenceImage else { return }
        let mode = mode ?? Self.currentSelectionCommitMode
        Task {
            await resolveVisionSubjectLayer(selection: selection, image: image, selectionMode: mode)
        }
    }

    @MainActor
    private func resolveVisionSubjectLayer(
        selection: VisionSubjectSelection,
        image: CGImage,
        selectionMode: SelectionMode
    ) async {
        visionSubjectStatusMessage = "Finding subject…"
        let layerID = UUID()
        do {
            let mask = try await Task.detached(priority: .userInitiated) { [inpaintIntent] in
                try Self.resolveVisionSubjectMask(
                    selection: selection,
                    image: image,
                    inpaintIntent: inpaintIntent
                )
            }.value
            let hadCommittedSelection = !inpaintMaskLayers.isEmpty
            recordSelectionUndoPoint()
            if selectionMode == .replace {
                inpaintMaskLayers.removeAll()
                visionSubjectMasks.removeAll()
                visionSubjectStatusMessage = nil
            }
            inpaintMaskLayers.append(
                InpaintMaskLayer(
                    id: layerID,
                    combineMode: selectionMode.combineMode,
                    primitive: .visionSubject(selection)
                )
            )
            visionSubjectMasks[layerID] = mask
            processArea = nil
            draftPolygonPoints.removeAll()
            visionSubjectStatusMessage = nil
            if !hadCommittedSelection {
                applyGenerativeFillDefaultsIfNeeded()
            }
        } catch {
            processArea = nil
            draftPolygonPoints.removeAll()
            visionSubjectStatusMessage = "No subject in selection"
        }
    }

    nonisolated private static func resolveVisionSubjectMask(
        selection: VisionSubjectSelection,
        image: CGImage,
        inpaintIntent: Flux2InpaintIntent
    ) throws -> CGImage {
        if #available(macOS 14.0, *) {
            let region = selection.visionRegion
            if inpaintIntent == .changeScene {
                return try Flux2SubjectMask.pickChangeSceneMask(from: image, region: region)
            }
            return try Flux2SubjectMask.pickSubjectInpaintMask(from: image, region: region)
        }
        throw Flux2Error.invalidConfiguration("Apple Vision subject detection requires macOS 14 or later.")
    }

    @MainActor
    func refreshVisionSubjectMaskCache() async {
        guard let image = primaryReferenceImage else { return }
        for layer in inpaintMaskLayers {
            guard case .visionSubject(let selection) = layer.primitive else { continue }
            if let cached = visionSubjectMasks[layer.id],
               cached.width == image.width,
               cached.height == image.height {
                continue
            }
            do {
                let mask = try await Task.detached(priority: .userInitiated) { [inpaintIntent] in
                    try Self.resolveVisionSubjectMask(
                        selection: selection,
                        image: image,
                        inpaintIntent: inpaintIntent
                    )
                }.value
                visionSubjectMasks[layer.id] = mask
            } catch {
                continue
            }
        }
    }

    func clearFillMask() {
        deselectSelections()
    }

    func clearProcessSelection() {
        deselectSelections()
    }

    func buildGenerativeFillMask(for image: CGImage) throws -> CGImage {
        var visionMasks = visionSubjectMasks
        for layer in inpaintMaskLayers {
            guard case .visionSubject(let selection) = layer.primitive else { continue }
            if let cached = visionMasks[layer.id],
               cached.width == image.width,
               cached.height == image.height {
                continue
            }
            visionMasks[layer.id] = try Self.resolveVisionSubjectMask(
                selection: selection,
                image: image,
                inpaintIntent: inpaintIntent
            )
        }

        return try ImageMaskBuilder.buildInpaintMask(
            definition: InpaintMaskDefinition(layers: inpaintMaskLayers),
            image: image,
            legacyRectangle: processArea,
            visionMasks: visionMasks
        )
    }
}
