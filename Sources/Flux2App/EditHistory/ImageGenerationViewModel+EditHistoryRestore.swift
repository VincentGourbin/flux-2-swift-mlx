import CoreGraphics
import Flux2Core
import Flux2Chains
import Foundation

extension ImageGenerationViewModel {
    func restoreFromHistoryEntry(_ entry: EditHistoryEntry, master: CGImage) throws {
        clearSelectionUndoHistory()
        beginEditHistoryRestore()
        defer {
            Task { @MainActor in
                self.endEditHistoryRestore()
            }
        }

        applyHistorySettings(entry.settings)
        applySpatialFromHistory(entry.spatial)
        applyGenerateRouteFromHistory(settings: entry.settings, spatial: entry.spatial)
        prompt = entry.prompt
        replacePrimaryReference(with: master, preservingSpatialWorkflow: true)
        if outpaintPadding.hasExpansion {
            updateOutpaintPadding(outpaintPadding)
        }
        generatedImage = master
        previewComparisonSide = .processed
        cacheFormattedComparisonImage(for: master)
        applySizingControls()
        Task { await refreshVisionSubjectMaskCache() }
    }

    func applyHistorySettings(_ settings: EditHistorySettings) {
        skipNextModelDefaultApplication = true
        selectedModel = Flux2Model(rawValue: settings.selectedModel) ?? selectedModel
        steps = settings.steps
        guidance = settings.guidance
        if let budget = settings.megapixelBudget {
            megapixelBudget = min(max(budget, Self.minMegapixelBudget), Self.maxMegapixelBudget)
        }
        upsamplePrompt = settings.upsamplePrompt
    }

    func applySpatialFromHistory(_ spatial: EditHistorySpatial) {
        contextArea = Self.clampUnitRect(spatial.contextArea.cgRect)
        processArea = spatial.processArea?.cgRect
        inpaintMaskLayers = spatial.inpaintMaskLayers ?? []
        fillContextMaskScale = spatial.fillContextMaskScale ?? 0
        outpaintPadding = spatial.outpaintPadding ?? .zero
        outpaintCanvasIsDefined = outpaintPadding.hasExpansion
        if let intentRaw = spatial.inpaintIntent,
           let intent = Flux2InpaintIntent(rawValue: intentRaw) {
            inpaintIntent = intent
        } else {
            inpaintIntent = (spatial.inpaintMaskLayers ?? []).isEmpty ? .modify : .fill
        }
        enrichInpaintPromptWithVLM = spatial.enrichInpaintPromptWithVLM
            ?? !(spatial.inpaintMaskLayers ?? []).isEmpty
        draftPolygonPoints.removeAll()
        draftLassoPoints.removeAll()
        visionSubjectMasks.removeAll()
        visionSubjectStatusMessage = nil
        isDrawingSelection = false
    }

    func applyGenerateRouteFromHistory(settings: EditHistorySettings, spatial: EditHistorySpatial) {
        switch EditHistoryRouteResolver.route(settings: settings, spatial: spatial) {
        case .outpaint:
            inpaintMaskTool = .cropCanvas
        case .localFill, .fullImage:
            inpaintMaskTool = .pointer
        }
    }
}

enum EditHistoryRouteResolver {
    static func route(settings: EditHistorySettings, spatial: EditHistorySpatial) -> I2IGenerateRoute {
        if let raw = settings.generateRoute,
           let route = I2IGenerateRoute(rawValue: raw) {
            return route
        }
        if spatial.outpaintPadding?.hasExpansion == true {
            return .outpaint
        }
        if !(spatial.inpaintMaskLayers ?? []).isEmpty {
            return .localFill
        }
        return .fullImage
    }
}
