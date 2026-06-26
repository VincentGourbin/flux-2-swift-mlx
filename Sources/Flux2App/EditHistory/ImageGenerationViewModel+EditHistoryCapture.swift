import CoreGraphics
import Flux2Core
import Foundation

extension ImageGenerationViewModel {
    func appendEditHistory(image: CGImage, kind: EditHistoryKind, label: String) {
        do {
            _ = try editHistoryStore.append(
                master: image,
                label: label,
                kind: kind,
                prompt: prompt,
                settings: captureCurrentHistorySettings(),
                spatial: captureCurrentHistorySpatial()
            )
            if editHistoryStore.lastPrunedStepCount > 0 {
                let count = editHistoryStore.lastPrunedStepCount
                statusMessage = "Oldest \(count) history step\(count == 1 ? "" : "s") dropped (max \(EditHistoryStore.maxEntryCount))."
            }
        } catch {
            errorMessage = "Failed to record edit history: \(error.localizedDescription)"
        }
    }

    func captureCurrentHistorySpatial() -> EditHistorySpatial {
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

    func captureCurrentHistorySettings() -> EditHistorySettings {
        EditHistorySettings(
            selectedModel: selectedModel.rawValue,
            steps: steps,
            guidance: guidance,
            megapixelBudget: megapixelBudget,
            upsamplePrompt: upsamplePrompt,
            generateRoute: generateRoute.rawValue
        )
    }

    func historyLabel(for kind: EditHistoryKind) -> String {
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
