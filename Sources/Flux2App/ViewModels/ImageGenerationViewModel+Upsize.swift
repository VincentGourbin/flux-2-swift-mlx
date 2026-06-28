/**
 * ImageGenerationViewModel+Upsize.swift
 * The Upsize control: scale or generatively enlarge the primary reference to the
 * megapixel budget, then adopt the result as the new working image.
 *
 * Two paths, one button:
 *   - Bicubic / Lanczos — instant Core Image resample (`ImageScaler`).
 *   - FLUX.2 4B / 9B / Dev — a bounded I2I generation at the budget, driven by a
 *     faithful "reproduce, don't reinvent" prompt (editable in Settings), run on
 *     its own pipeline so the user's edit model / prompt are never disturbed.
 *
 * Both replace the primary reference (preserving the spatial workflow) and record
 * an undoable edit-history step. The FLUX path frees the edit pipeline first so
 * the enlarge model and the edit model are never co-resident.
 */

import CoreGraphics
import Flux2Core
import Foundation
import MLX

/// A single entry in the Upsize dropdown. Resamplers carry no model; the
/// generative entries map to a `Flux2Model` run at BF16.
enum UpsizeMethod: String, CaseIterable, Identifiable, Hashable {
    case bicubic
    case lanczos
    case flux4B
    case flux9B
    case fluxDev

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .bicubic: return "Bicubic"
        case .lanczos: return "Lanczos"
        case .flux4B: return "FLUX.2 4B"
        case .flux9B: return "FLUX.2 9B"
        case .fluxDev: return "FLUX.2 Dev"
        }
    }

    /// The FLUX model this entry enlarges with, or `nil` for a plain resampler.
    var fluxModel: Flux2Model? {
        switch self {
        case .bicubic, .lanczos: return nil
        case .flux4B: return .klein4B
        case .flux9B: return .klein9B
        case .fluxDev: return .dev
        }
    }

    var isGenerative: Bool { fluxModel != nil }
}

extension ImageGenerationViewModel {
    // MARK: - Faithful upscale prompt (Settings-backed)

    static let upsizeFaithfulPromptKey = "upsizeFaithfulPrompt"

    /// Default faithful-upscale instruction. FLUX.2 is instruction-tuned, so this
    /// reads as an edit directive: add resolution, change nothing.
    static let defaultFaithfulUpscalePrompt =
        "Reproduce this image exactly — same composition, subjects, colors, and detail — "
        + "at higher resolution with sharper, finer detail. Add nothing and change nothing."

    /// The faithful prompt the user configured, falling back to the default when
    /// unset or blank.
    static var resolvedFaithfulUpscalePrompt: String {
        let stored = UserDefaults.standard.string(forKey: upsizeFaithfulPromptKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let stored, !stored.isEmpty { return stored }
        return defaultFaithfulUpscalePrompt
    }

    // MARK: - State

    /// True when the primary reference already meets or exceeds the budget, so a
    /// generative enlarge would have nothing to invent (only resampling applies).
    var isUpsizeDownscaling: Bool {
        guard let image = primaryReferenceImage else { return false }
        return image.width * image.height >= conditioningPixelBudget
    }

    /// Whether the Apply button can fire for the current method + image.
    var canApplyUpsize: Bool {
        guard hasPrimaryReference, !isPipelineBusy else { return false }
        if upsizeMethod.isGenerative, isUpsizeDownscaling { return false }
        return true
    }

    // MARK: - Apply

    func performUpsize() {
        guard canApplyUpsize else { return }
        if let model = upsizeMethod.fluxModel {
            startFluxUpsize(model)
        } else {
            applyResampleUpsize()
        }
    }

    /// Full-frame, favour-original target that fills the megapixel budget — the
    /// same size the model would generate at, snapped to the family alignment.
    private func upsizeTargetSize(for image: CGImage) -> PixelSize {
        var settings = ImagePreparationSettings()
        settings.sizingFavor = .original
        settings.sizingMethod = .crop
        settings.megapixelBudget = megapixelBudget
        settings.pixelAlignment = pixelAlignment
        settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        settings.clampValues()
        return ImagePreparation.generationSize(referenceImage: image, settings: settings)
    }

    // MARK: - Resample path (instant)

    private func applyResampleUpsize() {
        guard let image = primaryReferenceImage else { return }
        let target = upsizeTargetSize(for: image)
        do {
            let resized = upsizeMethod == .bicubic
                ? try ImageScaler.bicubic(image, to: target)
                : try ImageScaler.lanczos(image, to: target)
            replacePrimaryReference(with: resized, preservingSpatialWorkflow: true)
            appendEditHistory(image: resized, kind: .adopt, label: "Upsize (\(upsizeMethod.displayName))")
            statusMessage = "Upsized to \(target.width)×\(target.height) — \(upsizeMethod.displayName)"
        } catch {
            errorMessage = "Upsize failed: \(error.localizedDescription)"
        }
    }

    // MARK: - FLUX enlarge path (a bounded generation)

    private func startFluxUpsize(_ model: Flux2Model) {
        guard canApplyUpsize, let image = primaryReferenceImage else { return }
        isGenerating = true
        isPipelineBusy = true
        currentStep = 0
        totalSteps = model.defaultSteps
        statusMessage = "Preparing upsize…"
        let label = "Upsize (\(upsizeMethod.displayName))"
        generationTask = Task { [weak self] in
            await self?.runFluxUpsize(model: model, source: image, label: label)
        }
    }

    private func runFluxUpsize(model: Flux2Model, source: CGImage, label: String) async {
        defer {
            isGenerating = false
            isPipelineBusy = false
        }

        // Free the edit pipeline so the enlarge model and the edit model are never
        // resident at the same time (sequential by design).
        if pipeline != nil {
            await clearPipeline()
        }

        let prompt = Self.resolvedFaithfulUpscalePrompt
        let quantConfig = Flux2QuantizationConfig(textEncoder: textQuantization, transformer: .bf16)
        let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? UserDefaults.standard.string(forKey: "hfToken")
        let upscalePipeline = Flux2Pipeline(model: model, quantization: quantConfig, hfToken: hfToken)

        do {
            statusMessage = "Loading \(model.displayName)…"
            try await upscalePipeline.loadModels { _, message in
                Task { @MainActor in
                    guard self.isGenerating else { return }
                    self.statusMessage = message
                }
            }

            try Task.checkCancellation()

            // Full-frame prep at budget. `prepare()` renders the reference at native
            // scale and lets the model enlarge generatively; full-frame means no
            // composite-back, so the output is the budget-sized enlargement.
            var settings = ImagePreparationSettings()
            settings.sizingFavor = .original
            settings.sizingMethod = .crop
            settings.megapixelBudget = megapixelBudget
            settings.pixelAlignment = pixelAlignment
            settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
            settings.compositeBack = false
            settings.clampValues()
            let prepared = try ImagePreparation.prepare(referenceImages: [source], settings: settings)

            statusMessage = "Upsizing with \(model.displayName)…"
            let result = try await upscalePipeline.generateImageToImageWithResult(
                prompt: prompt,
                images: prepared.images,
                interpretImagePaths: nil,
                height: prepared.height,
                width: prepared.width,
                steps: model.defaultSteps,
                guidance: model.defaultGuidance,
                seed: nil,
                upsamplePrompt: false,
                checkpointInterval: nil,
                onProgress: { current, total in
                    Task { @MainActor in
                        guard self.isGenerating else { return }
                        self.currentStep = current
                        self.totalSteps = total
                        self.statusMessage = "Step \(current)/\(total)"
                    }
                },
                onCheckpoint: nil,
                onPromptUpsampled: nil
            )

            if isGenerating {
                replacePrimaryReference(with: result.image, preservingSpatialWorkflow: true)
                appendEditHistory(image: result.image, kind: .adopt, label: label)
                statusMessage = "Upsized with \(model.displayName) — \(result.image.width)×\(result.image.height)"
            }
        } catch is CancellationError {
            if isGenerating { statusMessage = "Cancelled" }
        } catch Flux2Error.generationCancelled {
            if isGenerating { statusMessage = "Cancelled" }
        } catch {
            if isGenerating {
                errorMessage = error.localizedDescription
                statusMessage = "Upsize failed"
            }
        }

        await upscalePipeline.clearAll()
        Memory.clearCache()
    }
}
