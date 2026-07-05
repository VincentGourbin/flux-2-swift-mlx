/**
 * ImageGenerationViewModel+Projects.swift
 * Project lifecycle: new / open / save (bundle + legacy), startup-project and
 * last-project restore, and the project<->view-model marshalling helpers.
 */

import AppKit
import CoreGraphics
import Flux2Chains
import Flux2Core
import ImageIO
import SwiftUI

extension ImageGenerationViewModel {
    // MARK: - Projects

    func newProject() {
        selectedFamily = .flux2
        selectedModel = .klein4B
        textQuantization = .mlx8bit
        transformerQuantization = .qint8
        prompt = ""
        upsamplePrompt = false
        clearPromptAfterGeneration = false
        upsampledPrompt = nil
        width = 1024
        height = 1024
        seed = ""
        megapixelBudget = 1.0
        preparationOverlayOpacity = 0.22
        processArea = nil
        contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        clearAllImageSlots()
        generatedImage = nil
        checkpointImages.removeAll()
        errorMessage = nil
        currentProjectURL = nil
        lastSavedImageURL = nil
        clearEditHistory()
        UserDefaults.standard.removeObject(forKey: Self.lastProjectURLKey)
        applyRecommendedDefaults(for: selectedModel)
        ImageSavePreferenceKeys.applyStoredDefaultsToWorking()
        statusMessage = "New project"
    }

    func saveProject() {
        do {
            let url: URL
            if let currentProjectURL, !Flux2ProjectDocument.isLegacyJSONProjectURL(currentProjectURL) {
                url = currentProjectURL
            } else {
                let panel = NSSavePanel()
                panel.allowedContentTypes = Flux2ProjectDocument.saveContentTypes
                panel.nameFieldStringValue = currentProjectURL.map {
                    Flux2ProjectDocument.normalizedBundleURL(from: $0).lastPathComponent
                } ?? Flux2ProjectDocument.defaultSaveName
                guard panel.runModal() == .OK, let selectedURL = panel.url else {
                    return
                }
                url = Flux2ProjectDocument.normalizedBundleURL(from: selectedURL)
            }

            try saveProject(to: url)
            currentProjectURL = url
            UserDefaults.standard.set(url.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Saved project to \(url.lastPathComponent)"
        } catch {
            errorMessage = "Failed to save project: \(error.localizedDescription)"
        }
    }

    func saveProjectAs() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = Flux2ProjectDocument.saveContentTypes
        let suggested = currentProjectURL.map { Flux2ProjectDocument.normalizedBundleURL(from: $0).lastPathComponent }
            ?? Flux2ProjectDocument.defaultSaveName
        panel.nameFieldStringValue = suggested

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        do {
            try saveProject(to: Flux2ProjectDocument.normalizedBundleURL(from: url))
            currentProjectURL = Flux2ProjectDocument.normalizedBundleURL(from: url)
            UserDefaults.standard.set(currentProjectURL?.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Saved project to \(currentProjectURL?.lastPathComponent ?? url.lastPathComponent)"
        } catch {
            errorMessage = "Failed to save project: \(error.localizedDescription)"
        }
    }

    func openProject() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = Flux2ProjectDocument.allowedOpenContentTypes
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.treatsFilePackagesAsDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        do {
            try loadProject(from: url)
            currentProjectURL = resolvedProjectURL(from: url)
            UserDefaults.standard.set(currentProjectURL?.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Opened project \(currentProjectURL?.lastPathComponent ?? url.lastPathComponent)"
        } catch {
            errorMessage = "Failed to open project: \(error.localizedDescription)"
        }
    }

    func loadStartupProjectIfAvailable() {
        if let envPath = ProcessInfo.processInfo.environment[Self.projectEnvironmentKey],
           !envPath.isEmpty {
            loadProjectFromEnvironment(path: envPath)
            return
        }
        loadLastProjectIfAvailable()
    }

    private func loadProjectFromEnvironment(path: String) {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            let message = "F2SM_PROJECT file not found: \(path)"
            errorMessage = message
            writeSmokeMarker(outcome: "error", detail: message)
            return
        }

        do {
            try loadProject(from: url)
            currentProjectURL = resolvedProjectURL(from: url)
            UserDefaults.standard.set(currentProjectURL?.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Opened project \(url.lastPathComponent) (F2SM_PROJECT)"
            writeSmokeMarker(
                outcome: "ok",
                detail: """
                project=\(url.path)
                references=\(assignedReferenceCount)
                prompt=\(prompt)
                \(editHistorySmokeSummary())
                """
            )
        } catch {
            let message = "Failed to open F2SM_PROJECT: \(error.localizedDescription)"
            errorMessage = message
            currentProjectURL = nil
            writeSmokeMarker(outcome: "error", detail: message)
        }
    }

    private func writeSmokeMarker(outcome: String, detail: String) {
        guard let markerPath = ProcessInfo.processInfo.environment[Self.smokeMarkerEnvironmentKey],
              !markerPath.isEmpty else {
            return
        }

        let body = "\(outcome)\n\(detail)\n"
        do {
            try body.write(toFile: markerPath, atomically: true, encoding: .utf8)
        } catch {
            NSLog("F2SM_SMOKE_MARKER write failed: \(error.localizedDescription)")
        }
    }

    func loadLastProjectIfAvailable() {
        guard let path = UserDefaults.standard.string(forKey: Self.lastProjectURLKey),
              !path.isEmpty else {
            return
        }

        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return
        }

        do {
            try loadProject(from: url)
            currentProjectURL = resolvedProjectURL(from: url)
            statusMessage = "Opened last project \(currentProjectURL?.lastPathComponent ?? url.lastPathComponent)"
        } catch {
            // Last-project restore should not block a fresh app launch.
            currentProjectURL = nil
        }
    }

    private func saveProject(to url: URL) throws {
        let bundleURL = Flux2ProjectDocument.normalizedBundleURL(from: url)
        let project = try makeProject(forBundle: true)
        let slotImages = imageSlots.compactMap { slot -> FluxGenerationProjectBundle.SlotImage? in
            guard slot.hasImage, let image = slot.image else { return nil }
            return FluxGenerationProjectBundle.SlotImage(id: slot.id, image: image)
        }
        try FluxGenerationProjectBundle.save(
            project: project,
            slotImages: slotImages,
            previewImage: previewDisplayImage,
            historyAssets: try historyAssetsForSave(),
            to: bundleURL
        )
    }

    private func resolvedProjectURL(from url: URL) -> URL {
        if FluxGenerationProjectBundle.isBundleURL(url) {
            return url.pathExtension == FluxGenerationProjectBundle.packageExtension
                ? url
                : url.deletingLastPathComponent()
        }
        return url
    }

    private func loadProject(from url: URL) throws {
        lastSavedImageURL = nil

        if url.pathExtension == "json" || url.lastPathComponent == FluxGenerationProjectBundle.manifestName {
            let data = try Data(contentsOf: url.pathExtension == "json" ? url : FluxGenerationProjectBundle.manifestURL(in: url))
            if let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let version = object["version"] as? Int,
               version < FluxGenerationProject.minimumLoadableVersion {
                let legacy = try JSONDecoder().decode(FluxGenerationProjectV1Legacy.self, from: data)
                applyProjectV1Shell(legacy)
                clearAllImageSlots()
                generatedImage = nil
                formattedComparisonImage = nil
                upsampledPrompt = nil
                checkpointImages.removeAll()
                errorMessage = nil
                statusMessage = "This project used the old image format (v1). Images were cleared — add them again in the Images palette."
                clearSelectionUndoHistory()
                clearEditHistory()
                applySizingControls()
                return
            }
        }

        let loaded = try FluxGenerationProject.load(at: url)
        applyProjectShell(from: loaded.project)
        try restoreImageSlots(from: loaded.project, bundleRoot: loaded.bundleRoot)
        loadEditHistory(from: loaded.project, bundleRoot: loaded.bundleRoot)
        Task { await refreshVisionSubjectMaskCache() }
        generatedImage = loaded.previewImage
        formattedComparisonImage = nil
        upsampledPrompt = nil
        checkpointImages.removeAll()
        errorMessage = nil
        clearSelectionUndoHistory()
        applySizingControls()
        applyLoadedHistoryPointer(bundleRoot: loaded.bundleRoot)
    }

    private func makeProject(forBundle: Bool = false) throws -> FluxGenerationProject {
        let records: [GenerationImageRecord]
        if forBundle {
            records = imageSlots.map { slot in
                let bundlePath = slot.hasImage
                    ? FluxGenerationProjectBundle.slotRelativePath(for: slot.id)
                    : nil
                var record = slot.toProjectRecord(bundlePath: bundlePath)
                record.sourcePath = nil
                record.pngBase64 = nil
                return record
            }
        } else {
            records = try imageSlots.map { slot in
                let pngBase64: String?
                if slot.hasImage, let image = slot.image {
                    pngBase64 = try pngData(from: image).base64EncodedString()
                } else {
                    pngBase64 = nil
                }
                return slot.toProjectRecord(pngBase64: pngBase64)
            }
        }

        return FluxGenerationProject(
            version: forBundle ? FluxGenerationProject.bundleVersion : FluxGenerationProject.minimumLoadableVersion,
            selectedModel: selectedModel.rawValue,
            textQuantization: textQuantization.rawValue,
            transformerQuantization: transformerQuantization.rawValue,
            prompt: prompt,
            upsamplePrompt: upsamplePrompt,
            width: width,
            height: height,
            steps: steps,
            guidance: guidance,
            seed: seed,
            preparationOverlayOpacity: preparationOverlayOpacity,
            megapixelBudget: megapixelBudget,
            clearPromptAfterGeneration: clearPromptAfterGeneration,
            selectedFamily: selectedFamily?.rawValue,
            processArea: processArea.map(FluxGenerationProject.NormalizedRect.init),
            contextArea: FluxGenerationProject.NormalizedRect(contextArea),
            editMode: nil,
            inpaintMaskTool: inpaintMaskTool.rawValue,
            outpaintPadding: outpaintPadding.hasExpansion ? outpaintPadding : nil,
            inpaintIntent: inpaintIntent.rawValue,
            enrichInpaintPromptWithVLM: enrichInpaintPromptWithVLM,
            fillContextMaskScale: hasLocalFillSelection ? fillContextMaskScale : nil,
            inpaintMaskLayers: inpaintMaskLayers.isEmpty ? nil : inpaintMaskLayers,
            images: records,
            selectedImageSlotID: selectedImageSlotID,
            currentHistoryIndex: editHistoryManifestFields().currentIndex,
            history: editHistoryManifestFields().entries
        )
    }

    func pngData(from image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else {
            throw NSError(domain: "FluxProject", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not encode reference image"])
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "FluxProject", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not finalize reference image"])
        }
        return data as Data
    }
}
