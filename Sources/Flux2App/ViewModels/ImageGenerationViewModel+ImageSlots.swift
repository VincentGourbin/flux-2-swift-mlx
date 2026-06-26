/**
 * ImageGenerationViewModel+ImageSlots.swift
 * Images palette tab model — roles, primary, per-slot formatting.
 */

import CoreGraphics
import Flux2Chains
import Flux2Core
import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

extension ImageGenerationViewModel {
    static let maxImageSlots = FluxGenerationProject.maxImageSlots

    var activeImageSlot: GenerationImageSlot? {
        if let selectedImageSlotID,
           let slot = imageSlots.first(where: { $0.id == selectedImageSlotID }) {
            return slot
        }
        return imageSlots.first
    }

    var primaryImageSlot: GenerationImageSlot? {
        imageSlots.first(where: \.isPrimary)
    }

    var primaryReferenceImage: CGImage? {
        guard let slot = primaryImageSlot, slot.role == .reference else { return nil }
        return slot.image
    }

    var hasPrimaryReference: Bool {
        primaryReferenceImage != nil
    }

    var assignedImageCount: Int {
        imageSlots.filter(\.hasImage).count
    }

    var assignedReferenceCount: Int {
        imageSlots.filter { $0.role == .reference && $0.hasImage }.count
    }

    var assignedInterpretCount: Int {
        imageSlots.filter { $0.role == .interpretive && $0.hasImage }.count
    }

    var canAddImageSlot: Bool {
        imageSlots.count < Self.maxImageSlots
    }

    /// Reference slots in generate order: primary first, then tab order.
    var orderedReferenceSlots: [GenerationImageSlot] {
        let references = imageSlots.filter { $0.role == .reference && $0.hasImage }
        guard let primary = references.first(where: \.isPrimary) else {
            return references
        }
        return [primary] + references.filter { $0.id != primary.id }
    }

    var referenceImages: [ReferenceImage] {
        orderedReferenceSlots.compactMap { $0.toReferenceImage() }
    }

    var isSpatialEditingActive: Bool {
        guard let active = activeImageSlot,
              let primary = primaryImageSlot,
              active.id == primary.id,
              primary.role == .reference,
              primary.hasImage else {
            return false
        }
        return true
    }

    var previewSourceImage: CGImage? {
        activeImageSlot?.image ?? primaryReferenceImage
    }

    var previewSizingFavor: ImageSizingFavor {
        activeImageSlot?.sizingFavor ?? .original
    }

    var previewSizingMethod: ImageSizingMethod {
        activeImageSlot?.sizingMethod ?? .crop
    }

    var imageAssignmentSummary: String {
        let references = imageSlots.filter { $0.role == .reference && $0.hasImage }
        let interprets = imageSlots.filter { $0.role == .interpretive && $0.hasImage }
        let unassigned = imageSlots.filter { $0.role == .unassigned && $0.hasImage }
        var parts: [String] = []

        if references.contains(where: \.isPrimary) {
            parts.append("1 primary reference")
        }
        let additionalReferences = references.count - (references.contains(where: \.isPrimary) ? 1 : 0)
        if additionalReferences > 0 {
            parts.append("\(additionalReferences) additional reference\(additionalReferences == 1 ? "" : "s")")
        }
        if !interprets.isEmpty {
            parts.append("\(interprets.count) interpret (VLM)")
        }
        if !unassigned.isEmpty {
            parts.append("\(unassigned.count) unassigned")
        }
        return parts.isEmpty ? "No images assigned" : parts.joined(separator: ", ")
    }

    func bootstrapImageSlotsIfNeeded() {
        if imageSlots.isEmpty {
            let slot = GenerationImageSlot.empty()
            imageSlots = [slot]
            selectedImageSlotID = slot.id
        } else if selectedImageSlotID == nil {
            selectedImageSlotID = imageSlots.first?.id
        }
    }

    func selectImageSlot(_ id: UUID) {
        selectedImageSlotID = id
    }

    func addImageSlot() {
        guard canAddImageSlot else { return }
        let slot = GenerationImageSlot.empty()
        imageSlots.append(slot)
        selectedImageSlotID = slot.id
    }

    func removeImageSlot(_ id: UUID) {
        guard imageSlots.count > 1 else { return }
        let wasPrimary = imageSlots.first(where: { $0.id == id })?.isPrimary ?? false
        imageSlots.removeAll { $0.id == id }
        if selectedImageSlotID == id {
            selectedImageSlotID = imageSlots.first?.id
        }
        if wasPrimary {
            promotePrimaryReferenceIfNeeded()
        }
        if imageSlots.allSatisfy({ !$0.hasImage }) {
            resetSpatialStateForEmptyImages()
        }
        applySizingControlsForPreview()
    }

    func setSlotTabLabel(_ id: UUID, label: String) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        imageSlots[index].customTabLabel = trimmed.isEmpty ? nil : trimmed
    }

    func clearImageSlot(_ id: UUID) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        let wasPrimary = imageSlots[index].isPrimary
        imageSlots[index].role = .unassigned
        imageSlots[index].isPrimary = false
        imageSlots[index].url = nil
        imageSlots[index].image = nil
        imageSlots[index].thumbnail = nil
        if wasPrimary {
            promotePrimaryReferenceIfNeeded()
        }
        if !imageSlots.contains(where: \.hasImage) {
            resetSpatialStateForEmptyImages()
        }
        applySizingControlsForPreview()
    }

    func clearAllImageSlots() {
        imageSlots = [GenerationImageSlot.empty()]
        selectedImageSlotID = imageSlots[0].id
        resetSpatialStateForEmptyImages()
        clearSelectionUndoHistory()
    }

    func setSlotRole(_ id: UUID, role: GenerationImageRole) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        imageSlots[index].role = role
        if role != .reference, imageSlots[index].isPrimary {
            imageSlots[index].isPrimary = false
            promotePrimaryReferenceIfNeeded()
        }
        reconcilePrimaryAssignment()
        applySizingControlsForPreview()
    }

    func setSlotPrimary(_ id: UUID, isPrimary: Bool) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        guard imageSlots[index].role == .reference else { return }

        if isPrimary {
            for slotIndex in imageSlots.indices {
                imageSlots[slotIndex].isPrimary = imageSlots[slotIndex].id == id
            }
        } else {
            imageSlots[index].isPrimary = false
            promotePrimaryReferenceIfNeeded()
        }
    }

    func setSlotSizingFavor(_ id: UUID, favor: ImageSizingFavor) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        cancelBarnDoorsIfActive()
        imageSlots[index].sizingFavor = favor
        applySizingControlsForPreview()
    }

    func setSlotSizingMethod(_ id: UUID, method: ImageSizingMethod) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        cancelBarnDoorsIfActive()
        imageSlots[index].sizingMethod = method
        applySizingControlsForPreview()
    }

    func setSlotPreparationScale(_ id: UUID, scale: Double) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        cancelBarnDoorsIfActive()
        imageSlots[index].preparationScale = min(max(scale, 0.1), 1.0)
        applySizingControlsForPreview()
    }

    func loadImageIntoSlot(_ id: UUID, from url: URL) {
        guard let data = try? Data(contentsOf: url),
              let cgImage = Self.cgImageFromData(data) else {
            errorMessage = "Failed to load image from \(url.lastPathComponent)"
            return
        }
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        assignImageToSlot(id, url: url, cgImage: cgImage, thumbnail: createThumbnail(from: nsImage))
    }

    func loadImageIntoSlot(_ id: UUID, from nsImage: NSImage) {
        guard let tiffData = nsImage.tiffRepresentation,
              let cgImage = Self.cgImageFromData(tiffData) else {
            errorMessage = "Failed to process dropped image"
            return
        }
        assignImageToSlot(id, url: nil, cgImage: cgImage, thumbnail: createThumbnail(from: nsImage))
    }

    func loadImageIntoSlot(_ id: UUID, cgImage: CGImage) {
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        assignImageToSlot(id, url: nil, cgImage: cgImage, thumbnail: createThumbnail(from: nsImage))
    }

    func addReferenceImage(from url: URL) {
        bootstrapImageSlotsIfNeeded()
        let targetID = selectedImageSlotID ?? imageSlots[0].id
        if let active = activeImageSlot, !active.hasImage {
            loadImageIntoSlot(targetID, from: url)
            return
        }
        if canAddImageSlot {
            addImageSlot()
            if let newID = selectedImageSlotID {
                loadImageIntoSlot(newID, from: url)
            }
        }
    }

    func addReferenceImage(from nsImage: NSImage) {
        bootstrapImageSlotsIfNeeded()
        let targetID = selectedImageSlotID ?? imageSlots[0].id
        if let active = activeImageSlot, !active.hasImage {
            loadImageIntoSlot(targetID, from: nsImage)
            return
        }
        if canAddImageSlot {
            addImageSlot()
            if let newID = selectedImageSlotID {
                loadImageIntoSlot(newID, from: nsImage)
            }
        }
    }

    func addReferenceImage(cgImage: CGImage) {
        bootstrapImageSlotsIfNeeded()
        let targetID = selectedImageSlotID ?? imageSlots[0].id
        if let active = activeImageSlot, !active.hasImage {
            loadImageIntoSlot(targetID, cgImage: cgImage)
            return
        }
        if canAddImageSlot {
            addImageSlot()
            if let newID = selectedImageSlotID {
                loadImageIntoSlot(newID, cgImage: cgImage)
            }
        }
    }

    func removeReferenceImage(_ id: UUID) {
        clearImageSlot(id)
    }

    func clearReferenceImages() {
        clearAllImageSlots()
    }

    private func assignImageToSlot(_ id: UUID, url: URL?, cgImage: CGImage, thumbnail: NSImage) {
        guard let index = imageSlots.firstIndex(where: { $0.id == id }) else { return }
        let isFirstImage = !imageSlots.contains(where: \.hasImage)
        imageSlots[index].url = url
        imageSlots[index].image = cgImage
        imageSlots[index].thumbnail = thumbnail
        if isFirstImage {
            imageSlots[index].role = .reference
            imageSlots[index].isPrimary = true
        }
        reconcilePrimaryAssignment()
        resetOutpaintCanvas()
        cancelBarnDoorsIfActive()
        ensurePreparationDefaults()
        applySizingControlsForPreview()
    }

    private func reconcilePrimaryAssignment() {
        let primaryIndices = imageSlots.indices.filter { imageSlots[$0].isPrimary }
        if primaryIndices.count > 1 {
            let keep = primaryIndices.first!
            for index in imageSlots.indices where index != keep {
                imageSlots[index].isPrimary = false
            }
        }
        if imageSlots.contains(where: { $0.isPrimary && $0.role != .reference }) {
            for index in imageSlots.indices where imageSlots[index].isPrimary && imageSlots[index].role != .reference {
                imageSlots[index].isPrimary = false
            }
            promotePrimaryReferenceIfNeeded()
        }
        if !imageSlots.contains(where: \.isPrimary) {
            promotePrimaryReferenceIfNeeded()
        }
    }

    private func promotePrimaryReferenceIfNeeded() {
        guard !imageSlots.contains(where: { $0.isPrimary && $0.role == .reference }) else { return }
        if let index = imageSlots.firstIndex(where: { $0.role == .reference && $0.hasImage }) {
            for slotIndex in imageSlots.indices {
                imageSlots[slotIndex].isPrimary = slotIndex == index
            }
        }
    }

    private func resetSpatialStateForEmptyImages() {
        processArea = nil
        contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        inpaintMaskTool = .pointer
        resetOutpaintCanvas()
    }

    func slotPreparationSettings(
        for slot: GenerationImageSlot,
        includeLiveArea: Bool
    ) -> ImagePreparationSettings {
        var settings = ImagePreparationSettings()
        settings.sizingFavor = slot.sizingFavor
        settings.sizingMethod = slot.sizingMethod
        settings.preparationScale = slot.preparationScale
        settings.megapixelBudget = megapixelBudget
        settings.pixelAlignment = pixelAlignment
        settings.compositeBack = true
        if includeLiveArea {
            settings.contextArea = contextArea
            settings.processArea = processArea
        } else {
            settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        }
        settings.clampValues()
        return settings
    }

    func interpretPathsForGeneration() throws -> [String] {
        var paths: [String] = []
        for slot in imageSlots where slot.role == .interpretive && slot.hasImage {
            if let path = slot.url?.path, FileManager.default.fileExists(atPath: path) {
                paths.append(path)
                continue
            }
            guard let image = slot.image else { continue }
            let temp = FileManager.default.temporaryDirectory
                .appendingPathComponent("flux2-interpret-\(slot.id.uuidString).png")
            try pngData(from: image).write(to: temp, options: Data.WritingOptions.atomic)
            paths.append(temp.path)
        }
        return paths
    }

    func applySizingControlsForPreview() {
        applySizingControls()
    }
}

struct FluxGenerationProjectV1Legacy: Decodable {
    var selectedModel: String
    var textQuantization: String
    var transformerQuantization: String
    var prompt: String
    var upsamplePrompt: Bool
    var width: Int
    var height: Int
    var steps: Int
    var guidance: Float
    var seed: String
    var sizingFavor: String
    var sizingMethod: String
    var preparationScale: Double?
    var preparationOverlayOpacity: Double?
    var megapixelBudget: Double?
    var clearPromptAfterGeneration: Bool?
    var selectedFamily: String?
    var processArea: FluxGenerationProject.NormalizedRect?
    var contextArea: FluxGenerationProject.NormalizedRect
    var editMode: String?
    var inpaintMaskTool: String?
    var outpaintPadding: OutpaintPadding?
    var inpaintIntent: String?
    var enrichInpaintPromptWithVLM: Bool?
    var fillContextMaskScale: Double?
    var inpaintMaskLayers: [InpaintMaskLayer]?
}

extension ImageGenerationViewModel {
    func applyProjectShell(from project: FluxGenerationProject) {
        skipNextModelDefaultApplication = true
        selectedModel = Flux2Model(rawValue: project.selectedModel) ?? .klein4B
        selectedFamily = project.selectedFamily.flatMap(ModelFamily.init(rawValue:)) ?? selectedModel.family
        textQuantization = MistralQuantization(rawValue: project.textQuantization) ?? .mlx8bit
        transformerQuantization = TransformerQuantization(rawValue: project.transformerQuantization) ?? .qint8
        prompt = project.prompt
        upsamplePrompt = project.upsamplePrompt
        width = project.width
        height = project.height
        steps = project.steps
        guidance = project.guidance
        seed = project.seed
        preparationOverlayOpacity = project.preparationOverlayOpacity ?? 0.22
        megapixelBudget = min(max(project.megapixelBudget ?? 1.0, Self.minMegapixelBudget), Self.maxMegapixelBudget)
        clearPromptAfterGeneration = project.clearPromptAfterGeneration ?? false
        processArea = project.processArea?.cgRect
        contextArea = Self.clampUnitRect(project.contextArea.cgRect)
        let legacyRoute = I2IGenerateRoute.fromLegacyProjectValue(project.editMode)
        inpaintIntent = project.inpaintIntent.flatMap(Flux2InpaintIntent.init(rawValue:)) ?? .modify
        if let toolRaw = project.inpaintMaskTool,
           let tool = InpaintMaskTool(rawValue: toolRaw) {
            inpaintMaskTool = tool
        } else if legacyRoute == .outpaint {
            inpaintMaskTool = .cropCanvas
        } else {
            inpaintMaskTool = .pointer
        }
        enrichInpaintPromptWithVLM = project.enrichInpaintPromptWithVLM
            ?? (!(project.inpaintMaskLayers ?? []).isEmpty || legacyRoute == .localFill)
        fillContextMaskScale = project.fillContextMaskScale ?? 0
        inpaintMaskLayers = project.inpaintMaskLayers ?? []
        outpaintPadding = project.outpaintPadding ?? .zero
        outpaintCanvasIsDefined = outpaintPadding.hasExpansion
        draftPolygonPoints.removeAll()
        visionSubjectMasks.removeAll()
        visionSubjectStatusMessage = nil
    }

    func applyProjectV1Shell(_ legacy: FluxGenerationProjectV1Legacy) {
        applyProjectShell(from: FluxGenerationProject(
            selectedModel: legacy.selectedModel,
            textQuantization: legacy.textQuantization,
            transformerQuantization: legacy.transformerQuantization,
            prompt: legacy.prompt,
            upsamplePrompt: legacy.upsamplePrompt,
            width: legacy.width,
            height: legacy.height,
            steps: legacy.steps,
            guidance: legacy.guidance,
            seed: legacy.seed,
            preparationOverlayOpacity: legacy.preparationOverlayOpacity,
            megapixelBudget: legacy.megapixelBudget,
            clearPromptAfterGeneration: legacy.clearPromptAfterGeneration,
            selectedFamily: legacy.selectedFamily,
            processArea: legacy.processArea,
            contextArea: legacy.contextArea,
            editMode: legacy.editMode,
            inpaintMaskTool: legacy.inpaintMaskTool,
            outpaintPadding: legacy.outpaintPadding,
            inpaintIntent: legacy.inpaintIntent,
            enrichInpaintPromptWithVLM: legacy.enrichInpaintPromptWithVLM,
            fillContextMaskScale: nil,
            inpaintMaskLayers: legacy.inpaintMaskLayers,
            images: [GenerationImageRecord()],
            selectedImageSlotID: nil
        ))
    }

    func restoreImageSlots(from project: FluxGenerationProject, bundleRoot: URL? = nil) throws {
        var restored: [GenerationImageSlot] = []
        for record in project.images {
            let cgImage: CGImage?
            if let bundleRoot,
               let bundlePath = record.bundlePath {
                let assetURL = bundleRoot.appendingPathComponent(bundlePath, isDirectory: false)
                if FileManager.default.fileExists(atPath: assetURL.path) {
                    cgImage = try? ProjectBundleImageWriter.loadCGImage(from: assetURL)
                } else {
                    cgImage = nil
                }
            } else if let path = record.sourcePath,
               FileManager.default.fileExists(atPath: path),
               let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
               let image = Self.cgImageFromData(data) {
                cgImage = image
            } else if let pngBase64 = record.pngBase64,
                      let data = Data(base64Encoded: pngBase64),
                      let image = Self.cgImageFromData(data) {
                cgImage = image
            } else {
                cgImage = nil
            }

            let thumbnail: NSImage?
            if let cgImage {
                let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
                thumbnail = createThumbnail(from: nsImage)
            } else {
                thumbnail = nil
            }

            restored.append(GenerationImageSlot.fromProjectRecord(record, cgImage: cgImage, thumbnail: thumbnail))
        }

        if restored.isEmpty {
            let slot = GenerationImageSlot.empty()
            imageSlots = [slot]
            selectedImageSlotID = slot.id
        } else {
            imageSlots = restored
            if let selected = project.selectedImageSlotID,
               restored.contains(where: { $0.id == selected }) {
                selectedImageSlotID = selected
            } else {
                selectedImageSlotID = restored.first?.id
            }
        }
        reconcilePrimaryAssignment()
    }
}
