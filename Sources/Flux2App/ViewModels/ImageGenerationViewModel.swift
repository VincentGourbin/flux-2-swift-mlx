/**
 * ImageGenerationViewModel.swift
 * ViewModel for Flux.2 image generation (T2I and I2I)
 */

import SwiftUI
import Flux2Core
import Flux2Chains
import FluxTextEncoders
import CoreGraphics
import ImageIO
import MLX
import UniformTypeIdentifiers

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Generation Mode

enum GenerationWorkflow {
    case textToImage
    case imageToImage
}

enum GenerationMode: String, CaseIterable {
    case textToImage = "Text to Image"
    case imageToImage = "Image to Image"
}

/// A model family groups inference models that share an architecture and is the
/// source of truth for architecture-wide constants — most importantly the
/// pixel-alignment factor (VAE scale × patch size) that output and conditioning
/// sizes floor to. Family is the first control in Model Configuration; the other
/// model pickers gate on it, but with a single family today it's pre-selected so
/// the gating is dormant until a second family is added.
enum ModelFamily: String, CaseIterable, Identifiable {
    case flux2 = "FLUX.2"

    var id: String { rawValue }
    var displayName: String { rawValue }

    /// Pixel-alignment factor. FLUX.2 = VAE 8× × patch 2 × the reference
    /// encoder's extra ÷2 flooring = 32. (Qwen-Image-Edit also lands on 32.)
    var pixelAlignment: Int {
        switch self {
        case .flux2: return 32
        }
    }
}

extension Flux2Model {
    /// The family this inference model belongs to.
    var family: ModelFamily {
        switch self {
        case .dev, .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV:
            return .flux2
        }
    }
}

// MARK: - Image Generation ViewModel

@MainActor
class ImageGenerationViewModel: ObservableObject {
    // MARK: - Model Selection
    /// Chosen architecture family — source of truth for the pixel-alignment
    /// factor and which models are selectable. Pre-selected to the only family
    /// today; the model / text-encoder / transformer pickers gate on
    /// `isFamilySelected`, so the gating stays dormant until a second family
    /// exists. Inferred from the model when an existing project loads.
    @Published var selectedFamily: ModelFamily? = .flux2
    @Published var selectedModel: Flux2Model = .klein4B
    @Published var textQuantization: MistralQuantization = .mlx8bit
    @Published var transformerQuantization: TransformerQuantization = .qint8

    // MARK: - Prompt
    @Published var prompt: String = ""
    @Published var upsamplePrompt: Bool = false
    /// Clear the prompt automatically once a run finishes successfully.
    @Published var clearPromptAfterGeneration: Bool = false
    /// The enhanced prompt when upsampling is on, surfaced live (set the moment
    /// the VLM returns it, before the long denoise) and cleared at the start of
    /// each run. `nil` whenever the prompt wasn't upsampled.
    @Published var upsampledPrompt: String?

    // MARK: - T2I Parameters
    @Published var width: Int = 1024
    @Published var height: Int = 1024
    @Published var steps: Int = 50
    @Published var guidance: Float = 4.0
    @Published var seed: String = ""  // Empty = random

    // MARK: - I2I Parameters
    @Published var imageSlots: [GenerationImageSlot] = []
    @Published var selectedImageSlotID: UUID?
    @Published var preparationOverlayOpacity: Double = 0.22
    @Published var processArea: CGRect?
    @Published var contextArea: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    @Published var inpaintIntent: Flux2InpaintIntent = .modify
    @Published var enrichInpaintPromptWithVLM: Bool = false
    /// Generative-fill Qwen context slider: -1 = minimum, 0 = auto (default), 1 = full image.
    @Published var fillContextMaskScale: Double = 0
    @Published var inpaintMaskLayers: [InpaintMaskLayer] = []
    @Published var inpaintMaskTool: InpaintMaskTool = .pointer
    @Published var draftPolygonPoints: [CGPoint] = []
    @Published var draftLassoPoints: [CGPoint] = []
    @Published var isDrawingSelection: Bool = false
    /// Set when the user defines an expanded canvas for outpainting (Crop tool).
    @Published var outpaintCanvasIsDefined: Bool = false
    @Published var outpaintPadding: OutpaintPadding = .zero
    /// Resolved Vision silhouettes keyed by mask-layer id (preview + generate).
    @Published var visionSubjectMasks: [UUID: CGImage] = [:]
    @Published var visionSubjectStatusMessage: String?
    let selectionUndoStore = SelectionUndoStore()
    let editHistoryStore = EditHistoryStore()
    var isApplyingSelectionUndo = false
    #if canImport(AppKit)
    /// Selection mode (replace/add/subtract) captured when a polygon draft begins.
    var polygonSelectionCommitMode: SelectionMode?
    #endif
    var fillContextMaskScaleUndoBaseline: Double?

    /// Target generation budget in megapixels (the *maximum* total pixels). The
    /// barn doors set the aspect ratio; the generation fills this budget at that
    /// aspect, so a small barn-door region no longer shrinks the output — it just
    /// means the conditioning crop is upscaled to hit the budget. Also drives the
    /// reference conditioning area (output == reference keeps the composite aligned).
    @Published var megapixelBudget: Double = 1.0 {
        didSet {
            // Assigning to an @Published property inside its own didSet re-enters
            // the setter (property-wrapper-backed properties don't get the
            // observer-suppression that plain stored properties do), so an
            // unguarded self-clamp here recurses until the stack overflows.
            // Guard on the clamped value: an out-of-range set re-enters exactly
            // once, and applySizingControls() runs a single time.
            let clamped = min(max(megapixelBudget, ImagePreparationSettings.minMegapixelBudget), ImagePreparationSettings.maxMegapixelBudget)
            guard clamped == megapixelBudget else {
                megapixelBudget = clamped
                return
            }
            cancelBarnDoorsIfActive()
            if outpaintPadding.hasExpansion {
                updateOutpaintPadding(outpaintPadding)
            }
            applySizingControls()
        }
    }

    // MARK: - State
    @Published var isGenerating: Bool = false
    /// True while a generation Task still holds the shared pipeline, including
    /// the brief background wind-down after a cancel. Gates starting a new run
    /// so two generations never touch the pipeline at once. `isGenerating` is
    /// the user-facing flag (cleared the instant Cancel is pressed); this one
    /// trails it until the abandoned compute actually releases.
    @Published var isPipelineBusy: Bool = false
    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 0
    @Published var generatedImage: CGImage?
    /// Formatted input scaled to match the latest output dimensions for aligned A/B preview.
    @Published var formattedComparisonImage: CGImage?
    @Published var previewZoomScale: Double = 1.0
    @Published var previewComparisonSide: PreviewComparisonSide = .processed
    @Published var inputSaveSource: ImageInputSaveSource = .formatted
    @Published var errorMessage: String?
    @Published var statusMessage: String = ""
    @Published var currentProjectURL: URL?

    /// URL of the most recently saved generated image. Drives "Open Folder" and
    /// names the companion "-input" file. Session-only; never persisted.
    @Published var lastSavedImageURL: URL?

    /// Window title: the loaded project's file name (no extension), or a
    /// placeholder when nothing has been saved/opened yet.
    var projectDisplayName: String {
        currentProjectURL?.deletingPathExtension().lastPathComponent ?? "Untitled Project"
    }

    /// Image shown in the output preview (processed result or aligned formatted comparison).
    var previewDisplayImage: CGImage? {
        guard let generatedImage else { return nil }
        if previewComparisonSide == .formatted, let formattedComparisonImage {
            return formattedComparisonImage
        }
        return generatedImage
    }

    var canTogglePreviewComparison: Bool {
        generatedImage != nil && formattedComparisonImage != nil
    }

    /// Whether the preview pane is showing a generation result (or checkpoints).
    var hasPreviewContent: Bool {
        generatedImage != nil || !checkpointImages.isEmpty
    }

    // MARK: - Checkpoints
    @Published var checkpointImages: [CheckpointImage] = []
    @Published var checkpointInterval: Int = 10  // Save checkpoint every N steps
    @Published var showCheckpoints: Bool = true

    // MARK: - Pipeline
    var pipeline: Flux2Pipeline?
    var generationTask: Task<Void, Never>?
    var skipNextModelDefaultApplication = false

    // MARK: - Init with defaults
    let loadsEnvironmentProject: Bool
    let workflow: GenerationWorkflow

    init(loadsEnvironmentProject: Bool = false, workflow: GenerationWorkflow = .textToImage) {
        self.loadsEnvironmentProject = loadsEnvironmentProject
        self.workflow = workflow
        applyRecommendedDefaults(for: selectedModel)
        if loadsEnvironmentProject {
            loadStartupProjectIfAvailable()
        } else {
            loadLastProjectIfAvailable()
        }
        restoreSessionStateIfNeeded()
        bootstrapImageSlotsIfNeeded()
    }

    // MARK: - Computed Properties

    var requiresReferenceImages: Bool {
        workflow == .imageToImage
    }

    var seedValue: UInt64? {
        guard !seed.isEmpty else { return nil }
        return UInt64(seed)
    }

    var canGenerate: Bool {
        guard !isPipelineBusy, isFamilySelected else { return false }
        if workflow == .imageToImage, !hasPrimaryReference {
            return false
        }
        if hasPrimaryReference {
            switch generateRoute {
            case .localFill:
                guard hasFillMask else { return false }
                return !prompt.isEmpty || enrichInpaintPromptWithVLM
            case .outpaint:
                return outpaintCanvasIsDefined && !prompt.isEmpty
            case .fullImage:
                break
            }
        }
        guard !prompt.isEmpty else { return false }
        return true
    }

    /// Inferred from the active tool and whether a selection exists — not user-facing.
    var generateRoute: I2IGenerateRoute {
        if inpaintMaskTool == .cropCanvas { return .outpaint }
        if hasLocalFillSelection { return .localFill }
        return .fullImage
    }

    var hasLocalFillSelection: Bool {
        isDrawingSelection || hasActiveSelection
    }

    var canUseCanvasTools: Bool {
        isSpatialEditingActive
    }

    var sizingFavor: ImageSizingFavor {
        primaryImageSlot?.sizingFavor ?? .original
    }

    var sizingMethod: ImageSizingMethod {
        primaryImageSlot?.sizingMethod ?? .crop
    }

    var preparationScale: Double {
        primaryImageSlot?.preparationScale ?? 1.0
    }

    /// Barn-door chrome applies on full-image prompt edit only — not during generative fill.
    var barnDoorToolsApply: Bool {
        guard isSpatialEditingActive, inpaintMaskTool != .cropCanvas else { return false }
        return !hasLocalFillSelection
    }

    /// Qwen framing rect for generative fill (independent of Live Area barn doors).
    var fillVLMContextArea: CGRect {
        ImagePreparation.fillVLMContextArea(
            maskLayers: inpaintMaskLayers,
            processArea: processArea,
            draftPolygonPoints: draftPolygonPoints,
            scale: CGFloat(fillContextMaskScale)
        )
    }

    /// White context-mask preview — hidden at the auto default (slider centered).
    var showsFillContextMaskOverlay: Bool {
        hasLocalFillSelection
            && enrichInpaintPromptWithVLM
            && abs(fillContextMaskScale) > 0.0001
    }

    /// Mistral text-only upsampling — full-image I2I and outpaint, not masked fill.
    var isUpsamplePromptApplicable: Bool {
        guard workflow == .imageToImage else { return true }
        switch generateRoute {
        case .fullImage, .outpaint: return true
        case .localFill: return false
        }
    }

    /// Qwen image-aware inpaint prompt — masked fill only.
    var isEnrichInpaintPromptApplicable: Bool {
        workflow == .imageToImage && generateRoute == .localFill
    }

    var recommendedSteps: Int {
        switch selectedModel {
        case .dev: return 28
        case .klein4B, .klein4BBase, .klein9B, .klein9BBase, .klein9BKV: return 4
        }
    }

    static let defaultMegapixelBudget: Double = 1.0

    var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(currentStep) / Double(totalSteps)
    }

    /// True only during the brief window after Cancel while the abandoned
    /// generation Task is still releasing the shared pipeline (`isGenerating`
    /// clears instantly on Cancel; `isPipelineBusy` trails until the compute
    /// actually unwinds). The Generate control surfaces this as a "Resetting"
    /// progress bar so the user can't fire a new run into a busy pipeline.
    var isResetting: Bool {
        isPipelineBusy && !isGenerating
    }

    /// Estimated peak memory based on current configuration
    var estimatedPeakMemoryGB: Int {
        let textEncoderMem: Int
        let transformerMem: Int

        switch selectedModel {
        case .dev:
            textEncoderMem = textQuantization.estimatedMemoryGB
            transformerMem = transformerQuantization == .bf16 ? 64 : 32
        case .klein4B, .klein4BBase:
            // Qwen3-4B is smaller
            textEncoderMem = 5  // Qwen3-4B 8bit
            transformerMem = transformerQuantization == .bf16 ? 8 : 4
        case .klein9B, .klein9BBase, .klein9BKV:
            textEncoderMem = 10  // Qwen3-8B 8bit
            transformerMem = 18  // Only bf16 available
        }

        // Peak is max of either phase + VAE + working memory
        return max(textEncoderMem, transformerMem) + 3 + 5
    }

    /// Get the appropriate transformer variant for current selection
    var selectedTransformerVariant: ModelRegistry.TransformerVariant {
        ModelRegistry.TransformerVariant.variant(for: selectedModel, quantization: transformerQuantization)
    }

    var selectableModels: [Flux2Model] {
        Flux2Model.allCases.filter { $0.isForInference && (selectedFamily == nil || $0.family == selectedFamily) }
    }

    /// Whether a family has been chosen. Model / encoder / transformer pickers
    /// and the Generate action gate on this.
    var isFamilySelected: Bool {
        selectedFamily != nil
    }

    /// Pixel-alignment factor for the active family. Falls back to the FLUX.2
    /// default before a family is chosen so incidental sizing math stays sane.
    var pixelAlignment: Int {
        selectedFamily?.pixelAlignment ?? ImagePreparation.generationSizeMultiple
    }

    var compatibleTransformerQuantizations: [TransformerQuantization] {
        Self.compatibleTransformerQuantizations(for: selectedModel)
    }

    var adjustedGenerationSize: (width: Int, height: Int) {
        ImagePreparation.referenceMatchedSize(
            width: width,
            height: height,
            maxArea: conditioningPixelBudget,
            multiple: pixelAlignment
        )
    }

    var processAreaDescription: String {
        guard primaryReferenceImage != nil else {
            return "Selection: none"
        }

        if hasLocalFillSelection, !inpaintMaskLayers.isEmpty {
            return "Selection active"
        }

        if processArea != nil {
            return "Drawing selection…"
        }

        return "Selection: none"
    }

    /// How much the resolution cap shrinks the working image before generative fill.
    var fillDownscaleFactor: Double {
        guard let image = primaryReferenceImage else { return 1 }
        let area = Double(image.width * image.height)
        let cap = megapixelBudget * 1_000_000
        guard area > cap, area > 0 else { return 1 }
        return (cap / area).squareRoot()
    }

    var contextAreaDescription: String {
        guard primaryReferenceImage != nil else {
            return "Context: add a reference image"
        }

        let size = adjustedGenerationSize
        return "Context sent to model: \(size.width)x\(size.height)"
    }

    var outpaintCanvasDescription: String {
        guard let image = primaryReferenceImage else {
            return "Canvas: add a reference image"
        }
        guard outpaintPadding.hasExpansion else {
            return "Canvas: drag edges on the preview to expand"
        }
        let size = outpaintPadding.canvasSize(sourceWidth: image.width, sourceHeight: image.height)
        let mp = Double(outpaintPadding.totalPixels(sourceWidth: image.width, sourceHeight: image.height)) / 1_000_000
        return "Canvas: \(size.width)×\(size.height) (\(String(format: "%.2f", mp)) MP) — +L\(outpaintPadding.left) +R\(outpaintPadding.right) +T\(outpaintPadding.top) +B\(outpaintPadding.bottom)"
    }

    static let lastProjectURLKey = "lastGenerationProjectURL"
    /// Absolute path to a generation project JSON file. When set, opens on launch
    /// instead of restoring `lastGenerationProjectURL` (VM smoke / agent hooks).
    static let projectEnvironmentKey = "F2SM_PROJECT"
    /// When set alongside `F2SM_PROJECT`, writes `ok` or `error` plus detail on load.
    static let smokeMarkerEnvironmentKey = "F2SM_SMOKE_MARKER"

    // MARK: - Image Management

    func applySizingControls() {
        guard let primary = primaryImageSlot, primary.role == .reference, let image = primary.image else { return }
        var settings = slotPreparationSettings(for: primary, includeLiveArea: true)
        if hasLocalFillSelection {
            settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        }
        let size = ImagePreparation.generationSize(referenceImage: image, settings: settings)
        width = size.width
        height = size.height
    }

    /// Linear factor by which the barn-door crop must be scaled to fill the
    /// budget. > 1 means the selected region is smaller than the target and gets
    /// upscaled (a softer reference); surfaced as a gentle advisory.
    var conditioningUpscaleFactor: Double {
        guard let image = primaryReferenceImage else { return 1 }
        let sourceRect = integralPixelRect(from: contextArea, in: image)
        let sourcePixels = Double(sourceRect.width * sourceRect.height)
        guard sourcePixels > 0 else { return 1 }
        let target = adjustedGenerationSize
        let targetPixels = Double(target.width * target.height)
        return (targetPixels / sourcePixels).squareRoot()
    }


    func setSizingFavor(_ favor: ImageSizingFavor) {
        guard let primary = primaryImageSlot else { return }
        setSlotSizingFavor(primary.id, favor: favor)
    }

    func setSizingMethod(_ method: ImageSizingMethod) {
        guard let primary = primaryImageSlot else { return }
        setSlotSizingMethod(primary.id, method: method)
    }

    func resetGuidanceToModelDefault() {
        guidance = selectedModel.defaultGuidance
    }

    func enforceAvailableModelDefaults(downloadedTransformers: Set<String>, downloadedTextModels: Set<String>) {
        if !selectedModel.isForInference {
            selectedModel = .klein4B
        }

        if !compatibleTransformerQuantizations.contains(transformerQuantization) {
            transformerQuantization = Self.defaultTransformerQuantization(for: selectedModel)
        }

        if selectedModel == .dev,
           let downloadedTextQuantization = Self.preferredDownloadedTextQuantization(in: downloadedTextModels),
           !downloadedTextModels.contains(Self.textModelId(for: textQuantization)) {
            textQuantization = downloadedTextQuantization
        }
    }

    func downloadedTextQuantizations(in downloadedTextModels: Set<String>) -> [MistralQuantization] {
        let downloaded = MistralQuantization.allCases.filter {
            downloadedTextModels.contains(Self.textModelId(for: $0))
        }
        return downloaded.isEmpty ? MistralQuantization.allCases : downloaded
    }

    static func compatibleTransformerQuantizations(for model: Flux2Model) -> [TransformerQuantization] {
        switch model {
        case .dev, .klein4B:
            return [.qint8, .bf16]
        case .klein9B, .klein9BKV:
            return [.bf16]
        case .klein4BBase, .klein9BBase:
            return [.bf16]
        }
    }

    private static func defaultTransformerQuantization(for model: Flux2Model) -> TransformerQuantization {
        compatibleTransformerQuantizations(for: model).first ?? .bf16
    }

    private static func preferredDownloadedTextQuantization(in downloadedTextModels: Set<String>) -> MistralQuantization? {
        [.mlx8bit, .mlx6bit, .mlx4bit, .bf16].first {
            downloadedTextModels.contains(textModelId(for: $0))
        }
    }

    private static func textModelId(for quantization: MistralQuantization) -> String {
        let variant: ModelVariant
        switch quantization {
        case .bf16:
            variant = .bf16
        case .mlx8bit:
            variant = .mlx8bit
        case .mlx6bit:
            variant = .mlx6bit
        case .mlx4bit:
            variant = .mlx4bit
        }

        return TextEncoderModelRegistry.shared.model(withVariant: variant)?.id ?? variant.rawValue
    }

    func shouldApplyDefaultsForModelChange() -> Bool {
        if skipNextModelDefaultApplication {
            skipNextModelDefaultApplication = false
            return false
        }
        return true
    }

    /// Restore barn doors to full frame, or reset the fill context-mask slider.
    func resetBarnDoors() {
        if hasLocalFillSelection {
            resetFillContextMaskScale()
        } else {
            resetContextArea()
        }
    }

    /// Reopen the barn doors (full image) and clear any selection.
    func resetContextArea() {
        contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        processArea = nil
        applySizingControls()
    }

    /// Barn doors reset when the working canvas changes (budget, sizing, outpaint).
    func cancelBarnDoorsIfActive() {
        let fullFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
        guard contextArea != fullFrame else { return }
        contextArea = fullFrame
    }

    func updateOutpaintPadding(_ padding: OutpaintPadding) {
        guard let image = primaryReferenceImage else { return }
        let maxPixels = Int(megapixelBudget * 1_000_000)
        let clamped = padding.clamped(
            sourceWidth: image.width,
            sourceHeight: image.height,
            maxPixels: maxPixels
        )
        let changed = clamped != outpaintPadding
        outpaintPadding = clamped
        outpaintCanvasIsDefined = clamped.hasExpansion
        if changed {
            cancelBarnDoorsIfActive()
        }
    }

    func resetOutpaintCanvas() {
        outpaintPadding = .zero
        outpaintCanvasIsDefined = false
    }

    /// Update the barn-door (Live Area) region from a barn-door edge drag.
    func setContextArea(_ rect: CGRect) {
        guard !hasLocalFillSelection else { return }
        contextArea = Self.clampUnitRect(rect)
        applySizingControls()
    }

    func resetFillContextMaskScale() {
        fillContextMaskScale = 0
    }

    /// Set or update the process selection (the dashed marquee).
    func setProcessArea(_ rect: CGRect?) {
        guard let rect else {
            processArea = nil
            return
        }
        processArea = Self.clampUnitRect(rect)
    }

    static func clampUnitPoint(_ point: CGPoint) -> CGPoint {
        CGPoint(
            x: min(max(point.x, 0), 1),
            y: min(max(point.y, 0), 1)
        )
    }

    // Dormant in prompt-edit mode until a paste-back marquee UI ships.
    func resetProcessContext() {
        resetContextArea()
    }

    func ensurePreparationDefaults() {
        guard hasPrimaryReference else { return }
        contextArea = ImagePreparation.clampUnitRect(contextArea)
        applySizingControls()
    }

    private func pixelRect(from normalizedRect: CGRect, in image: CGImage) -> CGRect {
        CGRect(
            x: normalizedRect.minX * CGFloat(image.width),
            y: normalizedRect.minY * CGFloat(image.height),
            width: normalizedRect.width * CGFloat(image.width),
            height: normalizedRect.height * CGFloat(image.height)
        )
    }

    static func clampUnitRect(_ rect: CGRect) -> CGRect {
        ImagePreparation.clampUnitRect(rect)
    }

    static let minMegapixelBudget = ImagePreparationSettings.minMegapixelBudget
    static let maxMegapixelBudget = ImagePreparationSettings.maxMegapixelBudget

    var conditioningPixelBudget: Int {
        ImagePreparation.conditioningPixelBudget(for: megapixelBudget)
    }

    func currentPreparationSettings() -> ImagePreparationSettings {
        guard let primary = primaryImageSlot, primary.role == .reference else {
            return ImagePreparationSettings()
        }
        return slotPreparationSettings(for: primary, includeLiveArea: true)
    }

    func prepareImageToImageInput() throws -> PreparedImageToImageInput {
        let refs = orderedReferenceSlots
        guard let primary = refs.first, let primaryImage = primary.image else {
            throw Flux2Error.invalidConfiguration("Add a primary reference image before generating")
        }
        guard refs.count <= selectedModel.maxReferenceImages else {
            throw Flux2Error.invalidConfiguration(
                "Maximum \(selectedModel.maxReferenceImages) reference images allowed for \(selectedModel.displayName)"
            )
        }

        var primarySettings = slotPreparationSettings(for: primary, includeLiveArea: true)
        if hasLocalFillSelection {
            primarySettings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        }

        let partial = try ImagePreparation.prepare(referenceImages: [primaryImage], settings: primarySettings)
        let additionalPrepared = try refs.dropFirst().compactMap { slot -> CGImage? in
            guard let image = slot.image else { return nil }
            let settings = slotPreparationSettings(for: slot, includeLiveArea: false)
            return try ImagePreparation.formatFullFrameReference(image, settings: settings)
        }

        return PreparedImageToImageInput(
            images: partial.images + additionalPrepared,
            width: partial.width,
            height: partial.height,
            compositionPlan: partial.compositionPlan
        )
    }

    func compositeGeneratedImage(_ generatedImage: CGImage, using plan: ImageCompositionPlan) throws -> CGImage {
        try ImagePreparation.composite(generatedImage, using: plan)
    }

    private func integralPixelRect(from normalizedRect: CGRect, in image: CGImage) -> CGRect {
        let pixel = CGRect(
            x: normalizedRect.minX * CGFloat(image.width),
            y: normalizedRect.minY * CGFloat(image.height),
            width: normalizedRect.width * CGFloat(image.width),
            height: normalizedRect.height * CGFloat(image.height)
        )
        let imageBounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let bounded = pixel.intersection(imageBounds)
        guard !bounded.isNull, bounded.width > 0, bounded.height > 0 else {
            return CGRect(x: 0, y: 0, width: max(1, min(image.width, 1)), height: max(1, min(image.height, 1)))
        }
        let minX = floor(bounded.minX)
        let minY = floor(bounded.minY)
        let maxX = ceil(bounded.maxX)
        let maxY = ceil(bounded.maxY)
        return CGRect(
            x: min(max(minX, 0), CGFloat(max(image.width - 1, 0))),
            y: min(max(minY, 0), CGFloat(max(image.height - 1, 0))),
            width: max(1, min(maxX, CGFloat(image.width)) - minX),
            height: max(1, min(maxY, CGFloat(image.height)) - minY)
        )
    }

    /// Decode image data using CGImageSource for pixel-exact results
    static func cgImageFromData(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    func createThumbnail(from image: NSImage) -> NSImage {
        let targetSize = NSSize(width: 100, height: 100)
        let thumbnail = NSImage(size: targetSize)
        thumbnail.lockFocus()
        image.draw(in: NSRect(origin: .zero, size: targetSize),
                   from: NSRect(origin: .zero, size: image.size),
                   operation: .copy,
                   fraction: 1.0)
        thumbnail.unlockFocus()
        return thumbnail
    }

}

// MARK: - Checkpoint Image Model

struct CheckpointImage: Identifiable {
    let id: UUID
    let image: CGImage
    let step: Int
    let timestamp: Date
}

// MARK: - Reference Image Model

struct ReferenceImage: Identifiable {
    let id: UUID
    let url: URL?
    let image: CGImage
    let thumbnail: NSImage
}
