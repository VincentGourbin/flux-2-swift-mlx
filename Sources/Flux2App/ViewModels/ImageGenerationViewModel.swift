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
    @Published private(set) var upsampledPrompt: String?

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
    private let selectionUndoStore = SelectionUndoStore()
    private var isApplyingSelectionUndo = false
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
    @Published private(set) var isPipelineBusy: Bool = false
    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 0
    @Published var generatedImage: CGImage?
    /// Formatted input scaled to match the latest output dimensions for aligned A/B preview.
    @Published private(set) var formattedComparisonImage: CGImage?
    @Published var previewZoomScale: Double = 1.0
    @Published var previewComparisonSide: PreviewComparisonSide = .processed
    @Published var inputSaveSource: ImageInputSaveSource = .formatted
    @Published var errorMessage: String?
    @Published var statusMessage: String = ""
    @Published var currentProjectURL: URL?

    /// URL of the most recently saved generated image. Drives "Open Folder" and
    /// names the companion "-input" file. Session-only; never persisted.
    @Published private(set) var lastSavedImageURL: URL?

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
    private var pipeline: Flux2Pipeline?
    private var generationTask: Task<Void, Never>?
    var skipNextModelDefaultApplication = false

    // MARK: - Init with defaults
    private let loadsEnvironmentProject: Bool
    private let workflow: GenerationWorkflow

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

    var hasFillMask: Bool {
        !inpaintMaskLayers.isEmpty
    }

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
            inpaintIntent: inpaintIntent
        )
    }

    private func recordSelectionUndoPoint() {
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
        visionSubjectMasks.removeAll()
        visionSubjectStatusMessage = nil
        isDrawingSelection = false
        objectWillChange.send()

        Task { await refreshVisionSubjectMaskCache() }
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

    private static let lastProjectURLKey = "lastGenerationProjectURL"
    /// Absolute path to a generation project JSON file. When set, opens on launch
    /// instead of restoring `lastGenerationProjectURL` (VM smoke / agent hooks).
    private static let projectEnvironmentKey = "F2SM_PROJECT"
    /// When set alongside `F2SM_PROJECT`, writes `ok` or `error` plus detail on load.
    private static let smokeMarkerEnvironmentKey = "F2SM_SMOKE_MARKER"

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

    private func prepareSelectionCommit(mode: SelectionMode) {
        if mode == .replace {
            inpaintMaskLayers.removeAll()
            visionSubjectMasks.removeAll()
            visionSubjectStatusMessage = nil
        }
    }

    private func appendCommittedLayer(_ primitive: InpaintMaskPrimitive, mode: SelectionMode) {
        recordSelectionUndoPoint()
        prepareSelectionCommit(mode: mode)
        inpaintMaskLayers.append(
            InpaintMaskLayer(
                combineMode: mode.combineMode,
                primitive: primitive
            )
        )
        processArea = nil
        applyGenerativeFillDefaultsIfNeeded()
    }

    func commitFillRectangle(_ rect: CGRect, mode: SelectionMode? = nil) {
        guard inpaintMaskTool != .visionSubject else { return }
        let resolvedMode = mode ?? Self.currentSelectionCommitMode
        let clamped = Self.clampUnitRect(rect)
        appendCommittedLayer(.rectangle(.init(clamped)), mode: resolvedMode)
    }

    #if canImport(AppKit)
    private var polygonSelectionCommitMode: SelectionMode?
    #endif

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
            applyGenerativeFillDefaultsIfNeeded()
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

    private func currentPreparationSettings() -> ImagePreparationSettings {
        guard let primary = primaryImageSlot, primary.role == .reference else {
            return ImagePreparationSettings()
        }
        return slotPreparationSettings(for: primary, includeLiveArea: true)
    }

    private func prepareImageToImageInput() throws -> PreparedImageToImageInput {
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

    private func compositeGeneratedImage(_ generatedImage: CGImage, using plan: ImageCompositionPlan) throws -> CGImage {
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

    private func loadStartupProjectIfAvailable() {
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
                detail: "project=\(url.path)\nreferences=\(assignedReferenceCount)\nprompt=\(prompt)"
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

    private func loadLastProjectIfAvailable() {
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
                applySizingControls()
                return
            }
        }

        let loaded = try FluxGenerationProject.load(at: url)
        applyProjectShell(from: loaded.project)
        try restoreImageSlots(from: loaded.project, bundleRoot: loaded.bundleRoot)
        Task { await refreshVisionSubjectMaskCache() }
        generatedImage = loaded.previewImage
        formattedComparisonImage = nil
        upsampledPrompt = nil
        checkpointImages.removeAll()
        errorMessage = nil
        clearSelectionUndoHistory()
        applySizingControls()
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
            selectedImageSlotID: selectedImageSlotID
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

    // MARK: - Generation

    /// Generate image (T2I or I2I based on reference images).
    ///
    /// Lifecycle (start guard + `isGenerating`) is owned by `startGeneration()`.
    /// The `defer` always releases the pipeline, even on cancel or error.
    func generate() async {
        defer {
            isGenerating = false
            isPipelineBusy = false
        }

        errorMessage = nil
        generatedImage = nil
        upsampledPrompt = nil
        checkpointImages.removeAll()
        currentStep = 0
        totalSteps = steps
        statusMessage = "Initializing pipeline..."

        do {
            // Create quantization config
            let quantConfig = Flux2QuantizationConfig(
                textEncoder: textQuantization,
                transformer: transformerQuantization
            )

            // Get HF token
            let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? UserDefaults.standard.string(forKey: "hfToken")

            // Create pipeline
            statusMessage = "Creating pipeline for \(selectedModel.displayName)..."
            pipeline = Flux2Pipeline(
                model: selectedModel,
                quantization: quantConfig,
                hfToken: hfToken
            )

            // Load models
            statusMessage = "Loading models..."
            try await pipeline!.loadModels { progress, message in
                Task { @MainActor in
                    guard self.isGenerating else { return }
                    self.statusMessage = message
                }
            }

            // Honor a cancel requested during loading before doing expensive work.
            try Task.checkCancellation()

            // Generate
            var image: CGImage
            let interpretPaths = try interpretPathsForGeneration()

            // Checkpoint callback
            var checkpointCallback: (@Sendable (Int, CGImage) -> Void)? = nil
            if showCheckpoints {
                checkpointCallback = { [weak self] step, checkpointImage in
                    Task { @MainActor in
                        guard let self, self.isGenerating else { return }
                        self.addCheckpoint(image: checkpointImage, step: step)
                    }
                }
            }

            // Surface the enhanced prompt the moment the VLM returns it, before
            // the long denoise, so the UI shows it live during generation.
            let promptUpsampledCallback: Flux2PromptUpsampleCallback = { [weak self] enhanced in
                Task { @MainActor in
                    guard let self, self.isGenerating else { return }
                    self.upsampledPrompt = enhanced
                }
            }

            if workflow == .textToImage {
                // Text-to-Image
                statusMessage = "Generating image..."
                let result = try await pipeline!.generateTextToImageWithResult(
                    prompt: prompt,
                    interpretImagePaths: interpretPaths.isEmpty ? nil : interpretPaths,
                    height: height,
                    width: width,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue,
                    upsamplePrompt: upsamplePrompt,
                    checkpointInterval: showCheckpoints ? checkpointInterval : nil,
                    onProgress: { current, total in
                        Task { @MainActor in
                            guard self.isGenerating else { return }
                            self.currentStep = current
                            self.totalSteps = total
                            self.statusMessage = "Step \(current)/\(total)"
                        }
                    },
                    onCheckpoint: checkpointCallback,
                    onPromptUpsampled: promptUpsampledCallback
                )
                image = result.image
                if result.wasUpsampled { upsampledPrompt = result.usedPrompt }
            } else if generateRoute == .localFill {
                guard hasFillMask else {
                    throw Flux2Error.invalidConfiguration("Draw a selection on the reference image before generating.")
                }

                if enrichInpaintPromptWithVLM {
                    statusMessage = "Loading Qwen3.5 VLM..."
                    try await ensureQwen35VLMLoaded()
                }

                guard let sourceImage = primaryReferenceImage else {
                    throw Flux2Error.invalidConfiguration("Add a primary reference image before generating.")
                }
                statusMessage = "Preparing selection..."
                let mask = try buildGenerativeFillMask(for: sourceImage)

                statusMessage = "Running generative fill..."
                let maxPixels = Int(megapixelBudget * 1_000_000)
                let chain = Flux2MaskedInpaintingChain(
                    pipeline: pipeline!,
                    prompt: prompt.isEmpty ? "fill in this region" : prompt,
                    image: sourceImage,
                    mask: mask,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue,
                    upsamplePrompt: upsamplePrompt,
                    enrichPromptWithVLM: enrichInpaintPromptWithVLM,
                    intent: inpaintIntent,
                    vlmContextArea: enrichInpaintPromptWithVLM ? fillVLMContextArea : nil,
                    maxPixels: maxPixels,
                    checkpointInterval: showCheckpoints ? checkpointInterval : nil,
                    onProgress: { current, total in
                        Task { @MainActor in
                            guard self.isGenerating else { return }
                            self.currentStep = current
                            self.totalSteps = total
                            self.statusMessage = "Step \(current)/\(total)"
                        }
                    },
                    onCheckpoint: checkpointCallback
                )
                let result = try await chain.run()
                image = result.image
                if result.wasUpsampled {
                    upsampledPrompt = result.usedPrompt
                } else if enrichInpaintPromptWithVLM,
                          result.usedPrompt != prompt,
                          !result.usedPrompt.isEmpty {
                    upsampledPrompt = result.usedPrompt
                }
                if let notice = result.notice {
                    statusMessage = "Generation complete — \(notice)"
                }
            } else if generateRoute == .outpaint {
                guard outpaintCanvasIsDefined else {
                    throw Flux2Error.invalidConfiguration("Expand the canvas on the preview before generating.")
                }

                guard let sourceImage = primaryReferenceImage else {
                    throw Flux2Error.invalidConfiguration("Add a primary reference image before generating.")
                }
                statusMessage = "Running outpaint..."
                let maxPixels = Int(megapixelBudget * 1_000_000)
                let chain = Flux2OutpaintingChain(
                    pipeline: pipeline!,
                    image: sourceImage,
                    top: outpaintPadding.top,
                    bottom: outpaintPadding.bottom,
                    left: outpaintPadding.left,
                    right: outpaintPadding.right,
                    prompt: prompt,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue,
                    upsamplePrompt: upsamplePrompt,
                    maxPixels: maxPixels,
                    onProgress: { current, total in
                        Task { @MainActor in
                            guard self.isGenerating else { return }
                            self.currentStep = current
                            self.totalSteps = total
                            self.statusMessage = "Step \(current)/\(total)"
                        }
                    }
                )
                let result = try await chain.run()
                image = result.image
                if result.wasUpsampled { upsampledPrompt = result.usedPrompt }
                if let notice = result.notice {
                    statusMessage = "Generation complete — \(notice)"
                }
            } else {
                // Image-to-Image (full-frame + Image Preparation)
                statusMessage = "Preparing image-to-image input..."
                let preparedInput = try prepareImageToImageInput()

                if upsamplePrompt {
                    statusMessage = "Upsampling prompt (Mistral VLM)..."
                } else {
                    statusMessage = "Generating with \(preparedInput.images.count) reference image(s)..."
                }
                let result = try await pipeline!.generateImageToImageWithResult(
                    prompt: prompt,
                    images: preparedInput.images,
                    interpretImagePaths: interpretPaths.isEmpty ? nil : interpretPaths,
                    height: preparedInput.height,
                    width: preparedInput.width,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue,
                    upsamplePrompt: upsamplePrompt,
                    checkpointInterval: showCheckpoints ? checkpointInterval : nil,
                    onProgress: { current, total in
                        Task { @MainActor in
                            guard self.isGenerating else { return }
                            self.currentStep = current
                            self.totalSteps = total
                            self.statusMessage = "Step \(current)/\(total)"
                        }
                    },
                    onCheckpoint: checkpointCallback,
                    onPromptUpsampled: promptUpsampledCallback
                )
                image = result.image
                if result.wasUpsampled { upsampledPrompt = result.usedPrompt }

                if let compositionPlan = preparedInput.compositionPlan {
                    statusMessage = "Compositing context area..."
                    image = try compositeGeneratedImage(image, using: compositionPlan)
                }
            }

            // Only publish if this run wasn't cancelled mid-flight. If Cancel was
            // pressed during the final step, `isGenerating` is already false and
            // we discard the result rather than flashing it over "Cancelled".
            if isGenerating {
                cacheFormattedComparisonImage(for: image)
                generatedImage = image
                previewComparisonSide = .processed
                if generateRoute == .localFill {
                    fillContextMaskScale = 0
                }
                if statusMessage.isEmpty || statusMessage.hasPrefix("Step ") {
                    statusMessage = "Generation complete!"
                }
                if clearPromptAfterGeneration {
                    prompt = ""
                }
            }

        } catch is CancellationError {
            if isGenerating { statusMessage = "Cancelled" }
        } catch Flux2Error.generationCancelled {
            if isGenerating { statusMessage = "Cancelled" }
        } catch {
            if isGenerating {
                errorMessage = error.localizedDescription
                statusMessage = "Generation failed"
            }
        }
    }

    /// Start a cancellable generation run. The Generate button calls this; the
    /// in-flight Task is owned here so `cancel()` can stop it at any time.
    func startGeneration() {
        guard canGenerate else { return }
        isGenerating = true
        isPipelineBusy = true
        generationTask = Task { [weak self] in
            await self?.generate()
        }
    }

    /// Cancel an in-flight generation immediately. The UI is freed right away and
    /// the in-flight result is discarded; the background diffusion step unwinds on
    /// its own (the denoising loop checks `Task.checkCancellation()` between steps)
    /// and releases the pipeline when it does.
    func cancel() {
        guard isGenerating else { return }
        generationTask?.cancel()
        isGenerating = false
        currentStep = 0
        totalSteps = 0
        statusMessage = "Cancelled"
    }

    /// Save generated image to file
    func saveImage() {
        guard let image = generatedImage else { return }

        do {
            let url = try ImageSaveService.save(
                image,
                metadata: ImageSaveMetadata(prompt: prompt)
            )
            lastSavedImageURL = url
            statusMessage = "Saved to \(url.path)"
        } catch {
            errorMessage = "Failed to save: \(error.localizedDescription)"
        }
    }

    /// Save the input variant chosen in the Save Input picker (raw / formatted / prepared).
    func saveInputImage() {
        do {
            let (image, suffix) = try resolveInputSaveImage(for: inputSaveSource)
            let url = try ImageSaveService.save(
                image,
                metadata: ImageSaveMetadata(prompt: prompt),
                stemSuffix: suffix
            )
            statusMessage = "Saved \(inputSaveSource.rawValue.lowercased()) input to \(url.lastPathComponent)"
        } catch {
            errorMessage = "Failed to save input: \(error.localizedDescription)"
        }
    }

    private func resolveInputSaveImage(for source: ImageInputSaveSource) throws -> (CGImage, String?) {
        switch source {
        case .raw:
            guard let image = primaryReferenceImage else {
                throw Flux2Error.invalidConfiguration("Add a reference image before saving the raw input.")
            }
            return (image, "-raw")
        case .formatted:
            return (try formattedFullInputImage(), "-formatted")
        case .prepared:
            return (try preparedInputImage(), "-prepared")
        }
    }

    /// Open the configured output folder in Finder, selecting the last save when known.
    func openOutputFolder() {
        #if canImport(AppKit)
        do {
            if let url = lastSavedImageURL {
                try ImageSaveService.revealOutputDirectoryInFinder(selecting: url)
            } else {
                try ImageSaveService.revealOutputDirectoryInFinder()
            }
        } catch {
            errorMessage = "Failed to open output folder: \(error.localizedDescription)"
        }
        #endif
    }

    /// Clear the generated image and checkpoints from the preview pane.
    func clearPreview() {
        generatedImage = nil
        formattedComparisonImage = nil
        previewComparisonSide = .processed
        checkpointImages.removeAll()
        statusMessage = "Preview cleared"
    }

    /// The reference image formatted per Image Formatting (Favour + crop/pad)
    /// across the *whole* frame — i.e. before the barn doors narrow it. Mirrors
    /// the output's budget-filled, alignment-floored sizing so the saved input
    /// shares the model's grid.
    func formattedFullInputImage() throws -> CGImage {
        guard let original = primaryReferenceImage else {
            throw Flux2Error.invalidConfiguration("Add a reference image before saving the input")
        }

        var settings = currentPreparationSettings()
        settings.contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        return try ImagePreparation.prepare(referenceImages: [original], settings: settings).images[0]
    }

    /// The barn-door crop plus formatting — the first reference image sent to the model.
    func preparedInputImage() throws -> CGImage {
        guard hasPrimaryReference else {
            throw Flux2Error.invalidConfiguration("Add a reference image before saving the prepared input")
        }
        return try prepareImageToImageInput().images[0]
    }

    private func cacheFormattedComparisonImage(for output: CGImage) {
        guard let original = primaryReferenceImage else {
            formattedComparisonImage = nil
            return
        }

        formattedComparisonImage = try? ImagePreparation.formatToCanvas(
            referenceImage: original,
            settings: currentPreparationSettings(),
            targetWidth: output.width,
            targetHeight: output.height
        )
    }

    func persistSessionState() {
        let state = captureGUIState()
        switch workflow {
        case .imageToImage:
            Flux2AppSessionStore.saveImageToImage(state)
        case .textToImage:
            Flux2AppSessionStore.saveTextToImage(state)
        }
    }

    private func restoreSessionStateIfNeeded() {
        if loadsEnvironmentProject,
           let projectPath = ProcessInfo.processInfo.environment["F2SM_PROJECT"],
           !projectPath.isEmpty {
            return
        }

        let state: Flux2GenerationGUIState?
        switch workflow {
        case .imageToImage:
            state = Flux2AppSessionStore.loadImageToImage()
        case .textToImage:
            state = Flux2AppSessionStore.loadTextToImage()
        }

        guard let state else { return }
        applyGUIState(state, projectLoaded: currentProjectURL != nil)
    }

    private func captureGUIState() -> Flux2GenerationGUIState {
        Flux2GenerationGUIState(
            selectedFamily: selectedFamily?.rawValue,
            selectedModel: selectedModel.rawValue,
            textQuantization: textQuantization.rawValue,
            transformerQuantization: transformerQuantization.rawValue,
            prompt: prompt,
            upsamplePrompt: upsamplePrompt,
            clearPromptAfterGeneration: clearPromptAfterGeneration,
            width: width,
            height: height,
            steps: steps,
            guidance: guidance,
            seed: seed,
            sizingFavor: sizingFavor.rawValue,
            sizingMethod: sizingMethod.rawValue,
            preparationScale: preparationScale,
            preparationOverlayOpacity: preparationOverlayOpacity,
            megapixelBudget: megapixelBudget,
            contextAreaX: Double(contextArea.minX),
            contextAreaY: Double(contextArea.minY),
            contextAreaWidth: Double(contextArea.width),
            contextAreaHeight: Double(contextArea.height),
            processAreaX: processArea.map { Double($0.minX) },
            processAreaY: processArea.map { Double($0.minY) },
            processAreaWidth: processArea.map { Double($0.width) },
            processAreaHeight: processArea.map { Double($0.height) },
            hasProcessArea: processArea != nil,
            editMode: nil,
            inpaintMaskTool: inpaintMaskTool.rawValue,
            outpaintPadding: outpaintPadding.hasExpansion ? outpaintPadding : nil,
            inpaintIntent: inpaintIntent.rawValue,
            enrichInpaintPromptWithVLM: enrichInpaintPromptWithVLM,
            fillContextMaskScale: hasLocalFillSelection ? fillContextMaskScale : 0,
            interpretImagePaths: (try? interpretPathsForGeneration()) ?? [],
            showCheckpoints: showCheckpoints,
            checkpointInterval: checkpointInterval,
            previewZoomScale: previewZoomScale,
            previewComparisonSide: previewComparisonSide.rawValue,
            inputSaveSource: inputSaveSource.rawValue
        )
    }

    private func applyGUIState(_ state: Flux2GenerationGUIState, projectLoaded: Bool) {
        if !projectLoaded {
            skipNextModelDefaultApplication = true
            if let family = state.selectedFamily.flatMap(ModelFamily.init(rawValue:)) {
                selectedFamily = family
            }
            selectedModel = Flux2Model(rawValue: state.selectedModel) ?? selectedModel
            textQuantization = MistralQuantization(rawValue: state.textQuantization) ?? textQuantization
            transformerQuantization = TransformerQuantization(rawValue: state.transformerQuantization) ?? transformerQuantization
            prompt = state.prompt
            upsamplePrompt = state.upsamplePrompt
            clearPromptAfterGeneration = state.clearPromptAfterGeneration
            width = state.width
            height = state.height
            steps = state.steps
            guidance = state.guidance
            seed = state.seed

            if workflow == .imageToImage {
                bootstrapImageSlotsIfNeeded()
                if let index = imageSlots.indices.first {
                    if let favor = state.sizingFavor.flatMap(ImageSizingFavor.init(rawValue:)) {
                        imageSlots[index].sizingFavor = favor
                    }
                    if let method = state.sizingMethod.flatMap(ImageSizingMethod.init(rawValue:)) {
                        imageSlots[index].sizingMethod = method
                    }
                    if let scale = state.preparationScale {
                        imageSlots[index].preparationScale = max(0.1, min(1.0, scale))
                    }
                }
                preparationOverlayOpacity = state.preparationOverlayOpacity ?? preparationOverlayOpacity
                megapixelBudget = state.megapixelBudget ?? megapixelBudget
                contextArea = Self.clampUnitRect(CGRect(
                    x: state.contextAreaX ?? contextArea.minX,
                    y: state.contextAreaY ?? contextArea.minY,
                    width: state.contextAreaWidth ?? contextArea.width,
                    height: state.contextAreaHeight ?? contextArea.height
                ))
                if state.hasProcessArea,
                   let x = state.processAreaX,
                   let y = state.processAreaY,
                   let width = state.processAreaWidth,
                   let height = state.processAreaHeight {
                    processArea = Self.clampUnitRect(CGRect(x: x, y: y, width: width, height: height))
                } else {
                    processArea = nil
                }
                let legacyRoute = I2IGenerateRoute.fromLegacyProjectValue(state.editMode)
                if let toolRaw = state.inpaintMaskTool,
                   let tool = InpaintMaskTool(rawValue: toolRaw) {
                    inpaintMaskTool = tool
                } else if legacyRoute == .outpaint {
                    inpaintMaskTool = .cropCanvas
                }
                outpaintPadding = state.outpaintPadding ?? .zero
                outpaintCanvasIsDefined = outpaintPadding.hasExpansion
                if outpaintPadding.hasExpansion {
                    updateOutpaintPadding(outpaintPadding)
                }
                inpaintIntent = state.inpaintIntent.flatMap(Flux2InpaintIntent.init(rawValue:)) ?? inpaintIntent
                enrichInpaintPromptWithVLM = state.enrichInpaintPromptWithVLM ?? enrichInpaintPromptWithVLM
                fillContextMaskScale = state.fillContextMaskScale
                applySizingControls()
            }
        } else if workflow == .imageToImage {
            preparationOverlayOpacity = state.preparationOverlayOpacity ?? preparationOverlayOpacity
        }

        showCheckpoints = state.showCheckpoints
        checkpointInterval = max(1, state.checkpointInterval)
        previewZoomScale = min(max(state.previewZoomScale, 0.25), 8.0)
        previewComparisonSide = PreviewComparisonSide(rawValue: state.previewComparisonSide) ?? .processed
        if let input = ImageInputSaveSource(rawValue: state.inputSaveSource) {
            inputSaveSource = input
        }
    }

    /// Clear pipeline to free memory
    func clearPipeline() async {
        await pipeline?.clearAll()
        pipeline = nil
        Memory.clearCache()
        statusMessage = "Models unloaded"
    }

    /// Load Qwen3.5 VLM (4-bit) when generative fill requests VLM prompt enrichment.
    private func ensureQwen35VLMLoaded() async throws {
        if FluxTextEncoders.shared.isQwen35VLMLoaded { return }

        let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? UserDefaults.standard.string(forKey: "hfToken")

        let downloader = TextEncoderModelDownloader(hfToken: hfToken)
        let path = try await downloader.downloadQwen35(variant: .qwen35_4B_4bit) { progress, message in
            Task { @MainActor in
                guard self.isGenerating else { return }
                self.statusMessage = "Qwen3.5: \(message) (\(Int(progress * 100))%)"
            }
        }
        try await FluxTextEncoders.shared.loadQwen35VLM(from: path.path)
    }

    // MARK: - Recommended Defaults (Black Forest Labs)

    /// Apply recommended defaults for a model (from official HuggingFace pages)
    func applyRecommendedDefaults(for model: Flux2Model) {
        switch model {
        case .dev:
            // Flux.2 Dev - https://huggingface.co/black-forest-labs/FLUX.2-dev
            // 28 steps is "a good trade-off", guidance 4.0
            textQuantization = .mlx8bit
            transformerQuantization = .qint8
            width = 1024
            height = 1024
            steps = 28
            guidance = 4.0
            checkpointInterval = 7

        case .klein4B, .klein4BBase:
            // Flux.2 Klein 4B - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
            // 4 steps, guidance 1.0, optimized for sub-second generation
            transformerQuantization = .qint8
            width = 1024
            height = 1024
            steps = 4
            guidance = 1.0
            checkpointInterval = 1

        case .klein9B, .klein9BBase, .klein9BKV:
            // Flux.2 Klein 9B - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
            // 4 steps, guidance 1.0, sub-second generation
            transformerQuantization = .bf16  // Only bf16 available
            width = 1024
            height = 1024
            steps = 4
            guidance = 1.0
            checkpointInterval = 1
        }
    }

    // MARK: - Presets

    /// Apply a memory-efficient preset (Klein 4B at 512x512)
    func applyLightweightPreset() {
        selectedModel = .klein4B
        applyRecommendedDefaults(for: .klein4B)
        width = 512
        height = 512
    }

    /// Apply a balanced preset (Klein 4B at 1024x1024)
    func applyBalancedPreset() {
        selectedModel = .klein4B
        applyRecommendedDefaults(for: .klein4B)
    }

    /// Apply a high quality preset (Dev at 1024x1024)
    func applyHighQualityPreset() {
        selectedModel = .dev
        applyRecommendedDefaults(for: .dev)
    }

    // MARK: - Checkpoints

    /// Clear checkpoint images
    func clearCheckpoints() {
        checkpointImages.removeAll()
    }

    /// Add a checkpoint image
    func addCheckpoint(image: CGImage, step: Int) {
        let checkpoint = CheckpointImage(
            id: UUID(),
            image: image,
            step: step,
            timestamp: Date()
        )
        checkpointImages.append(checkpoint)
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
