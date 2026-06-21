/**
 * ImageGenerationViewModel.swift
 * ViewModel for Flux.2 image generation (T2I and I2I)
 */

import SwiftUI
import Flux2Core
import FluxTextEncoders
import CoreGraphics
import ImageIO
import MLX
import UniformTypeIdentifiers

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Generation Mode

enum GenerationMode: String, CaseIterable {
    case textToImage = "Text to Image"
    case imageToImage = "Image to Image"
}

enum ImageSizingFavor: String, CaseIterable, Identifiable {
    case original = "Original"
    case horizontal = "Horizontal"
    case vertical = "Vertical"

    var id: String { rawValue }
}

enum ImageSizingMethod: String, CaseIterable, Identifiable {
    case crop = "Crop"
    case pad = "Pad"

    var id: String { rawValue }
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
    @Published var referenceImages: [ReferenceImage] = []
    @Published var interpretImageURLs: [URL] = []  // VLM interpretation images
    @Published var sizingFavor: ImageSizingFavor = .original
    @Published var sizingMethod: ImageSizingMethod = .crop
    @Published var preparationScale: Double = 1.0
    @Published var preparationOverlayOpacity: Double = 0.22
    @Published var processArea: CGRect?
    @Published var contextArea: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)
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
            let clamped = min(max(megapixelBudget, Self.minMegapixelBudget), Self.maxMegapixelBudget)
            guard clamped == megapixelBudget else {
                megapixelBudget = clamped
                return
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
    private var skipNextModelDefaultApplication = false

    // MARK: - Init with defaults
    private let loadsEnvironmentProject: Bool

    init(loadsEnvironmentProject: Bool = false) {
        self.loadsEnvironmentProject = loadsEnvironmentProject
        applyRecommendedDefaults(for: selectedModel)
        if loadsEnvironmentProject {
            loadStartupProjectIfAvailable()
        } else {
            loadLastProjectIfAvailable()
        }
    }

    // MARK: - Computed Properties

    var seedValue: UInt64? {
        guard !seed.isEmpty else { return nil }
        return UInt64(seed)
    }

    var canGenerate: Bool {
        !prompt.isEmpty && !isPipelineBusy && isFamilySelected
    }

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
        selectedFamily?.pixelAlignment ?? Self.generationSizeMultiple
    }

    var compatibleTransformerQuantizations: [TransformerQuantization] {
        Self.compatibleTransformerQuantizations(for: selectedModel)
    }

    var adjustedGenerationSize: (width: Int, height: Int) {
        Self.referenceMatchedSize(width: width, height: height, maxArea: conditioningPixelBudget, multiple: pixelAlignment)
    }

    // Dormant until the pipeline has native mask/inpaint support.
    var processAreaDescription: String {
        guard let image = referenceImages.first?.image,
              let processArea else {
            return "Process: none selected"
        }

        let rect = pixelRect(from: processArea, in: image)
        return "Process: \(Int(rect.width))x\(Int(rect.height)) at x=\(Int(rect.minX)) y=\(Int(rect.minY))"
    }

    var contextAreaDescription: String {
        guard referenceImages.first?.image != nil else {
            return "Context: add a reference image"
        }

        let size = adjustedGenerationSize
        return "Context sent to model: \(size.width)x\(size.height)"
    }

    private struct PreparedImageToImageInput {
        let images: [CGImage]
        let width: Int
        let height: Int
        let compositionPlan: ImageCompositionPlan?
    }

    private struct ImageCompositionPlan {
        let originalImage: CGImage
        let contextRect: CGRect
        let processRect: CGRect
        let transform: ImagePreparationTransform
    }

    private struct ImagePreparationTransform {
        let targetWidth: Int
        let targetHeight: Int
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat

        func canvasRect(forSourceRect sourceRect: CGRect) -> CGRect {
            CGRect(
                x: sourceRect.minX * scale + offsetX,
                y: sourceRect.minY * scale + offsetY,
                width: sourceRect.width * scale,
                height: sourceRect.height * scale
            )
        }

        func sourceRect(forCanvasRect canvasRect: CGRect) -> CGRect {
            CGRect(
                x: (canvasRect.minX - offsetX) / scale,
                y: (canvasRect.minY - offsetY) / scale,
                width: canvasRect.width / scale,
                height: canvasRect.height / scale
            )
        }
    }

    private struct GenerationProject: Codable {
        var version = 1
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
        var preparationOverlayOpacity: Double
        var megapixelBudget: Double?
        var clearPromptAfterGeneration: Bool?
        var selectedFamily: String?
        var processArea: CodableRect?
        var contextArea: CodableRect
        var interpretImagePaths: [String]
        var referenceImages: [ProjectReferenceImage]
    }

    private struct ProjectReferenceImage: Codable {
        var sourcePath: String?
        var pngBase64: String
    }

    private struct CodableRect: Codable {
        var x: Double
        var y: Double
        var width: Double
        var height: Double

        init(_ rect: CGRect) {
            x = rect.minX
            y = rect.minY
            width = rect.width
            height = rect.height
        }

        var cgRect: CGRect {
            CGRect(x: x, y: y, width: width, height: height)
        }
    }

    private static let lastProjectURLKey = "lastGenerationProjectURL"
    /// Absolute path to a generation project JSON file. When set, opens on launch
    /// instead of restoring `lastGenerationProjectURL` (VM smoke / agent hooks).
    private static let projectEnvironmentKey = "F2SM_PROJECT"
    /// When set alongside `F2SM_PROJECT`, writes `ok` or `error` plus detail on load.
    private static let smokeMarkerEnvironmentKey = "F2SM_SMOKE_MARKER"

    // MARK: - Image Management

    /// Add a reference image from URL using CGImageSource (pixel-exact, no NSImage re-rendering)
    func addReferenceImage(from url: URL) {
        guard referenceImages.count < selectedModel.maxReferenceImages else { return }

        // Use CGImageSource for pixel-exact loading (avoids NSImage roundtrip shifts)
        guard let data = try? Data(contentsOf: url),
              let cgImage = Self.cgImageFromData(data) else {
            errorMessage = "Failed to load image from \(url.lastPathComponent)"
            return
        }

        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        let refImage = ReferenceImage(
            id: UUID(),
            url: url,
            image: cgImage,
            thumbnail: createThumbnail(from: nsImage)
        )
        referenceImages.append(refImage)
        ensurePreparationDefaults()
    }

    /// Add a reference image from NSImage (drag & drop)
    /// Uses tiffRepresentation + CGImageSource to avoid cgImage(forProposedRect:) re-rendering
    func addReferenceImage(from nsImage: NSImage) {
        guard referenceImages.count < selectedModel.maxReferenceImages else { return }

        // Convert via TIFF data + CGImageSource to avoid cgImage(forProposedRect:) shifts
        guard let tiffData = nsImage.tiffRepresentation,
              let cgImage = Self.cgImageFromData(tiffData) else {
            errorMessage = "Failed to process dropped image"
            return
        }

        let refImage = ReferenceImage(
            id: UUID(),
            url: nil,
            image: cgImage,
            thumbnail: createThumbnail(from: nsImage)
        )
        referenceImages.append(refImage)
        ensurePreparationDefaults()
    }

    /// Add a reference image directly from a CGImage (no NSImage roundtrip)
    /// Used by "Use as Reference" to avoid pixel shifts on iterative I2I cycles
    func addReferenceImage(cgImage: CGImage) {
        guard referenceImages.count < selectedModel.maxReferenceImages else { return }

        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        let refImage = ReferenceImage(
            id: UUID(),
            url: nil,
            image: cgImage,
            thumbnail: createThumbnail(from: nsImage)
        )
        referenceImages.append(refImage)
        ensurePreparationDefaults()
    }

    /// Remove a reference image
    func removeReferenceImage(_ id: UUID) {
        referenceImages.removeAll { $0.id == id }
        if referenceImages.isEmpty {
            processArea = nil
            contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        } else {
            ensurePreparationDefaults()
        }
    }

    /// Clear all reference images
    func clearReferenceImages() {
        referenceImages.removeAll()
        processArea = nil
        contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
    }

    func applySizingControls() {
        guard let image = referenceImages.first?.image else { return }

        let sourceRect = integralPixelRect(from: contextArea, in: image)
        let sourceAspect = Double(sourceRect.width) / Double(sourceRect.height)
        let size = budgetFilledSize(sourceAspect: sourceAspect)
        width = size.width
        height = size.height
    }

    /// Size that fills the megapixel budget at `sourceAspect` (shaped by the
    /// Favour control), snapped to the family's pixel-alignment factor. The MP
    /// budget — not the crop's pixel count — is the target, so a small barn-door
    /// region no longer shrinks the output; it just implies the conditioning crop
    /// is upscaled (see conditioningUpscaleFactor).
    private func budgetFilledSize(sourceAspect: Double) -> (width: Int, height: Int) {
        let targetAspect: Double
        switch sizingFavor {
        case .original:
            targetAspect = sourceAspect
        case .horizontal:
            targetAspect = max(sourceAspect, 4.0 / 3.0)
        case .vertical:
            targetAspect = min(sourceAspect, 3.0 / 4.0)
        }

        let scale = min(max(preparationScale, 0.1), 1.0)
        let pixelBudget = Double(conditioningPixelBudget) * scale * scale
        let rawHeight = (pixelBudget / max(targetAspect, 0.0001)).squareRoot()
        let rawWidth = rawHeight * targetAspect

        return (
            Self.snapToMultiple(Int(rawWidth.rounded()), multiple: pixelAlignment),
            Self.snapToMultiple(Int(rawHeight.rounded()), multiple: pixelAlignment)
        )
    }

    /// Linear factor by which the barn-door crop must be scaled to fill the
    /// budget. > 1 means the selected region is smaller than the target and gets
    /// upscaled (a softer reference); surfaced as a gentle advisory.
    var conditioningUpscaleFactor: Double {
        guard let image = referenceImages.first?.image else { return 1 }
        let sourceRect = integralPixelRect(from: contextArea, in: image)
        let sourcePixels = Double(sourceRect.width * sourceRect.height)
        guard sourcePixels > 0 else { return 1 }
        let target = adjustedGenerationSize
        let targetPixels = Double(target.width * target.height)
        return (targetPixels / sourcePixels).squareRoot()
    }


    func setSizingFavor(_ favor: ImageSizingFavor) {
        sizingFavor = favor
        applySizingControls()
    }

    func setSizingMethod(_ method: ImageSizingMethod) {
        sizingMethod = method
        applySizingControls()
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

    /// Reopen the barn doors (full image) and clear any selection.
    func resetContextArea() {
        contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        processArea = nil
        applySizingControls()
    }

    /// Update the barn-door (context) region from a barn-door edge drag.
    func setContextArea(_ rect: CGRect) {
        contextArea = Self.clampUnitRect(rect)
        applySizingControls()
    }

    /// Set or update the process selection (the dashed marquee).
    func setProcessArea(_ rect: CGRect?) {
        guard let rect else {
            processArea = nil
            return
        }
        processArea = Self.clampUnitRect(rect)
    }

    func clearProcessSelection() {
        processArea = nil
    }

    // Dormant until the pipeline has native mask/inpaint support.
    func resetProcessContext() {
        resetContextArea()
    }

    private func ensurePreparationDefaults() {
        guard !referenceImages.isEmpty else { return }
        contextArea = Self.clampUnitRect(contextArea)
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
        let minSize: CGFloat = 0.01
        let width = min(max(rect.width, minSize), 1)
        let height = min(max(rect.height, minSize), 1)
        let x = min(max(rect.minX, 0), 1 - width)
        let y = min(max(rect.minY, 0), 1 - height)
        return CGRect(x: x, y: y, width: width, height: height)
    }

    /// Reference (context) images and the generated output bind 1:1 by raw
    /// patch index in the transformer (their position IDs differ only in the
    /// time coordinate — see `LatentUtils.generate*PositionIDs`). If the two
    /// grids differ in size the regenerated content binds to the output's
    /// top-left corner and lands offset when composited back (a clean seam at
    /// the context boundary). The generation target and the pipeline's reference
    /// conditioning area are therefore derived identically — same pixel budget,
    /// same alignment floor — so output == reference at any budget.
    static let generationSizeMultiple = 32
    /// Legacy/default conditioning budget (1 MP). The live budget now comes from
    /// `megapixelBudget`; this stays as the fallback for callers without one.
    static let referenceConditioningArea = 1024 * 1024
    static let minMegapixelBudget: Double = 0.25
    static let maxMegapixelBudget: Double = 4.0

    /// Total pixels the model conditions on and generates at, taken from the MP
    /// budget. Drives both the output size and the pipeline's reference cap so
    /// the two stay equal (composite alignment).
    var conditioningPixelBudget: Int {
        Int((min(max(megapixelBudget, Self.minMegapixelBudget), Self.maxMegapixelBudget) * 1_000_000).rounded())
    }

    static func snapToMultiple(_ value: Int, multiple: Int) -> Int {
        guard multiple > 0 else { return value }
        return max(multiple, ((value + multiple - 1) / multiple) * multiple)
    }

    /// The size the model actually reconstructs the context at, mirroring
    /// `Flux2Pipeline.encodeReferenceImages` exactly: scale down to the
    /// conditioning budget if needed, then floor to the alignment factor.
    /// Using this as the generation target keeps the output grid identical to
    /// the reference grid so composited patches stay aligned with the original.
    static func referenceMatchedSize(width: Int, height: Int, maxArea: Int = referenceConditioningArea, multiple: Int = generationSizeMultiple) -> (width: Int, height: Int) {
        var w = Double(max(1, width))
        var h = Double(max(1, height))
        let area = w * h
        if area > Double(maxArea) {
            let scale = (Double(maxArea) / area).squareRoot()
            w *= scale
            h *= scale
        }
        let flooredWidth = max(multiple, (Int(w) / multiple) * multiple)
        let flooredHeight = max(multiple, (Int(h) / multiple) * multiple)
        return (flooredWidth, flooredHeight)
    }

    private func prepareImageToImageInput() throws -> PreparedImageToImageInput {
        guard let firstReference = referenceImages.first else {
            throw Flux2Error.invalidConfiguration("Add a reference image before generating")
        }

        let original = firstReference.image
        let contextRect = integralPixelRect(from: contextArea, in: original)
        // Flux.2 has no mask input, so the model still sees the full context.
        // The paste-back step uses the dashed process selection when present so
        // edge artifacts from the generated context do not get stitched in.
        let processRect = integralProcessRect(in: original, contextRect: contextRect)
        let targetSize = adjustedGenerationSize

        let contextImage = try cropImage(original, to: contextRect)
        let transform = preparationTransform(
            sourceWidth: contextImage.width,
            sourceHeight: contextImage.height,
            targetWidth: targetSize.width,
            targetHeight: targetSize.height,
            method: sizingMethod
        )
        let preparedFirstImage = try renderImage(contextImage, targetWidth: targetSize.width, targetHeight: targetSize.height, transform: transform)

        let remainingImages = referenceImages.dropFirst().map(\.image)
        let plan = ImageCompositionPlan(
            originalImage: original,
            contextRect: contextRect,
            processRect: processRect,
            transform: transform
        )

        return PreparedImageToImageInput(
            images: [preparedFirstImage] + remainingImages,
            width: targetSize.width,
            height: targetSize.height,
            compositionPlan: plan
        )
    }

    private func compositeGeneratedImage(_ generatedImage: CGImage, using plan: ImageCompositionPlan) throws -> CGImage {
        let processInContext = plan.processRect.offsetBy(dx: -plan.contextRect.minX, dy: -plan.contextRect.minY)
        let mappedProcessRect = plan.transform.canvasRect(forSourceRect: processInContext)
        let canvasBounds = CGRect(x: 0, y: 0, width: generatedImage.width, height: generatedImage.height)
        let visibleCanvasRect = mappedProcessRect.intersection(canvasBounds)

        guard !visibleCanvasRect.isNull, visibleCanvasRect.width > 0, visibleCanvasRect.height > 0 else {
            throw Flux2Error.imageProcessingFailed("Process area falls outside the generated canvas")
        }

        let generatedCropRect = integralPixelRect(visibleCanvasRect, imageWidth: generatedImage.width, imageHeight: generatedImage.height)
        let generatedPatch = try cropImage(generatedImage, to: generatedCropRect)
        let visibleSourceRect = plan.transform.sourceRect(forCanvasRect: generatedCropRect)
        let destinationRect = integralPixelRect(
            visibleSourceRect.offsetBy(dx: plan.contextRect.minX, dy: plan.contextRect.minY),
            imageWidth: plan.originalImage.width,
            imageHeight: plan.originalImage.height
        )

        guard let context = makeImageContext(width: plan.originalImage.width, height: plan.originalImage.height) else {
            throw Flux2Error.imageProcessingFailed("Failed to create composition context")
        }

        context.interpolationQuality = .high
        context.draw(
            plan.originalImage,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: CGRect(x: 0, y: 0, width: plan.originalImage.width, height: plan.originalImage.height),
                canvasHeight: CGFloat(plan.originalImage.height)
            )
        )
        context.draw(
            generatedPatch,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: destinationRect,
                canvasHeight: CGFloat(plan.originalImage.height)
            )
        )

        guard let compositedImage = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to composite generated patch")
        }

        return compositedImage
    }

    private func preparationTransform(
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        method: ImageSizingMethod
    ) -> ImagePreparationTransform {
        let xScale = CGFloat(targetWidth) / CGFloat(max(sourceWidth, 1))
        let yScale = CGFloat(targetHeight) / CGFloat(max(sourceHeight, 1))
        let scale = method == .crop ? max(xScale, yScale) : min(xScale, yScale)
        let drawnWidth = CGFloat(sourceWidth) * scale
        let drawnHeight = CGFloat(sourceHeight) * scale

        return ImagePreparationTransform(
            targetWidth: targetWidth,
            targetHeight: targetHeight,
            scale: scale,
            offsetX: (CGFloat(targetWidth) - drawnWidth) / 2,
            offsetY: (CGFloat(targetHeight) - drawnHeight) / 2
        )
    }

    private func renderImage(_ image: CGImage, targetWidth: Int, targetHeight: Int, transform: ImagePreparationTransform) throws -> CGImage {
        guard let context = makeImageContext(width: targetWidth, height: targetHeight) else {
            throw Flux2Error.imageProcessingFailed("Failed to create prepared image context")
        }

        context.interpolationQuality = .high
        let drawRect = CGRect(
            x: transform.offsetX,
            y: transform.offsetY,
            width: CGFloat(image.width) * transform.scale,
            height: CGFloat(image.height) * transform.scale
        )
        context.draw(
            image,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: drawRect,
                canvasHeight: CGFloat(targetHeight)
            )
        )

        guard let renderedImage = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to render prepared image")
        }

        return renderedImage
    }

    private func cropImage(_ image: CGImage, to rect: CGRect) throws -> CGImage {
        let cropRect = integralPixelRect(rect, imageWidth: image.width, imageHeight: image.height)

        if let cropped = image.cropping(to: cropRect) {
            return cropped
        }

        guard let context = makeImageContext(width: Int(cropRect.width), height: Int(cropRect.height)) else {
            throw Flux2Error.imageProcessingFailed("Failed to create crop context")
        }

        let sourceDrawRect = CGRect(
            x: -cropRect.minX,
            y: -cropRect.minY,
            width: CGFloat(image.width),
            height: CGFloat(image.height)
        )
        context.draw(
            image,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: sourceDrawRect,
                canvasHeight: cropRect.height
            )
        )

        guard let croppedImage = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to crop image")
        }

        return croppedImage
    }

    private func integralProcessRect(in image: CGImage, contextRect: CGRect) -> CGRect {
        guard let processArea else {
            return contextRect
        }

        let rawProcessRect = pixelRect(from: Self.clampUnitRect(processArea), in: image)
        let clampedProcessRect = rawProcessRect.intersection(contextRect)

        guard !clampedProcessRect.isNull, clampedProcessRect.width > 0, clampedProcessRect.height > 0 else {
            return contextRect
        }

        return integralPixelRect(clampedProcessRect, imageWidth: image.width, imageHeight: image.height)
    }

    private func integralPixelRect(from normalizedRect: CGRect, in image: CGImage) -> CGRect {
        integralPixelRect(
            pixelRect(from: Self.clampUnitRect(normalizedRect), in: image),
            imageWidth: image.width,
            imageHeight: image.height
        )
    }

    private func integralPixelRect(_ rect: CGRect, imageWidth: Int, imageHeight: Int) -> CGRect {
        let imageBounds = CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight)
        let bounded = rect.intersection(imageBounds)

        guard !bounded.isNull, bounded.width > 0, bounded.height > 0 else {
            return CGRect(x: 0, y: 0, width: max(1, min(imageWidth, 1)), height: max(1, min(imageHeight, 1)))
        }

        let minX = floor(bounded.minX)
        let minY = floor(bounded.minY)
        let maxX = ceil(bounded.maxX)
        let maxY = ceil(bounded.maxY)

        return CGRect(
            x: min(max(minX, 0), CGFloat(max(imageWidth - 1, 0))),
            y: min(max(minY, 0), CGFloat(max(imageHeight - 1, 0))),
            width: max(1, min(maxX, CGFloat(imageWidth)) - minX),
            height: max(1, min(maxY, CGFloat(imageHeight)) - minY)
        )
    }

    private func makeImageContext(width: Int, height: Int) -> CGContext? {
        CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    }

    /// Decode image data using CGImageSource for pixel-exact results
    private static func cgImageFromData(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private func createThumbnail(from image: NSImage) -> NSImage {
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
        sizingFavor = .original
        sizingMethod = .crop
        preparationScale = 1.0
        megapixelBudget = 1.0
        preparationOverlayOpacity = 0.22
        processArea = nil
        contextArea = CGRect(x: 0, y: 0, width: 1, height: 1)
        referenceImages.removeAll()
        interpretImageURLs.removeAll()
        generatedImage = nil
        checkpointImages.removeAll()
        errorMessage = nil
        currentProjectURL = nil
        lastSavedImageURL = nil
        UserDefaults.standard.removeObject(forKey: Self.lastProjectURLKey)
        applyRecommendedDefaults(for: selectedModel)
        statusMessage = "New project"
    }

    func saveProject() {
        do {
            let url: URL
            if let currentProjectURL {
                url = currentProjectURL
            } else {
                let panel = NSSavePanel()
                panel.allowedContentTypes = [.json]
                panel.nameFieldStringValue = "flux_project.json"
                guard panel.runModal() == .OK, let selectedURL = panel.url else {
                    return
                }
                url = selectedURL
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
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = currentProjectURL?.lastPathComponent ?? "flux_project.json"

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        do {
            try saveProject(to: url)
            currentProjectURL = url
            UserDefaults.standard.set(url.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Saved project to \(url.lastPathComponent)"
        } catch {
            errorMessage = "Failed to save project: \(error.localizedDescription)"
        }
    }

    func openProject() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        do {
            try loadProject(from: url)
            currentProjectURL = url
            UserDefaults.standard.set(url.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Opened project \(url.lastPathComponent)"
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
            currentProjectURL = url
            UserDefaults.standard.set(url.path, forKey: Self.lastProjectURLKey)
            statusMessage = "Opened project \(url.lastPathComponent) (F2SM_PROJECT)"
            writeSmokeMarker(
                outcome: "ok",
                detail: "project=\(url.path)\nreferences=\(referenceImages.count)\nprompt=\(prompt)"
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
            currentProjectURL = url
            statusMessage = "Opened last project \(url.lastPathComponent)"
        } catch {
            // Last-project restore should not block a fresh app launch.
            currentProjectURL = nil
        }
    }

    private func saveProject(to url: URL) throws {
        let project = try makeProject()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(project)
        try data.write(to: url, options: .atomic)
    }

    private func loadProject(from url: URL) throws {
        lastSavedImageURL = nil
        let data = try Data(contentsOf: url)
        let project = try JSONDecoder().decode(GenerationProject.self, from: data)
        let restoredReferences = try project.referenceImages.map { stored in
            try referenceImage(from: stored)
        }

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
        sizingFavor = ImageSizingFavor(rawValue: project.sizingFavor) ?? .original
        sizingMethod = ImageSizingMethod(rawValue: project.sizingMethod) ?? .crop
        preparationScale = max(0.1, min(1.0, project.preparationScale ?? 1.0))
        preparationOverlayOpacity = project.preparationOverlayOpacity
        megapixelBudget = min(max(project.megapixelBudget ?? 1.0, Self.minMegapixelBudget), Self.maxMegapixelBudget)
        clearPromptAfterGeneration = project.clearPromptAfterGeneration ?? false
        processArea = project.processArea?.cgRect
        contextArea = Self.clampUnitRect(project.contextArea.cgRect)
        interpretImageURLs = project.interpretImagePaths.map { URL(fileURLWithPath: $0) }
        referenceImages = restoredReferences
        generatedImage = nil
        upsampledPrompt = nil
        checkpointImages.removeAll()
        errorMessage = nil
        // Recompute the budget-driven generation size from the loaded barn doors.
        applySizingControls()
    }

    private func makeProject() throws -> GenerationProject {
        GenerationProject(
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
            sizingFavor: sizingFavor.rawValue,
            sizingMethod: sizingMethod.rawValue,
            preparationScale: preparationScale,
            preparationOverlayOpacity: preparationOverlayOpacity,
            megapixelBudget: megapixelBudget,
            clearPromptAfterGeneration: clearPromptAfterGeneration,
            selectedFamily: selectedFamily?.rawValue,
            processArea: processArea.map(CodableRect.init),
            contextArea: CodableRect(contextArea),
            interpretImagePaths: interpretImageURLs.map(\.path),
            referenceImages: try referenceImages.map { reference in
                ProjectReferenceImage(
                    sourcePath: reference.url?.path,
                    pngBase64: try pngData(from: reference.image).base64EncodedString()
                )
            }
        )
    }

    private func referenceImage(from stored: ProjectReferenceImage) throws -> ReferenceImage {
        guard let data = Data(base64Encoded: stored.pngBase64),
              let cgImage = Self.cgImageFromData(data) else {
            throw NSError(domain: "FluxProject", code: 1, userInfo: [NSLocalizedDescriptionKey: "Project contains an invalid reference image"])
        }

        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        let url = stored.sourcePath.map { URL(fileURLWithPath: $0) }
        return ReferenceImage(
            id: UUID(),
            url: url,
            image: cgImage,
            thumbnail: createThumbnail(from: nsImage)
        )
    }

    private func pngData(from image: CGImage) throws -> Data {
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
            let interpretPaths = interpretImageURLs.map { $0.path }

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

            if referenceImages.isEmpty {
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
            } else {
                // Image-to-Image
                statusMessage = "Preparing image-to-image input..."
                let preparedInput = try prepareImageToImageInput()

                statusMessage = "Generating with \(preparedInput.images.count) reference image(s)..."
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
                generatedImage = image
                statusMessage = "Generation complete!"
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

    /// Open the configured output folder in Finder, selecting the last save when known.
    func openOutputFolder() {
        #if canImport(AppKit)
        do {
            if let url = lastSavedImageURL {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } else {
                let directory = try ImageSaveService.outputDirectory()
                NSWorkspace.shared.open(directory)
            }
        } catch {
            errorMessage = "Failed to open output folder: \(error.localizedDescription)"
        }
        #endif
    }

    /// Clear the generated image and checkpoints from the preview pane.
    func clearPreview() {
        generatedImage = nil
        checkpointImages.removeAll()
        statusMessage = "Preview cleared"
    }

    /// Save the formatted input image (Image Formatting applied to the whole
    /// reference, ignoring the barn doors) next to the last saved output, sharing
    /// its name + extension with a "-input" suffix.
    func saveInputImage() {
        guard let outputURL = lastSavedImageURL else {
            errorMessage = "Save the generated image first so the input can share its name."
            return
        }

        do {
            let input = try formattedFullInputImage()
            let url = try ImageSaveService.saveCompanion(input, alongside: outputURL, suffix: "-input")
            statusMessage = "Saved input to \(url.lastPathComponent)"
        } catch {
            errorMessage = "Failed to save input: \(error.localizedDescription)"
        }
    }

    /// The reference image formatted per Image Formatting (Favour + crop/pad)
    /// across the *whole* frame — i.e. before the barn doors narrow it. Mirrors
    /// the output's budget-filled, alignment-floored sizing so the saved input
    /// shares the model's grid.
    func formattedFullInputImage() throws -> CGImage {
        guard let original = referenceImages.first?.image else {
            throw Flux2Error.invalidConfiguration("Add a reference image before saving the input")
        }

        let fullAspect = Double(original.width) / Double(max(original.height, 1))
        let pre = budgetFilledSize(sourceAspect: fullAspect)
        let targetSize = Self.referenceMatchedSize(
            width: pre.width,
            height: pre.height,
            maxArea: conditioningPixelBudget,
            multiple: pixelAlignment
        )
        let transform = preparationTransform(
            sourceWidth: original.width,
            sourceHeight: original.height,
            targetWidth: targetSize.width,
            targetHeight: targetSize.height,
            method: sizingMethod
        )
        return try renderImage(original, targetWidth: targetSize.width, targetHeight: targetSize.height, transform: transform)
    }

    /// Clear pipeline to free memory
    func clearPipeline() async {
        await pipeline?.clearAll()
        pipeline = nil
        Memory.clearCache()
        statusMessage = "Models unloaded"
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
