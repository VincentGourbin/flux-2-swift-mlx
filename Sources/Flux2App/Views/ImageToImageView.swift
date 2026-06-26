/**
 * ImageToImageView.swift
 * Image-to-Image generation interface for Flux.2
 */

import SwiftUI
import Flux2Core
import Flux2Chains
import FluxTextEncoders
import UniformTypeIdentifiers

#if canImport(AppKit)
import AppKit
#endif

struct ImageToImageView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = ImageGenerationViewModel(loadsEnvironmentProject: true, workflow: .imageToImage)
    @StateObject private var paletteCoordinator = PaletteDetachCoordinator()
    @AppStorage("imageSaveUpscaleBy") private var imageSaveUpscaleBy = 1.0

    // Kept-but-hidden controls: the megapixel budget + barn doors now own
    // resolution and dimensions, so these are off by default (not deleted).
    private static let showScalingControl = false
    private static let showManualDimensions = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            HSplitView {
                VStack(spacing: 0) {
                    ImageToImageCanvasToolsSidebar(viewModel: viewModel)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(Color(nsColor: .windowBackgroundColor).opacity(0.92))

                    Divider()

                    ScrollView {
                        VStack(alignment: .leading, spacing: 10) {
                            PalettePanel(
                                storageKey: "i2i.images",
                                title: "Images",
                                systemImage: "photo.stack",
                                coordinator: paletteCoordinator,
                                headerTrailing: { imagesHeaderTrailing },
                                content: { ImagesPaletteView(viewModel: viewModel) }
                            )

                            PalettePanel(
                                storageKey: "i2i.workflow",
                                title: "Workflow",
                                systemImage: "arrow.triangle.branch",
                                coordinator: paletteCoordinator,
                                content: { workflowContextPaletteContent }
                            )

                            PalettePanel(
                                storageKey: "i2i.parameters",
                                title: "Generation Parameters",
                                systemImage: "slider.horizontal.3",
                                coordinator: paletteCoordinator,
                                content: { parametersPaletteContent }
                            )

                            PalettePanel(
                                storageKey: "i2i.outputOptions",
                                title: "Output Options",
                                systemImage: "slider.horizontal.below.rectangle",
                                coordinator: paletteCoordinator,
                                content: { outputOptionsPaletteContent }
                            )
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .frame(minWidth: 380, idealWidth: 450, maxWidth: 550)
                .clipped()

                outputSection
            }

            floatingPalettes
        }
        .onAppear {
            modelManager.refreshDownloadedModels()
            modelManager.refreshDownloadedDiffusionModels()
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: modelManager.downloadedTransformers,
                downloadedTextModels: modelManager.downloadedModels
            )
        }
        .onChange(of: viewModel.selectedModel) { _, newModel in
            if viewModel.shouldApplyDefaultsForModelChange() {
                viewModel.applyRecommendedDefaults(for: newModel)
            }
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: modelManager.downloadedTransformers,
                downloadedTextModels: modelManager.downloadedModels
            )
        }
        .onChange(of: modelManager.downloadedTransformers) { _, downloaded in
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: downloaded,
                downloadedTextModels: modelManager.downloadedModels
            )
        }
        .onChange(of: modelManager.downloadedModels) { _, downloaded in
            viewModel.enforceAvailableModelDefaults(
                downloadedTransformers: modelManager.downloadedTransformers,
                downloadedTextModels: downloaded
            )
        }
        .focusedSceneValue(\.generationProjectCommands, GenerationProjectCommands(
            newProject: { viewModel.newProject() },
            openProject: { viewModel.openProject() },
            saveProject: { viewModel.saveProject() },
            saveProjectAs: { viewModel.saveProjectAs() }
        ))
        .focusedSceneValue(\.generationProjectName, viewModel.projectDisplayName)
        .focusedSceneValue(\.generationModelConfiguration, viewModel)
        .onChange(of: viewModel.generateRoute) { _, route in
            if route == .localFill {
                viewModel.upsamplePrompt = false
            }
        }
        .onChange(of: viewModel.hasActiveSelection) { _, active in
            if active {
                viewModel.enrichInpaintPromptWithVLM = true
                viewModel.inpaintIntent = .fill
                viewModel.fillContextMaskScale = 0
            }
        }
        .onChange(of: viewModel.enrichInpaintPromptWithVLM) { _, _ in
            viewModel.clearActiveToolIfDisabled()
        }
        .focusedSceneValue(\.generationUnloadModels) {
            Task { await viewModel.clearPipeline() }
        }
        .onDisappear {
            viewModel.persistSessionState()
        }
        .onReceive(NotificationCenter.default.publisher(for: .flux2PersistSession)) { _ in
            viewModel.persistSessionState()
        }
    }

    // MARK: - Palette titles & floating overlays

    private var imagesHeaderTrailing: some View {
        Group {
            if viewModel.assignedImageCount > 0 {
                Button(action: { viewModel.clearAllImageSlots() }) {
                    Label("Clear All", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .labelStyle(.titleOnly)
            }
        }
    }

    @ViewBuilder
    private var floatingPalettes: some View {
        if paletteCoordinator.isDetached("i2i.images") {
            PaletteFloatingPanel(
                storageKey: "i2i.images",
                title: "Images",
                systemImage: "photo.stack",
                position: paletteCoordinator.positionBinding(for: "i2i.images"),
                coordinator: paletteCoordinator,
                headerTrailing: { imagesHeaderTrailing },
                content: { ImagesPaletteView(viewModel: viewModel) }
            )
        }
        if paletteCoordinator.isDetached("i2i.workflow") {
            PaletteFloatingPanel(
                storageKey: "i2i.workflow",
                title: "Workflow",
                systemImage: "arrow.triangle.branch",
                position: paletteCoordinator.positionBinding(for: "i2i.workflow"),
                coordinator: paletteCoordinator,
                content: { workflowContextPaletteContent }
            )
        }
        if paletteCoordinator.isDetached("i2i.parameters") {
            PaletteFloatingPanel(
                storageKey: "i2i.parameters",
                title: "Generation Parameters",
                systemImage: "slider.horizontal.3",
                position: paletteCoordinator.positionBinding(for: "i2i.parameters"),
                coordinator: paletteCoordinator,
                content: { parametersPaletteContent }
            )
        }
        if paletteCoordinator.isDetached("i2i.outputOptions") {
            PaletteFloatingPanel(
                storageKey: "i2i.outputOptions",
                title: "Output Options",
                systemImage: "slider.horizontal.below.rectangle",
                position: paletteCoordinator.positionBinding(for: "i2i.outputOptions"),
                coordinator: paletteCoordinator,
                content: { outputOptionsPaletteContent }
            )
        }
    }

    // MARK: - Contextual controls (inferred from tool + selection)

    @ViewBuilder
    private var workflowContextPaletteContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !viewModel.hasPrimaryReference {
                Text("Add a primary reference image to use workflow tools on the preview.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                if viewModel.assignedImageCount > 0 {
                    Text(viewModel.imageAssignmentSummary)
                        .font(.caption.bold())
                    if viewModel.assignedReferenceCount > viewModel.selectedModel.maxReferenceImages {
                        Text("Too many reference images for \(viewModel.selectedModel.displayName) (max \(viewModel.selectedModel.maxReferenceImages)).")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                    Divider()
                }

                switch viewModel.generateRoute {
                case .fullImage:
                    Text("Barn doors on the preview define the Live Area (context mask for generation). Select the Live Area tool to adjust them. The megapixel budget in Generation Parameters sets output resolution at that aspect ratio.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                case .localFill:
                    selectionControls
                case .outpaint:
                    Text(viewModel.outpaintCanvasDescription)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Drag the outer canvas edges on the preview. Padding snaps to 32 px. Megapixel budget in Generation Parameters caps the expanded canvas size.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    Button("Reset Canvas") {
                        viewModel.resetOutpaintCanvas()
                    }
                    .controlSize(.small)
                    .disabled(!viewModel.outpaintCanvasIsDefined)
                }
            }
        }
    }

    @ViewBuilder
    private var selectionControls: some View {
        if viewModel.enrichInpaintPromptWithVLM {
            Text("Selection = where to edit. Context mask = what Qwen sees when writing the prompt.")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
        }

        Text("Hold ⇧ to add to the selection and ⌥ to subtract. Click the canvas without drawing to clear.")
            .font(.caption)
            .foregroundStyle(.secondary)

        Picker("Intent", selection: $viewModel.inpaintIntent) {
            ForEach(Flux2InpaintIntent.allCases, id: \.self) { intent in
                Text(intent.displayName).tag(intent)
            }
        }
        .pickerStyle(.segmented)

        Text(viewModel.inpaintIntent.fillHelp)
            .font(.caption)
            .foregroundStyle(.secondary)

        if viewModel.enrichInpaintPromptWithVLM, viewModel.hasFillMask {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Context Mask Size")
                        .font(.caption)
                    Spacer()
                    if viewModel.showsFillContextMaskOverlay {
                        Text("Custom")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Auto")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                Slider(
                    value: $viewModel.fillContextMaskScale,
                    in: -1...1
                )
                HStack {
                    Text("Tight")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("Auto")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("Full image")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            Text("Live Area barn doors are hidden while a selection is active. Move the slider to preview the Qwen context frame (white overlay).")
                .font(.caption)
                .foregroundStyle(.secondary)
        }

        if viewModel.inpaintMaskTool == .polygon {
            HStack(spacing: 8) {
                if viewModel.draftPolygonPoints.count >= 3 {
                    Button("Close polygon") {
                        viewModel.closeDraftPolygon()
                    }
                    .controlSize(.small)
                }
                Text(polygonDraftHelp)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }

        if viewModel.inpaintMaskTool == .visionSubject {
            Text("Drag a lasso around the subject. Vision turns the hint into a marching-ants selection you can add to, subtract from, or undo.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }

        if let visionStatus = viewModel.visionSubjectStatusMessage {
            Text(visionStatus)
                .font(.caption)
                .foregroundStyle(visionStatus.contains("No subject") ? .orange : .secondary)
        }

        if !viewModel.hasFillMask {
            Text("Draw a selection on the preview before generating.")
                .font(.caption)
                .foregroundStyle(.orange)
        } else {
            Text(viewModel.processAreaDescription)
                .font(.caption)
                .foregroundStyle(.secondary)

            Button("Reset Selection") {
                viewModel.clearProcessSelection()
            }
            .controlSize(.small)
        }
    }

    // MARK: - Edit Mode Section (removed — kept for reference in git history)

    private var polygonDraftHelp: String {
        let count = viewModel.draftPolygonPoints.count
        if count == 0 { return "Click corners on the preview." }
        if count < 3 { return "\(count) point\(count == 1 ? "" : "s") — need at least 3." }
        return "\(count) points — Close polygon when ready."
    }

    private var visionSubjectDraftHelp: String {
        switch viewModel.inpaintMaskTool {
        case .visionSubject:
            if viewModel.draftPolygonPoints.count >= 3 {
                return "Close polygon to find the subject inside it."
            }
            return "Drag a box around the subject, or click polygon corners."
        default:
            return ""
        }
    }

    // MARK: - Megapixel budget / resolution cap

    private var megapixelBudgetHelp: String {
        switch viewModel.generateRoute {
        case .fullImage:
            return "Maximum total pixels to generate. Barn doors set the aspect ratio; this budget sets resolution."
        case .localFill:
            return "Maximum total pixels for the selection fill pass. Larger images are scaled down before denoising."
        case .outpaint:
            return "Caps the expanded canvas size for the outpaint pass."
        }
    }

    @ViewBuilder
    private func megapixelBudgetControls(help: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ResetOnDoubleClickSlider(
                    value: $viewModel.megapixelBudget,
                    range: ImageGenerationViewModel.minMegapixelBudget...ImageGenerationViewModel.maxMegapixelBudget,
                    step: 0.25,
                    defaultValue: ImageGenerationViewModel.defaultMegapixelBudget
                )

                TextField(
                    "MP",
                    value: $viewModel.megapixelBudget,
                    format: .number.precision(.fractionLength(2))
                )
                .textFieldStyle(.roundedBorder)
                .frame(width: 64)
                .multilineTextAlignment(.trailing)

                Text("MP")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text(help)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Parameters Section

    @ViewBuilder
    private var parametersPaletteContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            if viewModel.hasPrimaryReference {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Megapixel budget")
                        .font(.caption.bold())

                    megapixelBudgetControls(help: megapixelBudgetHelp)

                    if viewModel.hasLocalFillSelection, let primaryImage = viewModel.primaryReferenceImage {
                        FillResolutionInfoRow(
                            image: primaryImage,
                            megapixelBudget: viewModel.megapixelBudget
                        )
                    }
                }

                Divider()
            }

            // Dimensions — hidden: the barn-door aspect ratio + megapixel budget
            // now determine width/height. Kept behind a flag, not deleted.
            if Self.showManualDimensions {
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Width: \(viewModel.width)")
                            .font(.caption)
                        Slider(value: Binding(
                            get: { Double(viewModel.width) },
                            set: { viewModel.width = Int($0) }
                        ), in: 256...2048, step: 64)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Height: \(viewModel.height)")
                            .font(.caption)
                        Slider(value: Binding(
                            get: { Double(viewModel.height) },
                            set: { viewModel.height = Int($0) }
                        ), in: 256...2048, step: 64)
                    }
                }

                // Match reference button
                if let primaryImage = viewModel.primaryReferenceImage {
                    Button(action: {
                        viewModel.width = primaryImage.width
                        viewModel.height = primaryImage.height
                    }) {
                        Label("Match Reference Size", systemImage: "arrow.up.left.and.arrow.down.right")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            // Seed, Steps, and Guidance — one row
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Seed")
                        .font(.caption)
                    HStack(spacing: 4) {
                        #if canImport(AppKit)
                        NonAutofocusTextField(text: $viewModel.seed, placeholder: "Random", width: 60)
                            .frame(width: 60)
                        #else
                        TextField("Random", text: $viewModel.seed)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 60)
                        #endif
                        Button(action: {
                            viewModel.seed = String(UInt64.random(in: 0...UInt64.max))
                        }) {
                            Image(systemName: "dice")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .help("Generate random seed")
                    }
                }
                .fixedSize(horizontal: true, vertical: false)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Steps: \(viewModel.steps)")
                        .font(.caption)
                    ResetOnDoubleClickIntSlider(
                        value: $viewModel.steps,
                        range: 4...64,
                        step: 1,
                        defaultValue: viewModel.recommendedSteps
                    )
                }
                .frame(maxWidth: .infinity)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 4) {
                        Text("Guidance: \(String(format: "%.1f", viewModel.guidance))")
                            .font(.caption)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                        Button("Default") {
                            viewModel.resetGuidanceToModelDefault()
                        }
                        .controlSize(.mini)
                    }
                    ResetOnDoubleClickSlider(
                        value: Binding(
                            get: { Double(viewModel.guidance) },
                            set: { viewModel.guidance = Float($0) }
                        ),
                        range: 1...10,
                        step: 0.5,
                        defaultValue: Double(viewModel.selectedModel.defaultGuidance)
                    )
                }
                .frame(maxWidth: .infinity)
            }
        }
    }

    // MARK: - Output Options

    @ViewBuilder
    private var outputOptionsPaletteContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            LanczosUpscaleField(factor: $imageSaveUpscaleBy)
                .disabled(!viewModel.hasPrimaryReference)
                .opacity(viewModel.hasPrimaryReference ? 1 : 0.45)

            Divider()

            ImageSaveWorkingNamingPreferencesView(
                previewPrompt: viewModel.prompt.isEmpty ? "sample prompt" : viewModel.prompt
            )
        }
    }

    @ViewBuilder
    private var previewTrailingActions: some View {
        HStack(spacing: 8) {
            Picker("Compare", selection: $viewModel.previewComparisonSide) {
                Text("A").tag(PreviewComparisonSide.formatted)
                Text("B").tag(PreviewComparisonSide.processed)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 140)
            .disabled(!viewModel.canTogglePreviewComparison)
            .opacity(viewModel.canTogglePreviewComparison ? 1 : 0.45)
            .help("A: formatted input aligned to output · B: processed result")

            Button(action: { useAsReference() }) {
                Label("Use as Reference", systemImage: "arrow.uturn.left")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(
                viewModel.generatedImage == nil
                    || (!viewModel.canAddImageSlot && viewModel.activeImageSlot?.hasImage == true)
            )

            Button(action: { viewModel.saveImage() }) {
                Label("Save Preview", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(viewModel.generatedImage == nil)

            clearPreviewButton
        }
    }

    private var clearPreviewButton: some View {
        Button(action: { viewModel.clearPreview() }) {
            Label("Clear Preview", systemImage: "xmark.circle")
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(!viewModel.hasPreviewContent)
        .help("Clear the generated image from the preview pane")
    }

    // MARK: - Output Section

    @ViewBuilder
    private var outputSection: some View {
        VStack(spacing: 0) {
            ImageGenerationPromptSection(viewModel: viewModel)

            Divider()

            // Checkpoints row (if available)
            if viewModel.showCheckpoints && (viewModel.isGenerating || !viewModel.checkpointImages.isEmpty) {
                checkpointsSection
                    .layoutPriority(1)
                Divider()
            }

            // Main image display
            GeometryReader { geometry in
                if let cgImage = viewModel.previewDisplayImage ?? viewModel.generatedImage {
                    VStack(spacing: 0) {
                        PreviewZoomableImageView(
                            image: cgImage,
                            zoomScale: Binding(
                                get: { CGFloat(viewModel.previewZoomScale) },
                                set: { viewModel.previewZoomScale = Double($0) }
                            )
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                        previewMetricsFooter(image: viewModel.previewSourceImage ?? cgImage)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height)
                } else if let previewImage = viewModel.previewSourceImage {
                    VStack(spacing: 10) {
                        if viewModel.isSpatialEditingActive {
                        ImagePreparationPreview(
                            image: previewImage,
                            adjustedSize: viewModel.adjustedGenerationSize,
                            sizingMethod: viewModel.previewSizingMethod,
                            overlayOpacity: viewModel.preparationOverlayOpacity,
                            generateRoute: viewModel.generateRoute,
                            maskTool: viewModel.inpaintMaskTool,
                            outpaintPadding: Binding(
                                get: { viewModel.outpaintPadding },
                                set: { viewModel.updateOutpaintPadding($0) }
                            ),
                            megapixelBudget: viewModel.megapixelBudget,
                            maskLayers: viewModel.inpaintMaskLayers,
                            visionSubjectMasks: viewModel.visionSubjectMasks,
                            draftPolygonPoints: viewModel.draftPolygonPoints,
                            draftLassoPoints: viewModel.draftLassoPoints,
                            enrichInpaintPromptWithVLM: viewModel.enrichInpaintPromptWithVLM,
                            fillVLMContextArea: viewModel.fillVLMContextArea,
                            showsFillContextMaskOverlay: viewModel.showsFillContextMaskOverlay,
                            isDrawingSelection: Binding(
                                get: { viewModel.isDrawingSelection },
                                set: { viewModel.isDrawingSelection = $0 }
                            ),
                            contextArea: Binding(
                                get: { viewModel.contextArea },
                                set: { viewModel.setContextArea($0) }
                            ),
                            processArea: Binding(
                                get: { viewModel.processArea },
                                set: { viewModel.setProcessArea($0) }
                            ),
                            onCommitFillRectangle: { rect in
                                viewModel.commitFillRectangle(rect)
                            },
                            onPolygonPoint: { viewModel.addDraftPolygonPoint($0) },
                            onLassoPoint: { viewModel.appendLassoPoint($0) },
                            onCommitLasso: {
                                viewModel.commitLassoSelection()
                            },
                            onDeselect: { viewModel.deselectSelections() },
                            onResetBarnDoors: { viewModel.resetBarnDoors() }
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                        previewMetricsFooter(image: previewImage)
                        } else {
                            PreviewZoomableImageView(
                                image: previewImage,
                                zoomScale: Binding(
                                    get: { CGFloat(viewModel.previewZoomScale) },
                                    set: { viewModel.previewZoomScale = Double($0) }
                                )
                            )
                            .frame(maxWidth: .infinity, maxHeight: .infinity)

                            Text("Canvas tools apply on the Primary reference tab.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.bottom, 10)
                        }
                    }
                    .padding()
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .background(Color(nsColor: .windowBackgroundColor))
                } else {
                    VStack {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 64))
                            .foregroundStyle(.secondary.opacity(0.5))
                        Text("Generated image will appear here")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        if !viewModel.hasPrimaryReference {
                            Text("Add a primary reference image to start")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .padding(.top, 4)
                        }
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .background(Color(nsColor: .windowBackgroundColor))
                }
            }
        }
    }

    @ViewBuilder
    private func previewMetricsFooter(image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if viewModel.hasLocalFillSelection {
                FillResolutionInfoRow(
                    image: image,
                    megapixelBudget: viewModel.megapixelBudget
                )
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            PreparationSizeInfoRow(
                image: image,
                contextArea: viewModel.contextArea,
                adjustedSize: viewModel.adjustedGenerationSize,
                trailingControls: { previewTrailingActions }
            )
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal)
        .padding(.bottom, 10)
    }

    // MARK: - Checkpoints Section

    @ViewBuilder
    private var checkpointsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Checkpoints", systemImage: "clock.arrow.circlepath")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)

                Spacer()

                Button(action: { viewModel.clearCheckpoints() }) {
                    Text("Clear")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
            .padding(.horizontal)
            .padding(.top, 8)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    if viewModel.isGenerating && viewModel.checkpointImages.isEmpty {
                        VStack(spacing: 4) {
                            ProgressView()
                                .controlSize(.small)
                            Text(
                                viewModel.totalSteps > 0
                                    ? "Step \(viewModel.currentStep)/\(viewModel.totalSteps)"
                                    : "Generating…"
                            )
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        }
                        .frame(width: 80, height: 80)
                    }

                    ForEach(viewModel.checkpointImages) { checkpoint in
                        VStack(spacing: 2) {
                            Image(decorative: checkpoint.image, scale: 1.0)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 80, height: 80)
                                .cornerRadius(4)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 4)
                                        .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                                )

                            Text("Step \(checkpoint.step)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.horizontal)
            }
            .frame(height: 110)
        }
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }

    // MARK: - Helpers

    private func useAsReference() {
        guard let cgImage = viewModel.generatedImage else { return }
        viewModel.addReferenceImage(cgImage: cgImage)
    }
}

// MARK: - Supporting Views

struct ReferenceImageSlot: View {
    let image: ReferenceImage
    var edge: CGFloat = 100
    let onRemove: () -> Void

    var body: some View {
        Image(nsImage: image.thumbnail)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(maxWidth: .infinity)
            .frame(height: edge)
            .clipped()
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.accentColor, lineWidth: 2)
            )
            .overlay(alignment: .topTrailing) {
                Button(action: onRemove) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 18))
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .red)
                }
                .buttonStyle(.plain)
                .padding(4)
            }
    }
}

struct AddImageSlot: View {
    var edge: CGFloat = 100
    let onAdd: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: onAdd) {
            VStack(spacing: 8) {
                Image(systemName: "plus.circle.fill")
                    .font(edge > 120 ? .largeTitle : .title)
                Text("Add")
                    .font(.caption)
            }
            .foregroundStyle(isHovering ? Color.accentColor : .secondary)
            .frame(maxWidth: .infinity)
            .frame(height: edge)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [5]))
                    .foregroundStyle(isHovering ? Color.accentColor : .secondary)
            )
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
    }
}

struct EmptyImageSlot: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5]))
            .foregroundStyle(.secondary.opacity(0.3))
            .frame(width: 100, height: 100)
    }
}

struct InterpretImageThumbnail: View {
    let url: URL
    let onRemove: () -> Void

    var body: some View {
        Group {
            if let nsImage = NSImage(contentsOf: url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Rectangle()
                    .fill(.secondary.opacity(0.2))
            }
        }
        .frame(width: 50, height: 50)
        .clipped()
        .cornerRadius(4)
        .overlay(alignment: .topTrailing) {
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white, .red)
            }
            .buttonStyle(.plain)
            .padding(2)
        }
    }
}

// MARK: - Image Preparation Preview

struct ImagePreparationOptionButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: isSelected ? "largecircle.fill.circle" : "circle")
                .font(.caption)
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
        .contentShape(Rectangle())
    }
}

struct ImagePreparationPreview: View {
    let image: CGImage
    let adjustedSize: (width: Int, height: Int)
    let sizingMethod: ImageSizingMethod
    let overlayOpacity: Double
    let generateRoute: I2IGenerateRoute
    var maskTool: InpaintMaskTool = .rectangle
    @Binding var outpaintPadding: OutpaintPadding
    var megapixelBudget: Double = 1.0
    var maskLayers: [InpaintMaskLayer] = []
    var visionSubjectMasks: [UUID: CGImage] = [:]
    var draftPolygonPoints: [CGPoint] = []
    var draftLassoPoints: [CGPoint] = []
    var enrichInpaintPromptWithVLM: Bool = false
    var fillVLMContextArea: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    var showsFillContextMaskOverlay: Bool = false
    @Binding var isDrawingSelection: Bool
    @Binding var contextArea: CGRect
    @Binding var processArea: CGRect?
    var onCommitFillRectangle: ((CGRect) -> Void)?
    var onPolygonPoint: ((CGPoint) -> Void)?
    var onLassoPoint: ((CGPoint) -> Void)?
    var onCommitLasso: (() -> Void)?
    var onDeselect: (() -> Void)?
    var onResetBarnDoors: (() -> Void)?

    @State private var activeDrag: DragMode?
    @State private var selectionStart: CGPoint?
    @State private var contextBeforeDrag: CGRect?
    @State private var processBeforeDrag: CGRect?
    @State private var outpaintPaddingBeforeDrag: OutpaintPadding?
    @State private var lassoSampleDistance: CGFloat = 0.004
    @State private var heldSelectionModifier: SelectionModifierHint?

    private enum DragMode {
        case contextLeft
        case contextRight
        case contextTop
        case contextBottom
        case contextSelection
        case processSelection
        case lassoSelection
        case outpaintLeft
        case outpaintRight
        case outpaintTop
        case outpaintBottom
    }

    private var isSelectionTool: Bool {
        maskTool.isSelectionTool
    }

    private var showsBarnDoorChrome: Bool {
        generateRoute == .fullImage
    }

    private var allowsBarnDoorEditing: Bool {
        showsBarnDoorChrome && maskTool.isBarnDoorTool
    }

    private var showsSelectionOverlays: Bool {
        !maskLayers.isEmpty
            || processArea != nil
            || !draftPolygonPoints.isEmpty
            || !draftLassoPoints.isEmpty
    }

    var body: some View {
        GeometryReader { geometry in
            let isOutpaint = generateRoute == .outpaint
            let canvasRect = isOutpaint
                ? fittedCanvasRect(in: geometry.size, padding: outpaintPadding)
                : fittedImageRect(in: geometry.size)
            let imageRect = isOutpaint
                ? imageRectInsideCanvas(canvasRect: canvasRect, padding: outpaintPadding)
                : canvasRect
            let showsFormattingOverlay = showsBarnDoorChrome && activeDrag == nil && !formattingExcludedRects().isEmpty
            let showsBarnDoorOverlay = showsBarnDoorChrome && !isContextFullyOpen

            ZStack {
                Color(nsColor: .textBackgroundColor)

                if isOutpaint {
                    outpaintExpansionOverlay(canvasRect: canvasRect, imageRect: imageRect)
                        .allowsHitTesting(false)
                }

                Image(decorative: image, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: imageRect.width, height: imageRect.height)
                    .position(x: imageRect.midX, y: imageRect.midY)

                // Crop/pad preview bands — hidden while dragging so barn-door
                // moves don't spawn shifting rectangles.
                if showsFormattingOverlay {
                    formattingExclusionOverlay(in: imageRect)
                        .fill(
                            Color.black.opacity(overlayOpacity),
                            style: FillStyle(eoFill: false, antialiased: false)
                        )
                        .allowsHitTesting(false)
                }

                // Barn doors: darken only when the context is narrower than
                // the full image. Skip when fully open to avoid eoFill hairlines.
                if showsBarnDoorOverlay {
                    contextExclusionOverlay(outer: CGRect(x: 0, y: 0, width: 1, height: 1), inner: contextArea, in: imageRect)
                        .fill(
                            Color.black.opacity(0.72),
                            style: FillStyle(eoFill: false, antialiased: false)
                        )
                        .allowsHitTesting(false)
                }

                if showsFillContextMaskOverlay {
                    contextExclusionOverlay(
                        outer: CGRect(x: 0, y: 0, width: 1, height: 1),
                        inner: fillVLMContextArea,
                        in: imageRect
                    )
                    .fill(
                        Color.white.opacity(0.32),
                        style: FillStyle(eoFill: false, antialiased: false)
                    )
                    .allowsHitTesting(false)
                }

                if showsSelectionOverlays {
                    committedMaskOverlays(in: imageRect)
                        .allowsHitTesting(false)

                    if maskTool == .rectangle, let processArea {
                        selectionRectOverlay(for: processArea, in: imageRect, isDraft: true)
                            .allowsHitTesting(false)
                    }

                    if maskTool == .polygon, !draftPolygonPoints.isEmpty {
                        draftPolygonOverlay(in: imageRect)
                            .allowsHitTesting(false)
                    }

                    if maskTool == .visionSubject, !draftLassoPoints.isEmpty {
                        draftLassoOverlay(in: imageRect)
                            .allowsHitTesting(false)
                    }
                }

                if isOutpaint, outpaintPadding.hasExpansion {
                    Text(outpaintCanvasLabel)
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .position(x: canvasRect.midX, y: canvasRect.minY + 20)
                        .allowsHitTesting(false)
                } else if isOutpaint {
                    Text("Drag a canvas edge to expand")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .position(x: canvasRect.midX, y: canvasRect.minY + 20)
                        .allowsHitTesting(false)
                }

                selectionModifierBadge(in: imageRect)

            }
            .compositingGroup()
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
            )
            .contentShape(Rectangle())
            .gesture(dragGesture(canvasRect: canvasRect, imageRect: imageRect))
            .onContinuousHover { phase in
                #if canImport(AppKit)
                switch phase {
                case .active(let location):
                    refreshHeldSelectionModifier()
                    cursor(for: location, in: imageRect, canvasRect: canvasRect).set()
                case .ended:
                    heldSelectionModifier = nil
                    NSCursor.arrow.set()
                }
                #endif
            }
            #if canImport(AppKit)
            .background {
                ModifierFlagsChangeMonitor {
                    refreshHeldSelectionModifier()
                }
            }
            #endif
        }
    }

    #if canImport(AppKit)
    @ViewBuilder
    private func selectionModifierBadge(in imageRect: CGRect) -> some View {
        if let hint = heldSelectionModifier,
           let bounds = selectionBadgeBounds(in: imageRect),
           bounds.width > 4,
           bounds.height > 4 {
            SelectionModifierCornerBadge(hint: hint)
                .position(x: bounds.maxX - 7, y: bounds.maxY - 7)
        }
    }

    private func refreshHeldSelectionModifier() {
        guard selectionModifierBadgeContextActive else {
            heldSelectionModifier = nil
            return
        }
        heldSelectionModifier = SelectionModifierHint.from(flags: NSEvent.modifierFlags)
    }

    private var selectionModifierBadgeContextActive: Bool {
        isDrawingSelection
            || activeDrag == .processSelection
            || activeDrag == .lassoSelection
            || processArea != nil
            || !draftPolygonPoints.isEmpty
            || !draftLassoPoints.isEmpty
            || !maskLayers.isEmpty
    }

    private func selectionBadgeBounds(in imageRect: CGRect) -> CGRect? {
        if let processArea {
            return pixelAligned(displayRect(for: processArea, in: imageRect))
        }
        if !draftLassoPoints.isEmpty {
            return pixelAligned(boundingDisplayRect(points: draftLassoPoints, in: imageRect))
        }
        if !draftPolygonPoints.isEmpty {
            return pixelAligned(boundingDisplayRect(points: draftPolygonPoints, in: imageRect))
        }
        return committedSelectionBounds(in: imageRect)
    }

    private func committedSelectionBounds(in imageRect: CGRect) -> CGRect? {
        var union: CGRect?
        for layer in maskLayers {
            let layerRect: CGRect?
            switch layer.primitive {
            case .rectangle(let rect):
                layerRect = displayRect(for: rect.cgRect, in: imageRect)
            case .polygon(let points):
                layerRect = boundingDisplayRect(points: points.map(\.cgPoint), in: imageRect)
            case .visionSubject:
                layerRect = nil
            }
            if let layerRect {
                union = union.map { $0.union(layerRect) } ?? layerRect
            }
        }
        return union
    }

    private func boundingDisplayRect(points: [CGPoint], in imageRect: CGRect) -> CGRect {
        guard let first = points.first else { return .zero }
        var bounds = CGRect(origin: displayPoint(first, in: imageRect), size: .zero)
        for point in points.dropFirst() {
            let display = displayPoint(point, in: imageRect)
            bounds = bounds.union(CGRect(origin: display, size: .zero))
        }
        return bounds
    }
    #endif

    private var outpaintCanvasLabel: String {
        let size = outpaintPadding.canvasSize(sourceWidth: image.width, sourceHeight: image.height)
        return "Canvas \(size.width)×\(size.height)"
    }

    @ViewBuilder
    private func outpaintExpansionOverlay(canvasRect: CGRect, imageRect: CGRect) -> some View {
        ZStack {
            Path { path in
                path.addRect(canvasRect)
                path.addRect(imageRect)
            }
            .fill(Color.gray.opacity(0.4), style: FillStyle(eoFill: true, antialiased: false))

            Path { path in
                path.addRect(canvasRect)
            }
            .stroke(
                Color.accentColor,
                style: StrokeStyle(lineWidth: 2, dash: [8, 6])
            )
        }
    }

    private func fittedCanvasRect(in size: CGSize, padding: OutpaintPadding) -> CGRect {
        let canvasWidth = CGFloat(image.width + padding.left + padding.right)
        let canvasHeight = CGFloat(image.height + padding.top + padding.bottom)
        guard canvasWidth > 0, canvasHeight > 0, size.width > 0, size.height > 0 else {
            return .zero
        }

        let canvasAspect = canvasWidth / canvasHeight
        let viewAspect = size.width / size.height
        let width: CGFloat
        let height: CGFloat

        if canvasAspect > viewAspect {
            width = size.width
            height = width / canvasAspect
        } else {
            height = size.height
            width = height * canvasAspect
        }

        return CGRect(
            x: (size.width - width) / 2,
            y: (size.height - height) / 2,
            width: width,
            height: height
        )
    }

    private func imageRectInsideCanvas(canvasRect: CGRect, padding: OutpaintPadding) -> CGRect {
        let canvasWidth = CGFloat(image.width + padding.left + padding.right)
        let canvasHeight = CGFloat(image.height + padding.top + padding.bottom)
        guard canvasWidth > 0, canvasHeight > 0 else { return canvasRect }

        let scale = canvasRect.width / canvasWidth
        return CGRect(
            x: canvasRect.minX + CGFloat(padding.left) * scale,
            y: canvasRect.minY + CGFloat(padding.top) * scale,
            width: CGFloat(image.width) * scale,
            height: CGFloat(image.height) * scale
        )
    }

    private var isContextFullyOpen: Bool {
        let epsilon: CGFloat = 0.0001
        return abs(contextArea.minX) < epsilon
            && abs(contextArea.minY) < epsilon
            && abs(contextArea.width - 1) < epsilon
            && abs(contextArea.height - 1) < epsilon
    }

    private func fittedImageRect(in size: CGSize) -> CGRect {
        guard image.width > 0, image.height > 0, size.width > 0, size.height > 0 else {
            return .zero
        }

        let imageAspect = CGFloat(image.width) / CGFloat(image.height)
        let viewAspect = size.width / size.height
        let width: CGFloat
        let height: CGFloat

        if imageAspect > viewAspect {
            width = size.width
            height = width / imageAspect
        } else {
            height = size.height
            width = height * imageAspect
        }

        return CGRect(
            x: (size.width - width) / 2,
            y: (size.height - height) / 2,
            width: width,
            height: height
        )
    }

    /// Regions dimmed to preview Image Formatting (crop discard or pad bands).
    private func formattingExclusionOverlay(in imageRect: CGRect) -> Path {
        guard adjustedSize.width > 0, adjustedSize.height > 0 else {
            return Path()
        }

        let excluded = formattingExcludedRects()
        guard !excluded.isEmpty else { return Path() }

        var path = Path()
        for rect in excluded {
            let clipped = rect.intersection(contextArea)
            guard clipped.width > 0, clipped.height > 0 else { continue }
            path.addRect(displayRect(for: clipped, in: imageRect))
        }
        return path
    }

    private var contextPixelAspect: CGFloat {
        let ctx = contextArea
        let pixelWidth = ctx.width * CGFloat(image.width)
        let pixelHeight = ctx.height * CGFloat(image.height)
        guard pixelHeight > 0 else { return 1 }
        return pixelWidth / pixelHeight
    }

    private var targetPixelAspect: CGFloat {
        guard adjustedSize.height > 0 else { return 1 }
        return CGFloat(adjustedSize.width) / CGFloat(adjustedSize.height)
    }

    private func formattingExcludedRects() -> [CGRect] {
        let ctx = contextArea
        let ctxAspect = contextPixelAspect
        let targetAspect = targetPixelAspect

        guard abs(ctxAspect - targetAspect) > 0.0001 else { return [] }

        switch sizingMethod {
        case .crop:
            let visible = cropVisibleRect(in: ctx, ctxAspect: ctxAspect, targetAspect: targetAspect)
            return exclusionBands(outer: ctx, inner: visible)
        case .pad:
            return padExcludedRects(in: ctx, ctxAspect: ctxAspect, targetAspect: targetAspect)
        }
    }

    /// Crop: the kept region inside the barn-door context (matches
    /// `preparationTransform` with `.crop`).
    private func cropVisibleRect(in ctx: CGRect, ctxAspect: CGFloat, targetAspect: CGFloat) -> CGRect {
        if ctxAspect > targetAspect {
            let visibleWidth = targetAspect / ctxAspect * ctx.width
            return CGRect(
                x: ctx.minX + (ctx.width - visibleWidth) / 2,
                y: ctx.minY,
                width: visibleWidth,
                height: ctx.height
            )
        }

        let visibleHeight = ctxAspect / targetAspect * ctx.height
        return CGRect(
            x: ctx.minX,
            y: ctx.minY + (ctx.height - visibleHeight) / 2,
            width: ctx.width,
            height: visibleHeight
        )
    }

    /// Pad: letterbox/pillarbox bands around the context (matches
    /// `preparationTransform` with `.pad`).
    private func padExcludedRects(in ctx: CGRect, ctxAspect: CGFloat, targetAspect: CGFloat) -> [CGRect] {
        let frame: CGRect
        if ctxAspect > targetAspect {
            // Fits width; pads top and bottom in the output canvas.
            let frameHeight = ctx.width / targetAspect
            frame = CGRect(
                x: ctx.minX,
                y: ctx.minY - (frameHeight - ctx.height) / 2,
                width: ctx.width,
                height: frameHeight
            )
        } else {
            // Fits height; pads left and right.
            let frameWidth = ctx.height * targetAspect
            frame = CGRect(
                x: ctx.minX - (frameWidth - ctx.width) / 2,
                y: ctx.minY,
                width: frameWidth,
                height: ctx.height
            )
        }
        return exclusionBands(outer: frame, inner: ctx)
    }

    /// `outer` minus `inner` as up to four rectangular bands (eoFill is not
    /// used here — each band is filled individually).
    private func exclusionBands(outer: CGRect, inner: CGRect) -> [CGRect] {
        let clip = outer.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        let hole = inner.intersection(clip)
        guard clip.width > 0, clip.height > 0, hole.width > 0, hole.height > 0 else {
            return []
        }

        var bands: [CGRect] = []
        if hole.minY > clip.minY {
            bands.append(CGRect(x: clip.minX, y: clip.minY, width: clip.width, height: hole.minY - clip.minY))
        }
        if hole.maxY < clip.maxY {
            bands.append(CGRect(x: clip.minX, y: hole.maxY, width: clip.width, height: clip.maxY - hole.maxY))
        }
        if hole.minX > clip.minX {
            bands.append(CGRect(x: clip.minX, y: hole.minY, width: hole.minX - clip.minX, height: hole.height))
        }
        if hole.maxX < clip.maxX {
            bands.append(CGRect(x: hole.maxX, y: hole.minY, width: clip.maxX - hole.maxX, height: hole.height))
        }
        return bands
    }

    private func contextExclusionOverlay(outer: CGRect, inner: CGRect, in imageRect: CGRect) -> Path {
        let bands = exclusionBands(outer: outer, inner: inner)
        let contextRect = pixelAligned(displayRect(for: inner, in: imageRect))
        var path = Path()
        for band in bands {
            var rect = pixelAligned(displayRect(for: band, in: imageRect))
            rect = bleedBandRectTowardContext(rect, contextRect: contextRect)
            guard rect.width > 0, rect.height > 0 else { continue }
            path.addRect(rect)
        }
        return path
    }

    /// Expand a barn-door band 1px into the lit context so antialiased edges
    /// don't leave a grey hairline on the live region.
    private func bleedBandRectTowardContext(_ band: CGRect, contextRect: CGRect) -> CGRect {
        let bleed: CGFloat = 1
        var rect = band
        if abs(rect.maxX - contextRect.minX) <= bleed {
            rect.size.width += bleed
        }
        if abs(rect.minX - contextRect.maxX) <= bleed {
            rect.origin.x -= bleed
            rect.size.width += bleed
        }
        if abs(rect.maxY - contextRect.minY) <= bleed {
            rect.size.height += bleed
        }
        if abs(rect.minY - contextRect.maxY) <= bleed {
            rect.origin.y -= bleed
            rect.size.height += bleed
        }
        return rect
    }

    private func dragGesture(canvasRect: CGRect, imageRect: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                if activeDrag == nil {
                    activeDrag = dragMode(for: value.startLocation, canvasRect: canvasRect, imageRect: imageRect)
                    selectionStart = normalizedPoint(for: value.startLocation, in: imageRect)
                    if generateRoute == .outpaint {
                        outpaintPaddingBeforeDrag = outpaintPadding
                    }
                }

                let point = normalizedPoint(for: value.location, in: imageRect)

                switch activeDrag {
                case .contextLeft:
                    updateContextLeft(to: snapX(point.x), in: imageRect)
                case .contextRight:
                    updateContextRight(to: snapX(point.x), in: imageRect)
                case .contextTop:
                    updateContextTop(to: snapY(point.y), in: imageRect)
                case .contextBottom:
                    updateContextBottom(to: snapY(point.y), in: imageRect)
                case .contextSelection:
                    updateContextSelection(to: point, in: imageRect)
                case .processSelection:
                    isDrawingSelection = true
                    updateProcessSelection(to: point, in: imageRect)
                    #if canImport(AppKit)
                    refreshHeldSelectionModifier()
                    #endif
                case .lassoSelection:
                    isDrawingSelection = true
                    appendLassoPointIfNeeded(point)
                    #if canImport(AppKit)
                    refreshHeldSelectionModifier()
                    #endif
                case .outpaintLeft:
                    updateOutpaintLeft(to: value.location.x, canvasRect: canvasRect, imageRect: imageRect)
                case .outpaintRight:
                    updateOutpaintRight(to: value.location.x, canvasRect: canvasRect, imageRect: imageRect)
                case .outpaintTop:
                    updateOutpaintTop(to: value.location.y, canvasRect: canvasRect, imageRect: imageRect)
                case .outpaintBottom:
                    updateOutpaintBottom(to: value.location.y, canvasRect: canvasRect, imageRect: imageRect)
                case nil:
                    break
                }
            }
            .onEnded { value in
                defer {
                    isDrawingSelection = false
                    activeDrag = nil
                    selectionStart = nil
                    contextBeforeDrag = nil
                    processBeforeDrag = nil
                    outpaintPaddingBeforeDrag = nil
                    heldSelectionModifier = nil
                    #if canImport(AppKit)
                    NSCursor.arrow.set()
                    #endif
                }

                let endPoint = normalizedPoint(for: value.location, in: imageRect)
                let dragDistance = hypot(value.translation.width, value.translation.height)

                if generateRoute == .outpaint {
                    if dragDistance < 4, !canvasRect.contains(value.location) {
                        onDeselect?()
                    }
                    return
                }

                if maskTool == .polygon, isSelectionTool, generateRoute != .outpaint {
                    if dragDistance < 4, draftPolygonPoints.isEmpty, !maskLayers.isEmpty {
                        onDeselect?()
                        return
                    }
                    onPolygonPoint?(endPoint)
                    return
                }

                if maskTool == .visionSubject, activeDrag == .lassoSelection {
                    if draftLassoPoints.count >= 3 {
                        onCommitLasso?()
                    } else {
                        onDeselect?()
                    }
                    return
                }

                if maskTool == .rectangle, activeDrag == .processSelection, let rect = processArea {
                    let minWidthNorm = 12 / max(imageRect.width, 1)
                    let minHeightNorm = 12 / max(imageRect.height, 1)
                    if rect.width >= minWidthNorm, rect.height >= minHeightNorm {
                        onCommitFillRectangle?(rect)
                    } else {
                        onDeselect?()
                    }
                    return
                }

                if dragDistance < 4, imageRect.contains(value.location) {
                    if maskTool.isBarnDoorTool {
                        onResetBarnDoors?()
                    } else if maskTool == .pointer
                        || (maskTool.isSelectionTool && !maskLayers.isEmpty && draftPolygonPoints.isEmpty) {
                        onDeselect?()
                    }
                }
            }
    }

    private func appendLassoPointIfNeeded(_ point: CGPoint) {
        guard let last = draftLassoPoints.last else {
            onLassoPoint?(point)
            return
        }
        let dx = point.x - last.x
        let dy = point.y - last.y
        if hypot(dx, dy) >= lassoSampleDistance {
            onLassoPoint?(point)
        }
    }

    private func dragMode(for location: CGPoint, canvasRect: CGRect, imageRect: CGRect) -> DragMode? {
        if generateRoute == .outpaint {
            if let edge = outpaintCanvasEdge(at: location, in: canvasRect) {
                return edge
            }
            return nil
        }

        guard imageRect.contains(location) else { return nil }

        if isSelectionTool {
            switch maskTool {
            case .rectangle:
                return .processSelection
            case .polygon:
                return nil
            case .visionSubject:
                return .lassoSelection
            case .pointer, .liveArea, .cropCanvas:
                return nil
            }
        }

        if allowsBarnDoorEditing {
            if let edge = contextEdge(at: location, in: imageRect) {
                return edge
            }
            return .contextSelection
        }

        return nil
    }

    private func outpaintCanvasEdge(at location: CGPoint, in canvasRect: CGRect) -> DragMode? {
        let threshold: CGFloat = 20
        let distances: [(DragMode, CGFloat)] = [
            (.outpaintLeft, abs(location.x - canvasRect.minX)),
            (.outpaintRight, abs(location.x - canvasRect.maxX)),
            (.outpaintTop, abs(location.y - canvasRect.minY)),
            (.outpaintBottom, abs(location.y - canvasRect.maxY)),
        ]

        return distances
            .filter { $0.1 <= threshold }
            .min { $0.1 < $1.1 }?
            .0
    }

    private func updateOutpaintLeft(to x: CGFloat, canvasRect: CGRect, imageRect: CGRect) {
        var padding = outpaintPaddingBeforeDrag ?? outpaintPadding
        let scale = imagePixelScale(imageRect: imageRect)
        let leftDisplay = max(0, imageRect.minX - x)
        padding.left = Int((leftDisplay / scale).rounded())
        outpaintPadding = padding
    }

    private func updateOutpaintRight(to x: CGFloat, canvasRect: CGRect, imageRect: CGRect) {
        var padding = outpaintPaddingBeforeDrag ?? outpaintPadding
        let scale = imagePixelScale(imageRect: imageRect)
        let rightDisplay = max(0, x - imageRect.maxX)
        padding.right = Int((rightDisplay / scale).rounded())
        outpaintPadding = padding
    }

    private func updateOutpaintTop(to y: CGFloat, canvasRect: CGRect, imageRect: CGRect) {
        var padding = outpaintPaddingBeforeDrag ?? outpaintPadding
        let scale = imagePixelScale(imageRect: imageRect)
        let topDisplay = max(0, imageRect.minY - y)
        padding.top = Int((topDisplay / scale).rounded())
        outpaintPadding = padding
    }

    private func updateOutpaintBottom(to y: CGFloat, canvasRect: CGRect, imageRect: CGRect) {
        var padding = outpaintPaddingBeforeDrag ?? outpaintPadding
        let scale = imagePixelScale(imageRect: imageRect)
        let bottomDisplay = max(0, y - imageRect.maxY)
        padding.bottom = Int((bottomDisplay / scale).rounded())
        outpaintPadding = padding
    }

    private func imagePixelScale(imageRect: CGRect) -> CGFloat {
        guard image.width > 0 else { return 1 }
        return imageRect.width / CGFloat(image.width)
    }

    private func selectionRectOverlay(for normalizedRect: CGRect, in imageRect: CGRect, isDraft: Bool) -> some View {
        let rect = pixelAligned(displayRect(for: normalizedRect, in: imageRect))
        return ZStack {
            if isDraft {
                Path { path in
                    path.addRect(imageRect)
                    path.addRect(rect)
                }
                .fill(Color.black.opacity(0.4), style: FillStyle(eoFill: true, antialiased: false))
            }
            MarchingAntsRect(rect: rect)
        }
    }

    @ViewBuilder
    private func committedMaskOverlays(in imageRect: CGRect) -> some View {
        ForEach(maskLayers) { layer in
            switch layer.primitive {
            case .rectangle(let rect):
                selectionRectOverlay(for: rect.cgRect, in: imageRect, isDraft: false)
            case .polygon(let points):
                selectionPolygonOverlay(points: points.map(\.cgPoint), in: imageRect)
            case .visionSubject(let selection):
                if let mask = visionSubjectMasks[layer.id] {
                    visionSubjectSelectionOverlay(mask: mask, selection: selection, in: imageRect)
                } else {
                    visionSubjectPendingOverlay(in: imageRect)
                }
            }
        }
    }

    @ViewBuilder
    private func visionSubjectSelectionOverlay(
        mask: CGImage,
        selection: VisionSubjectSelection,
        in imageRect: CGRect
    ) -> some View {
        let points = InpaintMaskOutline.normalizedBoundaryPoints(from: mask)
        if points.count >= 3 {
            MarchingAntsPath(path: polygonPath(points: points, in: imageRect, closed: true))
        } else {
            switch selection {
            case .rectangle(let rect):
                selectionRectOverlay(for: rect.cgRect, in: imageRect, isDraft: false)
            case .polygon(let points):
                selectionPolygonOverlay(points: points.map(\.cgPoint), in: imageRect)
            }
        }
    }

    private func visionSubjectPendingOverlay(in imageRect: CGRect) -> some View {
        ProgressView()
            .controlSize(.small)
            .position(x: imageRect.midX, y: imageRect.midY)
            .help("Detecting subject from selection")
            .accessibilityLabel("Detecting subject from selection")
    }

    private func selectionPolygonOverlay(points: [CGPoint], in imageRect: CGRect) -> some View {
        let path = polygonPath(points: points, in: imageRect, closed: true)
        return MarchingAntsPath(path: path)
    }

    private func draftLassoOverlay(in imageRect: CGRect) -> some View {
        MarchingAntsPath(
            path: polygonPath(points: draftLassoPoints, in: imageRect, closed: draftLassoPoints.count >= 3)
        )
    }

    private func draftPolygonOverlay(in imageRect: CGRect) -> some View {
        ZStack {
            MarchingAntsPath(
                path: polygonPath(points: draftPolygonPoints, in: imageRect, closed: draftPolygonPoints.count >= 3)
            )
            ForEach(Array(draftPolygonPoints.enumerated()), id: \.offset) { _, point in
                Circle()
                    .fill(Color.white)
                    .overlay(Circle().stroke(Color.black, lineWidth: 1))
                    .frame(width: 7, height: 7)
                    .position(displayPoint(point, in: imageRect))
            }
        }
    }

    private func polygonPath(points: [CGPoint], in imageRect: CGRect, closed: Bool) -> Path {
        Path { path in
            guard let first = points.first else { return }
            path.move(to: displayPoint(first, in: imageRect))
            for point in points.dropFirst() {
                path.addLine(to: displayPoint(point, in: imageRect))
            }
            if closed, points.count >= 3 {
                path.closeSubpath()
            }
        }
    }

    private func displayPoint(_ normalized: CGPoint, in imageRect: CGRect) -> CGPoint {
        CGPoint(
            x: imageRect.minX + normalized.x * imageRect.width,
            y: imageRect.minY + normalized.y * imageRect.height
        )
    }

    private func contextEdge(at location: CGPoint, in imageRect: CGRect) -> DragMode? {
        let rect = displayRect(for: contextArea, in: imageRect)
        let threshold: CGFloat = 20

        guard rect.insetBy(dx: -threshold, dy: -threshold).contains(location) else {
            return nil
        }

        let distances: [(DragMode, CGFloat)] = [
            (.contextLeft, abs(location.x - rect.minX)),
            (.contextRight, abs(location.x - rect.maxX)),
            (.contextTop, abs(location.y - rect.minY)),
            (.contextBottom, abs(location.y - rect.maxY)),
        ]

        return distances
            .filter { $0.1 <= threshold }
            .min { $0.1 < $1.1 }?
            .0
    }

    #if canImport(AppKit)
    private func cursor(for location: CGPoint, in imageRect: CGRect, canvasRect: CGRect) -> NSCursor {
        if generateRoute == .outpaint {
            switch outpaintCanvasEdge(at: location, in: canvasRect) {
            case .outpaintLeft, .outpaintRight:
                return .resizeLeftRight
            case .outpaintTop, .outpaintBottom:
                return .resizeUpDown
            default:
                return canvasRect.contains(location) ? .crosshair : .arrow
            }
        }

        if isSelectionTool {
            return imageRect.contains(location) ? .crosshair : .arrow
        }

        if allowsBarnDoorEditing, let edge = contextEdge(at: location, in: imageRect) {
            switch edge {
            case .contextLeft, .contextRight:
                return .resizeLeftRight
            case .contextTop, .contextBottom:
                return .resizeUpDown
            default:
                break
            }
        }

        if maskTool.isBarnDoorTool {
            return imageRect.contains(location) ? .crosshair : .arrow
        }

        return .arrow
    }
    #endif

    /// Draw a brand-new barn-door region from a press-drag. Tiny drags (a stray
    /// click) restore the region captured at drag start instead of collapsing it.
    private func updateContextSelection(to point: CGPoint, in imageRect: CGRect) {
        guard let selectionStart else { return }
        if contextBeforeDrag == nil { contextBeforeDrag = contextArea }

        let start = CGPoint(x: snapX(selectionStart.x), y: snapY(selectionStart.y))
        let end = CGPoint(x: snapX(point.x), y: snapY(point.y))
        let rect = CGRect(
            x: min(start.x, end.x),
            y: min(start.y, end.y),
            width: abs(end.x - start.x),
            height: abs(end.y - start.y)
        )

        let minWidthNorm = 12 / max(imageRect.width, 1)
        let minHeightNorm = 12 / max(imageRect.height, 1)
        if rect.width >= minWidthNorm, rect.height >= minHeightNorm {
            contextArea = rect
        } else if let snapshot = contextBeforeDrag {
            contextArea = snapshot
        }
    }

    private func updateProcessSelection(to point: CGPoint, in imageRect: CGRect) {
        guard let selectionStart else { return }
        if processBeforeDrag == nil { processBeforeDrag = processArea }

        let start = CGPoint(x: snapX(selectionStart.x), y: snapY(selectionStart.y))
        let end = CGPoint(x: snapX(point.x), y: snapY(point.y))
        let rect = CGRect(
            x: min(start.x, end.x),
            y: min(start.y, end.y),
            width: abs(end.x - start.x),
            height: abs(end.y - start.y)
        )

        let minWidthNorm = 12 / max(imageRect.width, 1)
        let minHeightNorm = 12 / max(imageRect.height, 1)
        if rect.width >= minWidthNorm, rect.height >= minHeightNorm {
            processArea = rect
        } else if let snapshot = processBeforeDrag {
            processArea = snapshot
        }
    }

    private func updateContextLeft(to x: CGFloat, in imageRect: CGRect) {
        let limit = min(barnDoorLimitLeft(in: imageRect), contextArea.maxX - minimumContextWidth)
        let newLeft = min(max(x, 0), max(limit, 0))
        contextArea = CGRect(
            x: newLeft,
            y: contextArea.minY,
            width: contextArea.maxX - newLeft,
            height: contextArea.height
        )
    }

    private func updateContextRight(to x: CGFloat, in imageRect: CGRect) {
        let limit = max(barnDoorLimitRight(in: imageRect), contextArea.minX + minimumContextWidth)
        let newRight = max(min(x, 1), min(limit, 1))
        contextArea = CGRect(
            x: contextArea.minX,
            y: contextArea.minY,
            width: newRight - contextArea.minX,
            height: contextArea.height
        )
    }

    private func updateContextTop(to y: CGFloat, in imageRect: CGRect) {
        let limit = min(barnDoorLimitTop(in: imageRect), contextArea.maxY - minimumContextHeight)
        let newTop = min(max(y, 0), max(limit, 0))
        contextArea = CGRect(
            x: contextArea.minX,
            y: newTop,
            width: contextArea.width,
            height: contextArea.maxY - newTop
        )
    }

    private func updateContextBottom(to y: CGFloat, in imageRect: CGRect) {
        let limit = max(barnDoorLimitBottom(in: imageRect), contextArea.minY + minimumContextHeight)
        let newBottom = max(min(y, 1), min(limit, 1))
        contextArea = CGRect(
            x: contextArea.minX,
            y: contextArea.minY,
            width: contextArea.width,
            height: newBottom - contextArea.minY
        )
    }

    // Each barn door can close until the context reaches its minimum size.
    private func barnDoorLimitLeft(in imageRect: CGRect) -> CGFloat {
        contextArea.maxX - minimumContextWidth
    }

    private func barnDoorLimitRight(in imageRect: CGRect) -> CGFloat {
        contextArea.minX + minimumContextWidth
    }

    private func barnDoorLimitTop(in imageRect: CGRect) -> CGFloat {
        contextArea.maxY - minimumContextHeight
    }

    private func barnDoorLimitBottom(in imageRect: CGRect) -> CGFloat {
        contextArea.minY + minimumContextHeight
    }

    private var minimumContextWidth: CGFloat {
        CGFloat(16) / CGFloat(max(image.width, 16))
    }

    private var minimumContextHeight: CGFloat {
        CGFloat(16) / CGFloat(max(image.height, 16))
    }

    private func displayRect(for normalized: CGRect, in imageRect: CGRect) -> CGRect {
        pixelAligned(
            CGRect(
                x: imageRect.minX + normalized.minX * imageRect.width,
                y: imageRect.minY + normalized.minY * imageRect.height,
                width: normalized.width * imageRect.width,
                height: normalized.height * imageRect.height
            )
        )
    }

    /// Floor/ceil to whole pixels so stacked overlays don't leave 1px gaps.
    private func pixelAligned(_ rect: CGRect) -> CGRect {
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        let minX = (rect.minX * scale).rounded(.down) / scale
        let minY = (rect.minY * scale).rounded(.down) / scale
        let maxX = (rect.maxX * scale).rounded(.up) / scale
        let maxY = (rect.maxY * scale).rounded(.up) / scale
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private func normalizedPoint(for location: CGPoint, in imageRect: CGRect) -> CGPoint {
        CGPoint(
            x: min(max((location.x - imageRect.minX) / imageRect.width, 0), 1),
            y: min(max((location.y - imageRect.minY) / imageRect.height, 0), 1)
        )
    }

    private func snapX(_ x: CGFloat) -> CGFloat {
        snap(x, pixels: image.width)
    }

    private func snapY(_ y: CGFloat) -> CGFloat {
        snap(y, pixels: image.height)
    }

    private func snap(_ value: CGFloat, pixels: Int) -> CGFloat {
        let step = CGFloat(16) / CGFloat(max(pixels, 16))
        return min(max((value / step).rounded() * step, 0), 1)
    }
}

/// Aspect-ratio guidance for the generation target. Diffusion models are
/// trained on a distribution of shapes, so extreme ratios — at any megapixel
/// count — tend to stretch or duplicate content. The preview surfaces this as a
/// graduated caution (yellow) → warning (red) so the barn doors can be eased
/// back before generating.
enum AspectRatioAdvisory {
    case ok
    case caution(String)
    case severe(String)

    /// Longer-side ÷ shorter-side. Beyond `cautionRatio` quality starts to
    /// soften; beyond `severeRatio` distortion/duplication is likely.
    static let cautionRatio = 2.0
    static let severeRatio = 3.0

    init(width: Int, height: Int) {
        guard width > 0, height > 0 else {
            self = .ok
            return
        }

        let landscape = width >= height
        let ratio = landscape
            ? Double(width) / Double(height)
            : Double(height) / Double(width)
        let shape = landscape ? "wide" : "tall"
        let ratioText = landscape
            ? String(format: "%.1f:1", ratio)
            : String(format: "1:%.1f", ratio)

        switch ratio {
        case Self.severeRatio...:
            self = .severe("Extreme \(shape) ratio (\(ratioText)) — FLUX will likely stretch or duplicate content. Ease the barn doors toward a more even shape.")
        case Self.cautionRatio...:
            self = .caution("Unusual \(shape) ratio (\(ratioText)) — may soften quality. Consider easing the barn doors toward a more even shape.")
        default:
            self = .ok
        }
    }
}

struct PreparationSizeInfoRow<Trailing: View>: View {
    let image: CGImage
    let contextArea: CGRect?
    let adjustedSize: (width: Int, height: Int)
    @ViewBuilder let trailingControls: () -> Trailing

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 12) {
                sizeItem(label: "Original", value: "\(image.width)x\(image.height)")

                verticalSeparator

                sizeItem(label: "Context", value: contextValue)

                verticalSeparator

                sizeItem(label: "Adjusted", value: "\(adjustedSize.width)x\(adjustedSize.height)")

                verticalSeparator

                sizeItem(label: "Pixels", value: megapixelValue)

                Spacer(minLength: 8)

                trailingControls()
            }
            .foregroundStyle(.secondary)

            aspectRatioAdvisory
            upscaleAdvisory
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.75))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var aspectRatioAdvisory: some View {
        switch AspectRatioAdvisory(width: adjustedSize.width, height: adjustedSize.height) {
        case .ok:
            EmptyView()
        case .caution(let message):
            advisoryBanner(color: .yellow, icon: "exclamationmark.triangle.fill", message: message)
        case .severe(let message):
            advisoryBanner(color: .red, icon: "exclamationmark.octagon.fill", message: message)
        }
    }

    private func advisoryBanner(color: Color, icon: String, message: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(color)
            Text(message)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .font(.caption.weight(.medium))
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.14))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    /// Warns when the megapixel budget asks for more pixels than the source
    /// region holds — the model then synthesizes the extra detail.
    @ViewBuilder
    private var upscaleAdvisory: some View {
        let factor = upscaleFactor
        if factor >= 2.5 {
            advisoryBanner(
                color: .red,
                icon: "arrow.up.forward.square.fill",
                message: String(format: "Heavy upscale (%.1f×): the model will invent most of the detail in the processing area.", factor)
            )
        } else if factor >= 1.5 {
            advisoryBanner(
                color: .yellow,
                icon: "arrow.up.forward.square",
                message: String(format: "Upscaling %.1f×: fine detail in the processing area will be softened or synthesized.", factor)
            )
        }
    }

    private var megapixelValue: String {
        let mp = Double(adjustedSize.width * adjustedSize.height) / 1_000_000
        return String(format: "%.2f MP", mp)
    }

    private var sourcePixelDimensions: (width: Int, height: Int) {
        if let contextArea {
            let width = max(1, Int((contextArea.width * CGFloat(image.width)).rounded()))
            let height = max(1, Int((contextArea.height * CGFloat(image.height)).rounded()))
            return (width, height)
        }
        return (image.width, image.height)
    }

    private var upscaleFactor: Double {
        let source = sourcePixelDimensions
        let sourceArea = Double(source.width * source.height)
        guard sourceArea > 0 else { return 1 }
        let outputArea = Double(adjustedSize.width * adjustedSize.height)
        return (outputArea / sourceArea).squareRoot()
    }

    private var contextValue: String {
        guard let contextArea else {
            return "None"
        }

        let width = max(1, Int((contextArea.width * CGFloat(image.width)).rounded()))
        let height = max(1, Int((contextArea.height * CGFloat(image.height)).rounded()))
        return "\(width)x\(height)"
    }

    private var verticalSeparator: some View {
        Rectangle()
            .fill(Color.secondary.opacity(0.35))
            .frame(width: 1, height: 18)
    }

    private func sizeItem(label: String, value: String) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .fontWeight(.semibold)
            Text(value)
                .monospacedDigit()
        }
    }
}

/// Warns when generative fill must downscale the source to fit the resolution cap.
struct FillResolutionInfoRow: View {
    let image: CGImage
    let megapixelBudget: Double

    var body: some View {
        let factor = downscaleFactor
        if factor < 0.99 {
            HStack(alignment: .top, spacing: 6) {
                Image(systemName: "arrow.down.forward.square")
                    .foregroundStyle(.yellow)
                Text(
                    String(
                        format: "Resolution cap downscales the image to %.0f%% before fill — very fine detail may soften.",
                        factor * 100
                    )
                )
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.yellow.opacity(0.14))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    private var downscaleFactor: Double {
        let area = Double(image.width * image.height)
        let cap = megapixelBudget * 1_000_000
        guard area > cap, area > 0 else { return 1 }
        return (cap / area).squareRoot()
    }
}


#if canImport(AppKit)
/// Fires when ⇧ / ⌥ / other modifier keys change so the preview badge can update mid-drag.
private struct ModifierFlagsChangeMonitor: NSViewRepresentable {
    let onFlagsChange: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        context.coordinator.install(onFlagsChange: onFlagsChange)
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.onFlagsChange = onFlagsChange
    }

    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.remove()
    }

    final class Coordinator {
        var monitor: Any?
        var onFlagsChange: (() -> Void)?

        func install(onFlagsChange: @escaping () -> Void) {
            self.onFlagsChange = onFlagsChange
            remove()
            monitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
                self?.onFlagsChange?()
                return event
            }
        }

        func remove() {
            if let monitor {
                NSEvent.removeMonitor(monitor)
                self.monitor = nil
            }
        }
    }
}
#endif

#Preview {
    ImageToImageView()
        .environmentObject(ModelManager())
        .frame(width: 1200, height: 900)
}
