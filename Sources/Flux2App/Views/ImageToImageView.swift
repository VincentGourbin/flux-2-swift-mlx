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
    @StateObject private var viewModel = ImageGenerationViewModel(loadsEnvironmentProject: true)
    @State private var isTargetedForDrop = false
    @AppStorage("imageSaveUpscaleBy") private var imageSaveUpscaleBy = 1.0

    // Kept-but-hidden controls: the megapixel budget + barn doors now own
    // resolution and dimensions, so these are off by default (not deleted).
    private static let showScalingControl = false
    private static let showManualDimensions = false

    var body: some View {
        HSplitView {
            // Left panel: Controls
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Model Configuration — family + model choices drive everything below
                    modelSelectionSection

                    Divider()

                    // Reference Images
                    referenceImagesSection

                    Divider()

                    // Image Formatting — crop / pad / favour of the input image
                    imageFormattingSection

                    Divider()

                    // Edit workflow — prompt I2I vs masked inpaint
                    editModeSection

                    Divider()

                    // Processing Area — barn doors + megapixel budget (prompt edit)
                    // or mask region + resolution cap (masked inpaint)
                    processingAreaSection

                    Divider()

                    // Generation Parameters
                    parametersSection

                    Divider()

                    // AI Prompt (Interpret Images nested directly beneath)
                    promptSection

                    interpretImagesSection

                    Divider()

                    // Generate Button
                    generateSection
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minWidth: 380, idealWidth: 450, maxWidth: 550)
            .clipped()

            // Right panel: Output
            outputSection
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
        .onChange(of: viewModel.editMode) { _, mode in
            if mode == .generativeFill {
                viewModel.enrichInpaintPromptWithVLM = true
                viewModel.inpaintIntent = .modify
            }
        }
        .focusedSceneValue(\.generationUnloadModels) {
            Task { await viewModel.clearPipeline() }
        }
    }

    // MARK: - Reference Images Section

    @ViewBuilder
    private var referenceImagesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Reference Images (1-\(viewModel.selectedModel.maxReferenceImages))", systemImage: "photo.stack")
                    .font(.headline)

                Spacer()

                if !viewModel.referenceImages.isEmpty {
                    Button(action: { viewModel.clearReferenceImages() }) {
                        Label("Clear All", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            Text("Drop images here or click to add. The model will use these as reference for generation.")
                .font(.caption)
                .foregroundStyle(.secondary)

            // Drop zone with image slots (dynamic based on model)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(0..<viewModel.selectedModel.maxReferenceImages, id: \.self) { index in
                        if index < viewModel.referenceImages.count {
                            // Show existing image
                            ReferenceImageSlot(
                                image: viewModel.referenceImages[index],
                                onRemove: {
                                    viewModel.removeReferenceImage(viewModel.referenceImages[index].id)
                                }
                            )
                        } else if index == viewModel.referenceImages.count {
                            // Show add button for next slot
                            AddImageSlot(onAdd: { selectImage() })
                                .onDrop(of: [.image], isTargeted: $isTargetedForDrop) { providers in
                                    handleImageDrop(providers)
                                    return true
                                }
                        } else {
                            // Empty placeholder
                            EmptyImageSlot()
                        }
                    }
                }
                .padding(.vertical, 2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: 120)
        }
    }

    // MARK: - Image Formatting Section

    @ViewBuilder
    private var imageFormattingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Image Formatting", systemImage: "crop")
                .font(.headline)

            if !viewModel.referenceImages.isEmpty {
                GroupBox("Aspect Ratio") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .top, spacing: 14) {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Favour")
                                    .font(.caption.bold())

                                HStack(spacing: 8) {
                                    ForEach(ImageSizingFavor.allCases) { favor in
                                        ImagePreparationOptionButton(
                                            title: favor.rawValue,
                                            isSelected: viewModel.sizingFavor == favor
                                        ) {
                                            viewModel.setSizingFavor(favor)
                                        }
                                    }
                                }
                            }

                            Divider()
                                .frame(height: 34)

                            VStack(alignment: .leading, spacing: 6) {
                                Text("Method")
                                    .font(.caption.bold())

                                HStack(spacing: 8) {
                                    ForEach(ImageSizingMethod.allCases) { method in
                                        ImagePreparationOptionButton(
                                            title: method.rawValue,
                                            isSelected: viewModel.sizingMethod == method
                                        ) {
                                            viewModel.setSizingMethod(method)
                                        }
                                    }
                                }
                            }
                        }

                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // Hidden: the megapixel budget (Processing Area) now governs
                // output resolution, so the input-scaling slider is redundant.
                // Kept behind a flag in case it's needed again.
                if Self.showScalingControl {
                    GroupBox("Scaling") {
                        HStack(spacing: 10) {
                            Slider(
                                value: Binding(
                                    get: { viewModel.preparationScale },
                                    set: {
                                        viewModel.preparationScale = min(max($0, 0.1), 1.0)
                                        viewModel.applySizingControls()
                                    }
                                ),
                                in: 0.1...1.0,
                                step: 0.05
                            )

                            Text("\(Int((viewModel.preparationScale * 100).rounded()))%")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 44, alignment: .trailing)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            } else {
                Text("Add a reference image to set how it's cropped or padded.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Edit Mode Section

    @ViewBuilder
    private var editModeSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Edit workflow", systemImage: "square.dashed.inset.filled")
                .font(.headline)

            if viewModel.referenceImages.isEmpty {
                Text("Add a reference image to choose how edits are applied.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Picker("Mode", selection: $viewModel.editMode) {
                    ForEach(ImageEditMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                switch viewModel.editMode {
                case .promptEdit:
                    Text("Barn doors define the live area sent to the model; the result can be composited back into the full image.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                case .generativeFill:
                    Text("Draw a rectangle over a blemish or bad patch. The model regenerates only that region; everything else is preserved.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Toggle("Enrich prompt with Qwen3.5 VLM", isOn: $viewModel.enrichInpaintPromptWithVLM)
                        .font(.caption)

                    if viewModel.processArea == nil {
                        Text("Draw a fill region on the preview before generating.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    } else {
                        Text(viewModel.processAreaDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Processing Area Section

    @ViewBuilder
    private var processingAreaSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Processing Area", systemImage: "rectangle.dashed")
                .font(.headline)

            if !viewModel.referenceImages.isEmpty {
                switch viewModel.editMode {
                case .promptEdit:
                    Text("Barn doors set the aspect ratio of the region sent to the model; the megapixel budget sets its resolution.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                case .generativeFill:
                    Text("The megapixel budget caps the working resolution for generative fill. Larger images are scaled down before denoising.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                GroupBox(viewModel.editMode == .generativeFill ? "Resolution cap" : "Megapixel budget") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 10) {
                            Slider(
                                value: $viewModel.megapixelBudget,
                                in: ImageGenerationViewModel.minMegapixelBudget...ImageGenerationViewModel.maxMegapixelBudget,
                                step: 0.25
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

                        Text(
                            viewModel.editMode == .generativeFill
                                ? "Maximum total pixels for the fill pass."
                                : "Maximum total pixels to generate. The barn-door aspect ratio fills this budget."
                        )
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack {
                    if viewModel.editMode == .promptEdit {
                        Button("Reset Context") {
                            viewModel.resetContextArea()
                        }
                        .controlSize(.small)
                    } else {
                        Button("Clear Fill Region") {
                            viewModel.clearProcessSelection()
                        }
                        .controlSize(.small)
                        .disabled(viewModel.processArea == nil)
                    }

                    Spacer()
                }
            } else {
                Text("Add a reference image to set the processing area.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Model Selection Section

    @ViewBuilder
    private var modelSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Model Configuration", systemImage: "cpu")
                .font(.headline)

            // Family — chosen first; drives the pixel factor and which models are
            // available. Pre-selected today, so the gating below stays dormant.
            HStack {
                Text("Family:")
                    .frame(width: 100, alignment: .leading)
                Picker("", selection: $viewModel.selectedFamily) {
                    ForEach(ModelFamily.allCases) { family in
                        Text(family.displayName).tag(family as ModelFamily?)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }

            // Pixel-alignment factor — a read-only property of the family.
            if let family = viewModel.selectedFamily {
                HStack {
                    Text("Pixel factor:")
                        .frame(width: 100, alignment: .leading)
                    Text("\(family.pixelAlignment) px")
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .help("Output and conditioning dimensions are floored to this multiple. Determined by the model family; not adjustable.")
                    Spacer()
                }
            }

            // Model type picker
            HStack {
                Text("Model:")
                    .frame(width: 100, alignment: .leading)
                Picker("", selection: $viewModel.selectedModel) {
                    ForEach(viewModel.selectableModels, id: \.self) { model in
                        Text(modelSelectionTitle(for: model)).tag(model)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }
            .disabled(!viewModel.isFamilySelected)

            // Text encoder quantization (only for Dev)
            if viewModel.selectedModel == .dev {
                HStack {
                    Text("Text Encoder:")
                        .frame(width: 100, alignment: .leading)
                    Picker("", selection: $viewModel.textQuantization) {
                        ForEach(viewModel.downloadedTextQuantizations(in: modelManager.downloadedModels), id: \.self) { quant in
                            Text(quant.displayName).tag(quant)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity)
                }
                .disabled(!viewModel.isFamilySelected)
            }

            // Transformer quantization
            HStack {
                Text("Transformer:")
                    .frame(width: 100, alignment: .leading)
                Picker("", selection: $viewModel.transformerQuantization) {
                    ForEach(viewModel.compatibleTransformerQuantizations, id: \.self) { quant in
                        Text(transformerSelectionTitle(for: quant)).tag(quant)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            }
            .disabled(!viewModel.isFamilySelected)

            // Memory and download status
            HStack {
                Image(systemName: "memorychip")
                    .foregroundStyle(.secondary)
                Text("~\(viewModel.estimatedPeakMemoryGB)GB")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                let variant = viewModel.selectedTransformerVariant
                if modelManager.isTransformerDownloaded(variant) && modelManager.isVAEDownloaded {
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                } else {
                    VStack(alignment: .trailing, spacing: 4) {
                        if !modelManager.isTransformerDownloaded(variant) {
                            Button("Download Transformer") {
                                Task { await modelManager.downloadTransformer(variant) }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                        if !modelManager.isVAEDownloaded {
                            Button("Download VAE") {
                                Task { await modelManager.downloadVAE() }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                    }
                    .disabled(modelManager.isDownloading)
                }
            }

            if modelManager.isDownloading {
                ProgressView(value: modelManager.downloadProgress)
                Text(modelManager.downloadMessage)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let error = modelManager.errorMessage {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Prompt Section

    @ViewBuilder
    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if viewModel.editMode == .generativeFill {
                Label("Optional hint", systemImage: "text.cursor")
                    .font(.headline)
                Text("With Qwen3.5 VLM on, leave this empty — the VLM writes the Flux prompt from the image. Add a short hint only if you want to steer the fill.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Label("AI Prompt", systemImage: "text.cursor")
                    .font(.headline)
            }

            TextEditor(text: $viewModel.prompt)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.black.opacity(0.22), lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.18), radius: 2, x: 0, y: 1)
                .frame(minHeight: 96, maxHeight: 160)

            HStack(spacing: 16) {
                if viewModel.editMode != .generativeFill {
                    Toggle("Upsample prompt", isOn: $viewModel.upsamplePrompt)
                        .help("Enhance prompt using VLM to analyze reference images")
                }

                Toggle("Clear prompt after generation", isOn: $viewModel.clearPromptAfterGeneration)
                    .help("Empty the prompt automatically once a run finishes successfully")
            }
            .font(.caption)
            .toggleStyle(.checkbox)

            if let upsampled = viewModel.upsampledPrompt {
                upsampledPromptView(upsampled)
            }
        }
    }

    @ViewBuilder
    private func upsampledPromptView(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Upsampled prompt", systemImage: "sparkles")
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            ScrollView {
                Text(text)
                    .font(.caption)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 120)
            .padding(8)
            .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.accentColor.opacity(0.3), lineWidth: 1)
            )
        }
    }

    // MARK: - I2I Parameters Section (removed - strength is not used by Flux.2 conditioning mode)

    // MARK: - Parameters Section

    @ViewBuilder
    private var parametersSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Generation Parameters", systemImage: "slider.horizontal.3")
                .font(.headline)

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
                if let firstRef = viewModel.referenceImages.first {
                    Button(action: {
                        viewModel.width = firstRef.image.width
                        viewModel.height = firstRef.image.height
                    }) {
                        Label("Match Reference Size", systemImage: "arrow.up.left.and.arrow.down.right")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            // Steps and Guidance
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Steps: \(viewModel.steps)")
                        .font(.caption)
                    Slider(value: Binding(
                        get: { Double(viewModel.steps) },
                        set: { viewModel.steps = Int($0) }
                    ), in: 4...100, step: 1)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Guidance: \(String(format: "%.1f", viewModel.guidance))")
                            .font(.caption)
                        Spacer()
                        Button("Default") {
                            viewModel.resetGuidanceToModelDefault()
                        }
                        .controlSize(.mini)
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.guidance) },
                        set: { viewModel.guidance = Float($0) }
                    ), in: 1...10, step: 0.5)
                }
            }

            Divider()

            // Seed
            HStack {
                Text("Seed:")
                    .font(.caption)
                TextField("Random", text: $viewModel.seed)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                Button(action: {
                    viewModel.seed = String(UInt64.random(in: 0...UInt64.max))
                }) {
                    Image(systemName: "dice")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Interpret Images Section

    @ViewBuilder
    private var interpretImagesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Interpret Images (Optional)", systemImage: "eye")
                    .font(.subheadline.bold())

                Spacer()

                Button(action: selectInterpretImage) {
                    Label("Add", systemImage: "plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Text("VLM will analyze these images and inject descriptions into the prompt. Different from reference images - these provide semantic context only.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if !viewModel.interpretImageURLs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(viewModel.interpretImageURLs, id: \.self) { url in
                            InterpretImageThumbnail(url: url) {
                                viewModel.interpretImageURLs.removeAll { $0 == url }
                            }
                        }
                    }
                }
                .frame(height: 60)
            }
        }
    }

    // MARK: - Generate Section

    @ViewBuilder
    private var generateSection: some View {
        VStack(spacing: 12) {
            if viewModel.isResetting {
                ProgressView("Resetting…")
                    .progressViewStyle(.linear)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            } else {
                Button(action: {
                    viewModel.startGeneration()
                }) {
                    HStack {
                        if viewModel.isGenerating {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(viewModel.isGenerating ? "Generating..." : "Generate Image")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!canGenerate)
            }

            if viewModel.isGenerating {
                Button(role: .cancel) {
                    viewModel.cancel()
                } label: {
                    Text("Cancel")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .tint(.red)
            }

            if viewModel.isGenerating {
                VStack(spacing: 4) {
                    ProgressView(value: viewModel.progress)
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let error = viewModel.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                .padding(8)
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }
        }
    }

    // MARK: - Output Section

    @ViewBuilder
    private var outputSection: some View {
        VStack(spacing: 0) {
            HStack {
                Label(
                    viewModel.generatedImage == nil && !viewModel.referenceImages.isEmpty ? "Image Preview" : "Generated Image",
                    systemImage: "photo"
                )
                    .font(.headline)

                Spacer()

                if viewModel.generatedImage != nil {
                    Button(action: { viewModel.saveImage() }) {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button(action: { useAsReference() }) {
                        Label("Use as Reference", systemImage: "arrow.uturn.left")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.referenceImages.count >= viewModel.selectedModel.maxReferenceImages)
                }

                Button(action: { viewModel.clearPreview() }) {
                    Label("Clear Preview", systemImage: "xmark.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!viewModel.hasPreviewContent)
                .help("Clear the generated image from the preview pane")

                // Save-related controls (flush-right) once there's an image to work with.
                if !viewModel.referenceImages.isEmpty {
                    Divider()
                        .frame(height: 16)

                    LanczosUpscaleField(factor: $imageSaveUpscaleBy)

                    Button(action: { viewModel.openOutputFolder() }) {
                        Label("Open Folder", systemImage: "folder")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Open the image output folder in Finder")

                    Button(action: { viewModel.saveInputImage() }) {
                        Label("Save Input", systemImage: "square.and.arrow.down.on.square")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(viewModel.lastSavedImageURL == nil)
                    .help("Save the formatted input (before barn doors) as <name>-input")
                }
            }
            .padding()

            Divider()

            // Checkpoints row (if available)
            if viewModel.showCheckpoints && !viewModel.checkpointImages.isEmpty {
                checkpointsSection
                Divider()
            }

            // Main image display
            GeometryReader { geometry in
                if let cgImage = viewModel.generatedImage {
                    ScrollView([.horizontal, .vertical]) {
                        Image(decorative: cgImage, scale: 1.0)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(
                                maxWidth: geometry.size.width,
                                maxHeight: geometry.size.height
                            )
                    }
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .background(Color(nsColor: .windowBackgroundColor))
                } else if let firstRef = viewModel.referenceImages.first {
                    VStack(spacing: 10) {
                        ImagePreparationPreview(
                            image: firstRef.image,
                            adjustedSize: viewModel.adjustedGenerationSize,
                            sizingMethod: viewModel.sizingMethod,
                            overlayOpacity: viewModel.preparationOverlayOpacity,
                            editMode: viewModel.editMode,
                            contextArea: Binding(
                                get: { viewModel.contextArea },
                                set: { viewModel.setContextArea($0) }
                            ),
                            processArea: Binding(
                                get: { viewModel.processArea },
                                set: { viewModel.setProcessArea($0) }
                            )
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                        PreparationSizeInfoRow(
                            image: firstRef.image,
                            contextArea: viewModel.contextArea,
                            adjustedSize: viewModel.adjustedGenerationSize
                        )
                        .padding(.horizontal)
                        .padding(.bottom, 10)
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

                        if viewModel.referenceImages.isEmpty {
                            Text("Add at least one reference image to start")
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

    private var canGenerate: Bool {
        viewModel.canGenerate &&
        !viewModel.referenceImages.isEmpty &&
        modelManager.isTransformerDownloaded(viewModel.selectedTransformerVariant) &&
        modelManager.isVAEDownloaded
    }

    private func modelSelectionTitle(for model: Flux2Model) -> String {
        let hasDownloadedTransformer = ImageGenerationViewModel.compatibleTransformerQuantizations(for: model).contains { quantization in
            let variant = ModelRegistry.TransformerVariant.variant(for: model, quantization: quantization)
            return modelManager.isTransformerDownloaded(variant)
        }
        return hasDownloadedTransformer ? model.displayName : "\(model.displayName) (not downloaded)"
    }

    private func transformerSelectionTitle(for quantization: TransformerQuantization) -> String {
        let variant = ModelRegistry.TransformerVariant.variant(for: viewModel.selectedModel, quantization: quantization)
        let suffix = modelManager.isTransformerDownloaded(variant) ? "" : " (not downloaded)"
        return "\(quantization.displayName)\(suffix)"
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK {
            for url in panel.urls.prefix(viewModel.selectedModel.maxReferenceImages - viewModel.referenceImages.count) {
                viewModel.addReferenceImage(from: url)
            }
        }
    }

    private func selectInterpretImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true

        if panel.runModal() == .OK {
            viewModel.interpretImageURLs.append(contentsOf: panel.urls)
        }
    }

    private func handleImageDrop(_ providers: [NSItemProvider]) {
        for provider in providers {
            if provider.canLoadObject(ofClass: NSImage.self) {
                _ = provider.loadObject(ofClass: NSImage.self) { image, _ in
                    if let nsImage = image as? NSImage {
                        DispatchQueue.main.async {
                            viewModel.addReferenceImage(from: nsImage)
                        }
                    }
                }
            }
        }
    }

    private func useAsReference() {
        guard let cgImage = viewModel.generatedImage else { return }
        viewModel.addReferenceImage(cgImage: cgImage)
    }
}

// MARK: - Supporting Views

struct ReferenceImageSlot: View {
    let image: ReferenceImage
    let onRemove: () -> Void

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Image(nsImage: image.thumbnail)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 100, height: 100)
                .clipped()
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.accentColor, lineWidth: 2)
                )

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.white)
                    .background(Circle().fill(.red))
            }
            .buttonStyle(.plain)
            .offset(x: 8, y: -8)
        }
    }
}

struct AddImageSlot: View {
    let onAdd: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: onAdd) {
            VStack {
                Image(systemName: "plus.circle.fill")
                    .font(.title)
                Text("Add")
                    .font(.caption)
            }
            .foregroundStyle(isHovering ? Color.accentColor : .secondary)
            .frame(width: 100, height: 100)
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
        ZStack(alignment: .topTrailing) {
            if let nsImage = NSImage(contentsOf: url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 50, height: 50)
                    .clipped()
                    .cornerRadius(4)
            } else {
                Rectangle()
                    .fill(.secondary.opacity(0.2))
                    .frame(width: 50, height: 50)
                    .cornerRadius(4)
            }

            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .background(Circle().fill(.red).frame(width: 14, height: 14))
            }
            .buttonStyle(.plain)
            .offset(x: 4, y: -4)
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
    let editMode: ImageEditMode
    @Binding var contextArea: CGRect
    @Binding var processArea: CGRect?

    @State private var activeDrag: DragMode?
    @State private var selectionStart: CGPoint?
    /// Context rect captured at drag start, restored if the drag is too small
    /// to be an intentional new region (so a stray click doesn't collapse it).
    @State private var contextBeforeDrag: CGRect?
    @State private var processBeforeDrag: CGRect?

    private enum DragMode {
        case contextLeft
        case contextRight
        case contextTop
        case contextBottom
        case contextSelection
        case processSelection
    }

    private var usesBarnDoors: Bool { editMode == .promptEdit }
    private var usesFillMask: Bool { editMode == .generativeFill }

    var body: some View {
        GeometryReader { geometry in
            let imageRect = fittedImageRect(in: geometry.size)
            let showsFormattingOverlay = usesBarnDoors && activeDrag == nil && !formattingExcludedRects().isEmpty
            let showsBarnDoorOverlay = usesBarnDoors && !isContextFullyOpen

            ZStack {
                Color(nsColor: .textBackgroundColor)

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
                    contextExclusionOverlay(in: imageRect)
                        .fill(
                            Color.black.opacity(0.72),
                            style: FillStyle(eoFill: false, antialiased: false)
                        )
                        .allowsHitTesting(false)
                }

                if usesFillMask, let processArea {
                    inpaintMaskOverlay(for: processArea, in: imageRect)
                        .allowsHitTesting(false)
                }
            }
            .compositingGroup()
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
            )
            .contentShape(Rectangle())
            .gesture(dragGesture(in: imageRect))
            .onContinuousHover { phase in
                #if canImport(AppKit)
                switch phase {
                case .active(let location):
                    cursor(for: location, in: imageRect).set()
                case .ended:
                    NSCursor.arrow.set()
                }
                #endif
            }
        }
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

    private func contextExclusionOverlay(in imageRect: CGRect) -> Path {
        let bands = exclusionBands(
            outer: CGRect(x: 0, y: 0, width: 1, height: 1),
            inner: contextArea
        )
        let contextRect = pixelAligned(displayRect(for: contextArea, in: imageRect))
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

    private func dragGesture(in imageRect: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                if activeDrag == nil {
                    // Decide the drag mode from the press location. Barn-door
                    // edges win within their 20px zone; anywhere else inside the
                    // context area starts a selection.
                    activeDrag = dragMode(for: value.startLocation, in: imageRect)
                    selectionStart = normalizedPoint(for: value.startLocation, in: imageRect)
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
                    updateProcessSelection(to: point, in: imageRect)
                case nil:
                    break
                }
            }
            .onEnded { _ in
                activeDrag = nil
                selectionStart = nil
                contextBeforeDrag = nil
                processBeforeDrag = nil
                #if canImport(AppKit)
                NSCursor.arrow.set()
                #endif
            }
    }

    private func dragMode(for location: CGPoint, in imageRect: CGRect) -> DragMode? {
        guard imageRect.contains(location) else { return nil }

        if usesFillMask {
            return .processSelection
        }

        // Barn-door edges take priority within their 20px grab zone, so an edge
        // adjustment never starts a new region by accident.
        if let edge = contextEdge(at: location, in: imageRect) {
            return edge
        }

        return .contextSelection
    }

    private func inpaintMaskOverlay(for normalizedRect: CGRect, in imageRect: CGRect) -> some View {
        let rect = pixelAligned(displayRect(for: normalizedRect, in: imageRect))
        return ZStack {
            Path { path in
                path.addRect(imageRect)
                path.addRect(rect)
            }
            .fill(Color.black.opacity(0.45), style: FillStyle(eoFill: true, antialiased: false))

            Path { path in
                path.addRect(rect)
            }
            .stroke(
                Color.accentColor,
                style: StrokeStyle(lineWidth: 2, dash: [8, 6])
            )
        }
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
    private func cursor(for location: CGPoint, in imageRect: CGRect) -> NSCursor {
        if usesFillMask {
            return imageRect.contains(location) ? .crosshair : .arrow
        }

        // Within 20px of a barn-door edge -> resize. Anywhere else on the image
        // -> crosshair (draw a new region). Outside image -> arrow.
        switch contextEdge(at: location, in: imageRect) {
        case .contextLeft, .contextRight:
            return .resizeLeftRight
        case .contextTop, .contextBottom:
            return .resizeUpDown
        default:
            return imageRect.contains(location) ? .crosshair : .arrow
        }
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

struct PreparationSizeInfoRow: View {
    let image: CGImage
    let contextArea: CGRect?
    let adjustedSize: (width: Int, height: Int)

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                sizeItem(label: "Original", value: "\(image.width)x\(image.height)")

                verticalSeparator

                sizeItem(label: "Context", value: contextValue)

                verticalSeparator

                sizeItem(label: "Adjusted", value: "\(adjustedSize.width)x\(adjustedSize.height)")

                verticalSeparator

                sizeItem(label: "Pixels", value: megapixelValue)

                Spacer(minLength: 0)
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


#Preview {
    ImageToImageView()
        .environmentObject(ModelManager())
        .frame(width: 1200, height: 900)
}
