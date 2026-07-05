/**
 * ImageGenerationViewModel+Generation.swift
 * Generation lifecycle (T2I / I2I / local fill / outpaint), cancel, image &
 * input saving, preview caching, session state persistence, recommended
 * defaults, presets, and checkpoints.
 */

import AppKit
import CoreGraphics
import Flux2Chains
import Flux2Core
import FluxTextEncoders
import MLX
import SwiftUI

extension ImageGenerationViewModel {
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
                appendEditHistoryAfterGenerate(image: image)
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

    func cacheFormattedComparisonImage(for output: CGImage) {
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

    func restoreSessionStateIfNeeded() {
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
