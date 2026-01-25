/**
 * ImageGenerationViewModel.swift
 * ViewModel for Flux.2 image generation (T2I and I2I)
 */

import SwiftUI
import Flux2Core
import FluxTextEncoders
import CoreGraphics
import MLX

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Generation Mode

enum GenerationMode: String, CaseIterable {
    case textToImage = "Text to Image"
    case imageToImage = "Image to Image"
}

// MARK: - Image Generation ViewModel

@MainActor
class ImageGenerationViewModel: ObservableObject {
    // MARK: - Model Selection
    @Published var selectedModel: Flux2Model = .klein4B
    @Published var textQuantization: MistralQuantization = .mlx8bit
    @Published var transformerQuantization: TransformerQuantization = .qint8

    // MARK: - Prompt
    @Published var prompt: String = ""
    @Published var upsamplePrompt: Bool = false

    // MARK: - T2I Parameters
    @Published var width: Int = 1024
    @Published var height: Int = 1024
    @Published var steps: Int = 50
    @Published var guidance: Float = 4.0
    @Published var seed: String = ""  // Empty = random

    // MARK: - I2I Parameters
    @Published var referenceImages: [ReferenceImage] = []
    @Published var strength: Float = 0.8
    @Published var interpretImageURLs: [URL] = []  // VLM interpretation images

    // MARK: - State
    @Published var isGenerating: Bool = false
    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 0
    @Published var generatedImage: CGImage?
    @Published var errorMessage: String?
    @Published var statusMessage: String = ""

    // MARK: - Checkpoints
    @Published var checkpointImages: [CheckpointImage] = []
    @Published var checkpointInterval: Int = 10  // Save checkpoint every N steps
    @Published var showCheckpoints: Bool = true

    // MARK: - Pipeline
    private var pipeline: Flux2Pipeline?

    // MARK: - Init with defaults
    init() {
        applyRecommendedDefaults(for: selectedModel)
    }

    // MARK: - Computed Properties

    var seedValue: UInt64? {
        guard !seed.isEmpty else { return nil }
        return UInt64(seed)
    }

    var canGenerate: Bool {
        !prompt.isEmpty && !isGenerating
    }

    var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(currentStep) / Double(totalSteps)
    }

    /// Estimated peak memory based on current configuration
    var estimatedPeakMemoryGB: Int {
        let textEncoderMem: Int
        let transformerMem: Int

        switch selectedModel {
        case .dev:
            textEncoderMem = textQuantization.estimatedMemoryGB
            transformerMem = transformerQuantization == .bf16 ? 64 : 32
        case .klein4B:
            // Qwen3-4B is smaller
            textEncoderMem = 5  // Qwen3-4B 8bit
            transformerMem = transformerQuantization == .bf16 ? 8 : 4
        case .klein9B:
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

    // MARK: - Image Management

    /// Add a reference image from URL
    func addReferenceImage(from url: URL) {
        guard referenceImages.count < 3 else { return }

        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            errorMessage = "Failed to load image from \(url.lastPathComponent)"
            return
        }

        let refImage = ReferenceImage(
            id: UUID(),
            url: url,
            image: cgImage,
            thumbnail: createThumbnail(from: nsImage)
        )
        referenceImages.append(refImage)
    }

    /// Add a reference image from NSImage (drag & drop)
    func addReferenceImage(from nsImage: NSImage) {
        guard referenceImages.count < 3 else { return }

        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
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
    }

    /// Remove a reference image
    func removeReferenceImage(_ id: UUID) {
        referenceImages.removeAll { $0.id == id }
    }

    /// Clear all reference images
    func clearReferenceImages() {
        referenceImages.removeAll()
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

    // MARK: - Generation

    /// Generate image (T2I or I2I based on reference images)
    func generate() async {
        guard canGenerate else { return }

        isGenerating = true
        errorMessage = nil
        generatedImage = nil
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
                    self.statusMessage = message
                }
            }

            // Generate
            let image: CGImage
            let interpretPaths = interpretImageURLs.map { $0.path }

            // Checkpoint callback
            var checkpointCallback: (@Sendable (Int, CGImage) -> Void)? = nil
            if showCheckpoints {
                checkpointCallback = { [weak self] step, checkpointImage in
                    Task { @MainActor in
                        self?.addCheckpoint(image: checkpointImage, step: step)
                    }
                }
            }

            if referenceImages.isEmpty {
                // Text-to-Image
                statusMessage = "Generating image..."
                image = try await pipeline!.generateTextToImage(
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
                            self.currentStep = current
                            self.totalSteps = total
                            self.statusMessage = "Step \(current)/\(total)"
                        }
                    },
                    onCheckpoint: checkpointCallback
                )
            } else {
                // Image-to-Image
                statusMessage = "Generating with \(referenceImages.count) reference image(s)..."
                let cgImages = referenceImages.map { $0.image }

                image = try await pipeline!.generateImageToImage(
                    prompt: prompt,
                    images: cgImages,
                    interpretImagePaths: interpretPaths.isEmpty ? nil : interpretPaths,
                    height: height,
                    width: width,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue,
                    strength: strength,
                    upsamplePrompt: upsamplePrompt,
                    checkpointInterval: showCheckpoints ? checkpointInterval : nil,
                    onProgress: { current, total in
                        Task { @MainActor in
                            self.currentStep = current
                            self.totalSteps = total
                            self.statusMessage = "Step \(current)/\(total)"
                        }
                    },
                    onCheckpoint: checkpointCallback
                )
            }

            generatedImage = image
            statusMessage = "Generation complete!"

        } catch {
            errorMessage = error.localizedDescription
            statusMessage = "Generation failed"
        }

        isGenerating = false
    }

    /// Cancel generation (if possible)
    func cancel() {
        // Pipeline doesn't support cancellation yet, but we can clear state
        isGenerating = false
        statusMessage = "Cancelled"
    }

    /// Save generated image to file
    func saveImage() {
        guard let image = generatedImage else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "flux_generated_\(Date().timeIntervalSince1970).png"

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try saveImage(image, to: url)
                statusMessage = "Saved to \(url.lastPathComponent)"
            } catch {
                errorMessage = "Failed to save: \(error.localizedDescription)"
            }
        }
    }

    private func saveImage(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw NSError(domain: "ImageSave", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not create image destination"])
        }

        CGImageDestinationAddImage(destination, image, nil)

        if !CGImageDestinationFinalize(destination) {
            throw NSError(domain: "ImageSave", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not finalize image"])
        }
    }

    /// Clear pipeline to free memory
    func clearPipeline() async {
        await pipeline?.clearAll()
        pipeline = nil
        Memory.clearCache()
        statusMessage = "Pipeline cleared"
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

        case .klein4B:
            // Flux.2 Klein 4B - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
            // 4 steps, guidance 1.0, optimized for sub-second generation
            transformerQuantization = .qint8
            width = 1024
            height = 1024
            steps = 4
            guidance = 1.0
            checkpointInterval = 1

        case .klein9B:
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
