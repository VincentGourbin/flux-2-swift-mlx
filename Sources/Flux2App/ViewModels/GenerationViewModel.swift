// GenerationViewModel.swift - Generation logic and state
// Copyright 2025 Vincent Gourbin

import SwiftUI
import Flux2Core

/// ViewModel for image generation
@MainActor
class GenerationViewModel: ObservableObject {
    // MARK: - Input Parameters

    @Published var prompt: String = ""
    @Published var width: Int = 1024
    @Published var height: Int = 1024
    @Published var steps: Int = 50
    @Published var guidance: Float = 4.0
    @Published var useSeed: Bool = false
    @Published var seed: UInt64 = 0

    // Reference images for I2I mode
    @Published var referenceImages: [NSImage] = []

    // MARK: - Quantization Settings

    @AppStorage("textEncoderQuantization") var textQuantization: String = "8bit"
    @AppStorage("transformerQuantization") var transformerQuantization: String = "qint8"

    // MARK: - Generation State

    @Published var isGenerating: Bool = false
    @Published var currentStep: Int = 0
    @Published var progress: Double = 0.0
    @Published var generatedImage: NSImage?
    @Published var errorMessage: String?

    // MARK: - Pipeline

    private var pipeline: Flux2Pipeline?

    // MARK: - Generation Methods

    /// Generate image based on selected mode
    /// - Parameter mode: 0 for T2I, 1 for I2I
    func generate(mode: Int) async {
        guard !isGenerating else { return }
        guard !prompt.isEmpty else {
            errorMessage = "Please enter a prompt"
            return
        }

        isGenerating = true
        errorMessage = nil
        currentStep = 0
        progress = 0.0

        do {
            // Create quantization config
            let textQuant = MistralQuantization(rawValue: textQuantization) ?? .mlx8bit
            let transQuant = TransformerQuantization(rawValue: transformerQuantization) ?? .qint8

            let quantConfig = Flux2QuantizationConfig(
                textEncoder: textQuant,
                transformer: transQuant
            )

            // Create or reuse pipeline
            if pipeline == nil {
                pipeline = Flux2Pipeline(quantization: quantConfig)
            }

            let seedValue: UInt64? = useSeed ? seed : nil

            let cgImage: CGImage

            if mode == 0 {
                // Text-to-Image
                cgImage = try await pipeline!.generateTextToImage(
                    prompt: prompt,
                    height: height,
                    width: width,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue
                ) { [weak self] current, total in
                    Task { @MainActor in
                        self?.currentStep = current
                        self?.progress = Double(current) / Double(total)
                    }
                }
            } else {
                // Image-to-Image
                let refCGImages = referenceImages.compactMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
                guard !refCGImages.isEmpty else {
                    throw Flux2Error.invalidConfiguration("No reference images provided")
                }

                cgImage = try await pipeline!.generateImageToImage(
                    prompt: prompt,
                    images: refCGImages,
                    steps: steps,
                    guidance: guidance,
                    seed: seedValue
                ) { [weak self] current, total in
                    Task { @MainActor in
                        self?.currentStep = current
                        self?.progress = Double(current) / Double(total)
                    }
                }
            }

            // Convert to NSImage
            let size = NSSize(width: cgImage.width, height: cgImage.height)
            generatedImage = NSImage(cgImage: cgImage, size: size)

        } catch {
            errorMessage = error.localizedDescription
        }

        isGenerating = false
    }

    /// Save generated image to file
    func saveImage(to url: URL) throws {
        guard let image = generatedImage,
              let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else {
            throw Flux2Error.imageProcessingFailed("No image to save")
        }

        let data: Data?
        if url.pathExtension.lowercased() == "png" {
            data = bitmap.representation(using: .png, properties: [:])
        } else {
            data = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
        }

        guard let imageData = data else {
            throw Flux2Error.imageProcessingFailed("Failed to encode image")
        }

        try imageData.write(to: url)
    }

    /// Clear generated image
    func clearImage() {
        generatedImage = nil
    }

    /// Clear reference images
    func clearReferenceImages() {
        referenceImages.removeAll()
    }

    /// Add reference image
    func addReferenceImage(_ image: NSImage) {
        if referenceImages.count < 3 {
            referenceImages.append(image)
        }
    }

    // MARK: - Presets

    func applyPreset(_ preset: GenerationPreset) {
        width = preset.width
        height = preset.height
        steps = preset.steps
        guidance = preset.guidance
    }
}

// MARK: - Generation Presets

struct GenerationPreset {
    let name: String
    let width: Int
    let height: Int
    let steps: Int
    let guidance: Float

    static let standard = GenerationPreset(
        name: "Standard",
        width: 1024,
        height: 1024,
        steps: 50,
        guidance: 4.0
    )

    static let quick = GenerationPreset(
        name: "Quick",
        width: 512,
        height: 512,
        steps: 28,
        guidance: 3.5
    )

    static let highQuality = GenerationPreset(
        name: "High Quality",
        width: 1024,
        height: 1024,
        steps: 100,
        guidance: 4.5
    )

    static let portrait = GenerationPreset(
        name: "Portrait",
        width: 768,
        height: 1024,
        steps: 50,
        guidance: 4.0
    )

    static let landscape = GenerationPreset(
        name: "Landscape",
        width: 1024,
        height: 768,
        steps: 50,
        guidance: 4.0
    )

    static let allPresets: [GenerationPreset] = [
        .standard, .quick, .highQuality, .portrait, .landscape
    ]
}
