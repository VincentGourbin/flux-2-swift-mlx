/**
 * FluxTextEncoders.swift
 * Main entry point for FLUX.2 Text Encoders library
 *
 * Supports:
 * - Mistral Small 3.2 (FLUX.2 dev) - 24B VLM
 * - Qwen3 4B/8B (FLUX.2 Klein) - Text encoder
 *
 * Swift MLX implementation for Apple Silicon
 */

import Foundation
import MLX
import Tokenizers
import ImageIO
import CoreGraphics

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Public API

/// Main interface for FLUX.2 text encoder operations
/// Thread-safe: load/unload on main thread, inference can run on any thread
public final class FluxTextEncoders: @unchecked Sendable {
    /// Shared singleton instance
    public static let shared = FluxTextEncoders()
    public static let version = "0.1.0"

    private var model: MistralForCausalLM?
    private var vlmModel: MistralVLM?
    private var tokenizer: TekkenTokenizer?
    private var generator: MistralGenerator?
    private var extractor: EmbeddingExtractor?
    private var imageProcessor: ImageProcessor?
    
    // Qwen3/Klein support
    private var qwen3Model: Qwen3ForCausalLM?
    private var kleinExtractor: KleinEmbeddingExtractor?
    private var qwen3Tokenizer: Tokenizer?
    private var loadedKleinVariant: KleinVariant?
    private var qwen3Generator: Qwen3Generator?

    // Qwen3-VL/Klein VL support (experimental)
    private var qwen3VLModel: Qwen3VLForCausalLM?
    private var kleinVLExtractor: KleinVLEmbeddingExtractor?
    private var qwen3VLTokenizer: Tokenizer?
    private var qwen3VLGenerator: Qwen3VLGenerator?
    private var isKleinVLMode: Bool = false

    // Qwen3.5 VLM service
    private var qwen35VLM: Qwen35VLM?

    /// Whether VLM (vision) model is loaded
    public var isVLMLoaded: Bool {
        return vlmModel != nil && tokenizer != nil && imageProcessor != nil
    }
    
    /// Whether Qwen3/Klein model is loaded
    public var isKleinLoaded: Bool {
        return qwen3Model != nil && qwen3Tokenizer != nil && kleinExtractor != nil
    }
    
    /// Get the loaded Klein variant
    public var kleinVariant: KleinVariant? {
        return loadedKleinVariant
    }

    /// Access VLM directly for advanced use (e.g., LoRA evaluator custom prompts)
    public var qwen35VLMForEvaluation: Qwen35VLM? {
        return qwen35VLM
    }

    /// Public comparison parser for use by LoRA evaluator
    public func parseComparisonForEvaluation(_ text: String) -> FluxImageComparison {
        return parseComparisonResult(text)
    }

    /// Whether Qwen3-VL/Klein VL model is loaded
    public var isKleinVLLoaded: Bool {
        return qwen3VLModel != nil && qwen3VLTokenizer != nil && kleinVLExtractor != nil
    }

    private init() {}

    /// Check if model is loaded
    public var isModelLoaded: Bool {
        return model != nil && tokenizer != nil
    }

    /// Load model from path or download if needed
    @MainActor
    public func loadModel(
        variant: ModelVariant = .mlx8bit,
        hfToken: String? = nil,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws {
        let downloader = TextEncoderModelDownloader(hfToken: hfToken)
        let modelPath = try await downloader.download(variant: variant, progress: progress)

        try loadModel(from: modelPath.path)
    }

    /// Load model from local path
    @MainActor
    public func loadModel(from path: String) throws {
        FluxDebug.log("Loading model from \(path)")

        // Load tokenizer
        tokenizer = TekkenTokenizer(modelPath: path)

        // Load model
        model = try MistralForCausalLM.load(from: path)

        // Create generator and extractor
        if let model = model, let tokenizer = tokenizer {
            generator = MistralGenerator(model: model, tokenizer: tokenizer)
            extractor = EmbeddingExtractor(model: model, tokenizer: tokenizer)
        }

        FluxDebug.log("Model loaded successfully")
    }

    /// Load VLM (vision-language) model from path
    @MainActor
    public func loadVLMModel(from path: String) throws {
        let debug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

        if debug { print("[Core] Loading VLM from \(path)"); fflush(stdout) }

        // Load tokenizer
        tokenizer = TekkenTokenizer(modelPath: path)

        // Load VLM model
        vlmModel = try MistralVLM.load(from: path)

        // Initialize image processor
        imageProcessor = ImageProcessor(config: .pixtral)

        // Also set up text-only generator using the language model
        if let vlm = vlmModel, let tokenizer = tokenizer {
            generator = MistralGenerator(model: vlm.languageModel, tokenizer: tokenizer)
            extractor = EmbeddingExtractor(model: vlm.languageModel, tokenizer: tokenizer)
            model = vlm.languageModel
        }

        if debug { print("[Core] VLM loading complete!"); fflush(stdout) }
    }

    /// Load VLM from path or download if needed
    @MainActor
    public func loadVLMModel(
        variant: ModelVariant = .mlx4bit,
        hfToken: String? = nil,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws {
        let downloader = TextEncoderModelDownloader(hfToken: hfToken)
        let modelPath = try await downloader.download(variant: variant, progress: progress)

        try loadVLMModel(from: modelPath.path)
    }

    /// Unload model to free memory
    @MainActor
    public func unloadModel() {
        model = nil
        vlmModel = nil
        tokenizer = nil
        generator = nil
        extractor = nil
        imageProcessor = nil
        qwen3Model = nil
        kleinExtractor = nil
        qwen3Tokenizer = nil
        loadedKleinVariant = nil
        Memory.clearCache()
        FluxDebug.log("Model unloaded")
    }
    
    // MARK: - Klein/Qwen3 Loading
    
    /// Load Qwen3 model for Klein embeddings
    /// - Parameters:
    ///   - variant: Klein variant (klein4B or klein9B)
    ///   - modelPath: Local path to Qwen3 model
    @MainActor
    public func loadKleinModel(variant: KleinVariant, from modelPath: String) async throws {
        print("[Klein] Loading Qwen3 model for \(variant.displayName)")
        print("[Klein] Model path: \(modelPath)")

        // Verify path exists
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: modelPath) else {
            print("[Klein] ERROR: Path does not exist: \(modelPath)")
            throw FluxEncoderError.invalidInput("Model path does not exist: \(modelPath)")
        }

        // Check for required files
        let configPath = "\(modelPath)/config.json"
        let tokenizerPath = "\(modelPath)/tokenizer.json"
        print("[Klein] config.json exists: \(fileManager.fileExists(atPath: configPath))")
        print("[Klein] tokenizer.json exists: \(fileManager.fileExists(atPath: tokenizerPath))")

        // Load Qwen3 model
        print("[Klein] Loading model weights...")
        qwen3Model = try Qwen3ForCausalLM.load(from: modelPath)
        print("[Klein] Model weights loaded successfully")

        // CRITICAL: Limit GPU cache to prevent memory accumulation during repeated inference
        // This is essential for training where encode() is called many times
        // Without this limit, the GPU cache grows unbounded
        Memory.cacheLimit = 512 * 1024 * 1024  // 512 MB cache limit
        print("[Klein] GPU cache limit set to 512 MB")

        // Enable AGGRESSIVE memory optimization to prevent computation graph accumulation
        // Use aggressive preset: eval every 4 layers with cache clearing
        qwen3Model?.model.memoryConfig = .aggressive
        print("[Klein] Memory optimization enabled (aggressive: eval every 4 layers + cache clear)")

        // Load tokenizer using HuggingFace Tokenizers library
        // Use from(modelFolder:) for local paths (not from(pretrained:) which treats path as Hub ID)
        print("[Klein] Loading tokenizer from local path...")
        let modelFolderURL = URL(fileURLWithPath: modelPath)
        qwen3Tokenizer = try await AutoTokenizer.from(modelFolder: modelFolderURL)
        print("[Klein] Tokenizer loaded successfully")

        // Create Klein embedding extractor and Qwen3 generator
        if let model = qwen3Model, let tokenizer = qwen3Tokenizer {
            kleinExtractor = KleinEmbeddingExtractor(model: model, tokenizer: tokenizer, variant: variant)
            qwen3Generator = Qwen3Generator(model: model, tokenizer: tokenizer)
            loadedKleinVariant = variant
            print("[Klein] Extractor and generator created")
        }

        print("[Klein] Klein model loaded successfully for \(variant.displayName)")
    }
    
    /// Load Qwen3 model for Klein embeddings with automatic download
    /// - Parameters:
    ///   - variant: Klein variant (klein4B or klein9B)
    ///   - qwen3Variant: Specific Qwen3 model variant (default: recommended 8-bit)
    ///   - hfToken: HuggingFace token for downloads
    ///   - progress: Download progress callback
    @MainActor
    public func loadKleinModel(
        variant: KleinVariant,
        qwen3Variant: Qwen3Variant? = nil,
        hfToken: String? = nil,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws {
        // Get the appropriate Qwen3 model info
        let modelVariant: Qwen3Variant
        if let specified = qwen3Variant {
            modelVariant = specified
        } else {
            // Use recommended variant for the Klein variant
            modelVariant = variant == .klein4B ? .qwen3_4B_8bit : .qwen3_8B_8bit
        }
        
        guard let modelInfo = TextEncoderModelRegistry.shared.qwen3Model(withVariant: modelVariant) else {
            throw FluxEncoderError.invalidInput("Qwen3 model variant not found: \(modelVariant)")
        }

        print("[Klein] Loading variant: \(modelVariant), repoId: \(modelInfo.repoId)")

        // Download model (or get existing path)
        let downloader = TextEncoderModelDownloader(hfToken: hfToken)
        let modelPath = try await downloader.downloadQwen3(modelInfo, progress: progress)

        print("[Klein] Model path resolved: \(modelPath.path)")

        // Load from downloaded path
        try await loadKleinModel(variant: variant, from: modelPath.path)
    }
    
    /// Unload Klein model to free memory
    @MainActor
    public func unloadKleinModel() {
        qwen3Model = nil
        kleinExtractor = nil
        qwen3Tokenizer = nil
        loadedKleinVariant = nil
        qwen3Generator = nil
        // Also unload VL model if loaded
        qwen3VLModel = nil
        kleinVLExtractor = nil
        qwen3VLTokenizer = nil
        qwen3VLGenerator = nil
        isKleinVLMode = false
        Memory.clearCache()
        FluxDebug.log("Klein model unloaded")
    }

    // MARK: - Klein VL (Qwen3-VL) Loading

    /// Load Qwen3-VL model for Klein VL embeddings (experimental)
    /// This replaces the standard Qwen3 text encoder with Qwen3-VL (language component only)
    @MainActor
    public func loadKleinVLModel(variant: KleinVariant, from modelPath: String) async throws {
        print("[Klein-VL] Loading Qwen3-VL model for \(variant.displayName)")
        print("[Klein-VL] Model path: \(modelPath)")

        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: modelPath) else {
            throw FluxEncoderError.invalidInput("Model path does not exist: \(modelPath)")
        }

        // Load Qwen3-VL model (language component only — skips visual.* weights)
        print("[Klein-VL] Loading model weights (language only, skipping vision encoder)...")
        qwen3VLModel = try Qwen3VLForCausalLM.load(from: modelPath)
        print("[Klein-VL] Model weights loaded successfully")

        Memory.cacheLimit = 512 * 1024 * 1024
        qwen3VLModel?.model.memoryConfig = .aggressive
        print("[Klein-VL] Memory optimization enabled")

        // Load tokenizer
        print("[Klein-VL] Loading tokenizer...")
        let modelFolderURL = URL(fileURLWithPath: modelPath)
        qwen3VLTokenizer = try await AutoTokenizer.from(modelFolder: modelFolderURL)
        print("[Klein-VL] Tokenizer loaded")

        // Create VL embedding extractor and generator
        if let model = qwen3VLModel, let tokenizer = qwen3VLTokenizer {
            kleinVLExtractor = KleinVLEmbeddingExtractor(model: model, tokenizer: tokenizer, variant: variant)
            qwen3VLGenerator = Qwen3VLGenerator(model: model, tokenizer: tokenizer)
            loadedKleinVariant = variant
            isKleinVLMode = true
            print("[Klein-VL] VL extractor and generator created")
        }

        print("[Klein-VL] Klein VL model loaded successfully for \(variant.displayName)")
    }

    /// Load Qwen3-VL model for Klein VL embeddings with automatic download
    @MainActor
    public func loadKleinVLModel(
        variant: KleinVariant,
        qwen3VLVariant: Qwen3VLVariant? = nil,
        hfToken: String? = nil,
        progress: TextEncoderDownloadProgressCallback? = nil
    ) async throws {
        // Determine variant: explicit > already downloaded > default 8-bit
        let modelVariant: Qwen3VLVariant
        if let specified = qwen3VLVariant {
            modelVariant = specified
        } else {
            // Check if something is already downloaded
            let candidates: [Qwen3VLVariant] = variant == .klein4B
                ? [.qwen3VL_4B_8bit, .qwen3VL_4B_4bit]
                : [.qwen3VL_8B_8bit, .qwen3VL_8B_4bit]
            if let downloaded = candidates.first(where: { TextEncoderModelDownloader.isQwen3VLModelDownloaded(variant: $0) }) {
                modelVariant = downloaded
            } else {
                modelVariant = variant == .klein4B ? .qwen3VL_4B_8bit : .qwen3VL_8B_8bit
            }
        }

        guard let modelInfo = TextEncoderModelRegistry.shared.qwen3VLModel(withVariant: modelVariant) else {
            throw FluxEncoderError.invalidInput("Qwen3-VL model variant not found: \(modelVariant)")
        }

        print("[Klein-VL] Loading variant: \(modelVariant), repoId: \(modelInfo.repoId)")

        // Download model (or get existing path)
        let downloader = TextEncoderModelDownloader(hfToken: hfToken)
        let modelPath = try await downloader.downloadQwen3VL(modelInfo, progress: progress)

        print("[Klein-VL] Model path resolved: \(modelPath.path)")

        // Load from downloaded path
        try await loadKleinVLModel(variant: variant, from: modelPath.path)
    }

    /// Generate text using Qwen3-VL model
    public func generateQwen3VL(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = qwen3VLGenerator else {
            throw FluxEncoderError.invalidInput("Klein VL model not loaded")
        }
        return try generator.generate(prompt: prompt, parameters: parameters, onToken: onToken)
    }

    /// Extract Klein embeddings using Qwen3-VL
    public func extractKleinVLEmbeddings(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> MLXArray {
        guard let extractor = kleinVLExtractor else {
            throw FluxEncoderError.invalidInput("Klein VL model not loaded")
        }
        return try extractor.extractKleinEmbeddings(prompt: prompt, maxLength: maxLength)
    }

    // MARK: - Qwen3.5 VLM Service

    /// Whether Qwen3.5 VLM is loaded
    public var isQwen35VLMLoaded: Bool {
        return qwen35VLM != nil
    }

    /// Load Qwen3.5 VLM from local path
    @MainActor
    public func loadQwen35VLM(from modelPath: String) async throws {
        print("[Qwen3.5] Loading VLM from \(modelPath)...")

        Memory.cacheLimit = 512 * 1024 * 1024
        qwen35VLM = try await Qwen35VLM.load(from: modelPath)

        print("[Qwen3.5] VLM loaded successfully")
    }

    /// Unload Qwen3.5 VLM
    @MainActor
    public func unloadQwen35VLM() {
        qwen35VLM = nil
        Memory.clearCache()
        FluxDebug.log("Qwen3.5 VLM unloaded")
    }

    /// Analyze an image with Qwen3.5 VLM
    /// - Parameters:
    ///   - image: CGImage to analyze
    ///   - prompt: User prompt text
    ///   - systemPrompt: Optional system prompt to guide the analysis style
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - onToken: Streaming callback
    public func analyzeImageWithQwen35(
        image: CGImage,
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let vlm = qwen35VLM else {
            throw FluxEncoderError.invalidInput("Qwen3.5 VLM not loaded")
        }
        return try vlm.generate(
            image: image, prompt: prompt, systemPrompt: systemPrompt,
            maxTokens: maxTokens, temperature: temperature,
            onToken: onToken
        )
    }

    /// Analyze an image from file path with Qwen3.5 VLM
    public func analyzeImageWithQwen35(
        path: String,
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FluxEncoderError.invalidInput("Failed to load image: \(path)")
        }
        return try analyzeImageWithQwen35(
            image: cgImage, prompt: prompt, systemPrompt: systemPrompt,
            maxTokens: maxTokens, temperature: temperature, onToken: onToken
        )
    }

    /// Generate text with Qwen3.5 VLM (no image)
    public func generateWithQwen35(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let vlm = qwen35VLM else {
            throw FluxEncoderError.invalidInput("Qwen3.5 VLM not loaded")
        }
        return try vlm.generate(
            image: nil, prompt: prompt, systemPrompt: systemPrompt,
            maxTokens: maxTokens, temperature: temperature,
            onToken: onToken
        )
    }

    // MARK: - FLUX.2 Image Description Service

    /// System prompt for describing images in FLUX.2-compatible style
    /// Produces detailed visual descriptions optimized for image regeneration
    /// Covers both scene content AND visual style
    public static let fluxImageDescriptionSystemPrompt = """
    You are an expert image analyst for FLUX.2 by Black Forest Labs. Describe the provided image so that FLUX.2 can recreate it faithfully. You MUST describe both WHAT is depicted AND HOW it looks.

    SCENE (what is depicted):
    1. Action and narrative: what is happening, what are people/characters doing, interactions
    2. People/characters: number, identity, gender, age, pose, expression, clothing, position relative to each other
    3. Objects: what objects are present, their state, spatial relationships
    4. Setting: location, indoor/outdoor, time of day, environment details

    STYLE (how it looks):
    5. Art style: photographic, illustrated, 3D render, vector art, sketch, painting, etc.
    6. Composition: framing, perspective, camera angle, depth of field
    7. Colors and lighting: palette, contrast, light direction, shadows, color temperature
    8. Textures and materials: surfaces, fabrics, skin quality, line work
    9. Text in image: reproduce ALL visible text in quotation marks

    Write a single flowing paragraph, scene first then style. Be concrete and specific.
    Output only the description, nothing else.
    """

    /// Describe an image in FLUX.2-compatible style for regeneration
    /// - Parameters:
    ///   - image: CGImage to describe
    ///   - context: Optional additional context from the user (e.g., "focus on the background")
    ///   - maxTokens: Maximum description length
    /// - Returns: GenerationResult with FLUX.2-optimized image description
    public func describeImageForFlux(
        image: CGImage,
        context: String? = nil,
        maxTokens: Int = 300,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let prompt = context ?? "Describe this image."
        return try analyzeImageWithQwen35(
            image: image,
            prompt: prompt,
            systemPrompt: Self.fluxImageDescriptionSystemPrompt,
            maxTokens: maxTokens,
            temperature: 0,
            onToken: onToken
        )
    }

    /// Describe an image from file path in FLUX.2-compatible style
    public func describeImageForFlux(
        path: String,
        context: String? = nil,
        maxTokens: Int = 300,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw FluxEncoderError.invalidInput("Failed to load image: \(path)")
        }
        return try describeImageForFlux(
            image: cgImage, context: context, maxTokens: maxTokens, onToken: onToken
        )
    }

    // MARK: - FLUX.2 Image Comparison Service

    /// Result of comparing two images
    public struct FluxImageComparison: Sendable {
        public let sceneScore: Int
        public let styleScore: Int
        public let sceneReason: String
        public let styleReason: String
        public let rawResponse: String
    }

    /// System prompt for image comparison
    public static let fluxImageComparisonSystemPrompt = """
    You compare two images for FLUX.2 LoRA training evaluation. Image 1 is the REFERENCE (target). Image 2 is the GENERATED image (baseline without LoRA).

    Score each criterion from 0 to 100. Be STRICT and PRECISE — small differences matter for LoRA training decisions.

    SCENE score (content fidelity, 0-100):
    - 90-100: Identical subjects, poses, expressions, interactions, spatial layout
    - 70-89: Same subjects and general arrangement, minor differences in poses or details
    - 50-69: Similar concept but different number of subjects, different poses, or missing key elements
    - 30-49: Same general theme but substantially different composition and subjects
    - 0-29: Completely different scene

    STYLE score (visual fidelity, 0-100):
    - 90-100: Identical art style, line work, color palette, lighting, textures
    - 70-89: Same general style category but noticeable differences in execution (e.g., both vector art but different line weights)
    - 50-69: Similar style family but clearly different execution (e.g., flat vector vs 3D-shaded vector)
    - 30-49: Different style categories (e.g., hand-drawn sketch vs digital illustration)
    - 0-29: Completely different visual style (e.g., photograph vs cartoon)

    Respond ONLY with this exact JSON format, no other text:
    {"scene_score": N, "scene_reason": "brief explanation", "style_score": N, "style_reason": "brief explanation"}
    """

    /// Compare two images using FLUX.2 criteria
    /// - Parameters:
    ///   - reference: The original/reference image
    ///   - generated: The generated image to compare against reference
    ///   - onToken: Optional streaming callback
    /// - Returns: Structured comparison with scene and style scores (0-10)
    public func compareImagesForFlux(
        reference: CGImage,
        generated: CGImage,
        onToken: ((String) -> Bool)? = nil
    ) throws -> FluxImageComparison {
        guard let vlm = qwen35VLM else {
            throw FluxEncoderError.invalidInput("Qwen3.5 VLM not loaded")
        }

        let result = try vlm.generateMultiImage(
            images: [reference, generated],
            prompt: "Compare these two images.",
            systemPrompt: Self.fluxImageComparisonSystemPrompt,
            maxTokens: 300,
            temperature: 0,
            onToken: onToken
        )

        // Parse JSON from response
        return parseComparisonResult(result.text)
    }

    /// Compare two images from file paths
    public func compareImagesForFlux(
        referencePath: String,
        generatedPath: String,
        onToken: ((String) -> Bool)? = nil
    ) throws -> FluxImageComparison {
        guard let refSource = CGImageSourceCreateWithURL(URL(fileURLWithPath: referencePath) as CFURL, nil),
              let ref = CGImageSourceCreateImageAtIndex(refSource, 0, nil) else {
            throw FluxEncoderError.invalidInput("Failed to load reference image: \(referencePath)")
        }
        guard let genSource = CGImageSourceCreateWithURL(URL(fileURLWithPath: generatedPath) as CFURL, nil),
              let gen = CGImageSourceCreateImageAtIndex(genSource, 0, nil) else {
            throw FluxEncoderError.invalidInput("Failed to load generated image: \(generatedPath)")
        }
        return try compareImagesForFlux(reference: ref, generated: gen, onToken: onToken)
    }

    /// Parse comparison result from VLM output (JSON or regex fallback)
    private func parseComparisonResult(_ text: String) -> FluxImageComparison {
        // Try JSON extraction first
        if let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}") {
            let jsonStr = String(text[start...end])

            struct ComparisonJSON: Decodable {
                let scene_score: Int?
                let style_score: Int?
                let scene_reason: String?
                let style_reason: String?
            }

            if let data = jsonStr.data(using: .utf8),
               let parsed = try? JSONDecoder().decode(ComparisonJSON.self, from: data),
               parsed.scene_score != nil {
                return FluxImageComparison(
                    sceneScore: parsed.scene_score ?? -1,
                    styleScore: parsed.style_score ?? -1,
                    sceneReason: parsed.scene_reason ?? "",
                    styleReason: parsed.style_reason ?? "",
                    rawResponse: text
                )
            }
        }

        // Regex fallback: look for "scene.*N/100" or "scene.*N" patterns
        let sceneScore = extractScore(from: text, keyword: "scene")
        let styleScore = extractScore(from: text, keyword: "style")

        return FluxImageComparison(
            sceneScore: sceneScore,
            styleScore: styleScore,
            sceneReason: extractReason(from: text, keyword: "scene"),
            styleReason: extractReason(from: text, keyword: "style"),
            rawResponse: text
        )
    }

    private func extractScore(from text: String, keyword: String) -> Int {
        // Match patterns like "Scene: 75/100" or "scene_score: 75" or "Scene: 7/10"
        let patterns = [
            "\(keyword)[^0-9]*?(\\d+)/100",
            "\(keyword)_score[^0-9]*?(\\d+)",
            "\(keyword)[^0-9]*?(\\d+)\\s*/\\s*100",
            "\(keyword)[^0-9]*?(\\d+)/10"
        ]
        let lower = text.lowercased()
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: lower, range: NSRange(lower.startIndex..., in: lower)),
               match.numberOfRanges > 1,
               let range = Range(match.range(at: 1), in: lower) {
                return Int(lower[range]) ?? -1
            }
        }
        return -1
    }

    private func extractReason(from text: String, keyword: String) -> String {
        // Find text after "scene_reason" or "Scene:" and extract the explanation
        let patterns = [
            "\(keyword)_reason[\":\\s]+(.*?)[\",}]",
            "\(keyword).*?/10[^.]*\\.\\s*(.*?)(?:\\n|$)"
        ]
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]),
               let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
               match.numberOfRanges > 1,
               let range = Range(match.range(at: 1), in: text) {
                return String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return ""
    }

    // MARK: - Generation

    /// Generate text from prompt
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = generator else {
            throw FluxEncoderError.modelNotLoaded
        }
        return try generator.generate(prompt: prompt, parameters: parameters, onToken: onToken)
    }

    /// Generate with chat messages
    /// - Parameters:
    ///   - messages: Chat messages
    ///   - parameters: Generation parameters
    ///   - stream: If true, call onToken incrementally; if false, call once at end with complete text
    ///   - onToken: Callback for token output
    public func chat(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        stream: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = generator else {
            throw FluxEncoderError.modelNotLoaded
        }
        return try generator.chat(messages: messages, parameters: parameters, stream: stream, onToken: onToken)
    }

    // MARK: - Qwen3 Generation

    /// Generate text from prompt using Qwen3 model
    public func generateQwen3(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = qwen3Generator else {
            throw FluxEncoderError.kleinNotLoaded
        }
        return try generator.generate(prompt: prompt, parameters: parameters, onToken: onToken)
    }

    /// Generate with chat messages using Qwen3 model
    /// - Parameters:
    ///   - messages: Chat messages
    ///   - parameters: Generation parameters
    ///   - stream: If true, call onToken incrementally; if false, call once at end with complete text
    ///   - onToken: Callback for token output
    public func chatQwen3(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        stream: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = qwen3Generator else {
            throw FluxEncoderError.kleinNotLoaded
        }
        return try generator.chat(messages: messages, parameters: parameters, stream: stream, onToken: onToken)
    }

    /// Check if Qwen3 generation is available
    public var isQwen3GenerationAvailable: Bool {
        return qwen3Generator != nil
    }

    // MARK: - Vision

    /// Analyze an image with a text prompt
    /// - Parameters:
    ///   - image: NSImage to analyze
    ///   - prompt: Text prompt describing what to look for
    ///   - parameters: Generation parameters
    ///   - onToken: Callback for streaming tokens
    /// - Returns: Generated description/analysis
    /// Log memory for inference debugging (only when detailed profiling is enabled)
    private func logInferenceMemory(_ label: String) {
        guard FluxProfiler.shared.isEnabled || ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil else { return }
        let snapshot = FluxProfiler.snapshot()
        let mlxMB = Double(snapshot.mlxActive) / (1024 * 1024)
        let procMB = Double(snapshot.processFootprint) / (1024 * 1024)
        print("[VLM-INF] \(label): MLX=\(String(format: "%.1f", mlxMB))MB, Process=\(String(format: "%.1f", procMB))MB")
        fflush(stdout)
    }

    /// Analyze an image with a text prompt and optional system prompt
    /// - Parameters:
    ///   - image: NSImage to analyze
    ///   - prompt: Text prompt describing what to look for
    ///   - systemPrompt: Optional system prompt (e.g., for FLUX.2 I2I upsampling)
    ///   - parameters: Generation parameters
    ///   - onToken: Callback for streaming tokens
    /// - Returns: Generated description/analysis
    public func analyzeImage(
        image: NSImage,
        prompt: String,
        systemPrompt: String? = nil,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let vlm = vlmModel,
              let tokenizer = tokenizer,
              let processor = imageProcessor else {
            throw FluxEncoderError.vlmNotLoaded
        }

        let debug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

        if debug {
            print("[Analyze] Starting with prompt: \(prompt)")
            if let sys = systemPrompt { print("[Analyze] System prompt: \(sys.prefix(80))...") }
            fflush(stdout)
        }
        logInferenceMemory("START inference")

        // 1. Preprocess image
        let pixelValues = try processor.preprocess(image)
        logInferenceMemory("After image preprocess")

        // 2. Encode image to get number of image tokens
        // NHWC format: [batch, H, W, C]
        let (_, patchesH, patchesW) = vlm.encodeImage(pixelValues)
        if debug { print("[Analyze] Image encoded: \(patchesH)x\(patchesW) patches"); fflush(stdout) }
        logInferenceMemory("After image encode (vision tower)")
        let numImageTokens = vlm.getNumImageTokens(
            imageHeight: pixelValues.shape[1],
            imageWidth: pixelValues.shape[2]
        )

        // 3. Build input tokens with image token placeholders
        // IMPORTANT: We must insert actual image token IDs (10), not tokenize "[IMG]" string!
        // Format with system prompt: <s> [INST] {system_prompt}\n\n[IMG]...[IMG]\n{user_prompt} [/INST]
        // Format without system prompt: <s> [INST] [IMG]...[IMG]\n{user_prompt} [/INST]
        let imageTokenId = vlm.config.imageTokenIndex  // = 10

        // Build tokens directly:
        // - BOS token (1)
        // - [INST] token (3)
        // - optional system prompt + \n\n
        // - numImageTokens x image token (10)
        // - tokenized user prompt
        // - [/INST] token (4)
        var inputTokens: [Int] = []
        inputTokens.append(tokenizer.bosToken)  // <s>
        inputTokens.append(3)  // [INST]

        // Add system prompt if provided
        if let sysPrompt = systemPrompt {
            inputTokens.append(contentsOf: tokenizer.encode(sysPrompt + "\n\n", addSpecialTokens: false))
        }

        inputTokens.append(contentsOf: Array(repeating: imageTokenId, count: numImageTokens))
        inputTokens.append(contentsOf: tokenizer.encode("\n\(prompt) ", addSpecialTokens: false))
        inputTokens.append(4)  // [/INST]

        if debug {
            print("[Analyze] Input tokens: \(inputTokens.count) total (\(numImageTokens) image tokens)")
            print("[Analyze] First 10 tokens: \(inputTokens.prefix(10))")
            print("[Analyze] Last 10 tokens: \(inputTokens.suffix(10))")
            fflush(stdout)
        }

        let inputIds = MLXArray(inputTokens.map { Int32($0) }).expandedDimensions(axis: 0)

        // 5. Generate with vision
        let cache = vlm.createCache()
        logInferenceMemory("After KV cache creation")
        var generatedTokens: [Int] = []
        let maxTokens = parameters.maxTokens
        let startTime = Date()

        // First forward pass with image
        var logits = vlm(inputIds, pixelValues: pixelValues, cache: cache)
        logInferenceMemory("After first forward pass (prefill)")

        if debug {
            // Debug: Check logits stats
            print("[Debug] Logits shape: \(logits.shape)")
            let lastLogits = logits[0, -1, 0...]
            let logitsMean = MLX.mean(lastLogits).item(Float.self)
            let logitsStd = MLX.std(lastLogits).item(Float.self)
            let logitsMin = MLX.min(lastLogits).item(Float.self)
            let logitsMax = MLX.max(lastLogits).item(Float.self)
            print("[Debug] Last position logits: mean=\(logitsMean), std=\(logitsStd), min=\(logitsMin), max=\(logitsMax)")
            // Check top predictions
            let sortedIndices = MLX.argSort(lastLogits)
            let vocabSize = lastLogits.shape[0]
            let topK = min(5, vocabSize)
            let topIndices = sortedIndices[(vocabSize - topK)...]
            print("[Debug] Top \(topK) token indices: \(topIndices.asArray(Int32.self))")
            fflush(stdout)
        }

        for i in 0..<maxTokens {
            // Sample next token with repetition penalty
            let nextTokenLogits = logits[0, -1, 0...]
            let nextToken = sampleToken(logits: nextTokenLogits, parameters: parameters, generatedTokens: generatedTokens)

            // Force evaluation before sync - allows GPU work to complete
            MLX.eval(nextToken)
            let tokenId = nextToken.item(Int32.self)

            // Check for EOS
            if tokenId == Int32(tokenizer.eosToken) {
                break
            }

            generatedTokens.append(Int(tokenId))

            // Next forward pass (text only, using cache)
            let nextInput = MLXArray([tokenId]).expandedDimensions(axis: 0)
            logits = vlm(nextInput, pixelValues: nil, cache: cache)

            // Periodically clear GPU cache to prevent memory accumulation
            if (i + 1) % 20 == 0 {
                Memory.clearCache()
            }
        }

        // Decode all tokens at once for correct multi-byte character handling
        let outputText = tokenizer.decode(generatedTokens, skipSpecialTokens: true)

        // Call callback once with complete text (if provided)
        if let callback = onToken {
            _ = callback(outputText)
        }

        let totalTime = Date().timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / max(totalTime, 0.001)
        logInferenceMemory("After generation loop (\(generatedTokens.count) tokens)")

        // Clear KV cache to free memory
        cache.forEach { $0.clear() }
        logInferenceMemory("After KV cache clear")
        Memory.clearCache()
        logInferenceMemory("After GPU cache clear")

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: inputTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// Analyze image from file path
    /// - Parameters:
    ///   - path: Path to image file
    ///   - prompt: Text prompt describing what to look for
    ///   - systemPrompt: Optional system prompt (e.g., for FLUX.2 I2I upsampling)
    ///   - parameters: Generation parameters
    ///   - onToken: Callback for streaming tokens
    /// - Returns: Generated description/analysis
    public func analyzeImage(
        path: String,
        prompt: String,
        systemPrompt: String? = nil,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let processor = imageProcessor else {
            throw FluxEncoderError.vlmNotLoaded
        }

        let image = try processor.loadImage(from: path)
        return try analyzeImage(image: image, prompt: prompt, systemPrompt: systemPrompt, parameters: parameters, onToken: onToken)
    }

    /// Format vision prompt following Mistral chat template
    private func formatVisionPrompt(imageToken: String, userPrompt: String) -> String {
        // Mistral vision format: [INST] [IMG]...[IMG] prompt [/INST]
        return "[INST] \(imageToken)\n\(userPrompt) [/INST]"
    }

    /// Sample token from logits with repetition penalty
    private func sampleToken(
        logits: MLXArray,
        parameters: GenerateParameters,
        generatedTokens: [Int] = []
    ) -> MLXArray {
        var adjustedLogits = logits

        // Apply repetition penalty to recently generated tokens
        if parameters.repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
            // Get the last N tokens to penalize (use repetitionContextSize)
            let contextSize = min(parameters.repetitionContextSize, generatedTokens.count)
            let recentTokens = Array(generatedTokens.suffix(contextSize))

            // Create a set for O(1) lookup
            let tokenSet = Set(recentTokens)

            // Apply penalty: divide positive logits, multiply negative logits
            var logitsArray = adjustedLogits.asArray(Float.self)
            for tokenId in tokenSet {
                if tokenId >= 0 && tokenId < logitsArray.count {
                    if logitsArray[tokenId] > 0 {
                        logitsArray[tokenId] /= parameters.repetitionPenalty
                    } else {
                        logitsArray[tokenId] *= parameters.repetitionPenalty
                    }
                }
            }
            adjustedLogits = MLXArray(logitsArray)
        }

        // Apply temperature
        if parameters.temperature > 0 {
            adjustedLogits = adjustedLogits / parameters.temperature
        }

        // Apply softmax
        var probs = MLX.softmax(adjustedLogits, axis: -1)

        // Top-p sampling
        var sortedIndices: MLXArray? = nil
        if parameters.topP < 1.0 {
            sortedIndices = MLX.argSort(probs, axis: -1)
            let sortedProbs = MLX.takeAlong(probs, sortedIndices!, axis: -1)
            let cumProbs = MLX.cumsum(sortedProbs, axis: -1)

            // Find cutoff
            let mask = cumProbs .<= (1.0 - parameters.topP)
            let maskedProbs = MLX.where(mask, MLXArray(0.0), sortedProbs)

            // Renormalize
            let sum = MLX.sum(maskedProbs)
            probs = maskedProbs / sum
        }

        // Sample
        if parameters.temperature > 0 {
            let sampledIdx = MLXRandom.categorical(MLX.log(probs + 1e-10))
            // If we used top-p, map back from sorted space to vocabulary space
            if let indices = sortedIndices {
                return indices[sampledIdx]
            }
            return sampledIdx
        } else {
            // Greedy: if we used top-p, get argmax from sorted space and map back
            if let indices = sortedIndices {
                let sortedArgmax = MLX.argMax(probs, axis: -1)
                return indices[sortedArgmax]
            }
            return MLX.argMax(probs, axis: -1)
        }
    }

    /// Generate with streaming (AsyncStream)
    public func generateStream(
        prompt: String,
        parameters: GenerateParameters = .balanced
    ) throws -> AsyncStream<String> {
        guard let generator = generator else {
            throw FluxEncoderError.modelNotLoaded
        }
        return generator.generateStream(prompt: prompt, parameters: parameters)
    }

    // MARK: - Embeddings

    /// Extract embeddings from text
    public func extractEmbeddings(
        prompt: String,
        config: HiddenStatesConfig = .mfluxDefault
    ) throws -> MLXArray {
        guard let extractor = extractor else {
            throw FluxEncoderError.modelNotLoaded
        }
        return try extractor.extractEmbeddings(prompt: prompt, config: config)
    }

    /// Extract mflux-compatible embeddings
    public func extractMfluxEmbeddings(prompt: String) throws -> MLXArray {
        guard let extractor = extractor else {
            throw FluxEncoderError.modelNotLoaded
        }
        return try extractor.extractMfluxEmbeddings(prompt: prompt)
    }

    /// Extract FLUX.2-compatible embeddings (identical to mflux-gradio Python)
    /// - Parameters:
    ///   - prompt: User prompt text
    ///   - maxLength: Maximum sequence length (default: 512)
    /// - Returns: Embeddings tensor with shape [1, maxLength, 15360]
    public func extractFluxEmbeddings(
        prompt: String,
        maxLength: Int = FluxConfig.maxSequenceLength
    ) throws -> MLXArray {
        guard let extractor = extractor else {
            throw FluxEncoderError.modelNotLoaded
        }
        return try extractor.extractFluxEmbeddings(prompt: prompt, maxLength: maxLength)
    }

    /// Get FLUX-format token IDs for debugging/comparison with Python
    public func getFluxTokenIds(
        prompt: String,
        maxLength: Int = FluxConfig.maxSequenceLength
    ) throws -> [Int] {
        guard let extractor = extractor else {
            throw FluxEncoderError.modelNotLoaded
        }
        return extractor.getFluxTokenIds(prompt: prompt, maxLength: maxLength)
    }
    
    // MARK: - Klein Embeddings
    
    /// Extract FLUX.2 Klein embeddings using Qwen3 model
    /// - Parameters:
    ///   - prompt: User prompt text
    ///   - maxLength: Maximum sequence length (default: 512)
    /// - Returns: Embeddings tensor with shape [1, maxLength, outputDim]
    ///           Klein 4B: [1, 512, 7680]
    ///           Klein 9B: [1, 512, 12288]
    public func extractKleinEmbeddings(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> MLXArray {
        guard let extractor = kleinExtractor else {
            throw FluxEncoderError.kleinNotLoaded
        }
        return try extractor.extractKleinEmbeddings(prompt: prompt, maxLength: maxLength)
    }
    
    /// Get Klein-format token IDs for debugging/comparison with Python
    public func getKleinTokenIds(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> [Int] {
        guard let extractor = kleinExtractor else {
            throw FluxEncoderError.kleinNotLoaded
        }
        return try extractor.getKleinTokenIds(prompt: prompt, maxLength: maxLength)
    }
    
    /// Get Klein embedding dimension for loaded model
    public var kleinEmbeddingDimension: Int? {
        return kleinExtractor?.embeddingDimension
    }

    /// Extract FLUX.2-compatible embeddings with image (for Image-to-Image)
    /// This method produces embeddings that include both image and text features
    /// - Parameters:
    ///   - image: NSImage to include in embeddings
    ///   - prompt: User prompt text (editing instruction)
    /// - Returns: Embeddings tensor with shape [1, seq, 15360] where seq depends on image size
    #if canImport(AppKit)
    public func extractFluxEmbeddingsWithImage(
        image: NSImage,
        prompt: String
    ) throws -> MLXArray {
        guard let vlm = vlmModel,
              let tokenizer = tokenizer,
              let processor = imageProcessor else {
            throw FluxEncoderError.vlmNotLoaded
        }

        let debug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

        if debug { print("[FLUX I2I] Starting with prompt: \(prompt)"); fflush(stdout) }

        // 1. Preprocess image with FLUX-specific max size (768² as per BFL reference)
        // This limits the number of image tokens to a reasonable amount
        let pixelValues = try processor.preprocess(image, maxSize: FluxConfig.maxImageSizeUpsampling)

        // 2. Get number of image tokens from the projector output
        // Image tokens = (H/patch_size/merge_size) * (W/patch_size/merge_size)
        let numImageTokens = vlm.getNumImageTokens(
            imageHeight: pixelValues.shape[1],
            imageWidth: pixelValues.shape[2]
        )

        if debug { print("[FLUX I2I] Image will generate \(numImageTokens) tokens"); fflush(stdout) }

        // 3. Build input tokens with I2I system message
        // Format: <s> [INST] <<SYS>>\n{system}\n<</SYS>>\n\n[IMG]...[IMG] {prompt} [/INST]
        let imageTokenId = vlm.config.imageTokenIndex

        // Build messages for I2I mode
        let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
        let systemMessage = FluxConfig.systemMessageUpsamplingI2I

        // Encode system message part
        var inputTokens: [Int] = []
        inputTokens.append(tokenizer.bosToken)  // <s>
        inputTokens.append(3)  // [INST]
        inputTokens.append(contentsOf: tokenizer.encode("<<SYS>>\n\(systemMessage)\n<</SYS>>\n\n", addSpecialTokens: false))

        // Add ALL image tokens - do NOT truncate!
        // The image features and token positions MUST match
        inputTokens.append(contentsOf: Array(repeating: imageTokenId, count: numImageTokens))

        // Add prompt
        inputTokens.append(contentsOf: tokenizer.encode("\n\(cleanedPrompt) ", addSpecialTokens: false))
        inputTokens.append(4)  // [/INST]

        // Note: For I2I, we do NOT truncate or pad to a fixed length
        // The sequence length depends on the image size and must include all image tokens
        // FLUX.2 diffusion model handles variable-length conditioning

        if debug { print("[FLUX I2I] Total sequence length: \(inputTokens.count) tokens (image: \(numImageTokens))"); fflush(stdout) }

        // 4. Create input tensor
        let inputIds = MLXArray(inputTokens.map { Int32($0) }).expandedDimensions(axis: 0)

        // 5. Extract embeddings using VLM
        let embeddings = vlm.extractFluxEmbeddingsWithImage(
            pixelValues: pixelValues,
            inputIds: inputIds
        )

        if debug { print("[FLUX I2I] Embeddings shape: \(embeddings.shape)"); fflush(stdout) }

        return embeddings
    }
    #endif

    /// Extract FLUX.2-compatible embeddings with image from path (for Image-to-Image)
    /// - Parameters:
    ///   - imagePath: Path to image file
    ///   - prompt: User prompt text (editing instruction)
    /// - Returns: Embeddings tensor with shape [1, seq, 15360] where seq depends on image size
    public func extractFluxEmbeddingsWithImage(
        imagePath: String,
        prompt: String
    ) throws -> MLXArray {
        #if canImport(AppKit)
        guard let processor = imageProcessor else {
            throw FluxEncoderError.vlmNotLoaded
        }

        let image = try processor.loadImage(from: imagePath)
        return try extractFluxEmbeddingsWithImage(image: image, prompt: prompt)
        #else
        throw FluxEncoderError.vlmNotLoaded
        #endif
    }

    /// Export embeddings to file
    /// This is a standalone operation that doesn't require the full model to be loaded
    public func exportEmbeddings(
        _ embeddings: MLXArray,
        to path: String,
        format: ExportFormat = .binary
    ) throws {
        // Standalone export - doesn't require extractor or full model
        switch format {
        case .binary:
            // Export as raw float32 binary
            let flatEmbeddings = embeddings.reshaped([-1]).asArray(Float.self)
            let data = flatEmbeddings.withUnsafeBufferPointer { buffer in
                Data(buffer: buffer)
            }
            try data.write(to: URL(fileURLWithPath: path))

        case .numpy:
            // For .npy format, use MLX's save function
            try MLX.save(array: embeddings, url: URL(fileURLWithPath: path))

        case .json:
            // Export as JSON with shape and values
            let shape = embeddings.shape
            let flatEmbeddings = embeddings.reshaped([-1]).asArray(Float.self)
            let dict: [String: Any] = [
                "shape": shape.map { $0 },
                "values": flatEmbeddings
            ]
            let jsonData = try JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted)
            try jsonData.write(to: URL(fileURLWithPath: path))
        }
    }

    /// Load only the tokenizer without loading the full model
    /// Useful when we need tokenization but not MLX inference
    ///
    /// - Parameter modelPath: Path to model directory containing tekken.json
    public func loadTokenizerOnly(from modelPath: String) {
        tokenizer = TekkenTokenizer(modelPath: modelPath)
        FluxDebug.log("Tokenizer loaded from \(modelPath)")
    }

    // MARK: - Tokenization

    /// Encode text to tokens
    public func encode(_ text: String, addSpecialTokens: Bool = false) throws -> [Int] {
        guard let tokenizer = tokenizer else {
            throw FluxEncoderError.modelNotLoaded
        }
        return tokenizer.encode(text, addSpecialTokens: addSpecialTokens)
    }

    /// Decode tokens to text
    public func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) throws -> String {
        guard let tokenizer = tokenizer else {
            throw FluxEncoderError.modelNotLoaded
        }
        return tokenizer.decode(tokens, skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Model Info

    /// Get model configuration
    public var config: MistralTextConfig? {
        return model?.config
    }

    /// Print available models
    @MainActor
    public func printAvailableModels() {
        TextEncoderModelRegistry.shared.printAvailableModels()
    }
}

// MARK: - Errors

public enum FluxEncoderError: LocalizedError {
    case modelNotLoaded
    case vlmNotLoaded
    case kleinNotLoaded
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .vlmNotLoaded:
            return "VLM not loaded. Call loadVLMModel() first for vision capabilities."
        case .kleinNotLoaded:
            return "Klein model not loaded. Call loadKleinModel() first for Klein embeddings."
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        }
    }
}

// MARK: - Version Info

public struct MistralVersion {
    public static let version = "0.1.0"
    public static let modelName = "Mistral Small 3.2"
    public static let modelVersion = "24B-Instruct-2506"
}
