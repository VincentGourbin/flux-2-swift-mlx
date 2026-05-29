// Flux2OutpaintingChain.swift — High-level outpainting wrapper
// Copyright 2025 Vincent Gourbin
//
// Wraps `Flux2MaskedInpaintingChain` to expose a one-liner outpainting API
// matching Black Forest Labs' Flux outpainting service:
//
//   chain = Flux2OutpaintingChain(pipeline, image, top: 0, bottom: 0,
//                                 left: 384, right: 384, prompt: "...")
//   let result = try await chain.run()
//
// Everything painful — extending the canvas, seeding the new strips with
// neutral Gaussian noise (so RePaint's `originalNoised` has no semantic
// signal to bleed through the mask ramp), and building the *smart* mask
// (white strips, 32-px inner ramp inside the keep region, black keep) —
// happens inside.

import Foundation
import Flux2Core
import FluxTextEncoders  // FluxDebug logger; VLM service is reached via Flux2VLMPromptBuilder
@preconcurrency import MLX
import CoreGraphics

/// Outpaint an image by extending its canvas on any of the four sides.
///
/// Empirical recipe validated on klein-9b distilled, 4 steps, guidance 1.0
/// (see `project_flux2_chains.md`):
/// - Strips seeded with mid-grey Gaussian noise (no semantic signal).
/// - Smart mask: white strips, narrow gradient on a 32-px band *inside*
///   the keep region, black keep.
/// - I2I conditioning on the original image so the transformer's attention
///   continues style/colours/perspective into the strips.
/// - RePaint blend preserves the original pixels.
public struct Flux2OutpaintingChain: Flux2Chain {
    /// Pipeline to run on. Reused as-is; LoRA state and quantisation
    /// configured by the caller are honoured.
    public let pipeline: Flux2Pipeline

    /// Source image. Will be pasted at offset (left, top) on the extended
    /// canvas.
    public let image: CGImage

    /// Per-side padding in **pixels** to add to the original. Each value is
    /// silently rounded up to the next multiple of 32 (FLUX.2 requirement)
    /// before being applied.
    public let top: Int
    public let bottom: Int
    public let left: Int
    public let right: Int

    /// Text prompt describing the **full** extended scene.
    ///
    /// Follow the BFL prompting guidelines
    /// (<https://docs.bfl.ml/guides/prompting_guide_flux2>): *Subject +
    /// Action + Style + Context*, 30–80 words, leading words weigh more.
    /// Mention the content of the keep region too — the transformer
    /// attends to the I2I reference but the prompt is still its main
    /// steering signal. No negative prompts.
    public let prompt: String

    /// FLUX.2 generation parameters. Defaults target distilled klein.
    public let steps: Int
    public let guidance: Float
    public let seed: UInt64?
    /// Text-encoder-only prompt rewriting (Mistral / Klein-Qwen3). Does
    /// not look at ``image``. For *image-aware* enrichment that
    /// describes the kept region's lighting / camera / materials and
    /// instructs FLUX.2 to continue them, use ``enrichPromptWithVLM``
    /// instead — when both are set, the VLM wins and a warning is
    /// logged. Default `false`.
    public let upsamplePrompt: Bool

    /// When `true`, ask the bundled Qwen3.5 VLM to look at ``image`` and
    /// the list of sides being extended, then assemble a 30-80 word
    /// BFL-style prompt that continues the kept region's materials,
    /// perspective, lighting direction, and colour palette into the new
    /// strips.
    ///
    /// **The VLM is never required.** When it isn't loaded, a warning is
    /// logged via ``FluxDebug`` and the chain falls back to the
    /// caller's prompt + ``upsamplePrompt`` (existing behaviour).
    ///
    /// **Collision with ``upsamplePrompt``:** when both are `true`, the
    /// VLM wins. A warning is logged.
    ///
    /// The caller owns the VLM lifecycle (load via
    /// ``FluxTextEncoders/shared/loadQwen35VLM(from:)`` before
    /// ``run()``, unload when done). Default `false`.
    public let enrichPromptWithVLM: Bool
    public let onProgress: Flux2ProgressCallback?

    /// Width of the soft transition band, in pixels of the canvas. The band
    /// lives *inside* the keep region so the strips themselves carry a
    /// hard mask = 1.0 (pure paint, no seed contamination). Default 32 px
    /// is a good compromise on a 1-2 M-pixel canvas.
    public let transitionPixels: Int

    /// Maximum total pixel count for the working canvas. The default 4 M
    /// pixels is generous; the chain forwards this to the inpainting
    /// chain's `maxPixels`. Set lower if you want to cap memory/time.
    public let maxPixels: Int

    /// Configure an outpainting run.
    ///
    /// - Parameters:
    ///   - pipeline: Pipeline to drive. Reused as-is — load its models and any
    ///     LoRA *before* calling `run()` (or `run()` will load them lazily).
    ///   - image: Source image. Will be pasted at `(left, top)` on the
    ///     extended canvas and preserved bit-exact by the RePaint blend.
    ///   - top: Pixels to add at the top. Silently rounded up to the next
    ///     multiple of 32 (FLUX.2 requirement).
    ///   - bottom: Pixels to add at the bottom (same rounding).
    ///   - left: Pixels to add on the left (same rounding).
    ///   - right: Pixels to add on the right (same rounding).
    ///   - prompt: Text describing the **full** extended scene. Mention the
    ///     content of the keep region too — the I2I reference gives the
    ///     transformer visual anchoring, but the prompt is still its main
    ///     steering signal.
    ///   - steps: Denoising step count. `4` matches klein distilled defaults.
    ///   - guidance: Classifier-free guidance scale.
    ///   - seed: Random seed for reproducibility. `nil` for non-deterministic.
    ///   - upsamplePrompt: Text-encoder-only prompt rewriting. See the
    ///     property doc for the difference vs ``enrichPromptWithVLM``
    ///     and the collision rule. Default `false`.
    ///   - enrichPromptWithVLM: Opt-in image-aware prompt rewriting via
    ///     the bundled Qwen3.5 VLM. Strictly optional — when the VLM is
    ///     not loaded the chain falls back to the verbatim prompt and
    ///     logs a warning. Default `false`.
    ///   - transitionPixels: Width of the soft transition band, in pixels of
    ///     the keep region. The band lives *inside* the keep so the strips
    ///     themselves carry mask = 1.0 (pure paint, no seed contamination).
    ///   - maxPixels: Hard cap on the working canvas size. Overridden
    ///     internally to `canvas_w × canvas_h` when smaller, so the canvas
    ///     never gets downscaled.
    ///   - onProgress: Forwarded to the pipeline's denoising loop.
    public init(
        pipeline: Flux2Pipeline,
        image: CGImage,
        top: Int = 0,
        bottom: Int = 0,
        left: Int = 0,
        right: Int = 0,
        prompt: String,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        enrichPromptWithVLM: Bool = false,
        transitionPixels: Int = 32,
        maxPixels: Int = 4 * 1024 * 1024,
        onProgress: Flux2ProgressCallback? = nil
    ) {
        self.pipeline = pipeline
        self.image = image
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.prompt = prompt
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.upsamplePrompt = upsamplePrompt
        self.enrichPromptWithVLM = enrichPromptWithVLM
        self.transitionPixels = transitionPixels
        self.maxPixels = maxPixels
        self.onProgress = onProgress
    }

    /// Execute the chain end-to-end.
    ///
    /// Builds the extended canvas (mid-grey Gaussian noise outside the kept
    /// region, image pasted at `(left, top)`), builds the smart mask, then
    /// delegates to `Flux2MaskedInpaintingChain` with the original image
    /// passed as an I2I reference so the transformer's attention continues
    /// the kept content into the new strips.
    ///
    /// - Returns: The extended image plus prompt metadata.
    /// - Throws:
    ///   - `Flux2ChainError.invalidInput` if any padding is negative, every
    ///     padding is zero, or the resulting canvas is not a multiple of 32
    ///     on either axis (the source image itself may need padding by the
    ///     caller in that case).
    ///   - Whatever the underlying pipeline can throw (model not loaded,
    ///     memory, generation cancellation).
    public func run() async throws -> Flux2GenerationResult {
        guard top >= 0, bottom >= 0, left >= 0, right >= 0 else {
            throw Flux2ChainError.invalidInput("Padding values must be non-negative")
        }
        guard top + bottom + left + right > 0 else {
            throw Flux2ChainError.invalidInput("At least one side must have non-zero padding")
        }

        // FLUX.2 needs multiples of 32 on each axis. Round paddings UP so the
        // user's request is honoured (we never crop their image).
        let t = Self.roundUpToMultipleOf32(top)
        let b = Self.roundUpToMultipleOf32(bottom)
        let l = Self.roundUpToMultipleOf32(left)
        let r = Self.roundUpToMultipleOf32(right)

        let canvasW = image.width + l + r
        let canvasH = image.height + t + b
        guard canvasW % 32 == 0, canvasH % 32 == 0 else {
            // Image itself may not be a multiple of 32; that's the caller's
            // problem on the original side, but our additions are correct.
            throw Flux2ChainError.invalidInput(
                "Canvas dimensions \(canvasW)x\(canvasH) are not multiples of 32 — the input image's own dimensions need to be padded by the caller, or supply paddings that compensate."
            )
        }

        // Build the canvas with the source pasted at (l, t) and the rest
        // filled with neutral Gaussian noise. Noise is generated once on
        // the CPU; we don't need anything fancy.
        guard let canvas = Self.buildOutpaintCanvas(
            sourceImage: image,
            canvasWidth: canvasW,
            canvasHeight: canvasH,
            offsetX: l,
            offsetY: t,
            noiseSeed: seed ?? 0
        ) else {
            throw Flux2ChainError.invalidInput("Failed to build extended canvas")
        }

        // Smart mask: 1.0 in the strips, 0.0 deep in the keep, narrow
        // gradient on the keep side of the boundary.
        guard let mask = Self.buildSmartMask(
            canvasWidth: canvasW,
            canvasHeight: canvasH,
            keepX: l,
            keepY: t,
            keepWidth: image.width,
            keepHeight: image.height,
            transitionPixels: transitionPixels
        ) else {
            throw Flux2ChainError.invalidInput("Failed to build smart mask")
        }

        // VLM enrichment runs at this level (not delegated to the inner
        // inpaint chain) because we know which sides are being extended,
        // and we want the VLM to inspect the ORIGINAL image (not the
        // padded canvas with noise strips). The inner chain is then
        // called with `enrichPromptWithVLM: false` so there's no
        // double-pass.
        let (resolvedPrompt, resolvedUpsample) = await resolveFinalPromptAndUpsample(
            paddings: (t: t, b: b, l: l, r: r)
        )

        let inpaint = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: resolvedPrompt,
            image: canvas,
            mask: mask,
            referenceImages: [image],  // I2I conditioning continues the scene
            useImageAsReference: false,  // explicit refs already supplied
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: resolvedUpsample,
            enrichPromptWithVLM: false,  // already handled here, don't double-process
            maxPixels: max(maxPixels, canvasW * canvasH),
            onProgress: onProgress
        )
        return try await inpaint.run()
    }

    // MARK: - VLM enrichment resolution

    /// Same contract as the inpainting chain's resolver, but the VLM is
    /// given the ORIGINAL ``image`` (not the padded canvas) and the set
    /// of active sides so it can focus on the right edges.
    ///
    /// Outcomes (in order of preference):
    /// 1. `enrichPromptWithVLM == true` + VLM loaded + builder returns a
    ///    non-empty string → use that prompt, force `upsamplePrompt = false`
    ///    downstream. If ``upsamplePrompt`` was also `true`, warn — VLM wins.
    /// 2. `enrichPromptWithVLM == true` + VLM not loaded → warn, fall
    ///    through to (3).
    /// 3. Default → caller's prompt + caller's ``upsamplePrompt``
    ///    (existing behaviour, byte-identical to before this feature).
    private func resolveFinalPromptAndUpsample(
        paddings: (t: Int, b: Int, l: Int, r: Int)
    ) async -> (String, Bool) {
        guard enrichPromptWithVLM else {
            return (prompt, upsamplePrompt)
        }
        guard FluxTextEncoders.shared.isQwen35VLMLoaded else {
            FluxDebug.error("[Flux2OutpaintingChain] enrichPromptWithVLM=true but Qwen3.5 VLM is not loaded. Falling back to caller's prompt. Load the VLM via FluxTextEncoders.shared.loadQwen35VLM(from:) before run() to enable image-aware prompt enrichment.")
            return (prompt, upsamplePrompt)
        }
        if upsamplePrompt {
            FluxDebug.error("[Flux2OutpaintingChain] Both enrichPromptWithVLM and upsamplePrompt are true — VLM wins (image-aware enrichment supersedes text-only upsampling). Disable one of the two to silence this warning.")
        }

        var sides = Set<OutpaintSide>()
        if paddings.t > 0 { sides.insert(.top) }
        if paddings.b > 0 { sides.insert(.bottom) }
        if paddings.l > 0 { sides.insert(.left) }
        if paddings.r > 0 { sides.insert(.right) }
        // `paddings` is guaranteed to have at least one positive value by
        // the validation at the top of run(), so `sides` is non-empty.

        do {
            let built = try await Flux2VLMPromptBuilder.buildOutpaintPrompt(
                source: image,
                userInstruction: prompt,
                sides: sides
            )
            guard let final = built?.trimmingCharacters(in: .whitespacesAndNewlines), !final.isEmpty else {
                FluxDebug.error("[Flux2OutpaintingChain] VLM returned an empty prompt — falling back to caller's prompt.")
                return (prompt, upsamplePrompt)
            }
            FluxDebug.info("[Flux2OutpaintingChain] VLM-enriched prompt (sides=\(sides.sorted(by: { $0.rawValue < $1.rawValue }).map(\.rawValue).joined(separator: ","))): \(final)")
            return (final, false)
        } catch {
            FluxDebug.error("[Flux2OutpaintingChain] VLM enrichment failed: \(error). Falling back to caller's prompt.")
            return (prompt, upsamplePrompt)
        }
    }

    // MARK: - Canvas / mask builders

    /// Exposed as `internal` (instead of `private`) so the unit tests in the
    /// same module can exercise the geometry without spinning up a pipeline.
    internal static func roundUpToMultipleOf32(_ x: Int) -> Int {
        x == 0 ? 0 : ((x + 31) / 32) * 32
    }

    /// Build an RGB canvas of `canvasWidth × canvasHeight`, paint mid-grey
    /// Gaussian noise everywhere, then composite the source image at
    /// `(offsetX, offsetY)`.
    internal static func buildOutpaintCanvas(
        sourceImage: CGImage,
        canvasWidth: Int,
        canvasHeight: Int,
        offsetX: Int,
        offsetY: Int,
        noiseSeed: UInt64
    ) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = canvasWidth * 4
        var pixels = [UInt8](repeating: 0, count: canvasWidth * canvasHeight * 4)

        // Box-Muller for Gaussian samples, seeded from `noiseSeed` so the
        // canvas is reproducible.
        var state = noiseSeed &+ 0x9E37_79B9_7F4A_7C15
        func nextU64() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        func nextUnit() -> Double {
            // Avoid 0 so log() is finite.
            let u = (Double(nextU64() >> 11) + 1.0) / Double(UInt64(1) << 53)
            return u
        }
        func gauss() -> Double {
            let u1 = nextUnit(), u2 = nextUnit()
            return (-2.0 * log(u1)).squareRoot() * cos(2.0 * .pi * u2)
        }
        for i in stride(from: 0, to: pixels.count, by: 4) {
            // mean 128, sigma 32 — keeps values mostly in [32, 224].
            let g = 128.0 + gauss() * 32.0
            let v = UInt8(max(0, min(255, Int(g.rounded()))))
            pixels[i] = v
            pixels[i + 1] = v
            pixels[i + 2] = v
            pixels[i + 3] = 255
        }

        guard let context = CGContext(
            data: &pixels,
            width: canvasWidth,
            height: canvasHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        // CGContext is bottom-up; convert offsetY accordingly.
        let drawY = canvasHeight - offsetY - sourceImage.height
        context.draw(
            sourceImage,
            in: CGRect(x: offsetX, y: drawY, width: sourceImage.width, height: sourceImage.height)
        )
        return context.makeImage()
    }

    /// Build the smart grayscale mask:
    /// - White (255) on every strip outside the keep region.
    /// - Black (0) inside the keep region, except on a `transitionPixels`-wide
    ///   ramp adjacent to each side that actually has a strip.
    /// - Sides with zero padding get no transition band (no need — the keep
    ///   region simply runs to the canvas edge there).
    internal static func buildSmartMask(
        canvasWidth: Int,
        canvasHeight: Int,
        keepX: Int,
        keepY: Int,
        keepWidth: Int,
        keepHeight: Int,
        transitionPixels: Int
    ) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixels = [UInt8](repeating: 255, count: canvasWidth * canvasHeight)

        // Fill the keep rectangle with black.
        for y in keepY..<(keepY + keepHeight) {
            let row = y * canvasWidth
            for x in keepX..<(keepX + keepWidth) {
                pixels[row + x] = 0
            }
        }

        // Compute which sides actually need a transition band.
        let hasLeft = keepX > 0
        let hasRight = (keepX + keepWidth) < canvasWidth
        let hasTop = keepY > 0
        let hasBottom = (keepY + keepHeight) < canvasHeight

        // Inner horizontal ramps (left/right edges of the keep rectangle).
        let bandW = max(1, min(transitionPixels, keepWidth / 2))
        if hasLeft {
            for x in 0..<bandW {
                let v = UInt8((255 * (bandW - x) + bandW / 2) / bandW)
                let col = keepX + x
                for y in keepY..<(keepY + keepHeight) {
                    pixels[y * canvasWidth + col] = max(pixels[y * canvasWidth + col], v)
                }
            }
        }
        if hasRight {
            for x in 0..<bandW {
                let v = UInt8((255 * (bandW - x) + bandW / 2) / bandW)
                let col = keepX + keepWidth - 1 - x
                for y in keepY..<(keepY + keepHeight) {
                    pixels[y * canvasWidth + col] = max(pixels[y * canvasWidth + col], v)
                }
            }
        }
        // Inner vertical ramps (top/bottom edges of the keep rectangle).
        let bandH = max(1, min(transitionPixels, keepHeight / 2))
        if hasTop {
            for y in 0..<bandH {
                let v = UInt8((255 * (bandH - y) + bandH / 2) / bandH)
                let row = (keepY + y) * canvasWidth
                for x in keepX..<(keepX + keepWidth) {
                    pixels[row + x] = max(pixels[row + x], v)
                }
            }
        }
        if hasBottom {
            for y in 0..<bandH {
                let v = UInt8((255 * (bandH - y) + bandH / 2) / bandH)
                let row = (keepY + keepHeight - 1 - y) * canvasWidth
                for x in keepX..<(keepX + keepWidth) {
                    pixels[row + x] = max(pixels[row + x], v)
                }
            }
        }

        guard let context = CGContext(
            data: &pixels,
            width: canvasWidth,
            height: canvasHeight,
            bitsPerComponent: 8,
            bytesPerRow: canvasWidth,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        return context.makeImage()
    }
}
