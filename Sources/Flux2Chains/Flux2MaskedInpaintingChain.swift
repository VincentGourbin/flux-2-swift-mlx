// Flux2MaskedInpaintingChain.swift — RePaint-style masked inpainting
// Copyright 2025 Vincent Gourbin
//
// FLUX.2 has no dedicated Fill checkpoint (as of 2026-05). Diffusers'
// `FluxFillPipeline` channel-cat approach requires fine-tuned input channels,
// which FLUX.2 base/distilled doesn't have. The classical RePaint trick gets
// us most of the way for free: at every denoising step the region OUTSIDE the
// user mask is overwritten with the original-image latent re-noised to the
// current sigma. Only the inside-mask region accumulates new content.
//
// Implemented as a `Flux2Chain` using the `onStep:` hook on the standard T2I
// generation path — so progress callbacks, profiling, memory caps, and
// scheduler overrides all behave exactly like a normal generate() call.

import Foundation
import Flux2Core
@preconcurrency import MLX
@preconcurrency import MLXRandom
import CoreGraphics

/// Masked inpainting chain (RePaint-style per-step latent blending).
public struct Flux2MaskedInpaintingChain: Flux2Chain {
    /// Underlying pipeline. The chain does not own it — host apps can reuse
    /// the same pipeline across many chain runs without paying the model
    /// load cost each time.
    public let pipeline: Flux2Pipeline

    /// Text describing the **full** target image, **not** just the edit.
    ///
    /// FLUX.2 has no negative prompts and no dedicated edit instruction
    /// channel — the prompt is the only steering signal for the in-mask
    /// region. Follow the BFL prompting guidelines
    /// (<https://docs.bfl.ml/guides/prompting_guide_flux2>):
    ///
    /// - **Structure**: *Subject + Action + Style + Context*. Word order
    ///   matters: FLUX.2 weighs leading tokens more.
    /// - **Length**: ~30–80 words. Describe the **whole** scene including
    ///   the kept surroundings so the model has lighting/perspective cues
    ///   for the inpainted region.
    /// - **No negatives**: describe what you want, never what to avoid.
    ///
    /// Short prompts like `"a duck"` give the model no spatial / lighting /
    /// integration cues and produce floating, mismatched subjects. Prefer:
    /// *"A mallard duck standing on grey weathered concrete pavement at the
    /// base of a rough limestone wall, soft midday sunlight from the upper
    /// left casting a clear cast shadow on the ground, scattered grass
    /// clippings around, naturalistic photography, shallow depth of field."*
    ///
    /// For dynamic prompt expansion at inference time, set
    /// ``upsamplePrompt`` to `true`.
    public let prompt: String
    public let image: CGImage
    /// Mask image, same dimensions as `image` (resized otherwise). What
    /// "mask" means depends on ``maskConvention``:
    /// - default `grayscaleWhiteInpaint`: separate grayscale image, white
    ///   = inpaint, black = keep. Soft greys honoured.
    /// - `alphaTransparentInpaint`: an "erased copy" of the source. Alpha
    ///   = 0 (transparent) = inpaint, alpha = 1 (opaque) = keep. RGB
    ///   ignored.
    public let mask: CGImage
    /// How ``mask`` is interpreted. See ``Flux2MaskConvention``. Default
    /// `grayscaleWhiteInpaint` for back-compat. Switch to
    /// `alphaTransparentInpaint` when the mask is hand-authored in a photo
    /// editor (Photoshop / Affinity / Procreate) by erasing the part to
    /// redo — avoids maintaining a parallel grayscale file in lockstep.
    public let maskConvention: Flux2MaskConvention
    /// Optional reference image(s) for the transformer to attend to in
    /// addition to the prompt. When provided, the chain switches generation
    /// from `.textToImage` to `.imageToImage(referenceImages)` while still
    /// applying the RePaint blend.
    ///
    /// **When to use:** for outpainting / scene extension. Pass the
    /// original (un-extended) image so the transformer's attention
    /// continues the kept content into the new strips (see
    /// `Flux2OutpaintingChain`).
    ///
    /// **When NOT to use:** for "replace object X with object Y" inpainting.
    /// Passing the source image (which still contains X under the mask)
    /// makes the model attend to X's pixels and bleed its texture/colour
    /// into Y — empirically verified on cat → duck (the duck inherits the
    /// tabby's grey-striped head). Leave nil and rely on a descriptive
    /// Flux 2-style prompt; see ``prompt``.
    public let referenceImages: [CGImage]?
    /// When `referenceImages` is nil, auto-condition the transformer on
    /// ``image`` itself.
    ///
    /// Default `false`. **This is intentional for the common case of
    /// object replacement**: feeding the source as a reference lets the
    /// to-be-replaced subject leak into the result via attention. Enable
    /// only when the goal is *repair* or *scene extension* and the masked
    /// region is empty / neutral — in that scenario the reference gives
    /// the model the lighting / perspective / palette context it needs to
    /// integrate the new content.
    public let useImageAsReference: Bool
    public let steps: Int
    public let guidance: Float
    public let seed: UInt64?
    /// When `true`, the active text encoder rewrites the prompt with an
    /// internal vision-language model (Qwen3.5 VLM in this framework)
    /// before encoding. Useful when the caller supplies a short edit
    /// instruction rather than a full Flux 2-style descriptive prompt.
    /// Cost: one extra VLM forward pass per call. Default `false` to
    /// preserve the caller's exact wording.
    public let upsamplePrompt: Bool
    /// Optional progress callback forwarded to `generateWithResult`.
    public let onProgress: Flux2ProgressCallback?

    /// Maximum total pixel count for the working resolution. Larger inputs are
    /// scaled down (preserving aspect) before being clamped to a multiple of
    /// 32. Default 1024² matches the existing I2I conventions.
    public let maxPixels: Int

    /// Configure a masked inpainting run.
    ///
    /// - Parameters:
    ///   - pipeline: Pipeline to drive. Reused as-is — load its models and any
    ///     LoRA *before* calling `run()` (or `run()` will load them lazily).
    ///   - prompt: Text describing the desired full image. The model
    ///     regenerates everything; the mask just forces the original back
    ///     outside the painted region.
    ///   - image: Source image. Pixels under the *black* part of the mask are
    ///     preserved bit-exact by the RePaint blend.
    ///   - mask: Mask image, same dimensions as `image` (it will be resized
    ///     otherwise). Interpretation depends on `maskConvention`:
    ///     - default `.grayscaleWhiteInpaint`: white = inpaint, black =
    ///       keep, soft greys honoured. Use a Gaussian blur on the edge
    ///       (≈ `image_width/30`) for a seamless transition.
    ///     - `.alphaTransparentInpaint`: alpha = 0 (transparent) = inpaint,
    ///       alpha = 1 (opaque) = keep. RGB content ignored.
    ///   - maskConvention: How `mask` is interpreted. Default
    ///     `.grayscaleWhiteInpaint` matches the existing API; pick
    ///     `.alphaTransparentInpaint` to let an "erased copy of the source"
    ///     drive the mask without a separate file.
    ///   - referenceImages: When non-nil and non-empty, the chain switches to
    ///     `.imageToImage` so the transformer attends to these references in
    ///     addition to the prompt. Used by `Flux2OutpaintingChain` to make
    ///     the model continue the kept region. Pass `nil` for the chain to
    ///     decide based on ``useImageAsReference``.
    ///   - useImageAsReference: When `referenceImages` is `nil`, auto-pass
    ///     ``image`` itself as the I2I reference. Default `false` — for
    ///     "replace object X with object Y" inpainting the source-as-ref
    ///     bleeds X into Y via attention. Enable for *repair / extend*
    ///     workflows where the masked region is empty.
    ///   - steps: Denoising step count. `4` matches klein distilled defaults.
    ///   - guidance: Classifier-free guidance scale. `1.0` for distilled
    ///     klein; raise to ≈ 3.5 with `.klein9BBase` or `.klein4BBase` to
    ///     trigger the classical CFG path on the underlying pipeline.
    ///   - seed: Random seed for reproducibility. `nil` for non-deterministic.
    ///   - upsamplePrompt: Rewrite the prompt via the bundled VLM before
    ///     encoding. Useful when the caller supplies a short edit
    ///     instruction. Default `false`.
    ///   - maxPixels: Cap on the working resolution. Larger inputs are scaled
    ///     down (aspect preserved) before being clamped to a multiple of 32.
    ///   - onProgress: Forwarded to the pipeline's denoising loop.
    public init(
        pipeline: Flux2Pipeline,
        prompt: String,
        image: CGImage,
        mask: CGImage,
        maskConvention: Flux2MaskConvention = .grayscaleWhiteInpaint,
        referenceImages: [CGImage]? = nil,
        useImageAsReference: Bool = false,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        upsamplePrompt: Bool = false,
        maxPixels: Int = 1024 * 1024,
        onProgress: Flux2ProgressCallback? = nil
    ) {
        self.pipeline = pipeline
        self.prompt = prompt
        self.image = image
        self.mask = mask
        self.maskConvention = maskConvention
        self.referenceImages = referenceImages
        self.useImageAsReference = useImageAsReference
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.upsamplePrompt = upsamplePrompt
        self.maxPixels = maxPixels
        self.onProgress = onProgress
    }

    /// Execute the chain.
    ///
    /// Loads the pipeline's models if needed, encodes the source image into
    /// the working latent space, builds the latent-aligned mask, registers
    /// the RePaint step hook, and runs `generateWithResult`. Returns a
    /// regular `Flux2GenerationResult` — `usedPrompt` and `wasUpsampled`
    /// reflect the same conventions as a plain `generate*` call.
    ///
    /// - Returns: The inpainted image plus prompt metadata.
    /// - Throws: Whatever the underlying pipeline can throw (model not
    ///   loaded, memory, generation cancellation).
    public func run() async throws -> Flux2GenerationResult {
        try await pipeline.loadModels()

        let (targetH, targetW) = Flux2Pipeline.resolveChainDimensions(
            width: image.width,
            height: image.height,
            maxPixels: maxPixels
        )

        // Encode the source image *once*, before the denoising loop starts.
        // The VAE stays resident so the post-denoising decode reuses it.
        let imageLatents = try await pipeline.encodeImageToPackedSequence(
            image,
            targetHeight: targetH,
            targetWidth: targetW
        )

        let maskLatents = await Flux2Pipeline.packMaskForLatentBlending(
            mask,
            targetHeight: targetH,
            targetWidth: targetW,
            convention: maskConvention
        )

        // RePaint blend: outside-mask region is forced back to (image latent
        // re-noised to sigmaNext). On the final step sigmaNext == 0 ⇒ the
        // original clean latent is restored (no hallucination outside mask).
        //
        // NOTE: the I2I path emits transformer noise predictions for the
        // *concatenated* (output + reference) sequence and slices them back
        // to the output portion before calling the hook, so this blend acts
        // exclusively on the output latents in both modes.
        let imageLatentsCaptured = imageLatents
        let maskLatentsCaptured = maskLatents
        let onStep: Flux2StepHook = { ctx, latents in
            let freshNoise = MLXRandom.normal(latents.shape)
            let sigmaNext = MLXArray(ctx.sigmaNext)
            let originalNoised = (1 - sigmaNext) * imageLatentsCaptured + sigmaNext * freshNoise
            return (1 - maskLatentsCaptured) * originalNoised + maskLatentsCaptured * latents
        }

        let mode: Flux2GenerationMode
        if let refs = referenceImages, !refs.isEmpty {
            mode = .imageToImage(images: refs)
        } else if useImageAsReference {
            mode = .imageToImage(images: [image])
        } else {
            mode = .textToImage
        }

        return try await pipeline.generateWithResult(
            mode: mode,
            prompt: prompt,
            interpretImagePaths: nil,
            height: targetH,
            width: targetW,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: upsamplePrompt,
            precomputedEmbeddings: nil,
            checkpointInterval: nil,
            onProgress: onProgress,
            onCheckpoint: nil,
            onStep: onStep
        )
    }
}
