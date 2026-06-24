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
import FluxTextEncoders  // FluxDebug logger; VLM service is reached via Flux2VLMPromptBuilder
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
    /// When `true`, the active text encoder rewrites the prompt with its
    /// own internal language model (Mistral / Klein-Qwen3) before
    /// encoding. **Text-only path** — does not look at ``image``. Useful
    /// when the caller supplies a short instruction; the text encoder
    /// expands it with stylistic detail.
    ///
    /// For *image-aware* enrichment (the prompt is built from the source
    /// image's actual lighting / camera / materials, following the BFL
    /// prompting guide), use ``enrichPromptWithVLM`` instead — see the
    /// collision rules below.
    ///
    /// Cost: one extra text-encoder forward pass per call. Default
    /// `false` to preserve the caller's exact wording.
    public let upsamplePrompt: Bool

    /// When `true`, ask the bundled Qwen3.5 VLM to look at ``image`` and
    /// rewrite ``prompt`` into a 30-80 word BFL-style Flux 2 prompt that
    /// inherits the source's photographic identity (camera angle,
    /// lighting direction, surface materials, colour palette, depth of
    /// field). The exact rewriting strategy depends on ``intent``.
    ///
    /// **The VLM is never required.** Behaviour by configuration:
    /// - VLM loaded → image-aware prompt is built and used; the
    ///   downstream `upsamplePrompt` is forced to `false` so the text
    ///   encoder doesn't re-process the already-finalised prompt.
    /// - VLM not loaded → a warning is logged via ``FluxDebug`` and the
    ///   chain falls back to the existing behaviour (caller's prompt,
    ///   honouring ``upsamplePrompt`` as documented). No throw.
    ///
    /// **Collision with ``upsamplePrompt``:** when both are `true`, the
    /// VLM wins because it produces the higher-quality, image-aware
    /// output. A warning is logged so the caller can clean up their
    /// configuration.
    ///
    /// Cost: one extra VLM forward pass (~few seconds on M-series). The
    /// caller is responsible for the VLM lifecycle — load it via
    /// ``FluxTextEncoders/shared/loadQwen35VLM(from:)`` before
    /// ``run()`` and unload when done. Default `false`.
    public let enrichPromptWithVLM: Bool

    /// Drives ``Flux2VLMPromptBuilder`` when ``enrichPromptWithVLM`` is
    /// `true`. Ignored otherwise. See ``Flux2InpaintIntent`` — the three
    /// cases (`.replace`, `.remove`, `.modify`) have *opposite* prompting
    /// requirements and must be told apart explicitly.
    ///
    /// Default `.replace` (most common case: swap object X for Y).
    public let intent: Flux2InpaintIntent
    /// Normalized barn-door region Qwen sees when enriching the prompt.
    /// `nil` or full-frame means the entire source image. Does not affect
    /// the fill pass itself — only VLM prompt writing.
    public let vlmContextArea: CGRect?
    /// Optional progress callback forwarded to `generateWithResult`.
    public let onProgress: Flux2ProgressCallback?
    /// Optional checkpoint callback forwarded to `generateWithResult`.
    public let onCheckpoint: Flux2CheckpointCallback?
    /// When set, the pipeline emits intermediate decodes every N steps.
    public let checkpointInterval: Int?

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
    ///   - upsamplePrompt: Text-encoder-only prompt rewriting. See the
    ///     property doc for the difference vs ``enrichPromptWithVLM``,
    ///     and for the collision rule when both are set. Default `false`.
    ///   - enrichPromptWithVLM: Opt-in image-aware prompt rewriting via
    ///     the bundled Qwen3.5 VLM. Strictly optional — when the VLM is
    ///     not loaded the chain falls back to the verbatim prompt and
    ///     logs a warning. Default `false`.
    ///   - intent: Drives the VLM system prompt when
    ///     `enrichPromptWithVLM == true`; ignored otherwise. Default
    ///     `.replace`.
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
        enrichPromptWithVLM: Bool = false,
        intent: Flux2InpaintIntent = .replace,
        vlmContextArea: CGRect? = nil,
        maxPixels: Int = 1024 * 1024,
        checkpointInterval: Int? = nil,
        onProgress: Flux2ProgressCallback? = nil,
        onCheckpoint: Flux2CheckpointCallback? = nil
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
        self.enrichPromptWithVLM = enrichPromptWithVLM
        self.intent = intent
        self.vlmContextArea = vlmContextArea
        self.maxPixels = maxPixels
        self.checkpointInterval = checkpointInterval
        self.onProgress = onProgress
        self.onCheckpoint = onCheckpoint
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

        // Resolve which prompt + upsample flag actually reach the pipeline.
        // VLM enrichment is strictly opt-in and gracefully falls back when
        // the VLM is not loaded — never throws or auto-loads.
        let (resolvedPrompt, resolvedUpsample, enrichmentNotice) = await resolveFinalPromptAndUpsample()

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

        let result = try await pipeline.generateWithResult(
            mode: mode,
            prompt: resolvedPrompt,
            interpretImagePaths: nil,
            height: targetH,
            width: targetW,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: resolvedUpsample,
            precomputedEmbeddings: nil,
            checkpointInterval: checkpointInterval,
            onProgress: onProgress,
            onCheckpoint: onCheckpoint,
            onStep: onStep
        )

        return Flux2GenerationResult(
            image: result.image,
            usedPrompt: result.usedPrompt,
            wasUpsampled: result.wasUpsampled,
            originalPrompt: result.originalPrompt,
            notice: enrichmentNotice
        )
    }

    // MARK: - VLM enrichment resolution

    /// Compute the prompt and the downstream `upsamplePrompt` that
    /// actually reach `Flux2Pipeline.generateWithResult`. Handles the
    /// strictly-optional VLM path, the collision rule with
    /// ``upsamplePrompt``, and the graceful fallback when the VLM isn't
    /// loaded.
    ///
    /// Outcomes (in order of preference):
    /// 1. `enrichPromptWithVLM == true` + VLM loaded + builder returns a
    ///    non-empty string → use that prompt, force `upsamplePrompt = false`
    ///    downstream (the prompt is already finalised). If
    ///    ``upsamplePrompt`` was also `true`, log a warning — VLM wins.
    /// 2. `enrichPromptWithVLM == true` + VLM not loaded → log a
    ///    warning, fall through to (3).
    /// 3. Default → caller's prompt + caller's ``upsamplePrompt`` (existing
    ///    behaviour, byte-identical to before this feature).
    private func resolveFinalPromptAndUpsample() async -> (String, Bool, String?) {
        guard enrichPromptWithVLM else {
            return (prompt, upsamplePrompt, nil)
        }
        guard FluxTextEncoders.shared.isQwen35VLMLoaded else {
            FluxDebug.error("[Flux2MaskedInpaintingChain] enrichPromptWithVLM=true but Qwen3.5 VLM is not loaded. Falling back to caller's prompt. Load the VLM via FluxTextEncoders.shared.loadQwen35VLM(from:) before run() to enable image-aware prompt enrichment.")
            return (prompt, upsamplePrompt, nil)
        }
        if upsamplePrompt {
            FluxDebug.error("[Flux2MaskedInpaintingChain] Both enrichPromptWithVLM and upsamplePrompt are true — VLM wins (image-aware enrichment supersedes text-only upsampling). Disable one of the two to silence this warning.")
        }
        do {
            let built = try await Flux2VLMPromptBuilder.buildInpaintPrompt(
                source: image,
                userInstruction: prompt,
                intent: intent,
                vlmContextArea: vlmContextArea
            )
            guard let final = built?.trimmingCharacters(in: .whitespacesAndNewlines), !final.isEmpty else {
                FluxDebug.error("[Flux2MaskedInpaintingChain] VLM returned no usable prompt (empty or safety refusal) — falling back to caller's prompt.")
                return (
                    prompt,
                    upsamplePrompt,
                    "Qwen3.5 could not write a prompt for this image (often a safety filter). Used your hint or a generic fill prompt instead."
                )
            }
            FluxDebug.info("[Flux2MaskedInpaintingChain] VLM-enriched prompt (\(intent.rawValue)): \(final)")
            return (final, false, nil)
        } catch {
            FluxDebug.error("[Flux2MaskedInpaintingChain] VLM enrichment failed: \(error). Falling back to caller's prompt.")
            return (prompt, upsamplePrompt, nil)
        }
    }
}
