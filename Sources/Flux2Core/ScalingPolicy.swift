/**
 * ScalingPolicy.swift
 * Single source of truth for generation sizing: per-family pixel alignment, the
 * megapixel-budget -> pixels clamp, image -> latent dimensions, and direction-
 * explicit snapping. Built on a plain `alignment` Int so it stays in Flux2Core
 * (ModelFamily lives in the app layer; the `init(family:)` convenience is added
 * there). Intended consumers: ImagePreparation, the pipeline, the chains,
 * training, and the app's NR-IQA / HQ-resize routing.
 *
 * The one invariant — `alignment` must be a positive multiple of the latent
 * factor (VAE 8 × patch 2 = 16) — is enforced here and nowhere else.
 *
 * Design: `ImagePreparation` owns the sizing/snap primitives (it has the most
 * callers); `ScalingPolicy` is the typed facade over them — one alignment, one
 * budget clamp, one floor/ceil pair (`snapUp`/`snapDown` route through
 * `ImagePreparation.snapToMultiple`/`floorToMultiple`), one latent map
 * (`LatentUtils.getLatentDimensions`). Phase 2 folded the fork's hand-rolled
 * floor copies into `floorToMultiple` and replaced the literal `32` defaults with
 * `ImagePreparation.generationSizeMultiple`. The latent `/16` copies that remain
 * live in upstream-tracked pipeline / chains / training files (`LatentUtils`,
 * `Flux2Pipeline`, `Flux2OutpaintingChain`, the LoRA trainer); they already
 * centralize on `LatentUtils.getLatentDimensions`, so consolidating them further
 * is deferred to avoid upstream-rebase friction for no behavior change.
 */

import CoreGraphics
import Foundation

/// Pixel dimensions in image space.
public typealias PixelSize = (width: Int, height: Int)

public struct ScalingPolicy: Sendable, Equatable {
    /// Generation multiple every output dimension snaps to (e.g. 32 for FLUX.2).
    public let alignment: Int
    /// VAE downsample (8) × patch size (2). Output dims must be a multiple of this.
    public let latentFactor: Int

    /// - Precondition: `alignment` is a positive multiple of `latentFactor`.
    public init(alignment: Int = ImagePreparation.generationSizeMultiple, latentFactor: Int = 16) {
        precondition(
            Self.isValidAlignment(alignment, latentFactor: latentFactor),
            "ScalingPolicy alignment \(alignment) must be a positive multiple of latentFactor \(latentFactor)"
        )
        self.alignment = alignment
        self.latentFactor = latentFactor
    }

    /// The single expression of the alignment invariant — exposed so callers and
    /// tests can check it without tripping the `init` precondition.
    public static func isValidAlignment(_ alignment: Int, latentFactor: Int) -> Bool {
        alignment > 0 && latentFactor > 0 && alignment % latentFactor == 0
    }

    // MARK: - Budget

    /// Megapixel budget -> conditioning pixel count, clamped to [0.25, 4.0] MP.
    public func budgetPixels(megapixelBudget: Double) -> Int {
        ImagePreparation.conditioningPixelBudget(for: megapixelBudget)
    }

    // MARK: - Target size

    /// The exact pixel dimensions generation will use for `image` under
    /// `settings` (budget + aspect favour + scale + live area applied). The
    /// policy's `alignment` is authoritative — it overrides `settings.pixelAlignment`.
    public func targetSize(for image: CGImage, settings: ImagePreparationSettings) -> PixelSize {
        var settings = settings
        settings.pixelAlignment = alignment
        return ImagePreparation.generationSize(referenceImage: image, settings: settings)
    }

    // MARK: - Latent dimensions

    /// Latent grid + patch count for a pixel size. Replaces hand-inlined
    /// `width / 16` / `height / 8` copies across the pipeline and training.
    public func latentDimensions(width: Int, height: Int) -> (latentW: Int, latentH: Int, numPatches: Int) {
        let dims = LatentUtils.getLatentDimensions(height: height, width: width)
        return (latentW: dims.latentW, latentH: dims.latentH, numPatches: dims.numPatches)
    }

    // MARK: - Snapping (direction-explicit)

    /// Round up to `alignment` (meet a minimum valid size). Minimum result = `alignment`.
    public func snapUp(_ value: Int) -> Int {
        ImagePreparation.snapToMultiple(value, multiple: alignment)
    }

    /// Round down to `alignment` (don't exceed source / budget). Minimum result = `alignment`.
    public func snapDown(_ value: Int) -> Int {
        ImagePreparation.floorToMultiple(value, multiple: alignment)
    }
}
