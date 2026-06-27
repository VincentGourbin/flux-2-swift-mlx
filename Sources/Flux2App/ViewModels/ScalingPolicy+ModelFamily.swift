/**
 * ScalingPolicy+ModelFamily.swift
 * App-layer bridge: build a Flux2Core `ScalingPolicy` from a `ModelFamily`
 * (which carries the per-family pixel alignment). Keeps Flux2Core free of the
 * app's ModelFamily type while giving callers the "for a family" entry points.
 */

import CoreGraphics
import Flux2Core

extension ScalingPolicy {
    /// Policy for a model family, using its per-family pixel alignment.
    init(family: ModelFamily) {
        self.init(alignment: family.pixelAlignment)
    }

    /// Authoritative generation dimensions for `image` + `family` under `settings`.
    static func targetSize(
        for image: CGImage,
        family: ModelFamily,
        settings: ImagePreparationSettings
    ) -> PixelSize {
        ScalingPolicy(family: family).targetSize(for: image, settings: settings)
    }

    /// Authoritative generation dimensions for `image` + `family` at an explicit
    /// megapixel budget, assuming a full-frame edit with default formatting
    /// (favour `.original`, scale `1.0`, no Live Area). The NR-IQA quality probe
    /// uses this so its measurement never inherits a project's live-area / favour /
    /// scale — only the budget + per-family alignment.
    static func targetSize(
        for image: CGImage,
        family: ModelFamily,
        megapixelBudget: Double
    ) -> PixelSize {
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = megapixelBudget
        return targetSize(for: image, family: family, settings: settings)
    }
}
