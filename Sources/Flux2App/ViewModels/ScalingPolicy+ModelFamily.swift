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
}
