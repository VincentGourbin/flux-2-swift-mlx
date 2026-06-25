import Foundation

/// Per-side pixel padding for ``Flux2OutpaintingChain`` (snapped to 32 px).
public struct OutpaintPadding: Codable, Sendable, Equatable {
    public var top: Int
    public var bottom: Int
    public var left: Int
    public var right: Int

    public static let zero = OutpaintPadding()

    public init(top: Int = 0, bottom: Int = 0, left: Int = 0, right: Int = 0) {
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
    }

    public var hasExpansion: Bool {
        top > 0 || bottom > 0 || left > 0 || right > 0
    }

    public func canvasSize(sourceWidth: Int, sourceHeight: Int) -> (width: Int, height: Int) {
        (
            sourceWidth + left + right,
            sourceHeight + top + bottom
        )
    }

    public func totalPixels(sourceWidth: Int, sourceHeight: Int) -> Int {
        let size = canvasSize(sourceWidth: sourceWidth, sourceHeight: sourceHeight)
        return size.width * size.height
    }

    /// Round each non-zero side up to the next multiple of 32 (FLUX.2 grid).
    public func snapped() -> OutpaintPadding {
        OutpaintPadding(
            top: Self.snapSide(top),
            bottom: Self.snapSide(bottom),
            left: Self.snapSide(left),
            right: Self.snapSide(right)
        )
    }

    public static func snapSide(_ value: Int) -> Int {
        guard value > 0 else { return 0 }
        return ((value + 31) / 32) * 32
    }

    /// Snap sides and clamp total canvas pixels to ``maxPixels``.
    public func clamped(sourceWidth: Int, sourceHeight: Int, maxPixels: Int) -> OutpaintPadding {
        var padding = snapped()
        padding.top = max(0, padding.top)
        padding.bottom = max(0, padding.bottom)
        padding.left = max(0, padding.left)
        padding.right = max(0, padding.right)

        guard maxPixels > 0 else { return .zero }

        while padding.totalPixels(sourceWidth: sourceWidth, sourceHeight: sourceHeight) > maxPixels,
              padding.hasExpansion {
            if padding.bottom >= 32 {
                padding.bottom -= 32
            } else if padding.right >= 32 {
                padding.right -= 32
            } else if padding.top >= 32 {
                padding.top -= 32
            } else if padding.left >= 32 {
                padding.left -= 32
            } else {
                break
            }
        }
        return padding
    }
}
