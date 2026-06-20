import CoreGraphics

/// Bridges image-space rectangles (`CGImage.cropping`, pixel buffers, and the
/// SwiftUI preview use a top-left origin) to CoreGraphics draw rectangles
/// (`CGContext.draw` uses a bottom-left origin for the destination rect).
public enum ImageCoordinateMapper {
    public static func contextDrawRect(forTopLeftRect rect: CGRect, canvasHeight: CGFloat) -> CGRect {
        CGRect(
            x: rect.minX,
            y: canvasHeight - rect.maxY,
            width: rect.width,
            height: rect.height
        )
    }
}
