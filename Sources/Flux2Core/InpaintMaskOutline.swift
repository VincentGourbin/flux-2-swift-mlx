import CoreGraphics
import Foundation

/// Traces inpaint-mask boundaries for canvas preview (marching ants).
public enum InpaintMaskOutline {
    /// Normalized top-left points along the boundary of the inpaint region (white pixels).
    public static func normalizedBoundaryPoints(
        from mask: CGImage,
        maxSampleDimension: Int = 512,
        whiteThreshold: UInt8 = 128
    ) -> [CGPoint] {
        guard mask.width > 0, mask.height > 0 else { return [] }
        guard let samples = downsampledGrayBytes(from: mask, maxSampleDimension: maxSampleDimension) else {
            return []
        }

        let width = samples.width
        let height = samples.height
        let bytes = samples.bytes
        let scaleX = CGFloat(mask.width) / CGFloat(width)
        let scaleY = CGFloat(mask.height) / CGFloat(height)

        func isInside(_ x: Int, _ y: Int) -> Bool {
            guard x >= 0, y >= 0, x < width, y < height else { return false }
            return bytes[y * width + x] >= whiteThreshold
        }

        func isBoundary(_ x: Int, _ y: Int) -> Bool {
            guard isInside(x, y) else { return false }
            if !isInside(x, y - 1) { return true }
            if !isInside(x + 1, y) { return true }
            if !isInside(x, y + 1) { return true }
            if !isInside(x - 1, y) { return true }
            return false
        }

        guard let start = firstBoundaryPixel(width: width, height: height, isBoundary: isBoundary) else {
            return []
        }

        let sampled = traceBoundary(
            start: start,
            width: width,
            height: height,
            isInside: isInside
        )

        guard sampled.count >= 3 else { return [] }

        return sampled.map { point in
            CGPoint(
                x: (CGFloat(point.x) + 0.5) * scaleX / CGFloat(mask.width),
                y: (CGFloat(point.y) + 0.5) * scaleY / CGFloat(mask.height)
            )
        }
    }

    private struct SampledGray {
        var width: Int
        var height: Int
        var bytes: [UInt8]
    }

    private static func downsampledGrayBytes(
        from mask: CGImage,
        maxSampleDimension: Int
    ) -> SampledGray? {
        let sourceWidth = mask.width
        let sourceHeight = mask.height
        let longest = max(sourceWidth, sourceHeight)
        let scale = min(1, CGFloat(maxSampleDimension) / CGFloat(longest))
        let width = max(1, Int((CGFloat(sourceWidth) * scale).rounded()))
        let height = max(1, Int((CGFloat(sourceHeight) * scale).rounded()))

        let colorSpace = CGColorSpaceCreateDeviceGray()
        var bytes = [UInt8](repeating: 0, count: width * height)
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        context.interpolationQuality = .low
        context.draw(mask, in: CGRect(x: 0, y: 0, width: width, height: height))
        return SampledGray(width: width, height: height, bytes: bytes)
    }

    private static func firstBoundaryPixel(
        width: Int,
        height: Int,
        isBoundary: (Int, Int) -> Bool
    ) -> (x: Int, y: Int)? {
        for y in 0..<height {
            for x in 0..<width where isBoundary(x, y) {
                return (x, y)
            }
        }
        return nil
    }

    /// Moore-neighbor walk on 4-connected foreground boundary.
    private static func traceBoundary(
        start: (x: Int, y: Int),
        width: Int,
        height: Int,
        isInside: (Int, Int) -> Bool
    ) -> [(x: Int, y: Int)] {
        let directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        var path: [(x: Int, y: Int)] = []
        var current = start
        var directionIndex = 0
        let maxSteps = width * height * 2

        for _ in 0..<maxSteps {
            path.append(current)

            var turned = false
            for offset in 0..<4 {
                let candidate = (directionIndex + offset) % 4
                let delta = directions[candidate]
                let next = (current.x + delta.0, current.y + delta.1)
                if isInside(next.0, next.1) {
                    directionIndex = (candidate + 3) % 4
                    current = next
                    turned = true
                    break
                }
            }

            if !turned { break }
            if current == start, path.count > 2 { break }
        }

        return path
    }
}
