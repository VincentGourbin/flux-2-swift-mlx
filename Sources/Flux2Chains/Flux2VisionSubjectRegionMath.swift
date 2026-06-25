import CoreGraphics
import Foundation
import Flux2Core

/// Pure geometry helpers for Vision subject disambiguation.
enum VisionSubjectRegionMath {
    static func expandedRegion(
        _ region: VisionSubjectRegion,
        step: Int,
        stepFraction: Double
    ) -> VisionSubjectRegion {
        guard step > 0 else { return region }
        let scale = 1 + stepFraction * Double(step)
        switch region {
        case .rectangle(let rect):
            return .rectangle(expandedRect(rect, scale: scale))
        case .polygon(let points):
            guard !points.isEmpty else { return region }
            let centroid = polygonCentroid(points)
            let scaled = points.map { point in
                CGPoint(
                    x: centroid.x + (point.x - centroid.x) * scale,
                    y: centroid.y + (point.y - centroid.y) * scale
                )
            }
            return .polygon(clampedPoints(scaled))
        }
    }

    static func rasterizedSelectionMask(
        region: VisionSubjectRegion,
        width: Int,
        height: Int
    ) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: width * height)
        guard width > 0, height > 0 else { return bytes }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return bytes
        }

        context.setFillColor(gray: 1, alpha: 1)
        switch region {
        case .rectangle(let normalizedRect):
            let rect = pixelRect(from: normalizedRect, width: width, height: height)
            context.fill(rect)
        case .polygon(let points):
            guard points.count >= 3 else { return bytes }
            let path = CGMutablePath()
            let first = pixelPoint(points[0], width: width, height: height)
            path.move(to: first)
            for point in points.dropFirst() {
                path.addLine(to: pixelPoint(point, width: width, height: height))
            }
            path.closeSubpath()
            context.addPath(path)
            context.fillPath()
        }
        return bytes
    }

    static func overlapFraction(
        subjectMask: [UInt8],
        selectionMask: [UInt8],
        subjectThreshold: UInt8 = 128
    ) -> Double {
        guard subjectMask.count == selectionMask.count, !subjectMask.isEmpty else {
            return 0
        }
        var selectionCount = 0
        var overlapCount = 0
        for index in subjectMask.indices {
            if selectionMask[index] > 0 {
                selectionCount += 1
                if subjectMask[index] >= subjectThreshold {
                    overlapCount += 1
                }
            }
        }
        guard selectionCount > 0 else { return 0 }
        return Double(overlapCount) / Double(selectionCount)
    }

    private static func expandedRect(_ rect: CGRect, scale: Double) -> CGRect {
        let clamped = ImagePreparation.clampUnitRect(rect)
        let center = CGPoint(x: clamped.midX, y: clamped.midY)
        let halfW = clamped.width * scale / 2
        let halfH = clamped.height * scale / 2
        return ImagePreparation.clampUnitRect(
            CGRect(
                x: center.x - halfW,
                y: center.y - halfH,
                width: halfW * 2,
                height: halfH * 2
            )
        )
    }

    private static func polygonCentroid(_ points: [CGPoint]) -> CGPoint {
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        for point in points {
            sumX += point.x
            sumY += point.y
        }
        let count = CGFloat(points.count)
        return CGPoint(x: sumX / count, y: sumY / count)
    }

    private static func clampedPoints(_ points: [CGPoint]) -> [CGPoint] {
        points.map {
            CGPoint(
                x: min(max($0.x, 0), 1),
                y: min(max($0.y, 0), 1)
            )
        }
    }

    private static func pixelRect(from normalizedRect: CGRect, width: Int, height: Int) -> CGRect {
        CGRect(
            x: normalizedRect.minX * CGFloat(width),
            y: normalizedRect.minY * CGFloat(height),
            width: normalizedRect.width * CGFloat(width),
            height: normalizedRect.height * CGFloat(height)
        )
    }

    private static func pixelPoint(_ normalizedPoint: CGPoint, width: Int, height: Int) -> CGPoint {
        CGPoint(
            x: normalizedPoint.x * CGFloat(width),
            y: normalizedPoint.y * CGFloat(height)
        )
    }
}
