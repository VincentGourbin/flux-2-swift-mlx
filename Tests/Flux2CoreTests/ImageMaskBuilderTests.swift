import CoreGraphics
import Flux2Core
import XCTest

final class ImageMaskBuilderTests: XCTestCase {
    func testRectangleMaskWhiteInside() throws {
        let mask = try ImageMaskBuilder.rectangularInpaintMask(
            width: 100,
            height: 80,
            normalizedRect: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        )
        XCTAssertEqual(mask.width, 100)
        XCTAssertEqual(mask.height, 80)

        let center = sampleGray(mask, x: 50, y: 40)
        let corner = sampleGray(mask, x: 5, y: 5)
        XCTAssertGreaterThan(center, 200)
        XCTAssertLessThan(corner, 55)
    }

    func testPolygonMask() throws {
        let points = [
            CGPoint(x: 0.2, y: 0.2),
            CGPoint(x: 0.8, y: 0.2),
            CGPoint(x: 0.5, y: 0.8),
        ]
        let mask = try ImageMaskBuilder.polygonInpaintMask(
            width: 64,
            height: 64,
            normalizedPoints: points
        )
        let inside = sampleGray(mask, x: 32, y: 40)
        let outside = sampleGray(mask, x: 4, y: 4)
        XCTAssertGreaterThan(inside, 200)
        XCTAssertLessThan(outside, 55)
    }

    func testPolygonMaskRequiresThreePoints() {
        XCTAssertThrowsError(
            try ImageMaskBuilder.polygonInpaintMask(
                width: 32,
                height: 32,
                normalizedPoints: [CGPoint(x: 0.2, y: 0.2), CGPoint(x: 0.8, y: 0.2)]
            )
        )
    }

    private func sampleGray(_ image: CGImage, x: Int, y: Int) -> UInt8 {
        var bytes = [UInt8](repeating: 0, count: 1)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &bytes,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 1,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return 0
        }
        context.draw(
            image,
            in: CGRect(x: -CGFloat(x), y: -CGFloat(y), width: CGFloat(image.width), height: CGFloat(image.height))
        )
        return bytes[0]
    }
}
