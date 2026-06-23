import CoreGraphics
import XCTest
@testable import Flux2Core

final class ImageMaskBuilderTests: XCTestCase {
    func testRectangularInpaintMaskFillsWhiteInsideRegion() throws {
        let mask = try ImageMaskBuilder.rectangularInpaintMask(
            width: 100,
            height: 80,
            normalizedRect: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        )

        XCTAssertEqual(mask.width, 100)
        XCTAssertEqual(mask.height, 80)

        let center = sampleLuminance(mask, x: 50, y: 40)
        let corner = sampleLuminance(mask, x: 5, y: 5)
        XCTAssertGreaterThan(center, 200)
        XCTAssertLessThan(corner, 55)
    }

    private func sampleLuminance(_ image: CGImage, x: Int, y: Int) -> UInt8 {
        var pixel: UInt8 = 0
        guard let context = CGContext(
            data: &pixel,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 1,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            XCTFail("Failed to create sampling context")
            return 0
        }
        context.draw(image, in: CGRect(x: -x, y: -(image.height - y), width: image.width, height: image.height))
        return pixel
    }
}
