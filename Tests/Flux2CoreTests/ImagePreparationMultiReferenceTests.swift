import CoreGraphics
import Flux2Core
import XCTest

final class ImagePreparationMultiReferenceTests: XCTestCase {
    func testPrepareFormatsEveryReferenceImage() throws {
        let primary = Self.makeSolidImage(width: 800, height: 600)
        let secondary = Self.makeSolidImage(width: 1920, height: 1080)

        var settings = ImagePreparationSettings()
        settings.sizingFavor = .horizontal
        settings.sizingMethod = .pad
        settings.megapixelBudget = 1.0

        let prepared = try ImagePreparation.prepare(
            referenceImages: [primary, secondary],
            settings: settings
        )

        XCTAssertEqual(prepared.images.count, 2)

        for image in prepared.images {
            XCTAssertEqual(image.width % settings.pixelAlignment, 0)
            XCTAssertEqual(image.height % settings.pixelAlignment, 0)
        }

        XCTAssertNotEqual(prepared.images[1].width, secondary.width)
        XCTAssertNotEqual(prepared.images[1].height, secondary.height)
    }

    private static func makeSolidImage(width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let image = context.makeImage() else {
            fatalError("Failed to create test image")
        }
        return image
    }
}
