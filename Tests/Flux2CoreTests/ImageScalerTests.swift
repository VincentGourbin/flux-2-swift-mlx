import CoreGraphics
@testable import Flux2Core
import XCTest

/// The shared Lanczos resampler must hit exact target dimensions (the contract
/// the NR-IQA scorer and the output save bus both depend on).
final class ImageScalerTests: XCTestCase {
    private func makeImage(width: Int, height: Int) -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.setFillColor(CGColor(red: 0.5, green: 0.25, blue: 0.75, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    func testDownscaleHitsExactSize() throws {
        let image = makeImage(width: 100, height: 100)
        let result = try ImageScaler.lanczos(image, to: (width: 50, height: 50))
        XCTAssertEqual(result.width, 50)
        XCTAssertEqual(result.height, 50)
    }

    func testNonUniformResizeHitsExactSize() throws {
        let image = makeImage(width: 120, height: 80)
        let result = try ImageScaler.lanczos(image, to: (width: 200, height: 300))
        XCTAssertEqual(result.width, 200)
        XCTAssertEqual(result.height, 300)
    }

    func testZeroSizeThrows() {
        let image = makeImage(width: 10, height: 10)
        XCTAssertThrowsError(try ImageScaler.lanczos(image, to: (width: 0, height: 10)))
    }
}
