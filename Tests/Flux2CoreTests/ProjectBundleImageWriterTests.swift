import CoreGraphics
import Flux2Core
import XCTest

final class ProjectBundleImageWriterTests: XCTestCase {
    func testJXLWriteAndLoadRoundTrip() throws {
        try XCTSkipUnless(ProjectBundleImageWriter.isSupported(), "JPEG XL encoding is not available")

        let image = try makeTestImage(width: 48, height: 32)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-jxl-writer-\(UUID().uuidString).jxl")

        defer { try? FileManager.default.removeItem(at: url) }

        try ProjectBundleImageWriter.write(image, to: url, mode: .lossless)
        let decoded = try ProjectBundleImageWriter.loadCGImage(from: url)
        XCTAssertEqual(decoded.width, image.width)
        XCTAssertEqual(decoded.height, image.height)
    }

    private func makeTestImage(width: Int, height: Int) throws -> CGImage {
        var bytes = [UInt8](repeating: 128, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let image = context.makeImage() else {
            throw Flux2Error.imageProcessingFailed("Failed to allocate test image")
        }
        return image
    }
}
