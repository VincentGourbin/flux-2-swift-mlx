import CoreGraphics
@testable import Flux2Core
import XCTest

final class ImageCoordinateMapperTests: XCTestCase {
    func testTopLeftRectConvertsToCoreGraphicsDestinationRect() {
        let rect = CGRect(x: 10, y: 20, width: 30, height: 40)

        XCTAssertEqual(
            ImageCoordinateMapper.contextDrawRect(forTopLeftRect: rect, canvasHeight: 200),
            CGRect(x: 10, y: 140, width: 30, height: 40)
        )
    }

    func testMappedImageDrawLandsInRequestedTopLeftPixels() throws {
        let patch = try makeImage(
            width: 2,
            height: 2,
            pixels: [
                .red, .green,
                .blue, .white,
            ]
        )
        var canvas = [UInt8](repeating: 0, count: 4 * 4 * 4)
        let context = try makeContext(width: 4, height: 4, data: &canvas)
        context.interpolationQuality = .none

        context.draw(
            patch,
            in: ImageCoordinateMapper.contextDrawRect(
                forTopLeftRect: CGRect(x: 1, y: 0, width: 2, height: 2),
                canvasHeight: 4
            )
        )

        XCTAssertEqual(pixelName(canvas, width: 4, x: 1, y: 0), "red")
        XCTAssertEqual(pixelName(canvas, width: 4, x: 2, y: 0), "green")
        XCTAssertEqual(pixelName(canvas, width: 4, x: 1, y: 1), "blue")
        XCTAssertEqual(pixelName(canvas, width: 4, x: 2, y: 1), "white")
        XCTAssertEqual(pixelName(canvas, width: 4, x: 1, y: 2), "black")
    }

    func testUnmappedImageDrawLandsOnWrongVerticalSide() throws {
        let patch = try makeImage(
            width: 1,
            height: 1,
            pixels: [.red]
        )
        var canvas = [UInt8](repeating: 0, count: 4 * 4 * 4)
        let context = try makeContext(width: 4, height: 4, data: &canvas)
        context.interpolationQuality = .none

        context.draw(patch, in: CGRect(x: 1, y: 0, width: 1, height: 1))

        XCTAssertEqual(pixelName(canvas, width: 4, x: 1, y: 0), "black")
        XCTAssertEqual(pixelName(canvas, width: 4, x: 1, y: 3), "red")
    }
}

private enum TestPixel {
    case black
    case red
    case green
    case blue
    case white

    var rgba: [UInt8] {
        switch self {
        case .black: return [0, 0, 0, 255]
        case .red: return [255, 0, 0, 255]
        case .green: return [0, 255, 0, 255]
        case .blue: return [0, 0, 255, 255]
        case .white: return [255, 255, 255, 255]
        }
    }
}

private func makeImage(width: Int, height: Int, pixels: [TestPixel]) throws -> CGImage {
    let data = Data(pixels.flatMap(\.rgba))
    guard let provider = CGDataProvider(data: data as CFData),
          let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
          ) else {
        throw XCTSkip("Could not create test image")
    }
    return image
}

private func makeContext(width: Int, height: Int, data: inout [UInt8]) throws -> CGContext {
    guard let context = CGContext(
        data: &data,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw XCTSkip("Could not create test context")
    }
    return context
}

private func pixelName(_ data: [UInt8], width: Int, x: Int, y: Int) -> String {
    let index = (y * width + x) * 4
    let red = data[index]
    let green = data[index + 1]
    let blue = data[index + 2]

    switch (red, green, blue) {
    case (255, 0, 0): return "red"
    case (0, 255, 0): return "green"
    case (0, 0, 255): return "blue"
    case (255, 255, 255): return "white"
    default: return "black"
    }
}
