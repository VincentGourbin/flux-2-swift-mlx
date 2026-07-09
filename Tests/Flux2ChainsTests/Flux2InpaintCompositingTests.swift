// Flux2InpaintCompositingTests.swift — crop-and-stitch + pixel composite helpers
// Copyright 2025 Vincent Gourbin

import XCTest
import CoreGraphics
@testable import Flux2Chains
import Flux2Core

final class Flux2InpaintCompositingTests: XCTestCase {

    // MARK: - maskBoundingBox

    func testMaskBoundingBoxLocatesWhiteRegion() throws {
        // 200x100 mask, white rectangle at (60, 20)-(140, 60)
        let mask = try XCTUnwrap(Self.grayMask(width: 200, height: 100, whiteRect: CGRect(x: 60, y: 20, width: 80, height: 40)))
        let bbox = try XCTUnwrap(Flux2InpaintCompositing.maskBoundingBox(
            mask, convention: .grayscaleWhiteInpaint, imageWidth: 200, imageHeight: 100
        ))
        // Outward rounding tolerance: one scan pixel (mask fits in the scan cap, so ~1 px)
        XCTAssertLessThanOrEqual(abs(bbox.minX - 60), 3)
        XCTAssertLessThanOrEqual(abs(bbox.minY - 20), 3)
        XCTAssertLessThanOrEqual(abs(bbox.maxX - 140), 3)
        XCTAssertLessThanOrEqual(abs(bbox.maxY - 60), 3)
    }

    func testMaskBoundingBoxEmptyMaskReturnsNil() throws {
        let mask = try XCTUnwrap(Self.grayMask(width: 64, height: 64, whiteRect: nil))
        XCTAssertNil(Flux2InpaintCompositing.maskBoundingBox(
            mask, convention: .grayscaleWhiteInpaint, imageWidth: 64, imageHeight: 64
        ))
    }

    func testMaskBoundingBoxScalesToImageSpace() throws {
        // Mask at half the image resolution: bbox must come back in image coords
        let mask = try XCTUnwrap(Self.grayMask(width: 100, height: 50, whiteRect: CGRect(x: 30, y: 10, width: 40, height: 20)))
        let bbox = try XCTUnwrap(Flux2InpaintCompositing.maskBoundingBox(
            mask, convention: .grayscaleWhiteInpaint, imageWidth: 200, imageHeight: 100
        ))
        XCTAssertLessThanOrEqual(abs(bbox.minX - 60), 6)
        XCTAssertLessThanOrEqual(abs(bbox.maxX - 140), 6)
    }

    // MARK: - expandCropRegion

    func testExpandCropRegionMatchesImageAspect() {
        // Square bbox in a 2:1 image → region must be 2:1
        let region = Flux2InpaintCompositing.expandCropRegion(
            bbox: CGRect(x: 500, y: 200, width: 100, height: 100),
            padding: 32,
            imageWidth: 2000,
            imageHeight: 1000
        )
        let aspect = region.width / region.height
        XCTAssertEqual(Double(aspect), 2.0, accuracy: 0.05)
        // Contains the padded bbox
        XCTAssertLessThanOrEqual(region.minX, 500 - 32)
        XCTAssertGreaterThanOrEqual(region.maxX, 600 + 32)
        XCTAssertLessThanOrEqual(region.minY, 200 - 32)
        XCTAssertGreaterThanOrEqual(region.maxY, 300 + 32)
    }

    func testExpandCropRegionClampsToImageBounds() {
        // bbox in the top-left corner: region must stay inside the image
        let region = Flux2InpaintCompositing.expandCropRegion(
            bbox: CGRect(x: 5, y: 5, width: 50, height: 50),
            padding: 64,
            imageWidth: 800,
            imageHeight: 600
        )
        XCTAssertGreaterThanOrEqual(region.minX, 0)
        XCTAssertGreaterThanOrEqual(region.minY, 0)
        XCTAssertLessThanOrEqual(region.maxX, 800)
        XCTAssertLessThanOrEqual(region.maxY, 600)
        XCTAssertGreaterThan(region.width, 0)
        XCTAssertGreaterThan(region.height, 0)
    }

    func testExpandCropRegionHugeMaskCapsAtImageSize() {
        let region = Flux2InpaintCompositing.expandCropRegion(
            bbox: CGRect(x: 10, y: 10, width: 780, height: 580),
            padding: 64,
            imageWidth: 800,
            imageHeight: 600
        )
        XCTAssertEqual(Int(region.width), 800)
        XCTAssertEqual(Int(region.height), 600)
    }

    // MARK: - composite

    func testCompositeKeepsUnmaskedPixelsAndPastesMasked() throws {
        // Original: solid red 100x100. Generated: solid green 50x50 pasted at
        // (25, 25) with a hard white mask on its central 30x30.
        let original = try XCTUnwrap(Self.solidImage(width: 100, height: 100, r: 255, g: 0, b: 0))
        let generated = try XCTUnwrap(Self.solidImage(width: 50, height: 50, r: 0, g: 255, b: 0))
        let maskCrop = try XCTUnwrap(Self.grayMask(width: 50, height: 50, whiteRect: CGRect(x: 10, y: 10, width: 30, height: 30)))

        let out = try XCTUnwrap(Flux2InpaintCompositing.composite(
            original: original,
            generated: generated,
            cropRect: CGRect(x: 25, y: 25, width: 50, height: 50),
            maskCrop: maskCrop,
            convention: .grayscaleWhiteInpaint
        ))
        XCTAssertEqual(out.width, 100)
        XCTAssertEqual(out.height, 100)

        let px = try XCTUnwrap(Self.rgba(of: out))
        func pixel(_ x: Int, _ y: Int) -> (UInt8, UInt8, UInt8) {
            let i = (y * 100 + x) * 4
            return (px[i], px[i + 1], px[i + 2])
        }
        // Far outside the crop: untouched red
        XCTAssertEqual(pixel(5, 5).0, 255)
        XCTAssertEqual(pixel(5, 5).1, 0)
        // Inside crop but outside mask (crop-local (2,2) → global (27,27)): red
        XCTAssertGreaterThan(pixel(27, 27).0, 200)
        XCTAssertLessThan(pixel(27, 27).1, 50)
        // Centre of the masked region (crop-local (25,25) → global (50,50)): green
        XCTAssertLessThan(pixel(50, 50).0, 50)
        XCTAssertGreaterThan(pixel(50, 50).1, 200)
    }

    func testCompositeSoftMaskBlends() throws {
        let original = try XCTUnwrap(Self.solidImage(width: 40, height: 40, r: 0, g: 0, b: 0))
        let generated = try XCTUnwrap(Self.solidImage(width: 40, height: 40, r: 255, g: 255, b: 255))
        // Uniform 50% grey mask → every pixel blended ~50/50
        let maskCrop = try XCTUnwrap(Self.uniformGrayMask(width: 40, height: 40, value: 128))

        let out = try XCTUnwrap(Flux2InpaintCompositing.composite(
            original: original,
            generated: generated,
            cropRect: CGRect(x: 0, y: 0, width: 40, height: 40),
            maskCrop: maskCrop,
            convention: .grayscaleWhiteInpaint
        ))
        let px = try XCTUnwrap(Self.rgba(of: out))
        let center = (20 * 40 + 20) * 4
        XCTAssertEqual(Int(px[center]), 128, accuracy: 6)
    }

    func testCompositeRejectsOutOfBoundsCrop() throws {
        let original = try XCTUnwrap(Self.solidImage(width: 50, height: 50, r: 0, g: 0, b: 0))
        let generated = try XCTUnwrap(Self.solidImage(width: 20, height: 20, r: 255, g: 255, b: 255))
        let maskCrop = try XCTUnwrap(Self.uniformGrayMask(width: 20, height: 20, value: 255))
        XCTAssertNil(Flux2InpaintCompositing.composite(
            original: original,
            generated: generated,
            cropRect: CGRect(x: 40, y: 40, width: 20, height: 20),
            maskCrop: maskCrop,
            convention: .grayscaleWhiteInpaint
        ))
    }

    // MARK: - synthetic image helpers

    // Images are built from raw byte buffers in DEVICE colorspaces so that
    // drawing them into the helpers' device-colorspace contexts is a byte
    // identity — the tests can then assert exact values without being at the
    // mercy of sRGB→P3 / generic-gray gamma conversions.

    private static func solidImage(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> CGImage? {
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = r
            buf[i * 4 + 1] = g
            buf[i * 4 + 2] = b
            buf[i * 4 + 3] = 255
        }
        return rgbaImage(from: buf, width: width, height: height)
    }

    /// Grayscale mask from raw bytes, top-left origin coordinates.
    private static func grayMask(width: Int, height: Int, whiteRect: CGRect?) -> CGImage? {
        var buf = [UInt8](repeating: 0, count: width * height)
        if let rect = whiteRect {
            for y in Int(rect.minY)..<Int(rect.maxY) {
                for x in Int(rect.minX)..<Int(rect.maxX) {
                    buf[y * width + x] = 255
                }
            }
        }
        return grayImage(from: buf, width: width, height: height)
    }

    private static func uniformGrayMask(width: Int, height: Int, value: UInt8) -> CGImage? {
        grayImage(from: [UInt8](repeating: value, count: width * height), width: width, height: height)
    }

    private static func grayImage(from bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    private static func rgbaImage(from bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    private static func rgba(of image: CGImage) -> [UInt8]? {
        let w = image.width, h = image.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? buf : nil
    }
}
