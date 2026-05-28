// Flux2ChainsTests.swift — Unit tests for chain helpers
// Copyright 2025 Vincent Gourbin

import XCTest
@testable import Flux2Chains
import Flux2Core
import CoreGraphics

final class Flux2ChainHelpersTests: XCTestCase {

    // MARK: - Dimension resolution

    func testResolveDimensionsLeavesValidInputUntouched() {
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 512, height: 512)
        XCTAssertEqual(h, 512)
        XCTAssertEqual(w, 512)
    }

    func testResolveDimensionsClampsToMultipleOf32() {
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 1000, height: 600)
        XCTAssertEqual(w % 32, 0)
        XCTAssertEqual(h % 32, 0)
        XCTAssertLessThanOrEqual(w, 1000)
        XCTAssertLessThanOrEqual(h, 600)
    }

    func testResolveDimensionsScalesDownLargeInputs() {
        // 4K input should be scaled into ≤ 1024² area, multiple of 32.
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 3840, height: 2160)
        XCTAssertLessThanOrEqual(w * h, 1024 * 1024)
        XCTAssertEqual(w % 32, 0)
        XCTAssertEqual(h % 32, 0)
        // Aspect preserved within 5% (rounding allowed).
        let inputAspect = 3840.0 / 2160.0
        let outputAspect = Double(w) / Double(h)
        XCTAssertEqual(inputAspect, outputAspect, accuracy: 0.1)
    }

    func testResolveDimensionsRespectsCustomMaxPixels() {
        let (h, w) = Flux2Pipeline.resolveChainDimensions(width: 1024, height: 1024, maxPixels: 512 * 512)
        XCTAssertLessThanOrEqual(w * h, 512 * 512)
        XCTAssertEqual(w % 32, 0)
        XCTAssertEqual(h % 32, 0)
    }

    // MARK: - Mask packing

    func testPackMaskHasExpectedSequenceShape() async {
        // 512×512 image → 32×32 latent grid → 1024 tokens.
        let mask = makeUniformMask(width: 512, height: 512, value: 1.0)
        let packed = await Flux2Pipeline.packMaskForLatentBlending(mask, targetHeight: 512, targetWidth: 512)
        XCTAssertEqual(packed.shape, [1, 1024, 1])
    }

    func testPackMaskAllWhiteYieldsOnes() async {
        let mask = makeUniformMask(width: 128, height: 128, value: 1.0)
        let packed = await Flux2Pipeline.packMaskForLatentBlending(mask, targetHeight: 128, targetWidth: 128)
        // 128/16 = 8 → 64 tokens
        XCTAssertEqual(packed.shape, [1, 64, 1])
        // Sum should equal token count.
        let total = packed.sum().item(Float.self)
        XCTAssertEqual(total, 64.0, accuracy: 0.5)
    }

    func testPackMaskAllBlackYieldsZeros() async {
        let mask = makeUniformMask(width: 128, height: 128, value: 0.0)
        let packed = await Flux2Pipeline.packMaskForLatentBlending(mask, targetHeight: 128, targetWidth: 128)
        let total = packed.sum().item(Float.self)
        XCTAssertEqual(total, 0.0, accuracy: 0.5)
    }

    // MARK: - Mask packing (alpha convention)

    func testAlphaMaskFullyTransparentYieldsOnes() async {
        // Fully transparent input ⇒ everything should be inpainted (mask = 1).
        let mask = makeUniformAlphaMask(width: 128, height: 128, alpha: 0)
        let packed = await Flux2Pipeline.packMaskForLatentBlending(
            mask, targetHeight: 128, targetWidth: 128,
            convention: .alphaTransparentInpaint
        )
        let total = packed.sum().item(Float.self)
        XCTAssertEqual(total, 64.0, accuracy: 0.5,
                       "alpha=0 ⇒ inpaint=1.0 across all 64 tokens")
    }

    func testAlphaMaskFullyOpaqueYieldsZeros() async {
        // Fully opaque input ⇒ nothing inpainted (mask = 0 everywhere).
        let mask = makeUniformAlphaMask(width: 128, height: 128, alpha: 255)
        let packed = await Flux2Pipeline.packMaskForLatentBlending(
            mask, targetHeight: 128, targetWidth: 128,
            convention: .alphaTransparentInpaint
        )
        let total = packed.sum().item(Float.self)
        XCTAssertEqual(total, 0.0, accuracy: 0.5,
                       "alpha=255 ⇒ keep=0.0 across all tokens")
    }

    func testAlphaMaskSoftValueIsPreserved() async {
        // alpha=128 ⇒ ~0.5 soft mask.
        let mask = makeUniformAlphaMask(width: 128, height: 128, alpha: 128)
        let packed = await Flux2Pipeline.packMaskForLatentBlending(
            mask, targetHeight: 128, targetWidth: 128,
            convention: .alphaTransparentInpaint
        )
        let total = packed.sum().item(Float.self)
        // 1 - 128/255 ≈ 0.498; sum across 64 tokens ≈ 31.9.
        XCTAssertEqual(total, 64.0 * (1.0 - 128.0 / 255.0), accuracy: 1.0)
    }

    func testAlphaMaskIgnoresRGBContent() async {
        // Two masks with identical alpha but wildly different RGB must
        // produce identical packed mask arrays.
        let m0 = makeAlphaMask(width: 128, height: 128, alpha: 0, rgb: (0, 0, 0))
        let m1 = makeAlphaMask(width: 128, height: 128, alpha: 0, rgb: (255, 0, 255))
        let p0 = await Flux2Pipeline.packMaskForLatentBlending(
            m0, targetHeight: 128, targetWidth: 128, convention: .alphaTransparentInpaint
        )
        let p1 = await Flux2Pipeline.packMaskForLatentBlending(
            m1, targetHeight: 128, targetWidth: 128, convention: .alphaTransparentInpaint
        )
        XCTAssertEqual(p0.sum().item(Float.self), p1.sum().item(Float.self),
                       accuracy: 0.001,
                       "Alpha convention must ignore RGB — only the alpha plane decides")
    }

    // MARK: - Helpers

    private func makeUniformMask(width: Int, height: Int, value: Float) -> CGImage {
        let pixelValue: UInt8 = UInt8(max(0, min(255, Int(value * 255))))
        let pixelCount = width * height
        var pixels = [UInt8](repeating: pixelValue, count: pixelCount)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!
        return context.makeImage()!
    }

    private func makeUniformAlphaMask(width: Int, height: Int, alpha: UInt8) -> CGImage {
        makeAlphaMask(width: width, height: height, alpha: alpha, rgb: (0, 0, 0))
    }

    private func makeAlphaMask(width: Int, height: Int, alpha: UInt8, rgb: (UInt8, UInt8, UInt8)) -> CGImage {
        // RGBA, premultipliedLast — premultiplied means RGB stored as
        // RGB · alpha. We emit (r * a / 255, g * a / 255, b * a / 255, a)
        // so the source data is self-consistent and Core Graphics reads it
        // back as the intended alpha.
        let (r, g, b) = rgb
        let pa = Int(alpha)
        let pr = UInt8((Int(r) * pa) / 255)
        let pg = UInt8((Int(g) * pa) / 255)
        let pb = UInt8((Int(b) * pa) / 255)
        var pixels = [UInt8]()
        pixels.reserveCapacity(width * height * 4)
        for _ in 0..<(width * height) {
            pixels.append(pr); pixels.append(pg); pixels.append(pb); pixels.append(alpha)
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return context.makeImage()!
    }
}
