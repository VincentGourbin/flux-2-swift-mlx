// Flux2MaskedInpaintingChainTests.swift — Public-surface tests for the chain
// Copyright 2025 Vincent Gourbin
//
// The interesting behaviour of this chain (RePaint blend in the denoising
// loop, T2I vs I2I mode switch) is only observable through a loaded pipeline,
// which we don't want to require here. These tests cover the bits that are
// observable at construction time: defaults, init forwarding, and the
// reference-image presence used to gate the I2I switch.

import XCTest
@testable import Flux2Chains
import Flux2Core
import CoreGraphics

final class Flux2MaskedInpaintingChainTests: XCTestCase {

    func testDefaultsTargetKleinDistilled() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = solid(width: 32, height: 32)
        let chain = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: "x",
            image: img,
            mask: img
        )
        // Should match klein-9B distilled inference (CFG-distilled).
        XCTAssertEqual(chain.steps, 4)
        XCTAssertEqual(chain.guidance, 1.0)
        XCTAssertEqual(chain.maxPixels, 1024 * 1024)
        XCTAssertNil(chain.seed)
        XCTAssertNil(chain.referenceImages,
                     "Reference images default to nil so the chain picks the .textToImage mode")
    }

    func testInitForwardsExplicitArguments() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = solid(width: 32, height: 32)
        let refA = solid(width: 32, height: 32, value: 1)
        let refB = solid(width: 32, height: 32, value: 2)
        let chain = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: "hello",
            image: img,
            mask: img,
            referenceImages: [refA, refB],
            steps: 25,
            guidance: 3.5,
            seed: 1234,
            maxPixels: 2_400_000
        )
        XCTAssertEqual(chain.prompt, "hello")
        XCTAssertEqual(chain.steps, 25)
        XCTAssertEqual(chain.guidance, 3.5)
        XCTAssertEqual(chain.seed, 1234)
        XCTAssertEqual(chain.maxPixels, 2_400_000)
        XCTAssertEqual(chain.referenceImages?.count, 2)
    }

    func testMaskConventionDefaultsToGrayscale() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = solid(width: 32, height: 32)
        let chain = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: "x",
            image: img,
            mask: img
        )
        XCTAssertEqual(chain.maskConvention, .grayscaleWhiteInpaint,
                       "Back-compat: existing callers keep grayscale luminance reading")
    }

    func testMaskConventionForwardsAlpha() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = solid(width: 32, height: 32)
        let chain = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: "x",
            image: img,
            mask: img,
            maskConvention: .alphaTransparentInpaint
        )
        XCTAssertEqual(chain.maskConvention, .alphaTransparentInpaint)
    }

    func testEmptyReferenceImagesArrayBehavesLikeNil() async {
        // The chain treats nil and [] the same way for the mode switch: both
        // mean "no reference, run as T2I". We surface this through the
        // public property — the actual switch happens inside `run()`.
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = solid(width: 32, height: 32)
        let chain = Flux2MaskedInpaintingChain(
            pipeline: pipeline,
            prompt: "x",
            image: img,
            mask: img,
            referenceImages: []
        )
        XCTAssertEqual(chain.referenceImages?.count, 0,
                       "Empty arrays survive init verbatim — the chain's run() decides what to do with them")
    }

    // MARK: - Helpers

    private func solid(width: Int, height: Int, value: UInt8 = 128) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value; pixels[i + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return ctx.makeImage()!
    }
}
