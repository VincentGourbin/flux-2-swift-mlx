// Flux2OutpaintingChainTests.swift — Unit tests for outpainting chain helpers
// Copyright 2025 Vincent Gourbin
//
// CPU-only, no pipeline / no GPU. Exercises the pure-Swift geometry helpers
// the chain uses to build its canvas and smart mask. End-to-end inference
// is covered separately (and gated by model availability).

import XCTest
@testable import Flux2Chains
import Flux2Core
import CoreGraphics

final class Flux2OutpaintingChainTests: XCTestCase {

    // MARK: - roundUpToMultipleOf32

    func testRoundUpZeroStaysZero() {
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(0), 0,
                       "0 padding must stay 0 so we don't accidentally add a strip on sides with no requested padding")
    }

    func testRoundUpExactMultipleUnchanged() {
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(32), 32)
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(64), 64)
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(480), 480)
    }

    func testRoundUpJustBelowMultipleSnapsUp() {
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(1), 32)
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(31), 32)
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(33), 64)
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(63), 64)
        XCTAssertEqual(Flux2OutpaintingChain.roundUpToMultipleOf32(479), 480)
    }

    func testCanvasDimensionsPadsWorkingSizeToMultipleOf32() {
        let dimensions = Flux2OutpaintingChain.canvasDimensions(
            sourceWidth: 65,
            sourceHeight: 33,
            top: 0,
            bottom: 32,
            left: 32,
            right: 0
        )
        XCTAssertEqual(dimensions.requestedWidth, 97)
        XCTAssertEqual(dimensions.requestedHeight, 65)
        XCTAssertEqual(dimensions.workingWidth, 128)
        XCTAssertEqual(dimensions.workingHeight, 96)
    }

    // MARK: - buildOutpaintCanvas

    func testCanvasMatchesRequestedDimensions() {
        let src = makeSolidImage(width: 64, height: 32, value: 200)
        let canvas = Flux2OutpaintingChain.buildOutpaintCanvas(
            sourceImage: src,
            canvasWidth: 96, canvasHeight: 64,
            offsetX: 16, offsetY: 16, noiseSeed: 42
        )
        XCTAssertNotNil(canvas)
        XCTAssertEqual(canvas?.width, 96)
        XCTAssertEqual(canvas?.height, 64)
    }

    func testCanvasPlacesSourceAtRequestedOffset() throws {
        // Source: 32×32 of value 200. Canvas: 96×96, source at (32, 32).
        // Sample one pixel deep in the source region — should be 200.
        // Sample one pixel deep in the noise strip — should NOT be 200.
        let src = makeSolidImage(width: 32, height: 32, value: 200)
        let canvas = try XCTUnwrap(Flux2OutpaintingChain.buildOutpaintCanvas(
            sourceImage: src,
            canvasWidth: 96, canvasHeight: 96,
            offsetX: 32, offsetY: 32, noiseSeed: 42
        ))
        let insideKeep = sampleR(canvas, x: 48, y: 48)
        let leftStrip = sampleR(canvas, x: 8, y: 48)
        XCTAssertEqual(insideKeep, 200, accuracy: 4,
                       "Centre of the keep region should reflect the pasted source")
        XCTAssertNotEqual(leftStrip, 200,
                          "Left strip should be Gaussian noise, not the source value")
    }

    func testCanvasNoiseDeterministicWithSeed() throws {
        // Same seed → identical noise pattern outside the keep region.
        let src = makeSolidImage(width: 32, height: 32, value: 200)
        let a = try XCTUnwrap(Flux2OutpaintingChain.buildOutpaintCanvas(
            sourceImage: src, canvasWidth: 96, canvasHeight: 96,
            offsetX: 32, offsetY: 32, noiseSeed: 7
        ))
        let b = try XCTUnwrap(Flux2OutpaintingChain.buildOutpaintCanvas(
            sourceImage: src, canvasWidth: 96, canvasHeight: 96,
            offsetX: 32, offsetY: 32, noiseSeed: 7
        ))
        XCTAssertEqual(sampleR(a, x: 8, y: 8), sampleR(b, x: 8, y: 8))
        XCTAssertEqual(sampleR(a, x: 8, y: 80), sampleR(b, x: 8, y: 80))
    }

    func testCanvasNoiseDiffersBetweenSeeds() throws {
        let src = makeSolidImage(width: 32, height: 32, value: 200)
        let a = try XCTUnwrap(Flux2OutpaintingChain.buildOutpaintCanvas(
            sourceImage: src, canvasWidth: 96, canvasHeight: 96,
            offsetX: 32, offsetY: 32, noiseSeed: 1
        ))
        let b = try XCTUnwrap(Flux2OutpaintingChain.buildOutpaintCanvas(
            sourceImage: src, canvasWidth: 96, canvasHeight: 96,
            offsetX: 32, offsetY: 32, noiseSeed: 2
        ))
        // Sample a handful of strip pixels; at least one must differ.
        var anyDiff = false
        for (x, y) in [(2, 2), (2, 50), (90, 2), (90, 90)] {
            if sampleR(a, x: x, y: y) != sampleR(b, x: x, y: y) { anyDiff = true; break }
        }
        XCTAssertTrue(anyDiff, "Different seeds must produce different strip noise")
    }

    func testCropToRequestedCanvasRemovesTemporaryRightBottomPadding() throws {
        let padded = makeSolidImage(width: 128, height: 96, value: 180)
        let cropped = try XCTUnwrap(Flux2OutpaintingChain.cropToRequestedCanvas(
            padded,
            width: 97,
            height: 65
        ))
        XCTAssertEqual(cropped.width, 97)
        XCTAssertEqual(cropped.height, 65)
    }

    // MARK: - buildSmartMask

    func testSmartMaskBasicGeometryHorizontal() throws {
        // 96×32 canvas, keep is 32×32 at offset (32, 0). Left/right strips only.
        let mask = try XCTUnwrap(Flux2OutpaintingChain.buildSmartMask(
            canvasWidth: 96, canvasHeight: 32,
            keepX: 32, keepY: 0, keepWidth: 32, keepHeight: 32,
            transitionPixels: 4
        ))
        let row = 16
        // Far inside the left strip — pure white.
        XCTAssertEqual(sampleGray(mask, x: 4, y: row), 255)
        // Deep inside keep (away from any ramp) — pure black.
        XCTAssertEqual(sampleGray(mask, x: 48, y: row), 0)
        // Far inside the right strip — pure white.
        XCTAssertEqual(sampleGray(mask, x: 92, y: row), 255)
    }

    func testSmartMaskHasRampOnlyOnSidesWithPadding() throws {
        // Keep flush against the left edge (offsetX = 0): NO ramp on the left
        // side of the keep, but YES a ramp on the right.
        let mask = try XCTUnwrap(Flux2OutpaintingChain.buildSmartMask(
            canvasWidth: 96, canvasHeight: 32,
            keepX: 0, keepY: 0, keepWidth: 64, keepHeight: 32,
            transitionPixels: 4
        ))
        // First column of the keep (x=0) should be pure black — no left strip
        // exists, so the keep runs flush to the canvas edge.
        XCTAssertEqual(sampleGray(mask, x: 0, y: 16), 0)
        // Right inner band (last column of keep, x=63) should be near-white
        // because the ramp climbs back up as it approaches the right strip.
        XCTAssertGreaterThan(sampleGray(mask, x: 63, y: 16), 128)
        // The right strip (x≥64) must be pure white.
        XCTAssertEqual(sampleGray(mask, x: 80, y: 16), 255)
    }

    func testSmartMaskRampIsMonotonicInsideKeep() throws {
        // Keep occupies x ∈ [32, 64). With transitionPixels=4 the ramp on the
        // left side runs over x ∈ [32, 36). Values must be monotonically
        // decreasing (white → black) as we step into the keep.
        let mask = try XCTUnwrap(Flux2OutpaintingChain.buildSmartMask(
            canvasWidth: 96, canvasHeight: 32,
            keepX: 32, keepY: 0, keepWidth: 32, keepHeight: 32,
            transitionPixels: 4
        ))
        let row = 16
        let v0 = sampleGray(mask, x: 32, y: row)
        let v1 = sampleGray(mask, x: 33, y: row)
        let v2 = sampleGray(mask, x: 34, y: row)
        let v3 = sampleGray(mask, x: 35, y: row)
        XCTAssertGreaterThan(v0, v1)
        XCTAssertGreaterThan(v1, v2)
        XCTAssertGreaterThan(v2, v3)
    }

    func testSmartMaskRampWidthCappedToHalfKeep() throws {
        // Very small keep (4 px wide) with very wide requested transition
        // (16 px): the implementation caps the band at `keep_side / 2` so the
        // two ramps meet at the keep centre without wrapping around or
        // bleeding into the surrounding strips.
        let mask = try XCTUnwrap(Flux2OutpaintingChain.buildSmartMask(
            canvasWidth: 96, canvasHeight: 32,
            keepX: 46, keepY: 0, keepWidth: 4, keepHeight: 32,
            transitionPixels: 16
        ))
        // Strip pixels immediately outside the keep are pure white — the cap
        // must prevent the ramp from leaking past the keep boundary.
        XCTAssertEqual(sampleGray(mask, x: 45, y: 16), 255,
                       "Cap must not let the ramp leak into the left strip")
        XCTAssertEqual(sampleGray(mask, x: 50, y: 16), 255,
                       "Cap must not let the ramp leak into the right strip")
        // Inside the (tiny) keep, the two ramps cover everything and meet
        // mid-way — the outermost keep columns should be very near 255
        // (just transitioning back into the strips) and the innermost
        // columns near 128 (mid-ramp).
        XCTAssertGreaterThan(sampleGray(mask, x: 46, y: 16), 200,
                             "Outer column of the keep is at the start of the ramp")
        XCTAssertLessThan(sampleGray(mask, x: 47, y: 16), 200,
                          "Inner column of the keep is past the start of the ramp")
    }

    func testSmartMaskVerticalOnlyHasOnlyVerticalRamps() throws {
        // Top-only padding: 64×64 canvas, keep is 64×32 sitting flush against
        // the bottom (offset (0, 32)). The bottom of the keep meets the
        // canvas edge, so hasBottom is false and no bottom ramp should run.
        let mask = try XCTUnwrap(Flux2OutpaintingChain.buildSmartMask(
            canvasWidth: 64, canvasHeight: 64,
            keepX: 0, keepY: 32, keepWidth: 64, keepHeight: 32,
            transitionPixels: 4
        ))
        // Top strip — white.
        XCTAssertEqual(sampleGray(mask, x: 32, y: 4), 255)
        // Top edge of the keep — ramp value, < 255 (ramp climbs down into black).
        XCTAssertLessThan(sampleGray(mask, x: 32, y: 33), 255)
        XCTAssertGreaterThan(sampleGray(mask, x: 32, y: 33), 0)
        // Deep in the keep — pure black.
        XCTAssertEqual(sampleGray(mask, x: 32, y: 48), 0)
        // Bottom edge of the keep meets the canvas edge → no ramp → black.
        XCTAssertEqual(sampleGray(mask, x: 32, y: 63), 0)
    }

    // MARK: - run() input validation

    func testRunRejectsNegativePadding() async {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = makeSolidImage(width: 32, height: 32, value: 100)
        let chain = Flux2OutpaintingChain(
            pipeline: pipeline, image: img,
            left: -16, prompt: "x"
        )
        do {
            _ = try await chain.run()
            XCTFail("Expected Flux2ChainError.invalidInput for negative padding")
        } catch let err as Flux2ChainError {
            if case .invalidInput = err { /* ok */ } else { XCTFail("Wrong error case: \(err)") }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testRunRejectsAllZeroPadding() async {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = makeSolidImage(width: 32, height: 32, value: 100)
        let chain = Flux2OutpaintingChain(pipeline: pipeline, image: img, prompt: "x")
        do {
            _ = try await chain.run()
            XCTFail("Expected Flux2ChainError.invalidInput when no side has padding")
        } catch let err as Flux2ChainError {
            if case .invalidInput = err { /* ok */ } else { XCTFail("Wrong error case: \(err)") }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Public surface sanity

    func testChainStoresInitArgumentsVerbatim() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = makeSolidImage(width: 32, height: 32, value: 50)
        let chain = Flux2OutpaintingChain(
            pipeline: pipeline, image: img,
            top: 16, bottom: 32, left: 48, right: 0,
            prompt: "hello",
            steps: 7, guidance: 2.5, seed: 99,
            upsamplePrompt: true,
            enrichPromptWithVLM: true,
            transitionPixels: 16, maxPixels: 500_000
        )
        XCTAssertEqual(chain.top, 16)
        XCTAssertEqual(chain.bottom, 32)
        XCTAssertEqual(chain.left, 48)
        XCTAssertEqual(chain.right, 0)
        XCTAssertEqual(chain.prompt, "hello")
        XCTAssertEqual(chain.steps, 7)
        XCTAssertEqual(chain.guidance, 2.5)
        XCTAssertEqual(chain.seed, 99)
        XCTAssertTrue(chain.upsamplePrompt)
        XCTAssertTrue(chain.enrichPromptWithVLM)
        XCTAssertEqual(chain.transitionPixels, 16)
        XCTAssertEqual(chain.maxPixels, 500_000)
    }

    func testUpsamplePromptDefaultsOff() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = makeSolidImage(width: 32, height: 32, value: 50)
        let chain = Flux2OutpaintingChain(pipeline: pipeline, image: img, prompt: "x")
        XCTAssertFalse(chain.upsamplePrompt,
                       "Off by default — caller's exact wording is preserved unless they opt in")
    }

    func testEnrichPromptWithVLMDefaultsOff() {
        let pipeline = Flux2Pipeline(model: .klein9B)
        let img = makeSolidImage(width: 32, height: 32, value: 50)
        let chain = Flux2OutpaintingChain(pipeline: pipeline, image: img, prompt: "x")
        XCTAssertFalse(chain.enrichPromptWithVLM,
                       "Off by default — VLM enrichment is strictly opt-in so the chain runs without the VLM loaded")
    }

    // MARK: - Helpers

    private func makeSolidImage(width: Int, height: Int, value: UInt8) -> CGImage {
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i] = value
            pixels[i + 1] = value
            pixels[i + 2] = value
            pixels[i + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return ctx.makeImage()!
    }

    private func sampleR(_ image: CGImage, x: Int, y: Int) -> UInt8 {
        let w = image.width, h = image.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pixels[(y * w + x) * 4]
    }

    private func sampleGray(_ image: CGImage, x: Int, y: Int) -> UInt8 {
        let w = image.width, h = image.height
        var pixels = [UInt8](repeating: 0, count: w * h)
        let cs = CGColorSpaceCreateDeviceGray()
        let ctx = CGContext(
            data: &pixels, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w, space: cs,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pixels[y * w + x]
    }
}
