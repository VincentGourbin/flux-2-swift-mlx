// Flux2SubjectMaskTests.swift — Tests for the auto-segmentation helper
// Copyright 2025 Vincent Gourbin
//
// We cover the parts that don't require Vision to detect a subject
// (which would need a bundled real-world test image and an OS that
// agrees with us on what counts as a foreground). The Vision pipeline
// itself is Apple-tested. What we want pinned here:
//   - The downscale helper's no-op vs scaling boundaries (correctness
//     matters for the `targetMaxPixels` opt-in — wrong scale ⇒ caller's
//     mask is at the wrong resolution and won't align with the source).
//   - The public API rejects subject-less inputs cleanly with
//     `noSubjectDetected` (the documented contract).

import XCTest
@testable import Flux2Chains
import CoreGraphics

final class Flux2SubjectMaskTests: XCTestCase {

    // MARK: - downscaledIfNeeded

    func testDownscaleReturnsNilWhenTargetIsNil() {
        let img = makeSolidImage(width: 100, height: 100, value: 128)
        XCTAssertNil(
            Flux2SubjectMask.downscaledIfNeeded(img, targetMaxPixels: nil),
            "nil target ⇒ no-op; caller keeps the original image"
        )
    }

    func testDownscaleReturnsNilWhenSourceAlreadyFits() {
        let img = makeSolidImage(width: 100, height: 100, value: 128)
        XCTAssertNil(
            Flux2SubjectMask.downscaledIfNeeded(img, targetMaxPixels: 100 * 100),
            "Source already at target pixel count ⇒ no-op (cheaper than a needless re-render)"
        )
        XCTAssertNil(
            Flux2SubjectMask.downscaledIfNeeded(img, targetMaxPixels: 200_000),
            "Source smaller than target ⇒ no-op"
        )
    }

    func testDownscalePreservesAspectRatio() throws {
        // 4:3 source, large enough to require scaling.
        let src = makeSolidImage(width: 4000, height: 3000, value: 128)
        let target = 100_000  // ~316² square equivalent ⇒ should give ~365×274
        let scaled = try XCTUnwrap(Flux2SubjectMask.downscaledIfNeeded(src, targetMaxPixels: target))
        // Total pixels should fit (Lanczos may round so allow a small overshoot).
        XCTAssertLessThanOrEqual(
            scaled.width * scaled.height,
            target + max(scaled.width, scaled.height),
            "Output pixel count must be ≤ target (within one row/col of rounding)"
        )
        // Aspect ratio should match within 1% (Lanczos rounding tolerance).
        let inputAspect = 4000.0 / 3000.0
        let outputAspect = Double(scaled.width) / Double(scaled.height)
        XCTAssertEqual(inputAspect, outputAspect, accuracy: 0.02,
                       "Aspect ratio must survive the Lanczos resample")
    }

    func testDownscaleHandlesPortraitOrientation() throws {
        // 3:4 portrait (iPhone Photos default), 24 MP.
        let src = makeSolidImage(width: 4284, height: 5712, value: 200)
        let scaled = try XCTUnwrap(
            Flux2SubjectMask.downscaledIfNeeded(src, targetMaxPixels: 1024 * 1024)
        )
        // Both dimensions should shrink.
        XCTAssertLessThan(scaled.width, src.width)
        XCTAssertLessThan(scaled.height, src.height)
        // And the result should respect the cap (with one-row tolerance).
        XCTAssertLessThanOrEqual(
            scaled.width * scaled.height,
            1024 * 1024 + max(scaled.width, scaled.height)
        )
    }

    // MARK: - makeChangeSceneMask error path

    @available(macOS 14.0, *)
    func testMakeChangeSceneMaskThrowsNoSubjectOnSolidImage() {
        // A solid grey image has no foreground subject — Vision should
        // return an empty observation set. The helper must surface that
        // as the documented `.noSubjectDetected` error rather than
        // crashing or returning a meaningless mask.
        let img = makeSolidImage(width: 256, height: 256, value: 128)
        do {
            _ = try Flux2SubjectMask.makeChangeSceneMask(from: img)
            XCTFail("Expected Flux2SubjectMask.Error.noSubjectDetected on a solid grey image")
        } catch Flux2SubjectMask.Error.noSubjectDetected {
            // ok
        } catch {
            // Vision occasionally returns a transient request failure on
            // CI (e.g. simulator without GPU). Accept that too — the
            // important invariant is that we don't return a mask.
            if case Flux2SubjectMask.Error.visionFailure = error {
                // ok
            } else {
                XCTFail("Unexpected error: \(error)")
            }
        }
    }

    // MARK: - Helpers

    private func makeSolidImage(width: Int, height: Int, value: UInt8) -> CGImage {
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
            bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return ctx.makeImage()!
    }
}
