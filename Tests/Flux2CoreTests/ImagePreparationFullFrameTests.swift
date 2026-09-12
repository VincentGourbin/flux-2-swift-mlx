// ImagePreparationFullFrameTests.swift - Full-frame composite skip and floor primitive
// Copyright 2025 Vincent Gourbin
//
// Originally contributed by Bill Evans (@realnotsteve) in PR #99.

import CoreGraphics
@testable import Flux2Core
import XCTest

/// Pins the behavior the clean/size/invent pipeline relies on: a full-frame
/// edit outputs at the budget size instead of being down-sampled back to the
/// source.
final class ImagePreparationFullFrameTests: XCTestCase {
    private func makeImage(width: Int, height: Int) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return context.makeImage()!
    }

    func testFloorToMultiple() {
        XCTAssertEqual(ImagePreparation.floorToMultiple(64, multiple: 32), 64)
        XCTAssertEqual(ImagePreparation.floorToMultiple(63, multiple: 32), 32)
        XCTAssertEqual(ImagePreparation.floorToMultiple(95, multiple: 32), 64)
        XCTAssertEqual(ImagePreparation.floorToMultiple(10, multiple: 32), 32)   // never below the multiple
        XCTAssertEqual(ImagePreparation.floorToMultiple(0, multiple: 32), 32)
        XCTAssertEqual(ImagePreparation.floorToMultiple(100, multiple: 0), 100)  // guard: non-positive multiple
    }

    func testIsFullFramePredicate() {
        let image = makeImage(width: 200, height: 100)
        XCTAssertTrue(ImagePreparation.isFullFrame(
            processRect: CGRect(x: 0, y: 0, width: 200, height: 100), original: image))
        XCTAssertFalse(ImagePreparation.isFullFrame(
            processRect: CGRect(x: 10, y: 0, width: 190, height: 100), original: image))
        XCTAssertFalse(ImagePreparation.isFullFrame(
            processRect: CGRect(x: 0, y: 0, width: 100, height: 100), original: image))
    }

    func testFullFrameEditSkipsCompositePlan() throws {
        let image = makeImage(width: 1500, height: 1000)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        settings.compositeBack = true  // full contextArea + nil processArea => full-frame

        let result = try ImagePreparation.prepare(referenceImages: [image], settings: settings)
        XCTAssertNil(result.compositionPlan, "Full-frame edit must skip composite-back (Fix 2)")
        // Output lands at the budget size, not the source size.
        XCTAssertEqual(result.width % 32, 0)
        XCTAssertEqual(result.height % 32, 0)
        XCTAssertLessThanOrEqual(result.width * result.height, 1_000_000 + 32 * 32)
        XCTAssertLessThan(result.width, image.width)  // 1500x1000 source enlarges/shrinks to ~1MP budget
    }

    func testPartialEditKeepsCompositePlan() throws {
        let image = makeImage(width: 1500, height: 1000)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        settings.compositeBack = true
        settings.processArea = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

        let result = try ImagePreparation.prepare(referenceImages: [image], settings: settings)
        XCTAssertNotNil(result.compositionPlan, "Partial edit must composite back into the original")
    }

    func testFullFrameWithCompositeBackOffStaysNil() throws {
        let image = makeImage(width: 800, height: 800)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        settings.compositeBack = false

        let result = try ImagePreparation.prepare(referenceImages: [image], settings: settings)
        XCTAssertNil(result.compositionPlan)
    }

    func testClampUnitRectReplacesNaNAxisWithFullFrame() {
        let withNaN = CGRect(x: CGFloat.nan, y: 0.1, width: 0.8, height: 0.8)
        XCTAssertEqual(ImagePreparation.clampUnitRect(withNaN), CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    func testClampValuesReplacesNaNBudgetAndScaleWithDefaults() {
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = .nan
        settings.preparationScale = .nan
        settings.clampValues()
        XCTAssertEqual(settings.megapixelBudget, 1.0)
        XCTAssertEqual(settings.preparationScale, 1.0)
    }

    func testReferenceMatchedSizeRoundsRatherThanTruncates() {
        // width=1983, height=4, maxArea=1983 => scale = sqrt(1983/7932) = 0.5
        // exactly (both operands exactly representable), so the scaled width
        // is exactly 991.5 — a clean tie that lands in a *different* 32px
        // bucket depending on whether it's truncated or rounded:
        // truncate: Int(991.5) = 991 -> floorToMultiple = 960
        // round:    Int(991.5.rounded()) = 992 -> floorToMultiple = 992
        let (width, _) = ImagePreparation.referenceMatchedSize(
            width: 1983, height: 4, maxArea: 1983, multiple: 32)
        XCTAssertEqual(width, 992)
        XCTAssertNotEqual(width, 960, "still truncating instead of rounding")
    }

    func testPrepareThrowsWhenProcessAreaDoesNotOverlapLiveArea() throws {
        let image = makeImage(width: 1200, height: 900)
        var settings = ImagePreparationSettings()
        settings.contextArea = CGRect(x: 0, y: 0, width: 0.3, height: 0.3)   // Live Area: top-left corner
        settings.processArea = CGRect(x: 0.7, y: 0.7, width: 0.3, height: 0.3)  // does not overlap

        // Previously silently substituted the whole Live Area instead of
        // erroring — a typo'd, non-overlapping pair must be rejected, not
        // silently regenerate a different region than requested.
        XCTAssertThrowsError(try ImagePreparation.prepare(referenceImages: [image], settings: settings))
    }

    func testValidateComposableThrowsForEmptyIntersection() throws {
        let image = makeImage(width: 1200, height: 900)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        settings.processArea = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

        let prepared = try ImagePreparation.prepare(referenceImages: [image], settings: settings)
        let plan = try XCTUnwrap(prepared.compositionPlan)

        // A valid plan validates without throwing...
        XCTAssertNoThrow(try ImagePreparation.validateComposable(plan))

        // ...and this check runs before generation: it must reject a plan
        // whose process rect can't map onto its own canvas, without needing
        // a generated image at all (a completed multi-minute generation must
        // never be the first place this is discovered).
        let brokenPlan = ImageCompositionPlan(
            originalImage: plan.originalImage,
            contextRect: plan.contextRect,
            processRect: CGRect(x: 999_999, y: 999_999, width: 10, height: 10),
            transform: plan.transform
        )
        XCTAssertThrowsError(try ImagePreparation.validateComposable(brokenPlan))
    }

    func testCompositeRejectsGeneratedImageOfTheWrongSize() throws {
        let image = makeImage(width: 1200, height: 900)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        settings.processArea = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

        let prepared = try ImagePreparation.prepare(referenceImages: [image], settings: settings)
        let plan = try XCTUnwrap(prepared.compositionPlan)

        // A generated image of the wrong size must not be silently
        // mapped through geometry the plan wasn't built for.
        let wrongSize = makeImage(width: 64, height: 64)
        XCTAssertThrowsError(try ImagePreparation.composite(wrongSize, using: plan))
    }

    /// The public settings+image predicate must classify edits identically to the
    /// composite-back decision inside prepare() — the same resolved rects back both.
    func testPublicIsFullFramePredicateMatchesPrepareClassification() throws {
        let image = makeImage(width: 1200, height: 900)

        var full = ImagePreparationSettings()
        full.megapixelBudget = 1.0
        full.compositeBack = true
        XCTAssertTrue(ImagePreparation.isFullFrame(settings: full, image: image))
        XCTAssertNil(try ImagePreparation.prepare(referenceImages: [image], settings: full).compositionPlan)

        var partial = full
        partial.processArea = CGRect(x: 0.2, y: 0.2, width: 0.4, height: 0.4)
        XCTAssertFalse(ImagePreparation.isFullFrame(settings: partial, image: image))
        XCTAssertNotNil(try ImagePreparation.prepare(referenceImages: [image], settings: partial).compositionPlan)
    }
}
