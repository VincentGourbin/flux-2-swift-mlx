import CoreGraphics
@testable import Flux2Core
import XCTest

/// Fix 2 (full-frame composite skip) and the shared floor primitive that backs
/// `ScalingPolicy.snapDown` plus the never-upsample reference render. These pin
/// the behavior the clean/size/invent pipeline relies on: a full-frame edit
/// outputs at the budget size instead of being down-sampled back to the source.
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

    func testFloorMatchesScalingPolicySnapDown() {
        let policy = ScalingPolicy(alignment: 32)
        for value in [1, 31, 32, 33, 64, 95, 1000] {
            XCTAssertEqual(policy.snapDown(value), ImagePreparation.floorToMultiple(value, multiple: 32))
        }
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
