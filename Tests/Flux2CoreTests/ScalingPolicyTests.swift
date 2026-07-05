import CoreGraphics
@testable import Flux2Core
import XCTest

/// Behavioral coverage for the consolidated sizing policy. These pin the
/// invariant + the budget/snap/latent math so the follow-up dedup pass (folding
/// the hand-rolled 16/32 copies into the policy) can't drift.
final class ScalingPolicyTests: XCTestCase {
    private func makeImage(width: Int, height: Int) -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return context.makeImage()!
    }

    func testValidAlignmentInvariant() {
        XCTAssertTrue(ScalingPolicy.isValidAlignment(32, latentFactor: 16))
        XCTAssertTrue(ScalingPolicy.isValidAlignment(16, latentFactor: 16))
        XCTAssertTrue(ScalingPolicy.isValidAlignment(64, latentFactor: 16))
        XCTAssertFalse(ScalingPolicy.isValidAlignment(24, latentFactor: 16))
        XCTAssertFalse(ScalingPolicy.isValidAlignment(0, latentFactor: 16))
        XCTAssertFalse(ScalingPolicy.isValidAlignment(32, latentFactor: 0))
    }

    func testDefaultsMatchImagePreparation() {
        let policy = ScalingPolicy()
        XCTAssertEqual(policy.alignment, ImagePreparation.generationSizeMultiple)
        XCTAssertEqual(policy.latentFactor, 16)
    }

    func testBudgetPixelsClampMatchesImagePreparation() {
        let policy = ScalingPolicy()
        XCTAssertEqual(policy.budgetPixels(megapixelBudget: 1.0), 1_000_000)
        XCTAssertEqual(policy.budgetPixels(megapixelBudget: 0.1), 250_000)    // clamps up to 0.25 MP
        XCTAssertEqual(policy.budgetPixels(megapixelBudget: 8.0), 4_000_000)  // clamps down to 4.0 MP
        XCTAssertEqual(
            policy.budgetPixels(megapixelBudget: 2.5),
            ImagePreparation.conditioningPixelBudget(for: 2.5)
        )
    }

    func testSnapUpAndDown() {
        let policy = ScalingPolicy(alignment: 32)
        XCTAssertEqual(policy.snapUp(32), 32)
        XCTAssertEqual(policy.snapUp(33), 64)
        XCTAssertEqual(policy.snapUp(1), 32)     // floors up to the minimum alignment
        XCTAssertEqual(policy.snapDown(64), 64)
        XCTAssertEqual(policy.snapDown(63), 32)
        XCTAssertEqual(policy.snapDown(10), 32)  // never below the minimum alignment
    }

    func testLatentDimensions() {
        let policy = ScalingPolicy()
        let dims = policy.latentDimensions(width: 512, height: 512)
        XCTAssertEqual(dims.latentW, 64)            // 512 / 8
        XCTAssertEqual(dims.latentH, 64)
        XCTAssertEqual(dims.numPatches, 32 * 32)    // (64 / 2)^2
    }

    func testTargetSizeIsAlignedWithinBudgetAndMatchesDirectCall() {
        let policy = ScalingPolicy(alignment: 32)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        let image = makeImage(width: 2048, height: 1024)

        let size = policy.targetSize(for: image, settings: settings)
        XCTAssertEqual(size.width % 32, 0)
        XCTAssertEqual(size.height % 32, 0)
        XCTAssertLessThanOrEqual(size.width * size.height, 1_000_000 + 32 * 32)

        var direct = settings
        direct.pixelAlignment = 32
        let expected = ImagePreparation.generationSize(referenceImage: image, settings: direct)
        XCTAssertEqual(size.width, expected.width)
        XCTAssertEqual(size.height, expected.height)
    }

    func testTargetSizeOverridesSettingsAlignment() {
        // The policy's alignment wins even if the settings disagree.
        let policy = ScalingPolicy(alignment: 32)
        var settings = ImagePreparationSettings()
        settings.megapixelBudget = 1.0
        settings.pixelAlignment = 16
        let image = makeImage(width: 999, height: 777)

        let size = policy.targetSize(for: image, settings: settings)
        XCTAssertEqual(size.width % 32, 0)
        XCTAssertEqual(size.height % 32, 0)
    }
}
