import CoreGraphics
import Flux2Core
@testable import Flux2App
import XCTest

/// The app-layer `ScalingPolicy` bridge: per-family alignment plus the
/// full-frame budget convenience the NR-IQA quality probe consumes (so the
/// measurement never inherits a project's live-area / favour / scale).
final class ScalingPolicyModelFamilyTests: XCTestCase {
    private func makeImage(width: Int, height: Int) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return context.makeImage()!
    }

    func testFamilyPolicyUsesFamilyAlignment() {
        let policy = ScalingPolicy(family: .flux2)
        XCTAssertEqual(policy.alignment, ModelFamily.flux2.pixelAlignment)
    }

    func testBudgetConvenienceMatchesDefaultFullFrameSettings() {
        let image = makeImage(width: 2048, height: 1365)
        let budget = 1.0
        let alignment = ModelFamily.flux2.pixelAlignment

        let convenience = ScalingPolicy.targetSize(for: image, family: .flux2, megapixelBudget: budget)

        var settings = ImagePreparationSettings()
        settings.megapixelBudget = budget
        let viaSettings = ScalingPolicy.targetSize(for: image, family: .flux2, settings: settings)

        XCTAssertEqual(convenience.width, viaSettings.width)
        XCTAssertEqual(convenience.height, viaSettings.height)
        XCTAssertEqual(convenience.width % alignment, 0)
        XCTAssertEqual(convenience.height % alignment, 0)
        XCTAssertLessThanOrEqual(convenience.width * convenience.height, 1_000_000 + alignment * alignment)
    }

    func testBudgetConvenienceIgnoresLiveAreaFavourAndScale() {
        let image = makeImage(width: 2000, height: 1000)
        let probe = ScalingPolicy.targetSize(for: image, family: .flux2, megapixelBudget: 1.0)

        var skewed = ImagePreparationSettings()
        skewed.megapixelBudget = 1.0
        skewed.sizingFavor = .vertical
        skewed.preparationScale = 0.5
        skewed.contextArea = CGRect(x: 0.1, y: 0.1, width: 0.3, height: 0.3)
        let skewedSize = ScalingPolicy.targetSize(for: image, family: .flux2, settings: skewed)

        XCTAssertTrue(
            probe.width != skewedSize.width || probe.height != skewedSize.height,
            "Probe must not inherit a project's live-area / favour / scale"
        )
    }
}
