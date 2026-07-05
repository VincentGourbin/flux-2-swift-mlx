import CoreGraphics
import Flux2Core
import XCTest
@testable import Flux2Chains

final class VisionSubjectRegionMathTests: XCTestCase {
    func testExpandedRectGrowsFromCenter() {
        let region = VisionSubjectRegion.rectangle(CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2))
        guard case .rectangle(let expanded) = VisionSubjectRegionMath.expandedRegion(
            region,
            step: 1,
            stepFraction: 0.05
        ) else {
            return XCTFail("Expected rectangle region")
        }

        XCTAssertLessThan(expanded.minX, 0.4)
        XCTAssertLessThan(expanded.minY, 0.4)
        XCTAssertGreaterThan(expanded.width, 0.2)
        XCTAssertEqual(expanded.midX, 0.5, accuracy: 0.001)
        XCTAssertEqual(expanded.midY, 0.5, accuracy: 0.001)
    }

    func testOverlapFractionCountsSelectionCoverage() {
        let subject = [UInt8](repeating: 0, count: 100)
        var selection = [UInt8](repeating: 0, count: 100)
        selection[40] = 255
        selection[41] = 255
        selection[50] = 255

        var subjectWithOverlap = subject
        subjectWithOverlap[40] = 255
        subjectWithOverlap[50] = 255

        XCTAssertEqual(
            VisionSubjectRegionMath.overlapFraction(subjectMask: subject, selectionMask: selection),
            0,
            accuracy: 0.001
        )
        XCTAssertEqual(
            VisionSubjectRegionMath.overlapFraction(subjectMask: subjectWithOverlap, selectionMask: selection),
            2.0 / 3.0,
            accuracy: 0.001
        )
    }

    func testRasterizedSelectionMaskFillsRectangle() throws {
        let mask = VisionSubjectRegionMath.rasterizedSelectionMask(
            region: .rectangle(CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)),
            width: 4,
            height: 4
        )
        XCTAssertEqual(mask.filter { $0 > 0 }.count, 4)
    }
}
