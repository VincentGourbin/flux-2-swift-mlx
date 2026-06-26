import CoreGraphics
import Flux2Core
import XCTest

final class InpaintMaskOutlineTests: XCTestCase {
    func testRectangularMaskProducesFourCorners() throws {
        let mask = try ImageMaskBuilder.rectangularInpaintMask(
            width: 100,
            height: 80,
            normalizedRect: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        )
        let points = InpaintMaskOutline.normalizedBoundaryPoints(from: mask)
        XCTAssertGreaterThanOrEqual(points.count, 4)

        let xs = points.map(\.x)
        let ys = points.map(\.y)
        XCTAssertLessThan(xs.min() ?? 1, 0.3)
        XCTAssertGreaterThan(xs.max() ?? 0, 0.7)
        XCTAssertLessThan(ys.min() ?? 1, 0.3)
        XCTAssertGreaterThan(ys.max() ?? 0, 0.7)
    }
}
