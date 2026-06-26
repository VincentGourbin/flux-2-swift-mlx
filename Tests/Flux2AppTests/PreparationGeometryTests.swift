import CoreGraphics
import Flux2Core
@testable import Flux2App
import XCTest

/// Pure-math coverage for the Image Preparation coordinate transforms. These are
/// the conversions that selection/overlay bugs hide in, so they get exercised in
/// isolation (no view, no view-model).
final class PreparationGeometryTests: XCTestCase {
    private func assertEqual(
        _ lhs: CGRect,
        _ rhs: CGRect,
        accuracy: CGFloat = 0.0001,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.minX, rhs.minX, accuracy: accuracy, "minX", file: file, line: line)
        XCTAssertEqual(lhs.minY, rhs.minY, accuracy: accuracy, "minY", file: file, line: line)
        XCTAssertEqual(lhs.width, rhs.width, accuracy: accuracy, "width", file: file, line: line)
        XCTAssertEqual(lhs.height, rhs.height, accuracy: accuracy, "height", file: file, line: line)
    }

    private func assertEqual(
        _ lhs: CGPoint,
        _ rhs: CGPoint,
        accuracy: CGFloat = 0.0001,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.x, rhs.x, accuracy: accuracy, "x", file: file, line: line)
        XCTAssertEqual(lhs.y, rhs.y, accuracy: accuracy, "y", file: file, line: line)
    }

    // MARK: - Aspect fit

    func testFittedImageRectLandscapeLetterboxesVertically() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        let rect = geo.fittedImageRect(in: CGSize(width: 400, height: 400))
        // 2:1 image in a square view -> full width, centered band.
        assertEqual(rect, CGRect(x: 0, y: 100, width: 400, height: 200))
    }

    func testFittedImageRectPortraitLetterboxesHorizontally() {
        let geo = PreparationGeometry(imageWidth: 500, imageHeight: 1000)
        let rect = geo.fittedImageRect(in: CGSize(width: 400, height: 400))
        assertEqual(rect, CGRect(x: 100, y: 0, width: 200, height: 400))
    }

    func testFittedImageRectZeroSizeIsZero() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        XCTAssertEqual(geo.fittedImageRect(in: .zero), .zero)
        XCTAssertEqual(PreparationGeometry(imageWidth: 0, imageHeight: 0)
            .fittedImageRect(in: CGSize(width: 100, height: 100)), .zero)
    }

    func testFittedCanvasRectAccountsForPadding() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        let padding = OutpaintPadding(top: 0, bottom: 0, left: 250, right: 250)
        // Canvas is 1500x500 (3:1) fitted into 600x600.
        let rect = geo.fittedCanvasRect(in: CGSize(width: 600, height: 600), padding: padding)
        assertEqual(rect, CGRect(x: 0, y: 200, width: 600, height: 200))
    }

    func testImageRectInsideCanvasOffsetsByPadding() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        let padding = OutpaintPadding(top: 0, bottom: 0, left: 250, right: 250)
        let canvas = geo.fittedCanvasRect(in: CGSize(width: 600, height: 600), padding: padding)
        let imageRect = geo.imageRectInsideCanvas(canvasRect: canvas, padding: padding)
        // 1000px image inside a 1500px canvas at 0.4 scale -> inset 100pt each side.
        assertEqual(imageRect, CGRect(x: 100, y: 200, width: 400, height: 200))
    }

    func testImagePixelScale() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        let imageRect = geo.fittedImageRect(in: CGSize(width: 400, height: 400))
        XCTAssertEqual(geo.imagePixelScale(imageRect: imageRect), 0.4, accuracy: 0.0001)
    }

    // MARK: - Normalized <-> view round-trips

    func testDisplayPointAndNormalizedPointRoundTrip() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        let imageRect = geo.fittedImageRect(in: CGSize(width: 400, height: 400))
        let normalized = CGPoint(x: 0.5, y: 0.5)
        let display = geo.displayPoint(normalized, in: imageRect)
        assertEqual(display, CGPoint(x: 200, y: 200))
        assertEqual(geo.normalizedPoint(for: display, in: imageRect), normalized)
    }

    func testNormalizedPointClampsOutsideImage() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        let imageRect = CGRect(x: 0, y: 100, width: 400, height: 200)
        let clamped = geo.normalizedPoint(for: CGPoint(x: -100, y: 9999), in: imageRect)
        assertEqual(clamped, CGPoint(x: 0, y: 1))
    }

    func testDisplayRectProjectsNormalizedRect() {
        // backingScale 1 keeps pixelAligned a no-op for whole-number rects.
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500, backingScale: 1)
        let imageRect = CGRect(x: 0, y: 0, width: 400, height: 200)
        let rect = geo.displayRect(for: CGRect(x: 0.25, y: 0.5, width: 0.5, height: 0.25), in: imageRect)
        assertEqual(rect, CGRect(x: 100, y: 100, width: 200, height: 50))
    }

    // MARK: - Pixel alignment

    func testPixelAlignedExpandsToWholeDevicePixels() {
        let geo = PreparationGeometry(imageWidth: 100, imageHeight: 100, backingScale: 1)
        let aligned = geo.pixelAligned(CGRect(x: 0.4, y: 0.6, width: 2.2, height: 2.2))
        // minX/minY floor to 0; maxX = ceil(2.6)=3; maxY = ceil(2.8)=3.
        assertEqual(aligned, CGRect(x: 0, y: 0, width: 3, height: 3))
    }

    func testPixelAlignedRetinaSnapsToHalfPoints() {
        let geo = PreparationGeometry(imageWidth: 100, imageHeight: 100, backingScale: 2)
        let aligned = geo.pixelAligned(CGRect(x: 0.1, y: 0.1, width: 10.2, height: 10.2))
        // maxX = ceil(10.3*2)/2 = 21/2 = 10.5.
        assertEqual(aligned, CGRect(x: 0, y: 0, width: 10.5, height: 10.5))
    }

    // MARK: - Snapping (16px grid in normalized space)

    func testSnapRoundsToSixteenPixelGrid() {
        let geo = PreparationGeometry(imageWidth: 160, imageHeight: 160)
        // step = 16/160 = 0.1
        XCTAssertEqual(geo.snap(0.44, pixels: 160), 0.4, accuracy: 0.0001)
        XCTAssertEqual(geo.snap(0.46, pixels: 160), 0.5, accuracy: 0.0001)
        XCTAssertEqual(geo.snapX(0.46), 0.5, accuracy: 0.0001)
        XCTAssertEqual(geo.snapY(0.44), 0.4, accuracy: 0.0001)
    }

    func testSnapClampsToUnitInterval() {
        let geo = PreparationGeometry(imageWidth: 160, imageHeight: 160)
        XCTAssertEqual(geo.snap(-5, pixels: 160), 0, accuracy: 0.0001)
        XCTAssertEqual(geo.snap(5, pixels: 160), 1, accuracy: 0.0001)
    }

    func testMinimumContextStepIsSixteenPixels() {
        let geo = PreparationGeometry(imageWidth: 1000, imageHeight: 500)
        XCTAssertEqual(geo.minimumContextWidth, 16.0 / 1000.0, accuracy: 0.0001)
        XCTAssertEqual(geo.minimumContextHeight, 16.0 / 500.0, accuracy: 0.0001)
    }

    func testMinimumContextClampsTinyImages() {
        // Images under 16px wide should not produce a step > 1.
        let geo = PreparationGeometry(imageWidth: 8, imageHeight: 8)
        XCTAssertEqual(geo.minimumContextWidth, 1, accuracy: 0.0001)
        XCTAssertEqual(geo.minimumContextHeight, 1, accuracy: 0.0001)
    }
}
