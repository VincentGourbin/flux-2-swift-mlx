import CoreGraphics
import Flux2Core
import XCTest

final class AutoVLMContextAreaTests: XCTestCase {
    func testAutoVLMContextAreaPadsMaskBounds() {
        let rect = ImagePreparation.autoVLMContextArea(
            maskLayers: [
                InpaintMaskLayer(
                    combineMode: .add,
                    primitive: .rectangle(.init(CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2)))
                )
            ],
            processArea: nil
        )

        XCTAssertLessThan(rect.minX, 0.4)
        XCTAssertLessThan(rect.minY, 0.4)
        XCTAssertGreaterThan(rect.maxX, 0.6)
        XCTAssertGreaterThan(rect.maxY, 0.6)
        XCTAssertLessThanOrEqual(rect.maxX, 1)
        XCTAssertLessThanOrEqual(rect.maxY, 1)
    }

    func testVisionSubjectUsesSelectionBounds() {
        let rect = ImagePreparation.autoVLMContextArea(
            maskLayers: [
                InpaintMaskLayer(
                    combineMode: .add,
                    primitive: .visionSubject(.rectangle(.init(CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2))))
                )
            ],
            processArea: nil
        )

        XCTAssertLessThan(rect.minX, 0.4)
        XCTAssertLessThan(rect.minY, 0.4)
        XCTAssertGreaterThan(rect.maxX, 0.6)
        XCTAssertGreaterThan(rect.maxY, 0.6)
    }
}
