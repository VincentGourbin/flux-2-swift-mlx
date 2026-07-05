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

    func testMinimumVLMContextAreaIsTighterThanAuto() {
        let layers = [
            InpaintMaskLayer(
                combineMode: .add,
                primitive: .rectangle(.init(CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2)))
            )
        ]
        let auto = ImagePreparation.autoVLMContextArea(maskLayers: layers, processArea: nil)
        let minimum = ImagePreparation.minimumVLMContextArea(maskLayers: layers, processArea: nil)

        XCTAssertGreaterThan(auto.width, minimum.width)
        XCTAssertGreaterThan(auto.height, minimum.height)
        XCTAssertGreaterThanOrEqual(minimum.width, 0.15)
        XCTAssertGreaterThanOrEqual(minimum.height, 0.15)
    }

    func testFillVLMContextAreaScaleEndpoints() {
        let layers = [
            InpaintMaskLayer(
                combineMode: .add,
                primitive: .rectangle(.init(CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.2)))
            )
        ]
        let auto = ImagePreparation.autoVLMContextArea(maskLayers: layers, processArea: nil)
        let minimum = ImagePreparation.minimumVLMContextArea(maskLayers: layers, processArea: nil)

        XCTAssertEqual(
            ImagePreparation.fillVLMContextArea(maskLayers: layers, processArea: nil, scale: 0),
            auto
        )
        XCTAssertEqual(
            ImagePreparation.fillVLMContextArea(maskLayers: layers, processArea: nil, scale: -1),
            minimum
        )
        let full = ImagePreparation.fillVLMContextArea(maskLayers: layers, processArea: nil, scale: 1)
        XCTAssertEqual(full.minX, 0, accuracy: 0.0001)
        XCTAssertEqual(full.minY, 0, accuracy: 0.0001)
        XCTAssertEqual(full.width, 1, accuracy: 0.0001)
        XCTAssertEqual(full.height, 1, accuracy: 0.0001)
    }
}
