import Flux2Core
@testable import Flux2App
import XCTest

final class EditHistoryRouteResolverTests: XCTestCase {
    func testPrefersStoredGenerateRoute() {
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1)),
            inpaintMaskLayers: [sampleMaskLayer()]
        )
        let settings = EditHistorySettings(
            selectedModel: "klein-4b",
            steps: 4,
            guidance: 1,
            generateRoute: I2IGenerateRoute.fullImage.rawValue
        )

        XCTAssertEqual(
            EditHistoryRouteResolver.route(settings: settings, spatial: spatial),
            .fullImage
        )
    }

    func testInfersOutpaintFromPadding() {
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1)),
            outpaintPadding: OutpaintPadding(top: 8, bottom: 8, left: 8, right: 8)
        )
        let settings = EditHistorySettings(selectedModel: "klein-4b", steps: 4, guidance: 1)

        XCTAssertEqual(
            EditHistoryRouteResolver.route(settings: settings, spatial: spatial),
            .outpaint
        )
    }

    func testInfersLocalFillFromMaskLayers() {
        let spatial = EditHistorySpatial(
            contextArea: FluxGenerationProject.NormalizedRect(CGRect(x: 0, y: 0, width: 1, height: 1)),
            inpaintMaskLayers: [sampleMaskLayer()]
        )
        let settings = EditHistorySettings(selectedModel: "klein-4b", steps: 4, guidance: 1)

        XCTAssertEqual(
            EditHistoryRouteResolver.route(settings: settings, spatial: spatial),
            .localFill
        )
    }

    private func sampleMaskLayer() -> InpaintMaskLayer {
        InpaintMaskLayer(
            combineMode: .add,
            primitive: .rectangle(.init(CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.2)))
        )
    }
}
