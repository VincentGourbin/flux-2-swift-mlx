import Flux2Core
import Testing

@Test func outpaintPaddingSnapsToThirtyTwo() {
    let padding = OutpaintPadding(top: 1, bottom: 17, left: 31, right: 64).snapped()
    #expect(padding.top == 32)
    #expect(padding.bottom == 32)
    #expect(padding.left == 32)
    #expect(padding.right == 64)
}

@Test func outpaintPaddingClampsToMegapixelBudget() {
    let padding = OutpaintPadding(top: 512, bottom: 512, left: 0, right: 0)
        .clamped(sourceWidth: 1024, sourceHeight: 1024, maxPixels: 1_500_000)
    #expect(padding.totalPixels(sourceWidth: 1024, sourceHeight: 1024) <= 1_500_000)
    #expect(padding.hasExpansion)
}
