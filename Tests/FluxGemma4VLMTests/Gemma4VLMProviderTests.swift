// Gemma4VLMProviderTests.swift — Gemma 4 provider: pure-Swift surface + gated load
// Copyright 2025 Vincent Gourbin
//
// Everything here except the last suite runs without weights:
// - variant ↔ CLI spelling ↔ Gemma4Swift model mapping (what --gemma4-variant
//   resolves to, and the 6-bit default);
// - the multi-image batching, which is the one piece of real logic this target
//   adds: two differently-sized images must come out as one [2, 3, H, W] batch,
//   zero-padded top-left, or `chatStreamMultimodal` would reject the
//   marker/batch mismatch (or worse, scatter the wrong embeddings);
// - the reasoning-channel stripping.
//
// The load test is gated on FLUX2_GEMMA4_DIR (a local Gemma 4 weights
// directory), like ltx's Gemma4EnhancerProbeTests: loading is the whole test.

import XCTest
import CoreGraphics
import MLX
import FluxTextEncoders
@testable import FluxGemma4VLM

final class Gemma4VariantTests: XCTestCase {

    func testDefaultVariantIsSixBitE2B() {
        let provider = Gemma4VLMProvider()
        XCTAssertEqual(provider.model, .e2b6bit)
        XCTAssertTrue(provider.displayName.contains("E2B"))
        XCTAssertNil(provider.modelPath)
    }

    func testCLINamesRoundTrip() {
        for variant in Gemma4VLMProvider.Variant.allCases {
            XCTAssertEqual(Gemma4VLMProvider.Variant.fromCLIName(variant.cliName), variant)
        }
        XCTAssertEqual(Gemma4VLMProvider.Variant.fromCLIName("6bit"), .e2b6bit)
        XCTAssertEqual(Gemma4VLMProvider.Variant.fromCLIName("BF16"), .e2bBf16, "Parsing must be case-insensitive.")
        XCTAssertNil(Gemma4VLMProvider.Variant.fromCLIName("3bit"))
    }

    func testVariantsMapToE2BRepos() {
        XCTAssertEqual(Gemma4VLMProvider.Variant.e2b4bit.model.rawValue, "mlx-community/gemma-4-e2b-it-4bit")
        XCTAssertEqual(Gemma4VLMProvider.Variant.e2b6bit.model.rawValue, "mlx-community/gemma-4-e2b-it-6bit")
        XCTAssertEqual(Gemma4VLMProvider.Variant.e2b8bit.model.rawValue, "mlx-community/gemma-4-e2b-it-8bit")
        XCTAssertEqual(Gemma4VLMProvider.Variant.e2bBf16.model.rawValue, "mlx-community/gemma-4-e2b-it-bf16")
    }

    func testProviderIsUnloadedBeforeEnsureLoaded() {
        XCTAssertFalse(Gemma4VLMProvider().isLoaded)
    }

    func testRegisterDoesNotLoad() {
        let provider = FluxGemma4VLM.register(variant: .e2b4bit)
        XCTAssertTrue(FluxVLM.active === provider, "register() must take the seat immediately…")
        XCTAssertFalse(provider.isLoaded, "…but must not touch the weights: training loads/unloads around each phase.")
        FluxVLM.register(nil)
        XCTAssertTrue(FluxVLM.active === Qwen35VLMProvider.shared)
    }
}

final class Gemma4ImageBatchingTests: XCTestCase {

    func testSingleImagePassesThroughUnbatched() throws {
        let batch = try Gemma4VLMProvider.batchedPixelValues([Self.image(width: 96, height: 96)])
        XCTAssertEqual(batch.dim(0), 1)
        XCTAssertEqual(batch.dim(1), 3, "Channel-first [N, 3, H, W] is what the vision tower expects.")
    }

    func testTwoDifferentlySizedImagesArePaddedIntoOneBatch() throws {
        // A 1024² reference scored against a 512×768 render is the real case:
        // the processor sizes each to its own aspect-ratio box, so the shapes
        // differ before batching.
        let batch = try Gemma4VLMProvider.batchedPixelValues([
            Self.image(width: 480, height: 480),
            Self.image(width: 240, height: 480),
        ])
        XCTAssertEqual(batch.dim(0), 2, "Both images must land on the batch axis — one marker each.")
        XCTAssertEqual(batch.dim(1), 3)
        XCTAssertGreaterThan(batch.dim(2), 0)
        XCTAssertGreaterThan(batch.dim(3), 0)
    }

    func testPaddingIsZeroAndTopLeftAligned() throws {
        let wide = Self.image(width: 480, height: 240)
        let tall = Self.image(width: 240, height: 480)
        let batch = try Gemma4VLMProvider.batchedPixelValues([wide, tall])
        let h = batch.dim(2), w = batch.dim(3)

        // Each source is smaller than the union box in exactly one axis, so the
        // far corner of at least one slot must be padding.
        let cornerA = batch[0, 0, h - 1, w - 1].item(Float.self)
        let cornerB = batch[1, 0, h - 1, w - 1].item(Float.self)
        XCTAssertTrue(cornerA == 0 || cornerB == 0, "Padding must be zeros, not garbage.")

        // Top-left is real image content — a solid mid-blue fill, so non-zero.
        XCTAssertGreaterThan(batch[0, 2, 0, 0].item(Float.self), 0,
                             "Images must be top-left aligned inside the padded canvas.")
    }

    private static func image(width: Int, height: Int) -> CGImage {
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.setFillColor(CGColor(red: 0.25, green: 0.5, blue: 0.75, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()!
    }
}

final class Gemma4OutputCleanupTests: XCTestCase {

    func testKeepsOnlyTheAnswerAfterTheThoughtChannel() {
        let raw = "<|channel>thought The user wants a duck. Asphalt is grey.<channel|>A mallard duck stands on weathered grey asphalt."
        XCTAssertEqual(
            Gemma4VLMProvider.cleanGeneratedText(raw),
            "A mallard duck stands on weathered grey asphalt."
        )
    }

    func testStripsControlTokensAndTrims() {
        let raw = "  A mallard duck on asphalt.<end_of_turn>\n<eos>  "
        XCTAssertEqual(Gemma4VLMProvider.cleanGeneratedText(raw), "A mallard duck on asphalt.")
    }

    func testLeavesPlainAnswersUntouched() {
        let raw = "A mallard duck stands on weathered grey asphalt, 50mm, soft side light."
        XCTAssertEqual(Gemma4VLMProvider.cleanGeneratedText(raw), raw)
    }

    func testKeepsTextAfterTheLastChannelMarker() {
        // Two reasoning segments: only what follows the final close marker is the answer.
        let raw = "<|channel>thought a<channel|>draft<|channel>thought b<channel|>final answer"
        XCTAssertEqual(Gemma4VLMProvider.cleanGeneratedText(raw), "final answer")
    }
}

/// Loading is the whole test — same shape as ltx's enhancer probe.
///
/// ```
/// FLUX2_GEMMA4_DIR=~/Pictures/FluxforgeStudio/Models/mlx-community/gemma-4-e2b-it-6bit \
///   xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS' \
///     -skipPackagePluginValidation -only-testing:FluxGemma4VLMTests
/// ```
final class Gemma4LoadProbeTests: XCTestCase {

    func testLoadDescribeUnload() async throws {
        guard let dir = ProcessInfo.processInfo.environment["FLUX2_GEMMA4_DIR"] else {
            throw XCTSkip("Set FLUX2_GEMMA4_DIR to a local Gemma 4 weights directory to run this.")
        }
        let provider = Gemma4VLMProvider(modelPath: URL(fileURLWithPath: dir))
        try await provider.ensureLoaded()
        XCTAssertTrue(provider.isLoaded)

        let text = try await provider.analyzeImage(
            image: Self.image(),
            prompt: "What colour dominates this image? Answer in one word.",
            maxTokens: 16
        )
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        await provider.unload()
        XCTAssertFalse(provider.isLoaded)
    }

    private static func image() -> CGImage {
        let ctx = CGContext(
            data: nil, width: 224, height: 224,
            bitsPerComponent: 8, bytesPerRow: 224 * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.setFillColor(CGColor(red: 0.1, green: 0.2, blue: 0.9, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 224, height: 224))
        return ctx.makeImage()!
    }
}
