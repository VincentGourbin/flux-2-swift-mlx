// FluxVLMProviderTests.swift — Provider registry contract
// Copyright 2025 Vincent Gourbin
//
// No weights are involved: what matters here is the contract every enrichment
// call site depends on — `FluxVLM.active` is the bundled Qwen3.5 unless someone
// registered something else, and registering never loads or unloads anything.

import XCTest
import CoreGraphics
@testable import FluxTextEncoders

/// Records what it was asked, answers a canned string. Stands in for a real
/// provider so the registry and the shared helpers can be tested without a VLM.
private final class StubVLMProvider: FluxVLMProvider, @unchecked Sendable {
    let displayName = "stub-vlm"
    var isLoaded: Bool = true
    var ensureLoadedCalls = 0
    var unloadCalls = 0
    var lastImageCount: Int?
    var lastSystemPrompt: String?
    var lastMaxTokens: Int?
    var lastTemperature: Float?
    var answer: String

    init(answer: String = "stub answer") { self.answer = answer }

    func ensureLoaded() async throws { ensureLoadedCalls += 1 }
    func unload() async { unloadCalls += 1 }

    func generateText(
        images: [CGImage],
        prompt: String,
        systemPrompt: String?,
        enableThinking: Bool,
        maxTokens: Int,
        temperature: Float
    ) async throws -> String {
        lastImageCount = images.count
        lastSystemPrompt = systemPrompt
        lastMaxTokens = maxTokens
        lastTemperature = temperature
        return answer
    }
}

final class FluxVLMProviderTests: XCTestCase {

    override func tearDown() {
        // Global state: never leak a stub into another test.
        FluxVLM.register(nil)
        super.tearDown()
    }

    // MARK: - Registry

    func testDefaultProviderIsBundledQwen35() {
        XCTAssertFalse(FluxVLM.hasCustomProvider)
        XCTAssertTrue(FluxVLM.active === Qwen35VLMProvider.shared,
                      "With nothing registered, enrichment must keep running on the bundled Qwen3.5.")
        XCTAssertTrue(FluxVLM.active.displayName.contains("Qwen3.5"))
    }

    func testRegisteredProviderTakesOver() {
        let stub = StubVLMProvider()
        FluxVLM.register(stub)
        XCTAssertTrue(FluxVLM.hasCustomProvider)
        XCTAssertTrue(FluxVLM.active === stub)
    }

    func testRegisteringNilRestoresQwen35() {
        FluxVLM.register(StubVLMProvider())
        FluxVLM.register(nil)
        XCTAssertFalse(FluxVLM.hasCustomProvider)
        XCTAssertTrue(FluxVLM.active === Qwen35VLMProvider.shared)
    }

    func testQwen35ProviderReportsUnloadedInTestEnvironment() {
        // The singleton is never loaded in tests; the enrichment paths rely on
        // this being observable rather than throwing.
        XCTAssertFalse(Qwen35VLMProvider.shared.isLoaded)
    }

    func testQwen35ProviderVariantIsFourBitByDefault() {
        XCTAssertEqual(Qwen35VLMProvider.shared.variant, .qwen35_4B_4bit,
                       "Training and the chains have always downloaded the 4-bit variant.")
        XCTAssertEqual(Qwen35VLMProvider(variant: .qwen35_4B_8bit).variant, .qwen35_4B_8bit)
    }

    // MARK: - Shared helpers route through the active provider

    func testDescribeImageForFluxUsesSharedSystemPrompt() async throws {
        let stub = StubVLMProvider(answer: "a duck on asphalt, soft side light")
        let text = try await stub.describeImageForFlux(image: Self.solidImage(), context: "focus on the face")
        XCTAssertEqual(text, "a duck on asphalt, soft side light")
        XCTAssertEqual(stub.lastImageCount, 1)
        XCTAssertEqual(stub.lastSystemPrompt, FluxTextEncoders.fluxImageDescriptionSystemPrompt)
        XCTAssertEqual(stub.lastTemperature, 0, "Enrichment must stay greedy/deterministic.")
    }

    func testCompareImagesForFluxSendsTwoImagesAndParsesScores() async throws {
        let stub = StubVLMProvider(answer: """
        {"scene_score": 72, "scene_reason": "same subject, different pose", "style_score": 41, "style_reason": "flat vector vs 3D"}
        """)
        let comparison = try await stub.compareImagesForFlux(
            reference: Self.solidImage(), generated: Self.solidImage()
        )
        XCTAssertEqual(stub.lastImageCount, 2)
        XCTAssertEqual(stub.lastSystemPrompt, FluxTextEncoders.fluxImageComparisonSystemPrompt)
        XCTAssertEqual(comparison.sceneScore, 72)
        XCTAssertEqual(comparison.styleScore, 41)
        XCTAssertEqual(comparison.sceneReason, "same subject, different pose")
    }

    func testCompareImagesForFluxHonoursCustomSystemPrompt() async throws {
        // The LoRA evaluator passes a training-context rubric; it must not be
        // silently replaced by the generic one.
        let stub = StubVLMProvider(answer: "{\"scene_score\": 5, \"style_score\": 5}")
        _ = try await stub.compareImagesForFlux(
            reference: Self.solidImage(), generated: Self.solidImage(),
            systemPrompt: "CUSTOM RUBRIC"
        )
        XCTAssertEqual(stub.lastSystemPrompt, "CUSTOM RUBRIC")
    }

    func testTextOnlyHelperSendsNoImages() async throws {
        let stub = StubVLMProvider(answer: "{\"trigger_word\": \"sks\"}")
        let out = try await stub.generate(prompt: "analyse", systemPrompt: "JSON only", maxTokens: 100)
        XCTAssertEqual(out, "{\"trigger_word\": \"sks\"}")
        XCTAssertEqual(stub.lastImageCount, 0)
        XCTAssertEqual(stub.lastMaxTokens, 100)
    }

    // MARK: - Helpers

    private static func solidImage(width: Int = 16, height: Int = 16) -> CGImage {
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.8, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()!
    }
}
