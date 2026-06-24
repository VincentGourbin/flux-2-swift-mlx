// Flux2VLMPromptBuilderTests.swift — Public-surface tests for the VLM prompt builder
// Copyright 2025 Vincent Gourbin
//
// The actual VLM-driven prompt assembly is not covered here — it requires
// the Qwen3.5 VLM weights to be present and would make the test target
// model-dependent. Instead we exercise:
// - the graceful nil fallback when the VLM is not loaded (the load-bearing
//   "VLM is never required" contract from the chain APIs);
// - the system-prompt rules that anyone reviewing this PR should be able
//   to see at a glance (no negatives, no generic suffixes, BFL structure);
// - the user-message assembly (which is pure-Swift and operation-specific);
// - the output cleaner (quotes / whitespace stripping).

import XCTest
@testable import Flux2Chains
import CoreGraphics

final class Flux2VLMPromptBuilderTests: XCTestCase {

    // MARK: - Fallback contract

    func testInpaintBuilderReturnsNilWhenVLMNotLoaded() async throws {
        // The VLM service is a singleton; in the test environment it's
        // never loaded. The builder must therefore return nil — that's
        // the signal callers use to fall back to the verbatim prompt.
        let img = solidImage(width: 32, height: 32)
        let out = try await Flux2VLMPromptBuilder.buildInpaintPrompt(
            source: img,
            userInstruction: "replace the cat with a duck",
            intent: .replace
        )
        XCTAssertNil(out, "Builder must return nil when the Qwen3.5 VLM is not loaded — never throw, never auto-load.")
    }

    func testOutpaintBuilderReturnsNilWhenVLMNotLoaded() async throws {
        let img = solidImage(width: 32, height: 32)
        let out = try await Flux2VLMPromptBuilder.buildOutpaintPrompt(
            source: img,
            userInstruction: "extend with a sunset desert",
            sides: [.left, .right]
        )
        XCTAssertNil(out)
    }

    func testOutpaintBuilderReturnsNilWhenSidesEmpty() async throws {
        let img = solidImage(width: 32, height: 32)
        let out = try await Flux2VLMPromptBuilder.buildOutpaintPrompt(
            source: img,
            userInstruction: "extend",
            sides: []
        )
        XCTAssertNil(out, "No active sides ⇒ no-op (caller should not even call the builder).")
    }

    // MARK: - System prompt invariants

    func testSystemPromptsHonourBFLRules() {
        // Every system prompt must forbid negatives and generic suffixes
        // and must lock the output to a single prompt with no preamble.
        // These are the rules the BFL guide enforces — regressions here
        // would silently make the produced prompts worse without
        // breaking any test, so we pin them in.
        let prompts: [(String, String)] = [
            ("replace",     Flux2VLMPromptBuilder.replaceSystemPrompt),
            ("remove",      Flux2VLMPromptBuilder.removeSystemPrompt),
            ("modify",      Flux2VLMPromptBuilder.modifySystemPrompt),
            ("changeScene", Flux2VLMPromptBuilder.changeSceneSystemPrompt),
            ("outpaint",    Flux2VLMPromptBuilder.outpaintSystemPrompt),
        ]
        for (name, p) in prompts {
            XCTAssertTrue(p.contains("FLUX.2"), "[\(name)] must mention FLUX.2 — the model name primes the assistant for the right output style")
            XCTAssertTrue(p.contains("Subject + Action + Style + Context"), "[\(name)] must enforce the BFL structure")
            XCTAssertTrue(p.contains("30–80") || p.contains("30-80"), "[\(name)] must enforce the 30-80 word target from the BFL guide")
            XCTAssertTrue(p.lowercased().contains("never use generic suffixes") || p.lowercased().contains("no generic suffixes"), "[\(name)] must forbid generic suffixes like 'seamlessly extend'")
            XCTAssertTrue(p.lowercased().contains("never use negative phrases") || p.lowercased().contains("no negative"), "[\(name)] must forbid negative phrasing (FLUX.2 has no negatives)")
            XCTAssertTrue(p.lowercased().contains("output only"), "[\(name)] must lock the output to a single prompt — small models otherwise emit preambles")
        }
    }

    func testRemoveSystemPromptForbidsNamingTheObject() {
        // This is the load-bearing rule for the `.remove` case — naming
        // the object would make FLUX.2 reproduce it. If this string ever
        // drifts away, the regression won't surface until a user
        // notices the object reappearing.
        let p = Flux2VLMPromptBuilder.removeSystemPrompt.lowercased()
        XCTAssertTrue(
            p.contains("never name") || p.contains("do not name") || p.contains("never mention"),
            "remove system prompt must explicitly forbid naming the object being removed"
        )
    }

    // MARK: - User-message assembly (pure Swift, deterministic)

    func testReplaceUserMessageEchoesUserInstruction() {
        let msg = Flux2VLMPromptBuilder.userMessage(forInpaint: .replace, instruction: "a mallard duck")
        XCTAssertTrue(msg.contains("a mallard duck"))
    }

    func testRemoveUserMessageDoesNotAskVLMToEcho() {
        // For `.remove` the user's text (the object to remove) is passed
        // as context but the message instructs the VLM not to echo it.
        let msg = Flux2VLMPromptBuilder.userMessage(forInpaint: .remove, instruction: "the tabby cat")
        XCTAssertTrue(msg.contains("the tabby cat"), "Must pass the user's text as context so the VLM knows WHAT to ignore")
        XCTAssertTrue(msg.contains("Do NOT name it"), "Must instruct the VLM not to mention the removed subject in its output")
    }

    func testChangeSceneSystemPromptForbidsInventingAccessories() {
        // Load-bearing: `.modify` on a "put cat at the pool" prompt
        // hallucinated an "orange life vest" the user never asked for.
        // `.changeScene` must explicitly forbid that pattern.
        let p = Flux2VLMPromptBuilder.changeSceneSystemPrompt.lowercased()
        XCTAssertTrue(
            p.contains("never invent accessories"),
            "changeScene must forbid inventing accessories/outfits the user did not request"
        )
    }

    func testChangeSceneSystemPromptForbidsNamingTheSubject() {
        // Load-bearing — pinned in 2026-05-29 after the "two cats"
        // failure mode: when the prompt describes the preserved
        // subject, FLUX paints a duplicate next to the one the mask
        // preserved. The system prompt must explicitly forbid this.
        let p = Flux2VLMPromptBuilder.changeSceneSystemPrompt.lowercased()
        XCTAssertTrue(
            p.contains("never name or describe the subject")
            || p.contains("never name, describe, count, or hint at the existing subject")
            || p.contains("ignore the subject completely"),
            "changeScene must explicitly forbid naming the preserved subject — otherwise FLUX paints a duplicate"
        )
    }

    func testChangeSceneUserMessageReinforcesNoSubjectMention() {
        let msg = Flux2VLMPromptBuilder.userMessage(forInpaint: .changeScene, instruction: "at a swimming pool")
        XCTAssertTrue(msg.contains("at a swimming pool"))
        XCTAssertTrue(
            msg.lowercased().contains("do not name or describe"),
            "User message must reinforce the no-subject-mention rule alongside the system prompt"
        )
    }

    func testModifyUserMessageDescribesTheModification() {
        let msg = Flux2VLMPromptBuilder.userMessage(forInpaint: .modify, instruction: "blue collar instead of red")
        XCTAssertTrue(msg.contains("blue collar instead of red"))
        XCTAssertTrue(msg.lowercased().contains("keep the subject"), "Must remind the VLM to preserve the subject's identity")
    }

    func testReplaceUserMessageHandlesEmptyInstruction() {
        let msg = Flux2VLMPromptBuilder.userMessage(forInpaint: .replace, instruction: "   ")
        XCTAssertFalse(msg.isEmpty, "Empty/whitespace instruction must still produce a usable message")
        XCTAssertTrue(msg.lowercased().contains("replace"))
    }

    func testOutpaintUserMessageListsSidesDeterministically() {
        // Set<OutpaintSide> doesn't have a guaranteed iteration order;
        // the builder must serialise them in the enum's declaration order
        // so two callers with the same logical input get the same output
        // (reproducibility matters for caching, debugging, snapshot tests).
        let m1 = Flux2VLMPromptBuilder.userMessage(forOutpaintSides: [.right, .top], instruction: "x")
        let m2 = Flux2VLMPromptBuilder.userMessage(forOutpaintSides: [.top, .right], instruction: "x")
        XCTAssertEqual(m1, m2)
        // Top declared before right in the enum.
        let topIdx = m1.range(of: "top")!.lowerBound
        let rightIdx = m1.range(of: "right")!.lowerBound
        XCTAssertLessThan(topIdx, rightIdx, "Sides must be serialised in OutpaintSide.allCases order")
    }

    func testOutpaintUserMessageHandlesEmptyInstruction() {
        let msg = Flux2VLMPromptBuilder.userMessage(forOutpaintSides: [.left], instruction: "")
        XCTAssertFalse(msg.isEmpty)
        XCTAssertTrue(msg.contains("left"))
    }

    // MARK: - Output cleaner

    func testCleanFinalPromptStripsWhitespace() {
        XCTAssertEqual(Flux2VLMPromptBuilder.cleanFinalPrompt("  hello  \n"), "hello")
    }

    func testCleanFinalPromptStripsMatchingQuotes() {
        XCTAssertEqual(Flux2VLMPromptBuilder.cleanFinalPrompt("\"hello\""), "hello")
        XCTAssertEqual(Flux2VLMPromptBuilder.cleanFinalPrompt("“hello”"), "hello")
        XCTAssertEqual(Flux2VLMPromptBuilder.cleanFinalPrompt("'hello'"), "hello")
    }

    func testCleanFinalPromptLeavesMismatchedQuotesAlone() {
        // Don't accidentally remove an opening quote that's part of a
        // legitimate output (e.g., a quoted brand inside the prompt).
        XCTAssertEqual(Flux2VLMPromptBuilder.cleanFinalPrompt("a sign reading \"OPEN\""), "a sign reading \"OPEN\"")
    }

    func testValidatedPromptRejectsSafetyRefusal() {
        XCTAssertNil(
            Flux2VLMPromptBuilder.validatedPrompt(
                from: "I cannot generate content related to smoking or tobacco."
            )
        )
    }

    func testValidatedPromptAcceptsNormalPrompt() {
        XCTAssertEqual(
            Flux2VLMPromptBuilder.validatedPrompt(
                from: "Weathered grey asphalt pavement in soft midday sunlight."
            ),
            "Weathered grey asphalt pavement in soft midday sunlight."
        )
    }

    func testLooksLikeVLMRefusalDetectsCommonPrefixes() {
        XCTAssertTrue(Flux2VLMPromptBuilder.looksLikeVLMRefusal("I'm sorry, but I can't help with that."))
        XCTAssertFalse(Flux2VLMPromptBuilder.looksLikeVLMRefusal("A mallard duck on grey concrete in soft light."))
    }

    // MARK: - Helpers

    private func solidImage(width: Int, height: Int, value: UInt8 = 128) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for i in stride(from: 0, to: pixels.count, by: 4) {
            pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value; pixels[i + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        return ctx.makeImage()!
    }
}
