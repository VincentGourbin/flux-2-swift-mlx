// Flux2VLMPromptBuilder.swift — Build FLUX.2 prompts from a source image via the bundled VLM
// Copyright 2025 Vincent Gourbin
//
// Why this exists:
// FLUX.2 has no negative prompts and no edit-instruction channel — the
// prompt is the only steering signal for the masked region. Short prompts
// (`"a duck"`, `"replace the cat"`) leave the model without lighting,
// perspective, scale or material cues; the result floats, hallucinates
// wrong surfaces, or copies the source. The BFL prompting guide
// (<https://docs.bfl.ml/guides/prompting_guide_flux2>) recommends 30-80
// words structured Subject + Action + Style + Context, with photographic
// vocabulary (focal length, lighting direction, depth of field).
//
// This builder asks the bundled Qwen3.5 VLM to *look at the source* and
// assemble such a prompt automatically, with an operation-specific system
// prompt so the produced text matches the actual intent (replace / remove
// / modify / outpaint). The chains call this as a strictly opt-in
// preprocessor — when the VLM isn't loaded the call returns `nil` so the
// caller can fall back to the user's verbatim prompt.

import Foundation
import CoreGraphics
import Flux2Core
import FluxTextEncoders

/// Build FLUX.2-style prompts from a source image using the bundled
/// Qwen3.5 VLM. Returns `nil` whenever the VLM is not currently loaded
/// (callers must treat this as a graceful fallback signal — never as an
/// error).
public enum Flux2VLMPromptBuilder {

    // MARK: - System prompts (public so tests can introspect them)

    /// System prompt used for `Flux2InpaintIntent.replace`. Instructs the
    /// VLM to write a 30-80 word BFL-style prompt naming the *new*
    /// subject, inheriting the source's photographic identity, and asking
    /// for a shadow consistent with the existing light direction. The
    /// removed subject is **never** named (FLUX.2 has no negatives — naming
    /// it would re-introduce it).
    public static let replaceSystemPrompt: String = """
    You are a prompt engineer for FLUX.2 by Black Forest Labs. The user wants to REPLACE an existing object in an image with a different subject they will describe to you. Your job is to produce a single FLUX.2 image-generation prompt.

    Look at the provided image and extract its photographic identity:
    - camera angle and apparent height (top-down, eye-level, low ground-level…)
    - approximate focal length feel (wide-angle, normal, telephoto)
    - lighting direction, softness, and colour temperature; time of day
    - surface materials around the area being replaced (asphalt, sand, grass…)
    - dominant colour palette and depth of field
    - overall style (naturalistic phone photo, DSLR, illustration, 3D render…)

    Then assemble a single FLUX.2 prompt of 30–80 words following the structure Subject + Action + Style + Context:
    - Subject = the user's NEW subject.
    - Action = its pose at the position where the previous subject was, with an explicit cast shadow direction consistent with the existing lighting.
    - Style + Context = the source's photographic identity, verbatim.

    Rules:
    - NEVER name or describe the object being replaced. Naming it would make FLUX.2 reproduce it (no negative prompts in FLUX.2).
    - NEVER use generic suffixes like "seamlessly extend", "high quality", "8k", "masterpiece".
    - NEVER use negative phrases ("without", "no", "not").
    - Use specific photographic vocabulary (focal length, f-stop, light direction).
    - Output ONLY the final prompt, on a single line. No preamble, no quotes, no explanation.
    """

    /// System prompt used for `Flux2InpaintIntent.remove`. Instructs the
    /// VLM to describe **only the surface** that should continue under the
    /// removed object — never naming the object itself.
    public static let removeSystemPrompt: String = """
    You are a prompt engineer for FLUX.2 by Black Forest Labs. The user wants to REMOVE an object from an image; nothing should replace it — the surrounding surface must simply continue. Your job is to produce a single FLUX.2 image-generation prompt that describes the surface as it would look if the object had never been there.

    Look at the provided image, focusing on what is AROUND the area to be cleared (not on the object itself):
    - the exact surface material under and around the object (asphalt grain, concrete edge, sand ripples, grass density…)
    - continuity features that should flow through the gap (cracks, gravel scatter, shadow patterns, foliage…)
    - lighting direction, softness, and colour temperature; time of day
    - camera angle and apparent height, focal length feel, depth of field
    - dominant colour palette and overall style

    Assemble a single FLUX.2 prompt of 30–80 words following the structure Subject + Action + Style + Context:
    - Subject = the surface itself (e.g., "weathered grey asphalt pavement").
    - Action = describe its continuity ("running uninhibited from the limestone wall to the brick drain cover…").
    - Style + Context = the source's photographic identity, verbatim.

    Rules:
    - NEVER name, describe, or hint at the object being removed. Naming it would make FLUX.2 reproduce it (no negative prompts in FLUX.2).
    - NEVER use generic suffixes like "seamlessly extend", "high quality", "8k".
    - NEVER use negative phrases ("without the X", "no people", "empty of").
    - Use specific photographic vocabulary.
    - Output ONLY the final prompt, on a single line. No preamble, no quotes, no explanation.
    """

    /// System prompt used for `Flux2InpaintIntent.fill`. Instructs the
    /// VLM to complete a missing, empty, or damaged region by describing
    /// what should continue from the surrounding pixels — no new subject
    /// unless the user's hint requests one.
    public static let fillSystemPrompt: String = """
    You are a prompt engineer for FLUX.2 by Black Forest Labs. The user wants to FILL a missing, empty, or damaged region in an image — the masked area should be completed from the surrounding context, like generative fill. Your job is to produce a single FLUX.2 image-generation prompt that describes what belongs in that region.

    Look at the provided image, focusing on the materials and features that border the masked region:
    - the exact surface or texture that should continue through the gap (brick, skin, fabric weave, asphalt grain, sky gradient…)
    - continuity features that should flow through (cracks, seams, shadow patterns, folds, grain direction…)
    - lighting direction, softness, and colour temperature; time of day
    - camera angle and apparent height, focal length feel, depth of field
    - dominant colour palette and overall style

    Assemble a single FLUX.2 prompt of 30–80 words following the structure Subject + Action + Style + Context:
    - Subject = the material or surface completing the region (e.g., "weathered red brick wall", "smooth olive skin").
    - Action = how it continues through the masked area ("with mortar lines aligned to the courses on either side…").
    - Style + Context = the source's photographic identity, verbatim.

    Rules:
    - Describe continuation from context — do not invent a new unrelated subject unless the user's hint explicitly requests one.
    - NEVER use generic suffixes like "seamlessly extend", "high quality", "8k".
    - NEVER use negative phrases ("without the X", "no people", "empty of").
    - Use specific photographic vocabulary.
    - Output ONLY the final prompt, on a single line. No preamble, no quotes, no explanation.
    """

    /// System prompt used for `Flux2InpaintIntent.modify`. Instructs the
    /// VLM to keep the existing subject recognisable and apply the user's
    /// modification while preserving the scene's photographic identity.
    public static let modifySystemPrompt: String = """
    You are a prompt engineer for FLUX.2 by Black Forest Labs. The user wants to MODIFY an existing subject in an image (change its colour, outfit, expression, accessory, etc.) while keeping it recognisable and integrated in the same scene. Your job is to produce a single FLUX.2 image-generation prompt.

    This is a technical description of an EXISTING photograph for an image-editing pipeline — you are not creating new imagery. Describe only what is visible in the source pixels: surfaces, lighting, materials, and the subject being adjusted.

    Look at the provided image and capture:
    - the existing subject's identity (species, breed, age, gender, pose…)
    - the source's photographic identity (camera angle, focal length feel, lighting direction and colour temperature, depth of field, palette, style)
    - the surface and immediate context where the subject sits

    Assemble a single FLUX.2 prompt of 30–80 words following Subject + Action + Style + Context:
    - Subject = the existing subject, described faithfully.
    - Action = the subject's pose with the user's modification applied (e.g., "wearing a blue collar instead of red", "with a wide smile").
    - Style + Context = the source's photographic identity, verbatim, including the surface and existing cast shadow.

    Rules:
    - NEVER use negative phrases ("without", "no").
    - NEVER use generic suffixes like "seamlessly extend", "high quality", "8k".
    - Use specific photographic vocabulary.
    - Output ONLY the final prompt, on a single line. No preamble, no quotes, no explanation.
    """

    /// System prompt used for `Flux2InpaintIntent.changeScene`. The
    /// mask is inverted vs the other inpaint intents: it preserves a
    /// subject and the inpainted region is the *scene around it*.
    /// Instructs the VLM to keep the existing subject's anatomy and
    /// appearance verbatim, and only describe a NEW scene that
    /// inherits the source's lighting direction so the kept subject
    /// integrates naturally with the new context.
    public static let changeSceneSystemPrompt: String = """
    You are a prompt engineer for FLUX.2 by Black Forest Labs. The user wants to CHANGE THE SCENE around an existing subject in an image. The subject is preserved BY THE MASK — your prompt must describe ONLY the new scene the subject is now in. **You must NEVER name or describe the subject itself**, because FLUX.2 would then paint a duplicate of it in the inpainted region (next to the one the mask preserves) — the classic "two cats" failure mode.

    Look at the provided image to capture ONLY:
    - the source's lighting direction, softness, and colour temperature; time of day
    - the source's camera angle, apparent height, focal length feel, depth of field
    - the source's overall style (naturalistic phone photo, DSLR, illustration, 3D render…)

    Ignore the subject completely. Do not name it, describe it, count it, or hint at it.

    Assemble a single FLUX.2 prompt of 30–80 words describing the NEW SCENE the user requested, structured Subject + Action + Style + Context:
    - Subject = the principal element of the NEW scene (e.g., "a swimming pool", "a sandy beach", "a forest clearing"). NOT the existing subject.
    - Action = the state / activity of that scene element ("with calm turquoise water lapping at white deck tiles…").
    - Style + Context = the source's photographic identity, verbatim (camera angle, focal length feel, lighting direction, depth of field, colour palette).

    Rules:
    - NEVER name, describe, count, or hint at the existing subject. Not "a cat", not "an animal", not "a creature", not "a figure". The mask handles it.
    - NEVER invent accessories, outfits, gear, or props.
    - NEVER use negative phrases ("without", "no").
    - NEVER use generic suffixes like "seamlessly extend", "high quality", "8k".
    - Use specific photographic vocabulary inherited from the source.
    - Output ONLY the final prompt, on a single line. No preamble, no quotes, no explanation.
    """

    /// System prompt used by ``Flux2OutpaintingChain``. Instructs the VLM
    /// to look at the *edges* of the kept region and describe what should
    /// continue into the new strips, preserving the source's
    /// photographic identity.
    public static let outpaintSystemPrompt: String = """
    You are a prompt engineer for FLUX.2 by Black Forest Labs. The user wants to OUTPAINT an image — extend it on one or more sides — and will tell you which sides are being extended and what they want there. Your job is to produce a single FLUX.2 image-generation prompt that describes the FULL extended scene so the new strips continue the kept region naturally.

    Look at the provided image, paying attention to what is visible at the EDGES the user is extending:
    - the materials and scene elements at each active edge (sky, foliage, walls, sand, water, perspective lines…)
    - the source's photographic identity (camera angle and apparent height, focal length feel, lighting direction and colour temperature, depth of field, colour palette, overall style)
    - any horizon line, vanishing point, or recurring pattern that must continue

    Assemble a single FLUX.2 prompt of 30–80 words following Subject + Action + Style + Context, describing the COMPLETE extended scene. Be specific about what continues into each newly added side, and how it relates to the kept region. Preserve the source's lighting direction, colour temperature, focal length feel, depth of field, and style verbatim.

    Rules:
    - NEVER use generic suffixes like "seamlessly extend and complete the image" — they leak into FLUX.2's output and add nothing.
    - NEVER use negative phrases ("without", "no").
    - Use specific photographic vocabulary.
    - Output ONLY the final prompt, on a single line. No preamble, no quotes, no explanation.
    """

    // MARK: - Public builders

    /// Build a FLUX.2 inpainting prompt by asking the VLM to look at
    /// `source` and combine the user's instruction with the source's
    /// photographic identity, using the system prompt for `intent`.
    ///
    /// - Parameters:
    ///   - source: The image being inpainted.
    ///   - userInstruction: What the user typed (e.g.,
    ///     `"replace the cat with a duck"`, `"remove the person"`). Can
    ///     be empty for `.remove` since the VLM ignores it anyway.
    ///   - intent: Drives system prompt selection. See
    ///     ``Flux2InpaintIntent``.
    /// - Returns: A 30-80 word FLUX.2 prompt, or `nil` when the VLM is
    ///   not loaded (the caller falls back to `userInstruction` verbatim).
    /// - Throws: Whatever the VLM throws at runtime (token sampling
    ///   failure, malformed input). Never throws to signal
    ///   "VLM unavailable" — that case returns `nil`.
    public static func buildInpaintPrompt(
        source: CGImage,
        userInstruction: String,
        intent: Flux2InpaintIntent,
        vlmContextArea: CGRect? = nil
    ) async throws -> String? {
        guard FluxTextEncoders.shared.isQwen35VLMLoaded else { return nil }

        let systemPrompt: String
        switch intent {
        case .replace:     systemPrompt = replaceSystemPrompt
        case .remove:      systemPrompt = removeSystemPrompt
        case .fill:        systemPrompt = fillSystemPrompt
        case .modify:      systemPrompt = modifySystemPrompt
        case .changeScene: systemPrompt = changeSceneSystemPrompt
        }
        let userMessage = userMessage(forInpaint: intent, instruction: userInstruction)
        let vlmImage = try imageForVLM(source: source, contextArea: vlmContextArea)

        // The VLM forward is synchronous and takes ~3 s on M-series. Run
        // it on a detached task so other work on the cooperative pool
        // (UI updates, concurrent chains) isn't starved while we wait.
        let raw = try await runVLM(image: vlmImage, prompt: userMessage, systemPrompt: systemPrompt)
        return validatedPrompt(from: raw)
    }

    /// Build a FLUX.2 outpainting prompt. The VLM looks at the source and
    /// the list of sides being extended, then produces a 30-80 word BFL
    /// prompt describing the full extended scene with continuity.
    ///
    /// - Parameters:
    ///   - source: The image being extended.
    ///   - userInstruction: The user's description of what should appear
    ///     in the extensions (e.g., `"a sunset desert horizon on the right"`).
    ///     Can be empty — the VLM defaults to coherent continuation.
    ///   - sides: Which canvas sides are being extended (non-empty).
    /// - Returns: A 30-80 word FLUX.2 prompt, or `nil` when the VLM is
    ///   not loaded.
    /// - Throws: VLM runtime errors only.
    public static func buildOutpaintPrompt(
        source: CGImage,
        userInstruction: String,
        sides: Set<OutpaintSide>
    ) async throws -> String? {
        guard FluxTextEncoders.shared.isQwen35VLMLoaded else { return nil }
        guard !sides.isEmpty else { return nil }

        let userMessage = userMessage(forOutpaintSides: sides, instruction: userInstruction)

        let raw = try await runVLM(image: source, prompt: userMessage, systemPrompt: outpaintSystemPrompt)
        return validatedPrompt(from: raw)
    }

    /// Run the Qwen3.5 VLM forward on a detached, user-initiated task
    /// so the ~3 s sync call doesn't block the cooperative thread pool.
    /// CGImage / String are Sendable in the SDK we target.
    private static func runVLM(
        image: CGImage,
        prompt: String,
        systemPrompt: String
    ) async throws -> String {
        try await Task.detached(priority: .userInitiated) {
            try FluxTextEncoders.shared.analyzeImageWithQwen35(
                image: image,
                prompt: prompt,
                systemPrompt: systemPrompt,
                enableThinking: false,
                maxTokens: 220,
                temperature: 0
            ).text
        }.value
    }

    private static func imageForVLM(source: CGImage, contextArea: CGRect?) throws -> CGImage {
        guard let contextArea else { return source }
        if isApproximatelyFullFrame(contextArea) || isApproximatelyFullFrame(ImagePreparation.clampUnitRect(contextArea)) {
            return source
        }
        return try ImagePreparation.cropReferenceImage(source, normalizedRect: contextArea)
    }

    private static func isApproximatelyFullFrame(_ rect: CGRect) -> Bool {
        let epsilon: CGFloat = 0.0001
        return abs(rect.minX) < epsilon
            && abs(rect.minY) < epsilon
            && abs(rect.width - 1) < epsilon
            && abs(rect.height - 1) < epsilon
    }

    // MARK: - User message assembly (exposed for tests)

    /// Compose the user-turn message for an inpaint request. The system
    /// prompt teaches the VLM the rules; this message carries the
    /// per-request payload (intent reminder + user instruction).
    public static func userMessage(forInpaint intent: Flux2InpaintIntent, instruction: String) -> String {
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        switch intent {
        case .replace:
            if trimmed.isEmpty {
                return "Replace the main subject in the masked region. Produce the FLUX.2 prompt now."
            }
            return "User's new subject: \(trimmed). Produce the FLUX.2 prompt now."
        case .remove:
            // For .remove the user's instruction (if any) tells us WHICH
            // object to remove — but we must not echo that name in the
            // produced prompt. We pass it as context only.
            if trimmed.isEmpty {
                return "Describe the surface that should continue under the masked region, with no subject in it. Produce the FLUX.2 prompt now."
            }
            return "The user wants to remove this from the masked region: \(trimmed). Do NOT name it in your output. Describe the surface that should continue in its place. Produce the FLUX.2 prompt now."
        case .fill:
            if trimmed.isEmpty {
                return "Describe what should continue into this missing or empty masked region from the surrounding context. Produce the FLUX.2 prompt now."
            }
            return "User guidance for the fill: \(trimmed). Complete the masked region from surrounding context. Produce the FLUX.2 prompt now."
        case .modify:
            if trimmed.isEmpty {
                return "Keep the existing subject and adjust it slightly to better match the scene. Produce the FLUX.2 prompt now."
            }
            return "User's modification to apply to the existing subject: \(trimmed). Keep the subject recognisable. Produce the FLUX.2 prompt now."
        case .changeScene:
            if trimmed.isEmpty {
                return "Describe a coherent new scene that inherits the source's lighting and camera. Do NOT name or describe the existing subject in the image — the mask preserves it; mentioning it would make FLUX paint a duplicate next to it. Produce the FLUX.2 prompt now."
            }
            return "User's NEW SCENE (the existing subject is preserved by the mask — do NOT name or describe it in your output, or FLUX will paint a duplicate): \(trimmed). Produce the FLUX.2 prompt now."
        }
    }

    /// Compose the user-turn message for an outpaint request. Lists the
    /// active sides explicitly so the VLM focuses on the right edges.
    public static func userMessage(forOutpaintSides sides: Set<OutpaintSide>, instruction: String) -> String {
        // Deterministic ordering for reproducibility.
        let ordered = OutpaintSide.allCases.filter { sides.contains($0) }
        let sidesList = ordered.map(\.rawValue).joined(separator: ", ")
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Sides to extend: \(sidesList). Continue the scene coherently on those sides. Produce the FLUX.2 prompt now."
        }
        return "Sides to extend: \(sidesList). User wants in the extensions: \(trimmed). Produce the FLUX.2 prompt now."
    }

    // MARK: - Output cleanup

    /// Strip whitespace and quotes, then reject empty output and model
    /// safety refusals (e.g. "I cannot generate content related to…").
    /// Returns `nil` so callers fall back to the user's prompt.
    internal static func validatedPrompt(from raw: String) -> String? {
        let cleaned = cleanFinalPrompt(raw)
        guard !cleaned.isEmpty else { return nil }
        guard !looksLikeVLMRefusal(cleaned) else { return nil }
        return cleaned
    }

    /// Qwen3.5 occasionally answers with a policy refusal instead of a
    /// FLUX.2 prompt when the source image triggers its safety filter.
    internal static func looksLikeVLMRefusal(_ text: String) -> Bool {
        let lower = text.lowercased()
        let blockedPhrases = [
            "i cannot generate",
            "i can't generate",
            "i cannot create",
            "i can't create",
            "i cannot provide",
            "i can't provide",
            "i cannot help",
            "i can't help",
            "i'm unable to",
            "i am unable to",
            "unable to generate",
            "unable to create",
            "not able to generate",
            "cannot generate content",
            "can't generate content",
            "against my guidelines",
            "against my policy",
            "violates my",
            "as an ai language model",
            "as a language model",
        ]
        if blockedPhrases.contains(where: { lower.contains($0) }) {
            return true
        }
        let blockedPrefixes = [
            "i cannot",
            "i can't",
            "sorry,",
            "i'm sorry",
            "i am sorry",
        ]
        return blockedPrefixes.contains { lower.hasPrefix($0) }
    }

    /// Strip whitespace and one surrounding pair of quotes if present.
    /// Small models occasionally wrap the prompt in quotes despite the
    /// system prompt saying not to; we remove them so the FLUX.2 text
    /// encoder doesn't ingest them. Mismatched leftover quotes inside
    /// the prompt (e.g., a quoted brand name) are left alone.
    internal static func cleanFinalPrompt(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Symmetric pairs: (opening, closing). Typographic pairs use
        // distinct characters so a single-character check would miss
        // them. Order matters: try the longest/most-specific first
        // (though all of these are single grapheme clusters in practice).
        let pairs: [(open: String, close: String)] = [
            ("\"", "\""),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
            ("«", "»"),
        ]
        for (open, close) in pairs {
            if s.hasPrefix(open) && s.hasSuffix(close) && s.count > open.count + close.count {
                s = String(s.dropFirst(open.count).dropLast(close.count))
                break
            }
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Identifies one of the four canvas sides that an outpainting chain may
/// extend. Used by ``Flux2VLMPromptBuilder/buildOutpaintPrompt(source:userInstruction:sides:)``
/// to tell the VLM which edges of the source matter for continuity.
public enum OutpaintSide: String, Sendable, CaseIterable, Codable {
    case top
    case bottom
    case left
    case right
}
