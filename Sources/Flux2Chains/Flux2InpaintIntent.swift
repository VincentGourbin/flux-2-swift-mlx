// Flux2InpaintIntent.swift — Edit intent enum for VLM-guided prompt enrichment
// Copyright 2025 Vincent Gourbin
//
// FLUX.2 has no negative prompts and no dedicated edit instruction channel:
// the *only* steering signal for the masked region is the text prompt.
// What that prompt should look like depends on what the user is trying to
// do — and the three cases below have *opposite* requirements.

import Foundation

/// What the caller is trying to do inside the masked region of an inpaint.
///
/// Used by ``Flux2MaskedInpaintingChain`` when `enrichPromptWithVLM` is
/// enabled: the chain picks an intent-specific system prompt for the
/// bundled Qwen3.5 VLM so the produced FLUX.2 prompt matches the user's
/// actual goal. When `enrichPromptWithVLM` is `false`, this enum has no
/// effect — the caller's prompt is passed through verbatim.
///
/// Each case maps to a different prompting strategy because FLUX.2 reacts
/// very differently to "describe the new subject" vs "describe the empty
/// surface" vs "modify the existing subject":
///
/// - ``replace``: name the *new* subject in the BFL Subject+Action+Style
///   +Context structure; *never* mention the object being removed (naming
///   it would re-introduce it — FLUX.2 has no negatives). The VLM
///   describes the source's camera angle, lighting direction, materials,
///   palette, and depth of field so the new subject inherits the scene's
///   photographic identity, including a cast shadow consistent with the
///   existing light direction.
///
/// - ``remove``: describe *only* the surface that should continue under
///   what is being removed. The VLM looks at the pixels around the
///   masked region and writes a prompt about that surface alone — naming
///   the object would make it reappear.
///
/// - ``modify``: keep the existing subject recognisable but apply the
///   user's modification (colour change, outfit swap, expression
///   change). The VLM preserves the scene's photographic identity *and*
///   the subject's identity, applying the change as Action.
public enum Flux2InpaintIntent: String, Sendable, CaseIterable, Codable {
    /// The masked region currently contains an object the user wants to
    /// swap for a different subject. Example: *cat → duck*. The user
    /// supplies the new subject as the prompt.
    case replace

    /// The masked region currently contains an object the user wants
    /// gone, with the surrounding surface continuing seamlessly into the
    /// gap. Example: removing a person from a beach photo so only the
    /// sand remains. The user's prompt (if any) is consumed as context
    /// only; the produced FLUX.2 prompt never mentions the removed
    /// subject.
    case remove

    /// The masked region contains an existing subject the user wants to
    /// modify in place (colour, outfit, expression…) while keeping it
    /// recognisable and integrated. Example: *change the cat's collar
    /// from red to blue*. The user's prompt describes the modification.
    case modify

    /// The mask is **inverted** vs the other cases: it preserves a
    /// subject and the inpainted region is the **scene around it**. The
    /// user wants the same subject but relocated into a new context.
    /// Example: *put the cat at a swimming pool* — cat stays as-is,
    /// the asphalt + wall background becomes a pool deck. The user's
    /// prompt describes the new scene; the VLM is instructed to
    /// preserve the subject's anatomy verbatim and inherit the source's
    /// lighting direction so the new scene integrates naturally with
    /// the kept region.
    ///
    /// Distinct from ``modify`` because `.modify` invents a subject-
    /// level change (e.g., a "bright orange life vest" hallucinated
    /// from a "pool" cue), which is the wrong direction here.
    case changeScene
}

extension Flux2InpaintIntent {
    public var displayName: String {
        switch self {
        case .replace: "Replace"
        case .remove: "Remove"
        case .modify: "Modify"
        case .changeScene: "Change scene"
        }
    }

    public var fillHelp: String {
        switch self {
        case .replace:
            "Swap what is inside the mask for a new subject named in the prompt."
        case .remove:
            "Clear the masked region and continue the surrounding surface."
        case .modify:
            "Repair or adjust what is inside the mask while keeping it recognisable."
        case .changeScene:
            "Keep the detected subject and repaint the scene around it."
        }
    }
}
