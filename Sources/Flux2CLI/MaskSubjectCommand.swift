// MaskSubjectCommand.swift — `flux2 mask-subject` (auto-segmentation)
// Copyright 2025 Vincent Gourbin
//
// Generates a Flux-convention inpaint mask from auto-segmentation of
// the source image's foreground subject. Intended for the
// `.changeScene` workflow where the user wants to keep a subject and
// regenerate everything around it — a use case that breaks with
// hand-drawn blob masks (FLUX hallucinates a second-anatomy chimera in
// the soft mask ramp).

import Foundation
import ArgumentParser
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Flux2Chains

struct MaskSubject: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mask-subject",
        abstract: "Auto-segment the foreground subject and produce a Flux-convention inpaint mask (BLACK = subject = keep, WHITE = inpaint)."
    )

    @Option(name: [.short, .long], help: "Input image to segment.")
    var image: String

    @Option(name: [.short, .long], help: "Output PNG path for the generated mask.")
    var output: String

    @Option(name: .long, help: "Width of the Gaussian transition on the subject boundary, in pixels of the working image. 0 = bit-hard edge (risk of visible seam). Default 4.")
    var edgeSoftnessPixels: Int = 4

    @Option(name: .long, help: "Optional working-resolution cap (total pixels). When set, the source is downscaled with Lanczos before Vision runs and the mask is produced at the smaller size — useful when the downstream pipeline will downscale to ~1 MP anyway (saves up to 24× of work on a 24 MP iPhone shot). Omit to keep the mask at source dimensions (right choice for ad-hoc inspection).")
    var targetMaxPixels: Int?

    func run() async throws {
        @Sendable func logErr(_ msg: String) {
            FileHandle.standardError.write(Data((msg + "\n").utf8))
        }

        guard let cg = Self.loadCGImage(at: image) else {
            throw ValidationError("Could not decode image at \(image)")
        }
        logErr("Image: \(image) (\(cg.width)×\(cg.height))")
        if let cap = targetMaxPixels {
            logErr("Target max pixels: \(cap)")
        }

        let mask: CGImage
        do {
            mask = try Flux2SubjectMask.makeChangeSceneMask(
                from: cg,
                edgeSoftnessPixels: edgeSoftnessPixels,
                targetMaxPixels: targetMaxPixels
            )
        } catch Flux2SubjectMask.Error.noSubjectDetected {
            throw ValidationError("Vision found no foreground subject in \(image). Use a hand-drawn mask for this image.")
        } catch {
            throw ValidationError("Auto-segmentation failed: \(error)")
        }

        try Self.savePNG(mask, to: output)
        logErr("✓ Mask written → \(output) (\(mask.width)×\(mask.height))")
    }

    private static func loadCGImage(at path: String) -> CGImage? {
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private static func savePNG(_ image: CGImage, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw ValidationError("Could not create PNG destination at \(path)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ValidationError("Failed to write PNG at \(path)")
        }
    }
}
