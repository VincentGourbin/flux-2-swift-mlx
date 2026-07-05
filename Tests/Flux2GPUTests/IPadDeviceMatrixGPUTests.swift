// IPadDeviceMatrixGPUTests.swift — Sortie A8 (OPERATION THIMBLE TYPHOON)
//
// 16 GB iPad device-matrix smoke test (R6.2, Acceptance). The device-matrix
// test is SPLIT (OQ-3):
//
//   • The CI-runnable half — "Klein 4B qint8 generation at the iPad-16GB
//     defaults produces a non-nil image" — lives here and runs headless on
//     macOS CI against cached SwiftAcervo models (acervo-integration-ci
//     standard). It is MODEL-PRESENCE-GATED via `Acervo.isModelAvailable`, so
//     on a machine that has not primed the Phase-1 weights the whole suite is
//     cleanly SKIPPED (never a red failure).
//
//   • The "no jetsam on real 16 GB iPad hardware" half is a MANUAL on-device
//     checklist item — CI cannot assert iPad jetsam. See
//     `ON_DEVICE_CHECKLIST.md` at the repo root.
//
// The three Phase-1 components (A7 / CDN_PROVISIONING.md):
//   • transformer (qint8) → aydin99/FLUX.2-klein-4B-int8   (via .klein4B_8bit)
//   • VAE                 → black-forest-labs/FLUX.2-klein-4B (vae/ subfolder)
//   • text encoder        → lmstudio-community/Qwen3-4B-MLX-{8,4}bit
//
// Gate mechanism: a `.enabled(if:)` trait — NOT the `Issue.record`+`return`
// pattern used by the older GPU suites — so a missing model / missing Metal
// device disables the test (a clean skip) instead of recording a failure.

import CoreGraphics
import Foundation
import Metal
import SwiftAcervo
import Testing

@testable import Flux2Core

/// True only when a Metal device is present AND all three Phase-1 Klein 4B
/// qint8 iPad-tier components are on disk (cached / primed). Evaluated as a
/// swift-testing trait condition — `Acervo.isModelAvailable` is synchronous,
/// local-only I/O, so it is safe here.
func iPad16GBSmokeTestEnabled() -> Bool {
  guard MTLCreateSystemDefaultDevice() != nil else { return false }

  // `Acervo.isModelAvailable` resolves the shared models directory, which
  // `fatalError`s when NEITHER `ACERVO_MODELS_DIR` nor an App Group identifier
  // (`ACERVO_APP_GROUP_ID` / entitlement) is configured — the unentitled local
  // test-runner case. Guard on that first so an un-primed local machine skips
  // cleanly instead of crashing the xctest runner. In CI the integration
  // workflow forwards `TEST_RUNNER_ACERVO_MODELS_DIR` (→ `ACERVO_MODELS_DIR`),
  // so this passes and the real availability check runs.
  let env = ProcessInfo.processInfo.environment
  let hasModelsDir = (env[Acervo.modelsDirectoryOverrideVariable]?.isEmpty == false)
  let hasAppGroup = (env[Acervo.appGroupEnvironmentVariable]?.isEmpty == false)
  guard hasModelsDir || hasAppGroup else { return false }

  // Derive the repoIds from the registry so a variant rename can't silently
  // desync this gate. (klein4B, qint8) → .klein4B_8bit → aydin99/…-int8.
  let transformerRepoId =
    ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .qint8).repoId
  let vaeRepoId = ModelRegistry.VAEVariant.standard.repoId

  // Klein prefers the 8-bit Qwen3 encoder and falls back to 4-bit
  // (KleinTextEncoder.findDownloadedQwen3Variant); either satisfies the gate.
  let qwen3_8bit = "lmstudio-community/Qwen3-4B-MLX-8bit"
  let qwen3_4bit = "lmstudio-community/Qwen3-4B-MLX-4bit"

  return Acervo.isModelAvailable(transformerRepoId)
    && Acervo.isModelAvailable(vaeRepoId)
    && (Acervo.isModelAvailable(qwen3_8bit) || Acervo.isModelAvailable(qwen3_4bit))
}

@Suite("iPad 16GB Device Matrix (GPU)", .serialized)
struct IPadDeviceMatrixGPUTests {

  /// Klein 4B qint8 generation at the resolved iPad-16GB defaults (768², 4
  /// steps, guidance 1.0) must produce a non-nil 768×768 image.
  ///
  /// The generation knobs are pulled from the Sortie A3 tier-aware
  /// `forRAMGB:` helpers with `ramGB: 16` so the test exercises the REAL
  /// iPad-16GB config path (not hand-typed literals) regardless of the host's
  /// actual RAM — the iPad tier is a function of the passed RAM figure, not the
  /// CI runner's memory.
  @Test(
    "Klein 4B qint8 @ iPad-16GB defaults → non-nil 768² image",
    .enabled(if: iPad16GBSmokeTestEnabled()),
    .timeLimit(.minutes(10))
  )
  func klein4BQint8At768IPadDefaultsProducesImage() async throws {
    let ramGB = 16

    // A3 tier-aware defaults resolved for the iPad-16GB tier.
    let resolution = Flux2Pipeline.defaultResolution(forRAMGB: ramGB)
    let steps = Flux2Pipeline.defaultSteps(forRAMGB: ramGB)
    let guidance = Flux2Pipeline.defaultGuidance(forRAMGB: ramGB)
    let quantization = Flux2Pipeline.defaultQuantization(forRAMGB: ramGB)

    // Prove we are on the §5 "iPad 16 GB" column before spending inference time.
    #expect(resolution == 768, "iPad-16GB default resolution should be 768, got \(resolution)")
    #expect(steps == 4, "iPad-16GB default steps (Klein 4B recommended) should be 4, got \(steps)")
    #expect(guidance == 1.0, "iPad-16GB default guidance should be 1.0, got \(guidance)")
    #expect(
      quantization.transformer == .qint8,
      "iPad-16GB default transformer quant should be qint8, got \(quantization.transformer)")

    // Build the pipeline exactly as the iPad-16GB path would: forced Klein 4B
    // with the tier-resolved (qint8 transformer / mlx8bit encoder) quant.
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: quantization)
    try await pipeline.loadModels()
    #expect(pipeline.isLoaded, "Pipeline should report isLoaded after loadModels()")

    let image = try await pipeline.generateTextToImage(
      prompt: "a red balloon floating in a clear blue sky",
      height: resolution,
      width: resolution,
      steps: steps,
      guidance: guidance,
      seed: 42
    )

    // Non-nil image at the iPad-16GB resolution is the acceptance signal.
    #expect(image.width == 768, "Generated image width should be 768, got \(image.width)")
    #expect(image.height == 768, "Generated image height should be 768, got \(image.height)")
  }
}
