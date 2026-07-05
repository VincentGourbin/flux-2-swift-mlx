// IPad8GBDeviceMatrixGPUTests.swift — Sortie B5 (OPERATION THIMBLE TYPHOON)
//
// 8 GB iPad device-matrix smoke test (R6.2, Acceptance). Same SPLIT as A8
// (OQ-3):
//
//   • The CI-runnable half — "Klein 4B int4 generation at the iPad-8GB
//     defaults produces a non-nil image" — lives here and runs headless on
//     macOS CI against cached SwiftAcervo models (acervo-integration-ci
//     standard), MODEL-PRESENCE-GATED via `Acervo.isModelAvailable`. The 8 GB
//     int4 weights are NOT cached on a headless dev box, so this suite is
//     expected to SKIP CLEANLY on a machine that has not primed them (never a
//     red failure).
//
//   • The "no jetsam on real 8 GB iPad hardware" half is a MANUAL on-device
//     checklist item — CI cannot assert iPad jetsam. See the "B5 — 8 GB iPad
//     jetsam checklist" section of `ON_DEVICE_CHECKLIST.md` at the repo root.
//
// Honest status (requirements §8 OQ / EXECUTION_PLAN.md OQ-4): the 8 GB
// sub-tier (`MemoryConfig.MemoryTier.iPad8GB`) and its §5 8 GB knobs (512²
// default / 768² hard-max, int4 transformer, clearCache every 2 steps, max 1
// reference image — B3/B4) are fully IMPLEMENTED, but `MemoryConfig
// .enable8GBTier` still defaults OFF: B3's conservative working-set estimate
// (2.18 GB int4 transformer + ~2.28 GB text encoder + ~0.168 GB VAE + ~6 GB
// working set ≈ 10 GB) exceeds the 8 GB physical floor. This test exercises
// the 8 GB config via the `enable8GBTier:` PARAMETER on the `forRAMGB:`
// helpers — it does NOT flip the global `MemoryConfig.enable8GBTier` flag, so
// the shipped default stays OFF. This test (run on real 8 GB hardware with
// `phys_footprint` telemetry, A6) plus the manual jetsam checklist below are
// exactly the on-device measurement that would justify flipping the flag ON
// in a future follow-up.
//
// The three 8 GB-tier components (B1 registers the transformer; VAE / Qwen3
// encoder are the same Phase-1 components A7 already provisioned):
//   • transformer (int4) → themindstudio/flux2-klein-4b-mlx-4bit (via .klein4B_4bit)
//   • VAE                 → black-forest-labs/FLUX.2-klein-4B (vae/ subfolder)
//   • text encoder        → lmstudio-community/Qwen3-4B-MLX-{8,4}bit
//
// Gate mechanism: a `.enabled(if:)` trait — the A8/B2 pattern — so a missing
// model / missing Metal device disables the test (a clean skip) instead of
// recording a failure.

import CoreGraphics
import Foundation
import Metal
import SwiftAcervo
import Testing

@testable import Flux2Core

/// True only when a Metal device is present AND all three 8 GB-tier Klein 4B
/// int4 components are on disk (cached / primed). Mirrors the A8/B2 gate:
/// `Acervo.isModelAvailable` is synchronous, local-only I/O, so it is safe
/// here as a trait condition.
func iPad8GBSmokeTestEnabled() -> Bool {
  guard MTLCreateSystemDefaultDevice() != nil else { return false }

  // Same crash-guard as A8/B2: `Acervo.isModelAvailable` fatalErrors when
  // neither ACERVO_MODELS_DIR nor an App Group is configured (unentitled
  // local test-runner case). Guard first so an un-primed local machine skips
  // cleanly instead of crashing the xctest runner.
  let env = ProcessInfo.processInfo.environment
  let hasModelsDir = (env[Acervo.modelsDirectoryOverrideVariable]?.isEmpty == false)
  let hasAppGroup = (env[Acervo.appGroupEnvironmentVariable]?.isEmpty == false)
  guard hasModelsDir || hasAppGroup else { return false }

  // Derive the int4 repoId from the registry so a variant rename can't
  // silently desync this gate: (klein4B, .int4) → .klein4B_4bit →
  // themindstudio/flux2-klein-4b-mlx-4bit.
  let transformerRepoId =
    ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4).repoId
  let vaeRepoId = ModelRegistry.VAEVariant.standard.repoId

  // Klein prefers the 8-bit Qwen3 encoder and falls back to 4-bit
  // (KleinTextEncoder.findDownloadedQwen3Variant); either satisfies the gate.
  let qwen3_8bit = "lmstudio-community/Qwen3-4B-MLX-8bit"
  let qwen3_4bit = "lmstudio-community/Qwen3-4B-MLX-4bit"

  return Acervo.isModelAvailable(transformerRepoId)
    && Acervo.isModelAvailable(vaeRepoId)
    && (Acervo.isModelAvailable(qwen3_8bit) || Acervo.isModelAvailable(qwen3_4bit))
}

@Suite("iPad 8GB Device Matrix (GPU)", .serialized)
struct IPad8GBDeviceMatrixGPUTests {

  /// Klein 4B int4 generation at the resolved iPad-8GB defaults (512², 4
  /// steps, guidance 1.0) must produce a non-nil 512×512 image.
  ///
  /// The generation knobs are pulled from the Sortie B4 tier-aware
  /// `forRAMGB:` helpers with `ramGB: 8` AND `enable8GBTier: true` PASSED AS
  /// A PARAMETER — this exercises the real `.iPad8GB` config path for this
  /// test only. It never mutates the shared `MemoryConfig.enable8GBTier`
  /// global flag, which stays OFF (asserted below) per B3's conservative
  /// ~10 GB working-set estimate (requirements §8, EXECUTION_PLAN.md OQ-4).
  @Test(
    "Klein 4B int4 @ iPad-8GB defaults → non-nil 512² image",
    .enabled(if: iPad8GBSmokeTestEnabled()),
    .timeLimit(.minutes(10))
  )
  func klein4BInt4At512IPad8GBDefaultsProducesImage() async throws {
    let ramGB = 8
    let enable8GBTier = true

    // The shipped global default must stay OFF regardless of this test
    // exercising the sub-tier locally via the parameter overload.
    #expect(
      MemoryConfig.enable8GBTier == false,
      "MemoryConfig.enable8GBTier must remain OFF by default (B3); this test exercises the 8 GB sub-tier via the enable8GBTier: parameter only"
    )

    // B4 tier-aware defaults resolved for the iPad-8GB sub-tier.
    let resolution = Flux2Pipeline.defaultResolution(
      forRAMGB: ramGB, enable8GBTier: enable8GBTier)
    let steps = Flux2Pipeline.defaultSteps(forRAMGB: ramGB, enable8GBTier: enable8GBTier)
    let guidance = Flux2Pipeline.defaultGuidance(forRAMGB: ramGB, enable8GBTier: enable8GBTier)
    let quantization = Flux2Pipeline.defaultQuantization(
      forRAMGB: ramGB, enable8GBTier: enable8GBTier)

    // Prove we are on the §5 "iPad 8 GB" column before spending inference time.
    #expect(resolution == 512, "iPad-8GB default resolution should be 512, got \(resolution)")
    #expect(steps == 4, "iPad-8GB default steps (Klein 4B recommended) should be 4, got \(steps)")
    #expect(guidance == 1.0, "iPad-8GB default guidance should be 1.0, got \(guidance)")
    #expect(
      quantization.transformer == .int4,
      "iPad-8GB default transformer quant should be int4, got \(quantization.transformer)")

    let variant = ModelRegistry.TransformerVariant.variant(
      for: .klein4B, quantization: quantization.transformer)
    #expect(variant == .klein4B_4bit)
    #expect(variant.isPreQuantizedMLX, "int4 path must take the direct pre-quantized branch")

    // Build the pipeline exactly as the iPad-8GB path would: forced Klein 4B
    // with the sub-tier-resolved (int4 transformer / Qwen3 encoder) quant.
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: quantization)
    try await pipeline.loadModels()
    #expect(pipeline.isLoaded, "Pipeline should report isLoaded after loadModels()")

    let image = try await pipeline.generateTextToImage(
      prompt: "a single red apple on a white table",
      height: resolution,
      width: resolution,
      steps: steps,
      guidance: guidance,
      seed: 7
    )

    // Non-nil image at the iPad-8GB resolution is the acceptance signal.
    #expect(image.width == 512, "Generated image width should be 512, got \(image.width)")
    #expect(image.height == 512, "Generated image height should be 512, got \(image.height)")
  }
}
