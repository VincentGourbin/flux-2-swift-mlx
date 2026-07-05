// Int4DirectLoadGPUTests.swift — Sortie B2 (OPERATION THIMBLE TYPHOON)
//
// R2.3 acceptance: the direct pre-quantized int4 transformer load must NOT
// spike physical memory. The old fallback (bf16 + on-the-fly quantize) briefly
// materialized the full ~8 GB float transformer before quantizing; the B2
// direct path loads the packed MLX 4-bit weights (~2.18 GB) STRAIGHT into
// QuantizedLinear with no bf16 intermediate. We assert the process's
// `phys_footprint` captured at the transformer `weightLoadComplete` boundary
// (Sortie A6 telemetry) stays UNDER the 8 GB no-spike ceiling.
//
// This half REQUIRES the 2.18 GB themindstudio int4 weights (plus the VAE and
// Qwen3 encoder, since the transformer is only loaded during generation). Those
// are NOT cached on a headless dev box, so the suite is MODEL-PRESENCE-GATED via
// a `.enabled(if:)` trait (the A8 IPadDeviceMatrixGPUTests pattern): it SKIPS
// cleanly when the weights are absent and runs in CI against cached SwiftAcervo
// models. A clean skip locally is the expected result.

import Foundation
import Metal
import SwiftAcervo
import TestHelpers
import Testing

@testable import Flux2Core

/// The R2.3 no-spike ceiling: the requirement is "no ≥8 GB `phys_footprint`
/// spike", so the transformer-load footprint must be strictly under 8 GB.
private let noSpikeCeilingBytes: Int64 = 8 * 1_024 * 1_024 * 1_024

/// True only when a Metal device is present AND all three components needed to
/// drive the int4 transformer load — the themindstudio int4 transformer, the
/// VAE, and a Qwen3 text encoder — are on disk (cached / primed). Mirrors the
/// A8 gate: `Acervo.isModelAvailable` is synchronous local-only I/O.
func int4DirectLoadTestEnabled() -> Bool {
  guard MTLCreateSystemDefaultDevice() != nil else { return false }

  // Same crash-guard as A8: `Acervo.isModelAvailable` fatalErrors when neither
  // ACERVO_MODELS_DIR nor an App Group is configured (unentitled local runner).
  let env = ProcessInfo.processInfo.environment
  let hasModelsDir = (env[Acervo.modelsDirectoryOverrideVariable]?.isEmpty == false)
  let hasAppGroup = (env[Acervo.appGroupEnvironmentVariable]?.isEmpty == false)
  guard hasModelsDir || hasAppGroup else { return false }

  // Derive the int4 repoId from the registry so a rename can't silently desync
  // this gate: (klein4B, .int4) → .klein4B_4bit → themindstudio/…-mlx-4bit.
  let transformerRepoId =
    ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4).repoId
  let vaeRepoId = ModelRegistry.VAEVariant.standard.repoId
  let qwen3_8bit = "lmstudio-community/Qwen3-4B-MLX-8bit"
  let qwen3_4bit = "lmstudio-community/Qwen3-4B-MLX-4bit"

  return Acervo.isModelAvailable(transformerRepoId)
    && Acervo.isModelAvailable(vaeRepoId)
    && (Acervo.isModelAvailable(qwen3_8bit) || Acervo.isModelAvailable(qwen3_4bit))
}

@Suite("int4 Direct Load — no phys_footprint spike (GPU)", .serialized)
struct Int4DirectLoadGPUTests {

  /// Loading Klein 4B int4 via the direct pre-quantized path must keep the
  /// transformer `weightLoadComplete` `phys_footprint` under the 8 GB ceiling.
  @Test(
    "Klein 4B int4 direct load → transformer phys_footprint < 8 GB (no spike)",
    .enabled(if: int4DirectLoadTestEnabled()),
    .timeLimit(.minutes(10))
  )
  func int4DirectLoadHasNoPhysFootprintSpike() async throws {
    // ultraMinimal = { textEncoder: .mlx4bit, transformer: .int4 } — the int4
    // path. Prove the variant resolution + pre-quantized flag before spending
    // inference time (belt-and-suspenders with the pure-logic core tests).
    let quantization = Flux2QuantizationConfig.ultraMinimal
    #expect(quantization.transformer == .int4)
    let variant = ModelRegistry.TransformerVariant.variant(
      for: .klein4B, quantization: quantization.transformer)
    #expect(variant == .klein4B_4bit)
    #expect(variant.isPreQuantizedMLX, "int4 path must take the direct pre-quantized branch")

    // Capture telemetry so we can read the transformer weightLoadComplete
    // phys_footprint the direct-load path emits (via loadQuantizedTransformer).
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: quantization)
    pipeline.setTelemetry(reporter)

    try await pipeline.loadModels()

    // The transformer is loaded lazily during generation; a tiny generation
    // triggers loadTransformer() → the direct pre-quantized load path.
    _ = try await pipeline.generateTextToImage(
      prompt: "a single red apple on a white table",
      height: 512,
      width: 512,
      steps: 4,
      guidance: 1.0,
      seed: 7
    )

    // Find the transformer weightLoadComplete event and assert no spike.
    let events = await reporter.snapshot()
    var transformerFootprint: Int64?
    for event in events {
      if case .weightLoadComplete(let component, _, _, let physFootprint) = event,
        component == .transformer
      {
        transformerFootprint = physFootprint
      }
    }

    #expect(
      transformerFootprint != nil,
      "expected a transformer weightLoadComplete event carrying phys_footprint")

    if let footprint = transformerFootprint {
      #expect(
        footprint < noSpikeCeilingBytes,
        "int4 direct load spiked phys_footprint to \(footprint) bytes (≥ 8 GB ceiling \(noSpikeCeilingBytes)); the direct path must not materialize a bf16 intermediate"
      )
    }
  }
}
