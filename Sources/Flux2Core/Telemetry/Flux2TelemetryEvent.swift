import Foundation
@preconcurrency import MLX
import Tuberia  // for TuberiaTensorStat

/// Slim boundary-only telemetry surface for the Flux.2 image-generation pipeline.
///
/// Follows the cross-library chokepoint convention documented in
/// `AGENTS.md §11`: instrument boundaries, not internals. One event when each
/// major phase starts, one when it ends, plus a single anomaly signal at each
/// phase exit. Per-step / per-block / per-attention-head detail is **deferred**
/// — a `numericalAnomaly` or `errorThrown` here points the agent at the
/// region; finer instrumentation is added in a follow-up iteration only after
/// a real failure demands it.
///
/// The full event surface and emission spec live in
/// `REQUIREMENTS-instrumentation.md` §3.1 (the authoritative source).
public enum Flux2TelemetryEvent: Sendable {

  // --- Pipeline lifecycle ---
  // pipelineInit fires from Flux2Pipeline.init (sync), so hosts must call
  // setTelemetry() before the first generation to avoid losing this event.
  // pipelineDispose fires from an explicit `public func dispose() async`,
  // NOT deinit (deinit can't be async).
  case pipelineInit(model: String, quantization: String, vaeConfig: String)
  case pipelineDispose

  // --- Weight loading (one event per component, on success) ---
  case weightLoadComplete(component: WeightComponent, paramCount: Int, durationSeconds: Double)

  // --- Text encoding (boundary event with NaN/Inf check on the embedding) ---
  case textEncodeComplete(
    encoderName: String, finalPromptLength: Int, embeddingStat: TuberiaTensorStat,
    durationSeconds: Double)

  // --- Scheduler ---
  case schedulerConfigured(numInferenceSteps: Int, shift: Float, imageSeqLen: Int, mu: Float)

  // --- Denoise loop (start + end only; per-step events deferred) ---
  case denoiseLoopStart(
    variant: DenoiseVariant, totalSteps: Int, latentShape: [Int], latentDtype: String)
  case denoiseLoopEnd(
    variant: DenoiseVariant, totalSteps: Int, completedSteps: Int,
    finalLatentStat: TuberiaTensorStat, durationSeconds: Double)

  // --- VAE decode (boundary event with pixel-range check) ---
  case vaeDecodeComplete(pixelStat: TuberiaTensorStat, outputDims: [Int], durationSeconds: Double)

  // --- Anomaly side-channel (fires alongside any of the *Complete / *End events
  //     whose stat shows NaN, Inf, or out-of-range magnitude). One signal, not
  //     a per-step stream. ---
  case numericalAnomaly(phase: AnomalyPhase, kind: AnomalyKind, stat: TuberiaTensorStat)

  // --- Cancellation ---
  case generationCancelled(stepIndex: Int?)  // nil for pre-loop cancellation sites

  // --- Error side-channel — fires immediately before every Flux2Error throw ---
  case errorThrown(phase: ErrorPhase, errorDescription: String)

  public enum WeightComponent: String, Sendable {
    case textEncoderKlein  // Qwen3 (KleinTextEncoder)
    case textEncoderDev  // Mistral (DevTextEncoder / Flux2TextEncoder)
    case textEncoderTraining
    case transformer
    case vae
    case lora
  }

  public enum DenoiseVariant: String, Sendable {
    case textToImage
    case imageToImageKVExtractStep0  // single non-loop call; emits Start+End with totalSteps:1, completedSteps:1
    case imageToImageKVCached
    case imageToImageFullRecompute
  }

  public enum AnomalyPhase: String, Sendable {
    case textEncode
    case denoiseLoopEnd
    case vaeDecode
  }

  public enum AnomalyKind: String, Sendable {
    case nan
    case inf
    case outOfRange  // |x| > TuberiaTensorStat.defaultOutOfRangeThreshold
    case zeroLatent  // mean ≈ 0 && std ≈ 0
  }

  public enum ErrorPhase: String, Sendable {
    case modelNotLoaded
    case invalidConfiguration
    case insufficientMemory
    case modelNotDownloaded
    case generationCancelled
    case generationFailed
    case weightLoadFailed
    case vaeDecodeFailed
    case textEncoderFailed
    case vlmInterpretFailed
    case loraLoadFailed
    case imageProcessingFailed
    case other
  }
}
