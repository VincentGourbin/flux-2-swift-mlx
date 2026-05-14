import Tuberia

// Chosen location: separate file (option a) — keeps anomaly classification
// logic decoupled from the event-type declaration in Flux2TelemetryEvent.swift.

/// Maps a `TuberiaTensorStat` to an optional `Flux2TelemetryEvent.AnomalyKind`.
/// Returns `nil` if the stat looks numerically healthy.
///
/// Used alongside `textEncodeComplete` / `denoiseLoopEnd` / `vaeDecodeComplete`
/// emits to fire `.numericalAnomaly` side-channel events when the underlying
/// tensor shows distress. Logic follows the GLASS PIPES pattern (pixart
/// reference: `PixArtDiT.anomalyKind(for:)`).
public enum AnomalyCheck {
  /// Classify a tensor stat into an anomaly kind, or return `nil` if healthy.
  ///
  /// Priority order: NaN takes precedence over Inf, which takes precedence
  /// over out-of-range, which takes precedence over zero-latent.
  public static func classify(_ stat: TuberiaTensorStat) -> Flux2TelemetryEvent.AnomalyKind? {
    if stat.hasNaN { return .nan }
    if stat.hasInf { return .inf }
    if abs(stat.max) > TuberiaTensorStat.defaultOutOfRangeThreshold { return .outOfRange }
    if abs(stat.mean) < 1e-6 && stat.std < 1e-6 { return .zeroLatent }
    return nil
  }
}
