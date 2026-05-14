/// Sink protocol for `Flux2TelemetryEvent`s.
///
/// Hosts implement this to receive boundary-only telemetry from
/// `Flux2Pipeline` and its owned subcomponents. The reporter is injected via
/// `Flux2Pipeline.setTelemetry(_:)` (cross-library convention; see
/// `AGENTS.md §11.5`); the pipeline propagates it to every owned subcomponent.
///
/// The protocol is intentionally minimal — every event is delivered through
/// the same `capture(_:)` entry point so adapters can switch exhaustively over
/// the event enum without `default:` arms.
public protocol Flux2TelemetryReporter: Sendable {
  func capture(_ event: Flux2TelemetryEvent) async
}

/// No-op reporter used by tests and by hosts that want to confirm the
/// telemetry path is wired without actually consuming any events. Pairs with
/// `Flux2TelemetryNoopOverheadTests` to assert the boundary emit path costs
/// near-zero wall-clock time when the reporter does nothing.
public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter {
  public init() {}
  public func capture(_ event: Flux2TelemetryEvent) async {}
}
