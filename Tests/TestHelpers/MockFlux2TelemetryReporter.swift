import Flux2Core

/// Test-only reporter that captures every Flux2TelemetryEvent into an
/// append-only log. Not part of the public API.
///
/// The append-only log is implemented as an actor so concurrent capture
/// calls from the fire-and-forget Task in `Flux2Pipeline.init` and other
/// emission sites are serialized without data races.
///
/// ## Waiting for fire-and-forget dispatch
///
/// `pipelineInit` is dispatched via a single unstructured `Task { ... }` at
/// the end of `Flux2Pipeline.init`. After the sync init returns the Task may
/// not yet have delivered its event to this actor. Tests use a fixed sleep
/// before snapshotting the log:
///
///     try await Task.sleep(for: .milliseconds(100))
///     let events = await reporter.snapshot()
///
/// Bump to 250 ms if flakiness is observed on slow CI hardware.
public actor MockFlux2TelemetryReporter: Flux2TelemetryReporter {
  private(set) var events: [Flux2TelemetryEvent] = []

  public init() {}

  public func capture(_ event: Flux2TelemetryEvent) async {
    events.append(event)
  }

  public func snapshot() async -> [Flux2TelemetryEvent] {
    events
  }

  public func clear() async {
    events.removeAll()
  }
}
