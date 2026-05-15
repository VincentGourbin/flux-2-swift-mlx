// Flux2Telemetry.swift - Process-wide telemetry seam for CLI hosts
// Copyright 2025 intrusive-memory

import os.lock

/// Process-wide telemetry namespace for `flux-2-swift-mlx`.
///
/// CLI hosts (and other process-level observers) that cannot hold a reference
/// to every individual `Flux2Pipeline` instance use this seam instead:
///
/// ```swift
/// // At process startup — once, in e.g. CLITelemetryBootstrap.enable():
/// Flux2Telemetry.setReporter(myAdapter)
///
/// // At shutdown / teardown:
/// Flux2Telemetry.setReporter(nil)
/// ```
///
/// Every `Flux2Pipeline`, text encoder, transformer, scheduler, weight loader,
/// and LoRA manager consults `Flux2Telemetry.current` as a fallback when no
/// instance-level reporter has been installed via `setTelemetry(_:)`. The
/// **instance reporter always takes precedence**: if a caller installs both an
/// instance reporter and a process-wide reporter, the instance reporter wins.
///
/// ## Thread safety
///
/// `setReporter` and `current` are protected by an `OSAllocatedUnfairLock`,
/// the same primitive used by every instance-level seam in this library.
/// Both are safe to call from any thread or Swift concurrency context.
///
/// ## Reset semantics
///
/// Calling `Flux2Telemetry.setReporter(nil)` returns the process to the
/// "no reporter" state. Emission sites check `effectiveReporter` (instance ??
/// process-wide) at each call; after the process-wide reporter is cleared,
/// instances with no instance reporter silently drop events.
public enum Flux2Telemetry {

  private static let _lock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(
    initialState: nil)

  /// Install (or remove) the process-wide reporter.
  ///
  /// - Parameter reporter: The reporter to install, or `nil` to clear.
  public static func setReporter(_ reporter: (any Flux2TelemetryReporter)?) {
    _lock.withLock { $0 = reporter }
  }

  /// The currently installed process-wide reporter, or `nil` if none is set.
  public static var current: (any Flux2TelemetryReporter)? {
    _lock.withLock { $0 }
  }
}
