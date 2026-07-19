// BeaconOptions.swift — shared --beacon opt-in for CLI commands
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Flux2Core

/// Shared `--beacon` flag, pulled into commands via `@OptionGroup` so the
/// flag, its help text, and its activation live in exactly one place.
struct BeaconOptions: ParsableArguments {
    @Flag(name: .long, help: "Advertise activity to external monitors (writes a transient manifest in ~/Library/Application Support/ai-runtime-beacons/)")
    var beacon: Bool = false

    /// Call at the top of the command's run().
    func activate() {
        RuntimeBeacon.isEnabled = beacon
    }
}
