/**
 * DefaultsCommands.swift
 * FLUX.2 application menu — Defaults window.
 */

import SwiftUI

struct DefaultsCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(after: .appSettings) {
            Button("Defaults...") {
                openWindow(id: "image-save-defaults")
            }
        }
    }
}
