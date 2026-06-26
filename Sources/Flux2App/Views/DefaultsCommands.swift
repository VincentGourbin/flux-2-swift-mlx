/**
 * DefaultsCommands.swift
 * FLUX.2 application menu — Defaults window.
 */

import SwiftUI

struct DefaultsCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(after: .appSettings) {
            Button("Defaults...", systemImage: "slider.horizontal.3") {
                openWindow(id: "image-save-defaults")
            }
        }
    }
}
