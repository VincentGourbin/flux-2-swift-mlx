/**
 * Flux2App.swift
 * SwiftUI Application for FLUX.2 (Text Encoders + Image Generation)
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core

@main
struct Flux2App: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var modelManager = ModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
                .frame(minWidth: 900, minHeight: 700)
        }
        .defaultSize(width: 1200, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(replacing: .appInfo) {
                Button("About FLUX.2") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            .applicationName: "FLUX.2 Swift",
                            .applicationVersion: "1.0",
                            .credits: NSAttributedString(string: "Text encoders + Image generation powered by MLX Swift\nMistral Small 3.2 & Qwen3 for text encoding\nFlux.2 Dev/Klein for diffusion")
                        ]
                    )
                }
            }
        }

        Settings {
            SettingsView()
                .environmentObject(modelManager)
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Make this a regular app that appears in the Dock
        NSApplication.shared.setActivationPolicy(.regular)

        // Bring to front
        NSApplication.shared.activate(ignoringOtherApps: true)

        // Make the first window key and front
        if let window = NSApplication.shared.windows.first {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
