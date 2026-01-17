// Flux2App.swift - SwiftUI Demo Application
// Copyright 2025 Vincent Gourbin

import SwiftUI
import Flux2Core

@main
struct Flux2App: App {
    @StateObject private var modelManager = ModelViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
        }
        .windowStyle(.automatic)
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("Settings...") {
                    NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
            }
        }

        Settings {
            SettingsView()
                .environmentObject(modelManager)
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            GenerationView()
                .tabItem {
                    Label("Generate", systemImage: "wand.and.stars")
                }
                .tag(0)

            ModelManagerView()
                .tabItem {
                    Label("Models", systemImage: "arrow.down.circle")
                }
                .tag(1)
        }
        .frame(minWidth: 900, minHeight: 600)
    }
}

#Preview {
    ContentView()
        .environmentObject(ModelViewModel())
}
