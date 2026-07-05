/**
 * ContentView.swift
 * Main content view for Mistral App
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

struct ContentView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var chatViewModel = ChatViewModel()
    /// Open Image to Image when `F2SM_PROJECT` is set so smoke hooks run on launch.
    private static var initialSelectedTab: Int {
        if let path = ProcessInfo.processInfo.environment["F2SM_PROJECT"], !path.isEmpty {
            return 5
        }
        return Flux2AppSessionStore.loadShell()?.selectedTab ?? 0
    }
    @State private var selectedTab = Self.initialSelectedTab
    @StateObject private var imageToImageViewModel = ImageGenerationViewModel(
        loadsEnvironmentProject: true,
        workflow: .imageToImage
    )
    // Set by the focused image view; falls back to the app name on other tabs.
    @FocusedValue(\.generationProjectName) private var projectName

    var body: some View {
        NavigationSplitView {
            List(selection: Binding(
                get: { selectedTab },
                set: { selectedTab = $0 }
            )) {
                Section("Mode") {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                        .tag(0)
                    Label("Generate", systemImage: "text.cursor")
                        .tag(1)
                    Label("Vision", systemImage: "eye")
                        .tag(2)
                    Label("Qwen3 Chat", systemImage: "message.fill")
                        .tag(3)
                        .foregroundStyle(.orange)
                    Label("Text to Image", systemImage: "photo.badge.plus")
                        .tag(4)
                        .foregroundStyle(.purple)
                    Label("Image to Image", systemImage: "photo.on.rectangle.angled")
                        .tag(5)
                        .foregroundStyle(.purple)
                    Label("FLUX.2 Tools", systemImage: "cube.transparent")
                        .tag(6)
                    Label("Models", systemImage: "square.stack.3d.down.right")
                        .tag(7)
                }

                if selectedTab == 5 {
                    EditHistorySidebarSection(
                        viewModel: imageToImageViewModel,
                        historyStore: imageToImageViewModel.editHistoryStore
                    )
                }
            }
            .listStyle(.sidebar)
            .frame(minWidth: 220, idealWidth: 240)

        } detail: {
            // Main content
            VStack(spacing: 0) {
                // Model status bar - contextual based on selected tab
                ModelStatusBar(selectedTab: selectedTab)
                    .environmentObject(modelManager)

                Divider()

                // Content based on selection
                Group {
                    switch selectedTab {
                    case 0:
                        ChatView(viewModel: chatViewModel)
                    case 1:
                        GenerateView()
                    case 2:
                        VisionView()
                    case 3:
                        Qwen3ChatView()
                    case 4:
                        TextToImageView()
                    case 5:
                        ImageToImageView(viewModel: imageToImageViewModel)
                    case 6:
                        FluxToolsView()
                    case 7:
                        ModelsManagementView()
                    default:
                        ChatView(viewModel: chatViewModel)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle(projectName ?? "FLUX.2 Text Encoders")
        .onChange(of: selectedTab) { _, tab in
            Flux2AppSessionStore.saveShell(selectedTab: tab)
        }
    }
}

