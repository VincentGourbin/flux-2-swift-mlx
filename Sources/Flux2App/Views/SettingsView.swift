/**
 * SettingsView.swift
 * App Settings tab. Extracted from ContentView.swift.
 */

import SwiftUI
import AppKit
import FluxTextEncoders
import Flux2Core
import MLX

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("hfToken") private var hfToken = ""
    @AppStorage("imageSaveOutputRoot") private var imageSaveOutputRoot = ImageSaveService.defaultOutputRoot
    @AppStorage(ImageSavePreferenceKeys.preset) private var imageSavePreset = ImageSaveOutputRootPresetStore.factoryPresetName
    @AppStorage(ImageSavePreferenceKeys.outputRootPresetsJSON) private var outputRootPresetsJSON = ""
    @AppStorage("imageSaveFormat") private var imageSaveFormat = ImageSaveFormat.png24.rawValue
    @AppStorage("imageSaveUpscaleBy") private var imageSaveUpscaleBy = 1.0

    @State private var defaultOutputDraft = ImageSaveService.defaultOutputRoot

    private var outputRootPresets: [ImageSaveOutputRootPreset] {
        ImageSaveOutputRootPresetStore.decode(outputRootPresetsJSON)
    }

    var body: some View {
        Form {
            Section("HuggingFace") {
                SecureField("HF Token", text: $hfToken)
                    .textFieldStyle(.roundedBorder)
                Text("Required for gated models (FLUX.2 Dev bf16, Klein 9B). Create a read token at huggingface.co/settings/tokens, then accept each model’s license on its model page.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section {
                Divider()

                HStack {
                    Text("Variant: \(modelManager.selectedVariant?.rawValue ?? "None")")
                    Spacer(minLength: 12)
                    Text("Status: \(modelManager.isLoaded ? "Loaded" : "Not Loaded")")
                }

                Divider()
            }

            Section {
                LabeledContent("Format") {
                    Picker("Format", selection: $imageSaveFormat) {
                        ForEach(ImageSaveFormat.supportedCases) { format in
                            Text(format.rawValue).tag(format.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 280, alignment: .leading)
                }

                LabeledContent("Lanczos upscale") {
                    Stepper(
                        String(format: "%.2f×", imageSaveUpscaleBy),
                        value: $imageSaveUpscaleBy,
                        in: 1.0...8.0,
                        step: 0.25
                    )
                    .frame(maxWidth: 280, alignment: .leading)
                }

                Text("1× saves at native size. Path and filename defaults live in the Output palette and Defaults dialog.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Divider()

                LabeledContent("Default Folder") {
                    HStack(spacing: 8) {
                        TextField("", text: $defaultOutputDraft)
                            .textFieldStyle(.roundedBorder)
                            .labelsHidden()
                            .frame(maxWidth: .infinity)

                        Button("Choose…") {
                            chooseDefaultOutput()
                        }

                        Button("Set Default") {
                            setDefaultOutput()
                        }
                    }
                }

                LabeledContent("Presets") {
                    HStack(alignment: .center, spacing: 8) {
                        Picker("Presets", selection: $imageSavePreset) {
                            ForEach(outputRootPresets) { preset in
                                Text(preset.name).tag(preset.name)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: .infinity, alignment: .leading)

                        Button("Add Path…") {
                            addOutputRootPreset()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)

                        Button("Remove") {
                            removeOutputRootPreset()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                        .disabled(!outputRootPresets.contains(where: { $0.name == imageSavePreset }))

                        Button("Show in Finder") {
                            try? ImageSaveService.revealOutputDirectoryInFinder()
                        }
                    }
                }
            }
        }
        .padding()
        .frame(width: 640, height: 420)
        .onAppear {
            if ImageSaveFormat(rawValue: imageSaveFormat)?.isSupported != true {
                imageSaveFormat = ImageSaveFormat.png24.rawValue
            }
            ImageSaveOutputRootPresetStore.bootstrapIfNeeded()
            defaultOutputDraft = imageSaveOutputRoot
            ensureValidPresetSelection()
        }
    }

    private func ensureValidPresetSelection() {
        guard !outputRootPresets.contains(where: { $0.name == imageSavePreset }),
              let first = outputRootPresets.first else {
            return
        }
        imageSavePreset = first.name
    }

    private func setDefaultOutput() {
        imageSaveOutputRoot = ImageSaveOutputRootPresetStore.normalize(defaultOutputDraft)
        defaultOutputDraft = imageSaveOutputRoot
    }

    private func addOutputRootPreset() {
        guard let result = ImageSaveAddPathPanel.run() else { return }

        if outputRootPresets.contains(where: { $0.name == result.name }) {
            let alert = NSAlert()
            alert.messageText = "Preset name already exists"
            alert.informativeText = "Choose a different name for this path."
            alert.runModal()
            return
        }

        _ = ImageSaveOutputRootPresetStore.addPreset(name: result.name, path: result.path)
        imageSavePreset = result.name
    }

    private func removeOutputRootPreset() {
        _ = ImageSaveOutputRootPresetStore.removePreset(named: imageSavePreset)
        ensureValidPresetSelection()
    }

    private func chooseDefaultOutput() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true

        if panel.runModal() == .OK, let url = panel.url {
            defaultOutputDraft = ImageSaveOutputRootPresetStore.normalize(url.path)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ModelManager())
}
