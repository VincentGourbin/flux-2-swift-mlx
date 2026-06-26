/**
 * ImageSaveAddPathPanel.swift
 * Folder picker with a required preset name field for Settings → Add Path…
 */

import AppKit

enum ImageSaveAddPathPanel {
    struct Result {
        let name: String
        let path: String
    }

    @MainActor
    static func run() -> Result? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Add"
        panel.message = "Choose a folder, then enter a preset name."

        let coordinator = Coordinator()
        panel.accessoryView = coordinator.accessoryView
        panel.isAccessoryViewDisclosed = true
        panel.delegate = coordinator

        guard panel.runModal() == .OK, let url = panel.url else { return nil }

        let name = coordinator.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        return Result(
            name: name,
            path: ImageSaveOutputRootPresetStore.normalize(url.path)
        )
    }

    @MainActor
    private final class Coordinator: NSObject, NSOpenSavePanelDelegate {
        let nameField: NSTextField = {
            let field = NSTextField(string: "")
            field.placeholderString = "Required"
            return field
        }()

        lazy var accessoryView: NSView = {
            let container = NSView(frame: NSRect(x: 0, y: 0, width: 420, height: 36))
            let label = NSTextField(labelWithString: "name")
            label.font = .systemFont(ofSize: NSFont.systemFontSize)
            label.translatesAutoresizingMaskIntoConstraints = false
            nameField.translatesAutoresizingMaskIntoConstraints = false
            container.addSubview(label)
            container.addSubview(nameField)
            NSLayoutConstraint.activate([
                label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 12),
                label.centerYAnchor.constraint(equalTo: container.centerYAnchor),
                nameField.leadingAnchor.constraint(equalTo: label.trailingAnchor, constant: 8),
                nameField.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -12),
                nameField.centerYAnchor.constraint(equalTo: container.centerYAnchor),
                nameField.widthAnchor.constraint(greaterThanOrEqualToConstant: 220),
            ])
            return container
        }()

        var name: String { nameField.stringValue }

        func panel(_ sender: Any, validate url: URL) throws -> Bool {
            !nameField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }
}
