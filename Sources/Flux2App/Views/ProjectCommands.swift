/**
 * ProjectCommands.swift
 * macOS File menu actions for generation projects
 */

import SwiftUI

struct GenerationProjectCommands {
    let newProject: () -> Void
    let openProject: () -> Void
    let saveProject: () -> Void
    let saveProjectAs: () -> Void
}

private struct GenerationProjectCommandsKey: FocusedValueKey {
    typealias Value = GenerationProjectCommands
}

private struct GenerationProjectNameKey: FocusedValueKey {
    typealias Value = String
}

extension FocusedValues {
    var generationProjectCommands: GenerationProjectCommands? {
        get { self[GenerationProjectCommandsKey.self] }
        set { self[GenerationProjectCommandsKey.self] = newValue }
    }

    /// Display name of the active generation project, surfaced to the window
    /// title. Provided by whichever image view currently holds scene focus.
    var generationProjectName: String? {
        get { self[GenerationProjectNameKey.self] }
        set { self[GenerationProjectNameKey.self] = newValue }
    }
}

struct ProjectFileCommands: Commands {
    @FocusedValue(\.generationProjectCommands) private var projectCommands

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Project") {
                projectCommands?.newProject()
            }
            .keyboardShortcut("n", modifiers: [.command])
            .disabled(projectCommands == nil)

            Button("Open Project...") {
                projectCommands?.openProject()
            }
            .keyboardShortcut("o", modifiers: [.command])
            .disabled(projectCommands == nil)

            Divider()

            Button("Save Project") {
                projectCommands?.saveProject()
            }
            .keyboardShortcut("s", modifiers: [.command])
            .disabled(projectCommands == nil)

            Button("Save Project As...") {
                projectCommands?.saveProjectAs()
            }
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .disabled(projectCommands == nil)
        }
    }
}
