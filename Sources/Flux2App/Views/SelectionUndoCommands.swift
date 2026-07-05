/**
 * SelectionUndoCommands.swift
 * Edit menu undo/redo for I2I mask selections.
 */

import SwiftUI

struct GenerationSelectionUndoCommands {
    let undo: () -> Void
    let redo: () -> Void
    let canUndo: () -> Bool
    let canRedo: () -> Bool
}

private struct GenerationSelectionUndoCommandsKey: FocusedValueKey {
    typealias Value = GenerationSelectionUndoCommands
}

extension FocusedValues {
    var generationSelectionUndo: GenerationSelectionUndoCommands? {
        get { self[GenerationSelectionUndoCommandsKey.self] }
        set { self[GenerationSelectionUndoCommandsKey.self] = newValue }
    }
}

struct SelectionUndoCommands: Commands {
    @FocusedValue(\.generationSelectionUndo) private var selectionUndo

    var body: some Commands {
        CommandGroup(replacing: .undoRedo) {
            Button("Undo Selection") {
                selectionUndo?.undo()
            }
            .keyboardShortcut("z", modifiers: [.command])
            .disabled(selectionUndo?.canUndo() != true)

            Button("Redo Selection") {
                selectionUndo?.redo()
            }
            .keyboardShortcut("z", modifiers: [.command, .shift])
            .disabled(selectionUndo?.canRedo() != true)
        }
    }
}
