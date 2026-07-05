import SwiftUI

struct GenerationDocumentHistoryCommands {
    let stepBack: () -> Void
    let stepForward: () -> Void
    let canStepBack: () -> Bool
    let canStepForward: () -> Bool
}

private struct GenerationDocumentHistoryCommandsKey: FocusedValueKey {
    typealias Value = GenerationDocumentHistoryCommands
}

extension FocusedValues {
    var generationDocumentHistory: GenerationDocumentHistoryCommands? {
        get { self[GenerationDocumentHistoryCommandsKey.self] }
        set { self[GenerationDocumentHistoryCommandsKey.self] = newValue }
    }
}

struct DocumentHistoryCommands: Commands {
    @FocusedValue(\.generationDocumentHistory) private var documentHistory

    var body: some Commands {
        CommandMenu("History") {
            Button("Step Back") {
                documentHistory?.stepBack()
            }
            .keyboardShortcut("z", modifiers: [.control, .command])
            .disabled(documentHistory?.canStepBack() != true)

            Button("Step Forward") {
                documentHistory?.stepForward()
            }
            .keyboardShortcut("z", modifiers: [.control, .command, .shift])
            .disabled(documentHistory?.canStepForward() != true)
        }
    }
}
