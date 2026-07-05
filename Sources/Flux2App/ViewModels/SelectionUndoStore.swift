import CoreGraphics
import Flux2Chains
import Flux2Core
import Foundation

/// Serializable I2I selection state for session undo/redo (no rasters).
struct SelectionUndoSnapshot: Equatable {
    var inpaintMaskLayers: [InpaintMaskLayer]
    var processArea: CGRect?
    var draftPolygonPoints: [CGPoint]
    var draftLassoPoints: [CGPoint]
    var fillContextMaskScale: Double
    var inpaintIntent: Flux2InpaintIntent
    var enrichInpaintPromptWithVLM: Bool
}

/// Lightweight undo/redo stacks for generative-fill selections.
@MainActor
final class SelectionUndoStore {
    private(set) var undoStack: [SelectionUndoSnapshot] = []
    private(set) var redoStack: [SelectionUndoSnapshot] = []

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    func reset() {
        undoStack.removeAll()
        redoStack.removeAll()
    }

    func pushUndoPoint(_ snapshot: SelectionUndoSnapshot) {
        undoStack.append(snapshot)
        redoStack.removeAll()
    }

    func popUndo(current: SelectionUndoSnapshot) -> SelectionUndoSnapshot? {
        guard let previous = undoStack.popLast() else { return nil }
        redoStack.append(current)
        return previous
    }

    func popRedo(current: SelectionUndoSnapshot) -> SelectionUndoSnapshot? {
        guard let next = redoStack.popLast() else { return nil }
        undoStack.append(current)
        return next
    }
}
