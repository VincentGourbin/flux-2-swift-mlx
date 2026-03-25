// ResumableOptimizer.swift - Protocol for optimizers with checkpoint support
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Protocol for optimizers that support checkpointing (save/restore state)
/// Used to abstract over ResumableAdamW and ResumableLion in the training loop
/// Conforms to Updatable for use with MLX.compile() and eval()
public protocol ResumableOptimizer: AnyObject, Updatable {
    /// Current learning rate
    var learningRate: Float { get set }

    /// Current training step count
    var step: Int { get }

    /// Update model parameters using gradients
    func update(model: Module, gradients: ModuleParameters)

    /// Save optimizer state to a dictionary of arrays (for checkpointing)
    func saveState() -> [String: MLXArray]

    /// Restore optimizer state from a dictionary of arrays
    func restoreState(from state: [String: MLXArray]) throws
}
