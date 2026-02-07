// MLXCheckpoint.swift - Wrapper for mlx_checkpoint C function
// Enables gradient checkpointing to reduce memory usage during training

import Cmlx
import MLX

/// Wraps a function with gradient checkpointing.
///
/// During the forward pass, intermediate activations are not stored.
/// During the backward pass, they are recomputed as needed.
/// This trades compute time for memory savings (~30-50% less memory).
///
/// Usage:
/// ```swift
/// let checkpointedBlock = checkpoint { inputs in
///     return myTransformerBlock(inputs)
/// }
/// let output = checkpointedBlock(inputs)
/// ```
///
/// - Parameter f: The function to wrap with checkpointing
/// - Returns: A checkpointed version of the function
public func checkpoint(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
    // Create C closure from Swift function
    let inputClosure = new_mlx_closure(f)

    // Create checkpointed closure
    var checkpointedClosure = mlx_closure_new()
    let status = mlx_checkpoint(&checkpointedClosure, inputClosure)

    // Free the input closure (checkpointed version has its own reference)
    mlx_closure_free(inputClosure)

    guard status == 0 else {
        fatalError("mlx_checkpoint failed with status \(status)")
    }

    // Return a Swift function that invokes the checkpointed closure
    return { (inputs: [MLXArray]) -> [MLXArray] in
        // Convert inputs to C vector
        let inputVec = new_mlx_vector_array(inputs)
        defer { mlx_vector_array_free(inputVec) }

        // Call the checkpointed closure
        var outputVec = mlx_vector_array_new()
        let applyStatus = mlx_closure_apply(&outputVec, checkpointedClosure, inputVec)
        defer { mlx_vector_array_free(outputVec) }

        guard applyStatus == 0 else {
            fatalError("mlx_closure_apply failed with status \(applyStatus)")
        }

        // Convert output back to Swift arrays
        return mlx_vector_array_values(outputVec)
    }
}

/// Convenience overload for single input/output
public func checkpoint(_ f: @escaping (MLXArray) -> MLXArray) -> (MLXArray) -> MLXArray {
    let wrapped = checkpoint { (arrays: [MLXArray]) -> [MLXArray] in
        return [f(arrays[0])]
    }
    return { input in wrapped([input])[0] }
}

// MARK: - Helper functions (copied from MLX internal, needed for C interop)

/// Create a new mlx_vector_array from Swift arrays
private func new_mlx_vector_array(_ arrays: [MLXArray]) -> mlx_vector_array {
    withExtendedLifetime(arrays) {
        mlx_vector_array_new_data(arrays.map { $0.ctx }, arrays.count)
    }
}

/// Convert mlx_vector_array to Swift array
private func mlx_vector_array_values(_ vector_array: mlx_vector_array) -> [MLXArray] {
    (0 ..< mlx_vector_array_size(vector_array))
        .map { index in
            var ctx = mlx_array_new()
            mlx_vector_array_get(&ctx, vector_array, index)
            return MLXArray(ctx)
        }
}

/// Create a mlx_closure from a Swift function
private func new_mlx_closure(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> mlx_closure {
    class ClosureCaptureState {
        let f: ([MLXArray]) -> [MLXArray]
        init(_ f: @escaping ([MLXArray]) -> [MLXArray]) {
            self.f = f
        }
    }

    func free(ptr: UnsafeMutableRawPointer?) {
        Unmanaged<ClosureCaptureState>.fromOpaque(ptr!).release()
    }

    let payload = Unmanaged.passRetained(ClosureCaptureState(f)).toOpaque()

    func trampoline(
        resultOut: UnsafeMutablePointer<mlx_vector_array>?,
        vector_array: mlx_vector_array,
        payload: UnsafeMutableRawPointer?
    ) -> Int32 {
        let state = Unmanaged<ClosureCaptureState>.fromOpaque(payload!).takeUnretainedValue()
        let arrays = mlx_vector_array_values(vector_array)
        let result = state.f(arrays)

        if let resultOut {
            resultOut.pointee = new_mlx_vector_array(result)
        } else {
            fatalError("no resultOut pointer")
        }
        return 0
    }

    return mlx_closure_new_func_payload(trampoline, payload, free)
}
