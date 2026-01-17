// FlowMatchEulerScheduler.swift - Flow Matching Euler Scheduler
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX

/// Compute empirical mu for Flux.2 time shifting
/// Ported from diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2.py
public func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
    let a1: Float = 8.73809524e-05
    let b1: Float = 1.89833333
    let a2: Float = 0.00016927
    let b2: Float = 0.45666666

    if imageSeqLen > 4300 {
        let mu = a2 * Float(imageSeqLen) + b2
        return mu
    }

    let m_200 = a2 * Float(imageSeqLen) + b2
    let m_10 = a1 * Float(imageSeqLen) + b1

    let a = (m_200 - m_10) / 190.0
    let b = m_200 - 200.0 * a
    let mu = a * Float(numSteps) + b

    return mu
}

/// Flow Matching Euler Discrete Scheduler for Flux.2
///
/// Implements the flow matching ODE solver with Euler steps.
/// Based on diffusers FlowMatchEulerDiscreteScheduler
public class FlowMatchEulerScheduler: @unchecked Sendable {
    /// Number of training timesteps
    public let numTrainTimesteps: Int

    /// Shift parameter for timestep scheduling
    public let shift: Float

    /// Current inference timesteps (in [0, 1000] range for compatibility)
    public private(set) var timesteps: [Float] = []

    /// Current sigmas (noise levels in [0, 1] range)
    public private(set) var sigmas: [Float] = []

    /// Step index during sampling
    public private(set) var stepIndex: Int = 0

    public init(
        numTrainTimesteps: Int = 1000,
        shift: Float = 1.0
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
    }

    /// Set timesteps for inference with Flux.2 specific scheduling
    /// - Parameters:
    ///   - numInferenceSteps: Number of denoising steps
    ///   - imageSeqLen: Length of image sequence (for mu calculation)
    public func setTimesteps(numInferenceSteps: Int, imageSeqLen: Int? = nil) {
        // Compute mu based on image sequence length (Flux.2 specific)
        let mu: Float
        if let seqLen = imageSeqLen {
            mu = computeEmpiricalMu(imageSeqLen: seqLen, numSteps: numInferenceSteps)
        } else {
            // Default mu for 1024x1024 (4096 latent patches)
            mu = computeEmpiricalMu(imageSeqLen: 4096, numSteps: numInferenceSteps)
        }

        // Generate sigmas: linspace(1.0, 1/num_steps, num_steps)
        // This is the Flux.2 specific sigma schedule
        var sigmaValues: [Float] = []
        for i in 0..<numInferenceSteps {
            let sigma = 1.0 - Float(i) / Float(numInferenceSteps)
            sigmaValues.append(sigma)
        }

        // Apply time shifting with mu (exponential shift)
        sigmaValues = sigmaValues.map { sigma in
            timeShift(mu: mu, sigma: 1.0, t: sigma)
        }

        // Append terminal sigma (0)
        sigmaValues.append(0.0)

        self.sigmas = sigmaValues

        // Timesteps = sigmas * num_train_timesteps (for compatibility)
        self.timesteps = sigmaValues.map { $0 * Float(numTrainTimesteps) }

        self.stepIndex = 0

        Flux2Debug.log("Scheduler set: \(numInferenceSteps) steps, mu=\(mu)")
        Flux2Debug.verbose("Sigmas: \(sigmas.prefix(5))... to \(sigmas.suffix(2))")
    }

    /// Time shift function (exponential) - matches diffusers _time_shift_exponential
    private func timeShift(mu: Float, sigma: Float, t: Float) -> Float {
        // exponential time shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
        let expMu = exp(mu)
        let denominator = expMu + pow(1.0 / t - 1.0, sigma)
        return expMu / denominator
    }

    /// Perform one Euler step
    /// - Parameters:
    ///   - modelOutput: Predicted velocity from transformer
    ///   - timestep: Current timestep (not used, we track via stepIndex)
    ///   - sample: Current noisy sample
    /// - Returns: Updated sample for next step
    public func step(
        modelOutput: MLXArray,
        timestep: Float,
        sample: MLXArray
    ) -> MLXArray {
        guard stepIndex < sigmas.count - 1 else {
            return sample
        }

        let sigma = sigmas[stepIndex]
        let sigmaNext = sigmas[stepIndex + 1]

        // Euler step: x_{t-1} = x_t + (sigma_next - sigma) * v_t
        // where v_t is the velocity (model output)
        let dt = sigmaNext - sigma
        let nextSample = sample + dt * modelOutput

        stepIndex += 1

        return nextSample
    }

    /// Get the current sigma for this step
    public var currentSigma: Float {
        guard stepIndex < sigmas.count else { return 0 }
        return sigmas[stepIndex]
    }

    /// Get initial noise scale
    public func initNoiseSigma(for timestep: Float) -> Float {
        // For flow matching starting at t=1, the initial noise scale is 1
        return 1.0
    }

    /// Scale model input (identity for flow matching)
    public func scaleModelInput(_ sample: MLXArray, timestep: Float) -> MLXArray {
        sample
    }

    /// Add noise to latents for a given timestep
    public func addNoise(
        originalSamples: MLXArray,
        noise: MLXArray,
        timestep: Float
    ) -> MLXArray {
        // For flow matching: x_t = (1 - t) * x_0 + t * noise
        // where t is sigma (in [0, 1] range)
        let sigma = timestep / Float(numTrainTimesteps)
        let t = MLXArray(sigma)
        return (1 - t) * originalSamples + t * noise
    }

    /// Get velocity target for training
    public func getVelocity(
        sample: MLXArray,
        noise: MLXArray,
        timestep: Float
    ) -> MLXArray {
        // Velocity target: v = noise - sample
        noise - sample
    }

    /// Reset scheduler state
    public func reset() {
        stepIndex = 0
    }
}

// MARK: - Progress Tracking

extension FlowMatchEulerScheduler {
    /// Current progress (0.0 to 1.0)
    public var progress: Float {
        guard !timesteps.isEmpty else { return 0 }
        return Float(stepIndex) / Float(timesteps.count - 1)
    }

    /// Remaining steps
    public var remainingSteps: Int {
        max(0, timesteps.count - 1 - stepIndex)
    }
}
