// Training.swift - LoRA Training Module for Flux.2
// Copyright 2025 Vincent Gourbin

// This file serves as the module entry point for training functionality.

// MARK: - Configuration
// LoRATrainingConfig - Main configuration for training
// TrainingQuantization - Quantization options (bf16, int8, int4, nf4)
// LoRATargetLayers - Which layers to apply LoRA to
// SimpleLoRAConfig - Simplified config for Ostris-compatible training

// MARK: - Data Pipeline
// TrainingDataset - Load and iterate over training data
// CaptionParser - Parse captions from txt/jsonl files
// LatentCache - Pre-encode and cache VAE latents
// AspectRatioBucket - Handle multiple resolutions
// CachedLatentEntry - Pre-encoded latent with metadata
// CachedEmbeddingEntry - Pre-encoded text embedding

// MARK: - Model
// LoRALinear - LoRA layer implementation (inference)
// LoRAInjectedLinear - LoRA layer for training with separate A/B matrices

// MARK: - Training Loop
// SimpleLoRATrainer - Ostris-compatible training loop (no EMA, clean implementation)

// MARK: - Training Control (High-Level APIs)
// TrainingController - Control pause/resume/stop with file-based signaling
// TrainingSession - High-level session management for app integration
// TrainingState - Persistable training state for checkpoint/resume
// TrainingObserver - Protocol for observing training events
// TrainingStatus - Enum for training status (idle, running, paused, etc.)

// MARK: - Optimizer
// ResumableAdamW - AdamW with checkpoint state support

import Foundation

/// LoRA Training Module Version
public enum Training {
    public static let version = "0.3.0"

    /// Supported features
    public static let features: [String] = [
        // Core Training
        "LoRA training for Flux.2 models",
        "Support for Klein 4B and Klein 9B models",
        "Ostris-compatible training (no EMA)",
        "Differential Output Preservation (DOP)",
        "Gradient accumulation",
        "Bell-shaped loss weighting",
        "Content/Style/Balanced timestep sampling",
        "Multi-resolution bucketing",
        "Latent and text embedding caching",
        // Training Control
        "Checkpoint resume with optimizer state",
        "Pause/Resume with automatic checkpoint",
        "File-based cross-process control (.pause, .stop files)",
        "High-level TrainingSession API for app integration",
        "TrainingObserver protocol for UI updates"
    ]

    /// Minimum recommended memory by model and quantization
    public static func recommendedMemoryGB(
        for model: Flux2Model,
        quantization: TrainingQuantization
    ) -> Int {
        switch (model, quantization) {
        case (.klein4B, .nf4), (.klein4BBase, .nf4): return 8
        case (.klein4B, .int4), (.klein4BBase, .int4): return 8
        case (.klein4B, .int8), (.klein4BBase, .int8): return 12
        case (.klein4B, .bf16), (.klein4BBase, .bf16): return 16
        case (.klein9B, .nf4), (.klein9BBase, .nf4): return 12
        case (.klein9B, .int4), (.klein9BBase, .int4): return 12
        case (.klein9B, .int8), (.klein9BBase, .int8): return 16
        case (.klein9B, .bf16), (.klein9BBase, .bf16): return 24
        case (.dev, .nf4): return 18
        case (.dev, .int4): return 18
        case (.dev, .int8): return 24
        case (.dev, .bf16): return 48
        }
    }
}

