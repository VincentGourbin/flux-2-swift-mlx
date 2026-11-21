---
name: flux2-swift-mlx
description: Use when generating images with Flux.2 on Apple Silicon, working with MLX Swift, implementing text-to-image/image-to-image pipelines, LoRA training, or quantized model inference in Swift.
---

# Flux.2 Swift MLX

Native Swift implementation of Flux.2 image generation for Apple Silicon using MLX.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/VincentGourbin/flux-2-swift-mlx.git", from: "3.2.0")
]

// Add to your target dependencies
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "Flux2Core", package: "flux-2-swift-mlx"),
        .product(name: "FluxTextEncoders", package: "flux-2-swift-mlx"),
    ]
)
```

Requires macOS 15+ and Apple Silicon (M1/M2/M3/M4). Build with Xcode 26+ (not `swift build`) for Metal GPU support; command-line builds need `xcodebuild ... -skipPackagePluginValidation`. mlx-swift is pinned `exact: "0.33.0"`.

## Quick Start

```swift
import Flux2Core

// Create pipeline (Klein 4B for fast generation)
let pipeline = Flux2Pipeline(model: .klein4B)
try await pipeline.loadModels()

// Generate image
let image = try await pipeline.generateTextToImage(
    prompt: "a cat sitting on a chair",
    height: 1024,
    width: 1024,
    steps: 4,        // Klein uses 4 steps
    guidance: 1.0    // Klein uses guidance 1.0
)
```

## Model Selection

| Model | Steps | Guidance | Transformer (qint8) | Speed (1024x1024) | License |
|-------|-------|----------|---------------------|-------------------|---------|
| Klein 4B | 4 | 1.0 | ~4 GB | ~28s | Apache 2.0 |
| Klein 9B | 4 | 1.0 | ~9 GB | ~60s | Non-commercial |
| Dev (32B) | 28 | 4.0 | ~33 GB | ~30 min | Non-commercial |

## Quantization (On-the-fly)

All models support on-the-fly quantization to reduce memory usage:

```swift
import Flux2Core

// Quantization presets
let balanced = Flux2QuantizationConfig.balanced          // 8bit text + qint8 transformer
let memoryEfficient = Flux2QuantizationConfig.memoryEfficient  // 4bit text + qint8 transformer
let ultraMinimal = Flux2QuantizationConfig.ultraMinimal  // 4bit text + int4 transformer

// Custom quantization
let custom = Flux2QuantizationConfig(
    textEncoder: .mlx4bit,   // .bf16, .mlx8bit, .mlx6bit, .mlx4bit
    transformer: .int4       // .bf16, .qint8, .int4
)

// Pipeline with specific quantization
let pipeline = Flux2Pipeline(
    model: .klein9B,
    quantization: custom
)
try await pipeline.loadModels()
```

Measured transformer memory per quantization level:

| Model | bf16 | qint8 (-47%) | int4 (-72%) |
|-------|------|-------------|-------------|
| Klein 4B | 7.4 GB | 3.9 GB | 2.1 GB |
| Klein 9B | 17.3 GB | 9.2 GB | 4.9 GB |
| Dev (32B) | 61.5 GB | 32.7 GB | 17.3 GB |

## Klein 9B with 8-bit Quantization

Klein 9B provides better quality than Klein 4B. With qint8, it fits comfortably on 32GB+ Macs.

### Basic Klein 9B qint8 Usage

```swift
import Flux2Core

// Klein 9B with 8-bit quantization (balanced preset: 8bit text + qint8 transformer)
let pipeline = Flux2Pipeline(
    model: .klein9B,
    quantization: .balanced  // qint8 transformer (~9.2 GB) + 8bit Qwen3-8B (~8 GB)
)
try await pipeline.loadModels()

// Generate with Klein 9B's optimal parameters
let image = try await pipeline.generateTextToImage(
    prompt: "a beaver building a dam in a forest stream",
    height: 1024,
    width: 1024,
    steps: 4,        // Klein models are distilled — 4 steps is optimal
    guidance: 1.0    // Klein models use guidance 1.0
)
```

### Explicit qint8 Configuration for Klein 9B

```swift
import Flux2Core

// Explicit quantization: 8-bit text encoder + qint8 transformer
let quantConfig = Flux2QuantizationConfig(
    textEncoder: .mlx8bit,    // Qwen3-8B at 8-bit (~8 GB)
    transformer: .qint8       // Klein 9B transformer at qint8 (~9.2 GB, down from 17.3 GB bf16)
)

let pipeline = Flux2Pipeline(
    model: .klein9B,
    quantization: quantConfig
)
try await pipeline.loadModels()

let image = try await pipeline.generateTextToImage(
    prompt: "a detailed landscape with mountains and a lake",
    height: 1024,
    width: 1024,
    steps: 4,
    guidance: 1.0,
    seed: 42
)
```

### Klein 9B Memory Breakdown

The pipeline uses two-phase loading, so text encoder and transformer are never in memory simultaneously:

| Phase | Components | Memory |
|-------|-----------|--------|
| Phase 1 — Text Encoding | Qwen3-8B (8-bit) | ~8 GB |
| *(text encoder unloaded)* | | |
| Phase 2 — Generation | Klein 9B (qint8: 9.2 GB) + VAE (3 GB) | ~12 GB |
| **Peak** | **Maximum of phase 1 or phase 2** | **~12 GB** |

### Even More Memory-Efficient Klein 9B

```swift
// 4-bit text encoder + qint8 transformer for tighter memory
let tightConfig = Flux2QuantizationConfig(
    textEncoder: .mlx4bit,    // Qwen3-8B at 4-bit (~4 GB)
    transformer: .qint8       // Klein 9B at qint8 (~9.2 GB)
)

let pipeline = Flux2Pipeline(model: .klein9B, quantization: tightConfig)
try await pipeline.loadModels()
```

### Klein 9B with int4 for Minimum Memory

```swift
// int4 quantization reduces transformer memory by 72%
let minimalConfig = Flux2QuantizationConfig(
    textEncoder: .mlx4bit,    // Qwen3-8B at 4-bit (~4 GB)
    transformer: .int4        // Klein 9B at int4 (~4.9 GB, down from 17.3 GB)
)

let pipeline = Flux2Pipeline(model: .klein9B, quantization: minimalConfig)
try await pipeline.loadModels()

let image = try await pipeline.generateTextToImage(
    prompt: "a cat on a sunny windowsill",
    height: 1024,
    width: 1024,
    steps: 4,
    guidance: 1.0
)
```

## Text-to-Image

### Basic Generation

```swift
import Flux2Core

let pipeline = Flux2Pipeline(model: .dev)
try await pipeline.loadModels()

let image = try await pipeline.generateTextToImage(
    prompt: "a beautiful sunset over mountains",
    height: 1024,
    width: 1024,
    steps: 28,
    guidance: 4.0,
    seed: 42  // For reproducibility
)
```

### With Prompt Upsampling

```swift
// Get both image and the enhanced prompt
let result = try await pipeline.generateTextToImageWithResult(
    prompt: "a sunset",
    upsamplePrompt: true  // Enhance with Mistral/Qwen3
)

print("Original: \(result.originalPrompt)")
print("Enhanced: \(result.usedPrompt)")
let image = result.image
```

### With Progress and Checkpoints

```swift
let image = try await pipeline.generateTextToImage(
    prompt: "a beaver building a dam",
    height: 1024,
    width: 1024,
    steps: 4,
    guidance: 1.0,
    checkpointInterval: 2
) { currentStep, totalSteps in
    print("Step \(currentStep)/\(totalSteps)")
} onCheckpoint: { step, checkpointImage in
    saveImage(checkpointImage, to: "step_\(step).png")
}
```

## Image-to-Image

### Style Transfer

```swift
import Flux2Core

// Transform existing image
let image = try await pipeline.generateImageToImage(
    prompt: "transform into watercolor style",
    images: [referenceImage],
    strength: 0.7,  // 0.0 = preserve original, 1.0 = full transform
    steps: 28
)
```

### Multi-Image Conditioning

```swift
// Combine elements from multiple images
let image = try await pipeline.generateImageToImage(
    prompt: "Modify the cat on image 1 to wear the hat from image 2",
    images: [catImage, hatImage],  // Up to 4 images for Klein, 6 for Dev
    steps: 28,
    guidance: 4.0
)
```

### Image Interpretation (VLM)

```swift
// Use VLM to analyze an image and generate based on interpretation
let image = try await pipeline.generateTextToImage(
    prompt: "Describe what the red arrow is pointing at",
    interpretImage: mapImage,  // VLM analyzes this image
    height: 1024,
    width: 1024,
    steps: 28
)
```

## LoRA Adapters

### Loading a LoRA

```swift
import Flux2Core

let pipeline = Flux2Pipeline(model: .klein4B)
try await pipeline.loadModels()

// Load LoRA adapter
let loraConfig = LoRAConfig(
    filePath: "/path/to/lora.safetensors",
    scale: 1.0  // 0.5-1.5 typical range
)
try pipeline.loadLoRA(loraConfig)

// Generate with LoRA applied
let image = try await pipeline.generateTextToImage(
    prompt: "a photo of sks cat toy",
    height: 1024,
    width: 1024,
    steps: 4,
    guidance: 1.0
)
```

### LoRA with Activation Keyword

```swift
var config = LoRAConfig(filePath: "/path/to/object-removal.safetensors")
config.scale = 1.1
config.activationKeyword = "RMVOBJ"

try pipeline.loadLoRA(config)
```

## LoRA Training

Train custom LoRA adapters on Apple Silicon:

```swift
import Flux2Core

// Configure training
let config = LoRATrainingConfig(
    datasetPath: URL(fileURLWithPath: "examples/cat-toy/train"),
    rank: 32,
    alpha: 32.0,
    learningRate: 1e-4,
    maxSteps: 250,
    batchSize: 1,
    resolution: 512,
    gradientCheckpointing: true,  // Reduces memory ~50%
    outputPath: URL(fileURLWithPath: "output/my-lora")
)

// Start training
let trainer = SimpleLoRATrainer(config: config, model: .klein4B)
try await trainer.train { step, loss, gradNorm in
    print("Step \(step): loss=\(loss), gradNorm=\(gradNorm)")
}
```

### Training YAML Config

```yaml
model: klein-4b
dataset:
  path: examples/cat-toy/train
  trigger_word: "sks"
lora:
  rank: 32
  alpha: 32.0
  target_layers: all
training:
  max_steps: 250
  learning_rate: 1e-4
  resolution: [512]
  gradient_checkpointing: true
validation:
  enabled: true
  every_n_steps: 50
  prompt: "a photo of sks cat toy on a beach"
```

```bash
flux2 train-lora --config my_training.yaml
```

### Training Control

```bash
# Pause/resume training
touch output/my-lora/.pause
rm output/my-lora/.pause

# Force checkpoint save
touch output/my-lora/.checkpoint

# Stop gracefully
touch output/my-lora/.stop
```

## Memory Management

### Two-Phase Loading Architecture

The pipeline **never loads text encoder and transformer simultaneously**. This is the key to fitting large models on limited RAM:

1. **Phase 1 — Text Encoding**: Load text encoder → encode prompt → unload text encoder → free memory
2. **Phase 2 — Image Generation**: Load transformer + VAE → denoise → decode → output image

This means peak memory is the **maximum** of phase 1 or phase 2, not their sum.

### Total Memory Requirements per Model

| Model | Quantization | Phase 1 (Text) | Phase 2 (Transformer + VAE) | Peak Memory |
|-------|-------------|----------------|----------------------------|-------------|
| Klein 4B | int4 | ~4 GB | ~5 GB | ~8 GB |
| Klein 4B | qint8 | ~4 GB | ~7 GB | ~10 GB |
| Klein 4B | bf16 | ~8 GB | ~10 GB | ~16 GB |
| Klein 9B | int4 | ~4 GB | ~8 GB | ~16 GB |
| Klein 9B | qint8 | ~8 GB | ~12 GB | ~24 GB |
| Klein 9B | bf16 | ~8 GB | ~20 GB | ~32 GB |
| Dev (32B) | int4 | ~14 GB | ~20 GB | ~32 GB |
| Dev (32B) | qint8 | ~25 GB | ~36 GB | ~64 GB |
| Dev (32B) | bf16 | ~25 GB | ~65 GB | ~96 GB |

### Recommended Quantization for Available RAM

```swift
import Flux2Core

// Automatic recommendation based on system RAM
let memoryManager = Flux2MemoryManager.shared
print("System RAM: \(memoryManager.physicalMemoryGB) GB")

let recommended = memoryManager.recommendedConfig()
let pipeline = Flux2Pipeline(model: .klein9B, quantization: recommended)
```

| Available RAM | Recommended Model | Recommended Quantization |
|---------------|-------------------|--------------------------|
| 8–16 GB | Klein 4B | `.ultraMinimal` (4bit text + int4 transformer) |
| 16–24 GB | Klein 4B or Klein 9B | `.memoryEfficient` (4bit text + qint8 transformer) |
| 24–48 GB | Klein 9B | `.balanced` (8bit text + qint8 transformer) |
| 48–64 GB | Dev | `.memoryEfficient` (4bit text + qint8 transformer) |
| 64–96 GB | Dev | `.balanced` (8bit text + qint8 transformer) |
| 96+ GB | Dev | `.highQuality` (bf16 text + bf16 transformer) |

### Handling Out-of-Memory (OOM) Scenarios

```swift
import Flux2Core

// Strategy 1: Use more aggressive quantization
let pipeline = Flux2Pipeline(
    model: .klein4B,
    quantization: .ultraMinimal  // 4bit text + int4 transformer
)

// Strategy 2: Use memory optimization during denoising
let pipeline = Flux2Pipeline(
    model: .klein9B,
    quantization: .balanced,
    memoryOptimization: .aggressive  // Periodic eval + cache clearing during denoising
)

// Strategy 3: Reduce image resolution (memory scales with pixel count)
let image = try await pipeline.generateTextToImage(
    prompt: "a landscape",
    height: 512,   // 4x less memory than 1024x1024
    width: 512,
    steps: 4,
    guidance: 1.0
)

// Strategy 4: Clear GPU cache between generations
pipeline.clearCacheEveryNSteps = 2  // More frequent cache clearing
await pipeline.clearAll()           // Full cleanup between generations
```

### MemoryOptimizationConfig for Denoising

Controls how often the computation graph is evaluated during denoising steps, trading speed for memory:

```swift
import Flux2Core

// Presets (from fastest to most memory-efficient):
let light = MemoryOptimizationConfig.light           // Every 16 blocks
let moderate = MemoryOptimizationConfig.moderate      // Every 8 blocks (recommended)
let aggressive = MemoryOptimizationConfig.aggressive  // Every 4 blocks + cache clearing
let ultraLow = MemoryOptimizationConfig.ultraLowMemory // Every 2 blocks + cache clearing

let pipeline = Flux2Pipeline(
    model: .dev,
    quantization: .balanced,
    memoryOptimization: .aggressive
)
```

| System RAM | Recommended Preset |
|---|---|
| < 32 GB | `.ultraLowMemory` |
| 32–64 GB | `.aggressive` |
| 64–96 GB | `.moderate` |
| 96+ GB | `.light` or `.disabled` |

### Monitoring GPU Memory

```swift
import MLX

// Real-time Metal GPU memory stats
print("Active: \(MLX.Memory.activeMemory / 1_073_741_824) GB")
print("Peak: \(MLX.Memory.peakMemory / 1_073_741_824) GB")
print("Cache: \(MLX.Memory.cacheMemory / 1_073_741_824) GB")

// Set GPU cache limit
MLX.Memory.cacheLimit = 2 * 1_073_741_824  // 2 GB cache limit

// Clear GPU cache
MLX.Memory.clearCache()

// Reset peak counter before a specific operation
MLX.Memory.peakMemory = 0
// ... run generation ...
print("Operation peak: \(MLX.Memory.peakMemory / 1_073_741_824) GB")
```

### Pre-Generation Memory Checks

```swift
import Flux2Core

let memoryManager = Flux2MemoryManager.shared

// Check before text encoding phase
let textCheck = memoryManager.checkTextEncodingPhase(config: quantization)
if !textCheck.isOK {
    print("Warning: \(textCheck.suggestion)")
}

// Check before generation phase
let genCheck = memoryManager.checkImageGenerationPhase(config: quantization)
if !genCheck.isOK {
    print("Warning: \(genCheck.suggestion)")
}

// Check image dimensions
let sizeCheck = memoryManager.checkImageSize(width: 1024, height: 1024)
```

### Training Memory Optimization

For LoRA training, gradient checkpointing reduces activation memory by ~50% at the cost of ~2x slower forward passes:

```swift
let config = LoRATrainingConfig(
    datasetPath: datasetURL,
    gradientCheckpointing: true,  // Reduces memory ~50%, ~2x slower
    outputPath: outputURL
)
```

| Model | Without Checkpointing | With Checkpointing | Reduction |
|-------|----------------------|-------------------|-----------|
| Klein 4B | ~60 GB | ~42 GB | -30% |
| Klein 9B | N/A (OOM) | ~94 GB | Required |

## CLI Usage

```bash
# Install pre-built binary
unzip Flux2CLI-v3.2.0-macOS.zip   # executable + *.bundle resources, keep them together

# Text-to-Image (Klein 4B - fast, commercial OK)
flux2 t2i "a beaver building a dam" --model klein-4b

# Text-to-Image (Dev - maximum quality)
flux2 t2i "a beautiful sunset" --model dev --steps 28 --output sunset.png

# Image-to-Image with strength
flux2 i2i "transform into watercolor" --images photo.jpg --strength 0.7

# Multi-image conditioning
flux2 i2i "cat wearing this hat" --images cat.jpg --images hat.jpg

# With LoRA adapter
flux2 t2i "a photo of sks" --lora my-lora.safetensors --lora-scale 1.0

# Memory-efficient with int4 quantization
flux2 t2i "landscape painting" --transformer-quant int4 --text-quant 4bit

# With prompt upsampling
flux2 t2i "a cat" --upsample-prompt --model klein-4b

# Download models
flux2 download --model klein-4b
flux2 download --model dev --hf-token hf_xxxxx

# System info
flux2 info
```

## Model Downloading

```swift
import Flux2Core

// Models are downloaded automatically on first use
let pipeline = Flux2Pipeline(model: .klein4B)
try await pipeline.loadModels { component, progress in
    print("Downloading \(component): \(Int(progress * 100))%")
}

// Check if models are available
let isAvailable = ModelRegistry.isDownloaded(.transformer(.klein4B_bf16))
```

## Troubleshooting

Common issues and solutions:

```swift
// Out of memory → use more aggressive quantization
let pipeline = Flux2Pipeline(
    model: .klein4B,
    quantization: .ultraMinimal  // 4bit text + int4 transformer (~30GB)
)

// For training, enable gradient checkpointing to halve activation memory
let config = LoRATrainingConfig(
    datasetPath: datasetURL,
    gradientCheckpointing: true,  // Reduces memory ~50%, ~2x slower
    outputPath: outputURL
)

// Gated model access error → provide HuggingFace token
let pipeline = Flux2Pipeline(model: .dev, hfToken: "hf_xxxxx")
```

## Hardware Requirements

| Model | Minimum RAM | Recommended |
|-------|-------------|-------------|
| Klein 4B (int4) | 8 GB | 16 GB+ |
| Klein 4B (qint8) | 16 GB | 32 GB+ |
| Klein 9B (qint8) | 32 GB | 48 GB+ |
| Dev (qint8) | 64 GB | 96 GB+ |
| Dev (int4) | 32 GB | 64 GB+ |
