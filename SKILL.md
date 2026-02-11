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
    .package(url: "https://github.com/VincentGourbin/flux-2-swift-mlx.git", branch: "main")
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

Requires macOS 14+ and Apple Silicon (M1/M2/M3/M4). Build with Xcode (not `swift build`) for Metal GPU support.

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

The pipeline uses two-phase loading — text encoder loads first (then unloads), then transformer + VAE load for generation. This allows running on machines with less RAM than the total model size.

```swift
import Flux2Core

// Check system memory
let memoryManager = Flux2MemoryManager.shared
print("Physical RAM: \(memoryManager.physicalMemoryGB) GB")

// Get recommended quantization for available RAM
let recommended = memoryManager.recommendedConfig()

// Create pipeline with memory-optimized settings
let pipeline = Flux2Pipeline(
    model: .klein4B,
    quantization: recommended
)

// Clear memory when done
await pipeline.clearAll()
```

## CLI Usage

```bash
# Install pre-built binary
unzip Flux2CLI-v1.0.1-macOS.zip

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
