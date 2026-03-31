# Qwen3.5 VLM API Reference

Native Qwen3.5-4B Vision-Language Model running locally on Apple Silicon. Auto-downloaded (~3GB 4-bit, ~5GB 8-bit) on first use.

## Quick Start

```swift
import FluxTextEncoders

// Load VLM (auto-downloads if needed)
let downloader = TextEncoderModelDownloader()
let path = try await downloader.downloadQwen35(variant: .qwen35_4B_4bit)
try await FluxTextEncoders.shared.loadQwen35VLM(from: path.path)

// Analyze an image
let result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
    image: myCGImage,
    prompt: "What do you see?"
)
print(result.text)

// Unload when done
await MainActor.run { FluxTextEncoders.shared.unloadQwen35VLM() }
```

## APIs

### 1. Image Analysis (free-form)

Analyze any image with a custom prompt and optional system prompt.

```swift
let result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
    image: cgImage,
    prompt: "Describe the architecture in this photo",
    systemPrompt: "You are an architecture expert",  // optional
    enableThinking: false,  // skip reasoning, faster response
    maxTokens: 300,
    temperature: 0
)
// result.text = "The building features a neo-classical facade..."
```

**File path variant:**
```swift
let result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
    path: "/path/to/photo.png",
    prompt: "What is this?"
)
```

### 2. Text Generation (no image)

```swift
let result = try FluxTextEncoders.shared.generateWithQwen35(
    prompt: "What is the capital of France?",
    enableThinking: false,
    maxTokens: 50,
    temperature: 0
)
// result.text = "The capital of France is Paris."
```

### 3. FLUX.2 Image Description

Describes an image optimized for FLUX.2 regeneration — covers both **scene** (what is depicted) and **style** (how it looks). Thinking mode is disabled automatically.

```swift
let result = try FluxTextEncoders.shared.describeImageForFlux(
    image: cgImage,
    context: "Focus on the person's face"  // optional
)
// result.text = "A young man with short brown hair and rectangular
//   black-rimmed glasses, wearing a light blue t-shirt, soft natural
//   lighting, shallow depth of field..."
```

### 4. Image Comparison (0-100 scores)

Compare two images on **scene** (content fidelity) and **style** (visual fidelity). Returns structured scores. Thinking disabled automatically.

```swift
let comparison = try FluxTextEncoders.shared.compareImagesForFlux(
    reference: refImage,
    generated: genImage
)
print("Scene: \(comparison.sceneScore)/100")  // e.g. 65
print("Style: \(comparison.styleScore)/100")  // e.g. 85
print("Reason: \(comparison.sceneReason)")
```

**File path variant:**
```swift
let comparison = try FluxTextEncoders.shared.compareImagesForFlux(
    referencePath: "ref.png",
    generatedPath: "gen.png"
)
```

**Score rubric (0-100):**

| Range | Meaning |
|-------|---------|
| 90-100 | Identical |
| 70-89 | Same subject/style, minor differences |
| 50-69 | Similar concept, clearly different details |
| 30-49 | Same general theme, substantially different |
| 0-29 | Completely different |

### 5. Multi-Image (advanced)

Pass multiple images to the VLM in a single forward pass.

```swift
guard let vlm = FluxTextEncoders.shared.qwen35VLMForEvaluation else { return }
let result = try vlm.generateMultiImage(
    images: [image1, image2, image3],
    prompt: "Compare these three photos",
    enableThinking: false,
    maxTokens: 500,
    temperature: 0
)
```

## LoRA Training APIs

### Pre-Training Evaluation

Evaluate the gap between a reference image and the base model output, then recommend training parameters.

```swift
import Flux2Core

let context = LoRAContext(
    name: "Vincent",
    description: "A specific person with glasses and brown hair"
)

let evaluator = LoRAEvaluator()
let evaluation = try await evaluator.evaluate(
    referenceImage: refImage,
    context: context,
    model: .klein4B
) { progress in
    print(progress)
}

// Results
print("Scene: \(evaluation.sceneScore)/100")      // e.g. 45
print("Style: \(evaluation.styleScore)/100")       // e.g. 85
print("Trigger word: \(evaluation.triggerWord)")    // e.g. "sks"
print("Steps: \(evaluation.recommendation.steps)") // e.g. 1000
```

### Complete Training Setup (end-to-end)

Chains everything: reference photo → VLM describe → evaluate baseline → recommend → generate YAML.

```swift
let setupAPI = LoRATrainingSetup_API()
let setup = try await setupAPI.createEvaluatedTrainingConfig(
    referenceImagePath: "/path/to/photo.jpg",
    context: LoRAContext(name: "Vincent", description: "A specific person"),
    model: .klein4B,
    datasetPath: "./my_dataset",
    triggerWord: "VinZ"
) { progress in
    print(progress)
}

// The validation prompt was auto-generated from the reference photo
print(setup.validationPrompt)
// "VinZ, young man with short brown hair and rectangular black-rimmed glasses..."

// Export YAML with VLM scoring at every checkpoint
let yaml = setup.recommendation.toYAMLWithVLMScoring(
    model: .klein4B,
    triggerWord: "VinZ",
    validationPrompt: setup.validationPrompt,
    referenceImagePath: "/path/to/photo.jpg",
    checkpointEvery: 50
)
try yaml.write(toFile: "training_config.yaml", atomically: true, encoding: .utf8)
```

The generated YAML includes VLM-supervised validation:

```yaml
model:
  name: klein-4b
  quantization: bf16

lora:
  rank: 32
  alpha: 32.0

training:
  max_steps: 1000
  learning_rate: 0.0001

validation:
  prompts:
    - prompt: "VinZ, young man with short brown hair and glasses..."
      apply_trigger: false
      is_512: true
  every_n_steps: 50
  vlm_scoring:
    enabled: true
    reference_images:
      - /path/to/photo.jpg
    save_best_checkpoint: true
    compare_to_baseline: true
```

### Describe Reference for Validation

Generate a validation prompt from a reference photo. Useful when setting up training manually.

```swift
let setupAPI = LoRATrainingSetup_API()

// Load VLM first
try await FluxTextEncoders.shared.loadQwen35VLM(from: vlmPath)

let prompt = try setupAPI.describeReferenceForValidation(
    image: refImage,
    triggerWord: "VinZ"
)
// "VinZ, close-up portrait of a young man with short brown hair..."
```

### VLM Scoring During Training

When VLM scoring is enabled in the training config, the trainer automatically:

1. **Step 0**: Scores baseline images (before any LoRA training)
2. **Each checkpoint**: Generates validation images with LoRA, compares vs reference
3. **Best checkpoint**: Auto-saves the checkpoint with highest composite score
4. **Early stopping** (optional): Stops training if scores plateau or degrade

Score progression example:
```
Step   0: 65/100 (scene: 45, style: 85)  ← baseline
Step  25: 68/100 (scene: 65, style: 70)  ← learning!
Step  50: 72/100 (scene: 70, style: 74)  ← improving
Step  75: 71/100 (scene: 68, style: 73)  ← plateau
Step 100: 73/100 (scene: 72, style: 74)  ← best checkpoint saved
```

## Thinking Mode

Qwen3.5 supports a thinking/reasoning mode (default: enabled). For scoring and comparison tasks, thinking is disabled automatically. For free-form analysis, you can control it:

```swift
// With thinking (default) — model reasons before answering
let result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
    image: img, prompt: "What is this?",
    enableThinking: true  // default
)
// result.text includes reasoning, then answer

// Without thinking — direct answer, faster
let result = try FluxTextEncoders.shared.analyzeImageWithQwen35(
    image: img, prompt: "What is this?",
    enableThinking: false
)
// result.text is just the answer
```

## CLI

```bash
# Image analysis
flux2 test-qwen35 "What do you see?" --image photo.png

# Without thinking (faster)
flux2 test-qwen35 "What do you see?" --image photo.png --no-think

# FLUX.2 description (thinking disabled automatically)
flux2 test-qwen35 "Describe" --image photo.png --flux-describe

# Compare two images (0-100 scores)
flux2 test-qwen35 "Compare" --image ref.png --image2 gen.png --compare

# Pre-training evaluation
flux2 evaluate-lora --image ref.png \
  --name "Vincent" \
  --lora-description "A specific person with glasses" \
  --model klein-4b --output-dir ./eval

# Model variant selection
flux2 test-qwen35 "Hello" --variant 8bit  # higher quality (5GB)
flux2 test-qwen35 "Hello" --variant 4bit  # faster, less memory (3GB)
```

## Performance

| Mode | Speed | Notes |
|------|-------|-------|
| Text generation | ~45 tok/s | 4-bit on M2 Ultra |
| Image analysis | ~30 tok/s | Single image |
| Image comparison | ~25 tok/s | Two images |
| With thinking | ~25 tok/s | Tokens spent reasoning |
| Without thinking | ~35 tok/s | Direct response |

## Memory

| Variant | Size | Peak GPU |
|---------|------|----------|
| 4-bit | ~3 GB | ~4 GB |
| 8-bit | ~5 GB | ~6 GB |

The VLM is loaded/unloaded between training phases to share memory with the transformer and VAE.
