# VLM API Reference

The framework's image-aware services — FLUX.2 image description, 0-100
scene/style scoring, BFL prompt rewriting for the inpainting/outpainting
chains, training captions — run on a **VLM provider**. Two are available:

| Provider | Weights | Target to link | Notes |
|---|---|---|---|
| **Qwen3.5 4B** (default) | ~3 GB 4-bit / ~5 GB 8-bit | `FluxTextEncoders` (bundled) | Historical default; nothing to opt into |
| **Gemma 4 E2B-it** | ~3.6 GB 4-bit / **~4.2 GB 6-bit** / ~5.2 GB 8-bit / ~10 GB bf16 | `FluxGemma4VLM` (opt-in) | Runs on [gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx); brings `mlx-swift-lm` into the graph, hence the separate target |

Both answer the *same* system prompts and are parsed by the same code — see
[Providers](#providers). The sections below document the Qwen3.5-specific
entry points, which remain unchanged.

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

// Load whichever provider is active first
try await FluxVLM.active.ensureLoaded()

let prompt = try await setupAPI.describeReferenceForValidation(
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

## Providers

### Choosing one

`FluxVLM.active` is what every enrichment call uses. With nothing registered it
is the bundled Qwen3.5, so an app that never touches this API keeps its previous
behaviour exactly.

```swift
import FluxTextEncoders
import FluxGemma4VLM   // opt-in target

// Load Gemma 4 E2B-it 6-bit (downloads on first use) and take over from Qwen3.5
try await FluxGemma4VLM.activate(variant: .e2b6bit)

// …every enrichment call now runs on Gemma:
let text = try await FluxVLM.active.describeImageForFlux(image: photo)
let scores = try await FluxVLM.active.compareImagesForFlux(reference: ref, generated: gen)

// Hand the seat back (and free the weights)
await FluxGemma4VLM.deactivate()
```

Sandboxed apps that manage their own weights directory pass a path instead of a
variant, and can point the Gemma cache at their shared models folder (the layout
inside is `{org}/{model}`, the same HuggingFace owner namespace the FLUX
checkpoints use):

```swift
FluxGemma4VLM.setModelsDirectory(URL(fileURLWithPath: "…/FluxforgeStudio/Models"))
try await FluxGemma4VLM.activate(
    modelPath: URL(fileURLWithPath: "…/Models/mlx-community/gemma-4-e2b-it-6bit")
)
```

For the training paths (LoRA evaluation, VLM-guided checkpoint selection), which
load and unload the VLM around each generation phase, register without loading:

```swift
FluxGemma4VLM.register(variant: .e2b6bit)   // loaded on first use via ensureLoaded()
```

### The provider protocol

```swift
public protocol FluxVLMProvider: AnyObject, Sendable {
    var displayName: String { get }
    var isLoaded: Bool { get }
    func ensureLoaded() async throws
    func unload() async
    func generateText(
        images: [CGImage],          // 0 = text-only, 1 = analysis, 2 = comparison
        prompt: String,
        systemPrompt: String?,
        enableThinking: Bool,
        maxTokens: Int,
        temperature: Float
    ) async throws -> String
}
```

That is the whole contract. The system prompts, the JSON score parsing, the
FLUX.2 description rubric and the BFL rewriting rules live in
`FluxTextEncoders` as protocol extensions (`describeImageForFlux`,
`compareImagesForFlux`, `analyzeImage`, `generate`), so a new provider inherits
all of them and only has to know how to run a forward.

Registering a third VLM is therefore:

```swift
final class MyVLMProvider: FluxVLMProvider { /* four members */ }
FluxVLM.register(MyVLMProvider())
```

### What the chains do when nothing is loaded

Unchanged and load-bearing: `enrichPromptWithVLM: true` with no VLM resident
logs a warning and falls back to the caller's verbatim prompt. The chains never
auto-load a VLM, whichever provider is active.

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

### Gemma 4 provider

```bash
# Same three modes as test-qwen35, on Gemma 4 E2B-it (6-bit by default)
flux2 test-gemma4 "What do you see?" --image photo.png
flux2 test-gemma4 "Describe" --image photo.png --flux-describe
flux2 test-gemma4 "Compare" --image ref.png --image2 gen.png --compare

# Variant / local weights / custom cache
flux2 test-gemma4 "Hello" --variant 4bit
flux2 test-gemma4 "Hello" --model-path ~/Models/mlx-community/gemma-4-e2b-it-6bit
flux2 test-gemma4 "Hello" --models-dir ~/Pictures/FluxforgeStudio/Models

# Image-aware prompt rewriting on either provider
flux2 inpaint  -i in.jpg -m mask.png -p "replace the cat with a duck" -o out.png \
  --enrich-prompt-with-vlm --vlm-provider gemma4 --gemma4-variant 6bit
flux2 outpaint -i in.jpg --right 384 -p "…" -o out.png \
  --enrich-prompt-with-vlm --vlm-provider qwen35 --qwen35-variant 8bit
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
